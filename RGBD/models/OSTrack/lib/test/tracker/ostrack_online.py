import math
import numpy as np
from torchvision.ops import box_iou

from lib.models.ostrack.ostrack import build_ostrack
from lib.test.evaluation import Tracker
from lib.test.tracker.basetracker import BaseTracker
import torch
import pickle
import uuid
import torch.nn.functional as F

from lib.test.utils.hann import hann2d
from lib.test.utils.psr import psr_dynamic
from lib.test.utils.vis_token_mask import gen_visualization
from lib.train.actors.ostrack import generate_bbox_mask, generate_mask_cond
from lib.train.data.processing_utils import sample_target
# for debug
import cv2
import os

from lib.test.tracker.img_utils import Preprocessor
from lib.utils.box_ops import clip_box, box_cxcywh_to_xyxy, box_xywh_to_xyxy


class OSTrackOnline(BaseTracker):
    def __init__(self, params, dataset_name):
        super(OSTrackOnline, self).__init__(params)
        network = build_ostrack(params.cfg, training=False)
        network.load_state_dict(torch.load(self.params.checkpoint, map_location='cpu')['net'], strict=True)
        self.cfg = params.cfg
        self.network = network.cuda()
        self.network.eval()
        self.preprocessor = Preprocessor()
        self.state = None

        self.feat_sz = self.cfg.TEST.SEARCH_SIZE // self.cfg.MODEL.STRIDE
        # motion constrain
        self.output_window = hann2d(torch.tensor([self.feat_sz, self.feat_sz]).long(), centered=True).cuda()
        self.head_type = self.cfg.MODEL.HEAD_TYPE

        # for segmentation
        self.enable_seg = self.cfg.TRAIN.TRAIN_SEG
        self.enable_cls = self.cfg.TRAIN.TRAIN_CLS

        self.enable_redetect = self.cfg.TEST.REDETECT
        self.redetect_th = self.cfg.TEST.REDE_THRESH

        self.vis_attn = False

        # parameters used for template update
        self.enable_online_template = True
        self.online_templates = []
        self.update_intervals = 1
        self.online_template_nums = 1
        self.max_score_decay = 1.0

        self.main_lobe_score_ratio_thresh = 0.2
        self.main_lobe_area_threshold = self.cfg.TEST.MAIN_LOB_AREA_THR
        self.cls_threshold = 0.5

        # for debug
        self.debug = params.debug
        self.use_visdom = params.debug
        self.frame_id = 0
        if self.debug:
            if not self.use_visdom:
                self.save_dir = "debug"
                if not os.path.exists(self.save_dir):
                    os.makedirs(self.save_dir)
            else:
                if self.vis_attn:
                    self.add_hook()
                self._init_visdom(None, 1)
        # for save boxes from all queries
        self.save_all_boxes = params.save_all_boxes
        self.z_dict1 = {}

    def initialize(self, image, info: dict):
        # forward the template once
        z_patch_arr, resize_factor, z_amask_arr = sample_target(image, info['init_bbox'], self.params.template_factor,
                                                    output_sz=self.params.template_size)
        self.z_patch_arr = z_patch_arr
        template = self.preprocessor.process(z_patch_arr, z_amask_arr)
        with torch.no_grad():
            self.z_dict1 = template

        if self.enable_online_template:
            for i in range(self.online_template_nums):
                self.online_templates.append(template.tensors.clone())

        self.box_mask_z = None
        if 'prune' in self.cfg.MODEL.BACKBONE.TYPE:
            template_bbox = self.transform_bbox_to_crop(info['init_bbox'], resize_factor,
                                                        template.tensors.device).squeeze(1)
            self.box_mask_z = generate_mask_cond(self.cfg, 1, template.tensors.device, template_bbox)

        # save states
        self.state = info['init_bbox']
        self.frame_id = 0
        if self.save_all_boxes:
            '''save all predicted boxes'''
            all_boxes_save = info['init_bbox'] * self.cfg.MODEL.NUM_OBJECT_QUERIES
            return {"all_boxes": all_boxes_save}

    def track(self, image, info: dict = None, f1_reg=False):
        H, W, _ = image.shape
        self.frame_id += 1
        x_patch_arr, resize_factor, x_amask_arr = sample_target(image, self.state, self.params.search_factor,
                                                                output_sz=self.params.search_size)  # (x1, y1, w, h)
        search = self.preprocessor.process(x_patch_arr, x_amask_arr)

        templates = self.z_dict1.tensors
        search_input = search.tensors
        box_mask_z = self.box_mask_z
        if self.enable_online_template:
            templates = [templates, *self.online_templates]
            templates = torch.cat(templates, dim=0)
            search_input = search_input.expand(2, -1, -1, -1)
            if box_mask_z is not None:
                box_mask_z = box_mask_z.expand(2, -1)
        with torch.no_grad():
            out_dict_f1, _, cat_template_search, pred_score_map_f1, pred_motion, offset_map = self.network.forward(
                template=templates, search=search_input, box_mask_z=box_mask_z)

            pred_seg = None
            final_mask = None
            if self.enable_seg:
                seg_dict = self.network.forward_seg_head(cat_template_search, pred_score_map * self.output_window, out_dict['attn'])
                pred_mask = seg_dict['pred_masks']
                pred_seg = seg_dict['pred_masks']

                pred_mask = self.map_mask_back(pred_mask, image, resize_factor)
                final_mask = (pred_mask > 0.5).astype(np.uint8)

            cat_template_search = cat_template_search.mean(dim=0, keepdim=True)
            out_dict_merge, outputs_coord_merge, score_map_merge = self.network.forward_head(None, cat_template_search)
            pred_score_map = score_map_merge

            conf_score = 1.
            if self.enable_cls:
                # pred_box = out_dict_f1['pred_boxes'][0:1].squeeze(1)
                pred_box = self.network.box_head.cal_bbox(self.output_window * pred_score_map,
                                                          out_dict_merge['size_map'],
                                                          out_dict_merge['offset_map']).squeeze(1)
                # pred_box = torch.zeros_like(pred_box)
                # pred_box = self.network.box_head.cal_bbox(pred_score_map_f1 * self.output_window,
                #                                             out_dict_f1['size_map'],
                #                                             out_dict_f1['offset_map'])[0:1].squeeze(1)
                cls_dict = self.network.forward_cls_head(cat_template_search[0:1], pred_box)
                conf_score = cls_dict["pred_logits"].view(-1).sigmoid().item()
                # print(conf_score)

        update_previous_template = True
        psr, main_lobe_area, peak = psr_dynamic(score_map_merge.squeeze().cpu().numpy(), self.main_lobe_score_ratio_thresh)
        if main_lobe_area > self.main_lobe_area_threshold or conf_score < self.cls_threshold:
            update_previous_template = False

        if self.head_type == 'CORNER':
            pred_boxes = out_dict_merge['pred_boxes']
        else:
            # add hann windows
            self.window_influence = 0.49  # 0.176
            # response = (1 - self.window_influence) * pred_score_map + self.window_influence * self.output_window
            # response = pred_score_map * self.output_window * pred_motion
            # response = self.output_window * pred_motion
            response = self.output_window * pred_score_map
            pred_boxes = self.network.box_head.cal_bbox(response, out_dict_merge['size_map'], out_dict_merge['offset_map'])
            if f1_reg:
                pred_boxes = self.network.box_head.cal_bbox(response, out_dict_f1['size_map'][0:1], out_dict_f1['offset_map'][0:1])
            if self.enable_redetect:
                _, idx = torch.max(response.flatten(1), dim=1, keepdim=True)
                tracked_score = pred_score_map_f1[0:1].flatten(1)[0, idx]
                # print(tracked_score.item())
                if tracked_score.item() <= self.redetect_th:
                    self.pause_mode = True
                    pred_boxes = self.network.box_head.cal_bbox(pred_score_map_f1[0:1] * self.output_window, out_dict_f1['size_map'][0:1], out_dict_f1['offset_map'][0:1])

        pred_boxes = pred_boxes.view(-1, 4)
        # Baseline: Take the mean of all pred boxes as the final result
        pred_box = (pred_boxes.mean(
            dim=0) * self.params.search_size / resize_factor).tolist()  # (cx, cy, w, h) [0,1]
        # get the final box result
        self.state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)
        if self.enable_seg:
            final_mask = self.constraint_mask(final_mask, self.state)

        if self.enable_online_template and self.debug:
            for i in range(len(self.online_templates)):
                temple_i = self.online_templates[i]
                self.visdom.register(self.preprocessor.inverse_process(temple_i), 'image', 1, 'template_' + str(i))

        if self.enable_online_template and update_previous_template:
            z_patch_arr, _, z_amask_arr = sample_target(image, self.state, self.params.template_factor,
                                                        output_sz=self.params.template_size)  # (x1, y1, w, h)
            template_t = self.preprocessor.process(z_patch_arr, z_amask_arr)
            self.online_templates.pop()
            self.online_templates.append(template_t.tensors)

        # for debug
        if self.debug:
            if not self.use_visdom:
                x1, y1, w, h = self.state
                image_BGR = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                cv2.rectangle(image_BGR, (int(x1),int(y1)), (int(x1+w),int(y1+h)), color=(0,0,255), thickness=2)
                save_path = os.path.join(self.save_dir, "%04d.jpg" % self.frame_id)
                cv2.imwrite(save_path, image_BGR)
            else:
                caption = "Area: {:d}, Score: {:.2f}".format(main_lobe_area, conf_score)
                # self.visdom.register((image, info['gt_bbox'].tolist(), self.state, outputs_ref['target_bbox']), 'Tracking', 1, 'Tracking', caption=caption)
                self.visdom.register((image, info['gt_bbox'].tolist(), self.state), 'Tracking', 1, 'Tracking', caption=caption)

                self.visdom.register(torch.from_numpy(x_patch_arr).permute(2, 0, 1), 'image', 1, 'search_region')
                self.visdom.register(torch.from_numpy(self.z_patch_arr).permute(2, 0, 1), 'image', 1, 'template')
                self.visdom.register(pred_score_map.view(self.feat_sz, self.feat_sz), 'heatmap', 1, 'score_map')
                self.visdom.register((pred_score_map * self.output_window).view(self.feat_sz, self.feat_sz), 'heatmap', 1, 'score_map_hann')

                self.visdom.register(pred_score_map_f1[0].view(self.feat_sz, self.feat_sz), 'heatmap', 1, 'score_map_f1')
                self.visdom.register(pred_score_map_f1[1].view(self.feat_sz, self.feat_sz), 'heatmap', 1, 'score_map_fprev')

                while self.pause_mode:
                    if self.step:
                        self.step = False
                        break

        if self.save_all_boxes:
            '''save all predictions'''
            all_boxes = self.map_box_back_batch(pred_boxes * self.params.search_size / resize_factor, resize_factor)
            all_boxes_save = all_boxes.view(-1).tolist()  # (4N, )
            return {"target_bbox": self.state,
                    "all_boxes": all_boxes_save}
        else:
            return {"target_bbox": self.state,
                    "mask_pred": final_mask,
                    'update_flag': update_previous_template,
                    }

    def map_box_back(self, pred_box: list, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

    def map_box_back_batch(self, pred_box: torch.Tensor, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box.unbind(-1) # (N,4) --> (N,)
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return torch.stack([cx_real - 0.5 * w, cy_real - 0.5 * h, w, h], dim=-1)

    def constraint_mask(self, mask, bbox):
        """
        mask: shape (H, W)
        bbox: list [x1, y1, w, h]
        """
        x1 = np.int(np.floor(bbox[0]))
        y1 = np.int(np.floor(bbox[1]))
        x2 = np.int(np.ceil(bbox[0]+bbox[2]))
        y2 = np.int(np.ceil(bbox[1]+bbox[3]))
        mask[0:y1+1,:] = 0
        mask[y2:,:] = 0
        mask[:,0:x1+1] = 0
        mask[:,x2:] = 0
        return mask

    def map_mask_back(self, segmentation_scores, im, sample_scale, mode=cv2.BORDER_CONSTANT):
        """ Extracts a crop centered at target_bb box, of size search_area_factor times target_bb(Both height and width)

        args:
            im - cv image
            target_bb - target box [x, y, w, h]
            search_area_factor - Ratio of crop size to target size
            output_sz - (float) Size to which the extracted crop is resized (always square). If None, no resizing is done.

        returns:
            cv image - extracted crop
            float - the factor by which the crop has been resized to make the crop size equal output_size
        """
        H, W = (im.shape[0], im.shape[1])
        base = np.zeros((H, W))
        x_center, y_center = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        s_x = math.ceil(self.params.search_size / sample_scale)
        mask = segmentation_scores.squeeze().cpu().numpy()

        # Crop image

        if s_x < 1 or s_x < 1:
            raise Exception('Too small bounding box.')
        c = (s_x + 1) / 2

        x1 = int(np.floor(x_center - c + 0.5))
        x2 = int(x1 + s_x - 1)

        y1 = int(np.floor(y_center - c + 0.5))
        y2 = int(y1 + s_x -1)

        x1_pad = int(max(0., -x1))
        y1_pad = int(max(0., -y1))
        x2_pad = int(max(0., x2 - W + 1))
        y2_pad = int(max(0., y2 - H + 1))

        '''pad base'''
        base_padded = cv2.copyMakeBorder(base, y1_pad, y2_pad, x1_pad, x2_pad, mode)
        '''Resize mask'''
        mask_rsz = cv2.resize(mask, (s_x, s_x))
        '''fill region with mask'''
        base_padded[y1 + y1_pad:y2 + y1_pad + 1, x1 + x1_pad:x2 + x1_pad + 1] = mask_rsz.copy()
        '''crop base_padded to get final mask'''
        final_mask = base_padded[y1_pad:y1_pad + H, x1_pad:x1_pad + W]
        assert (final_mask.shape == (H, W))
        return final_mask

    def add_hook(self):
        conv_features, enc_attn_weights, dec_attn_weights = [], [], []

        for i in range(12):
            self.network.backbone.blocks[i].attn.register_forward_hook(
                # lambda self, input, output: enc_attn_weights.append(output[1])
                lambda self, input, output: enc_attn_weights.append(output[1])
            )

        self.enc_attn_weights = enc_attn_weights


def get_tracker_class():
    return OSTrackOnline
