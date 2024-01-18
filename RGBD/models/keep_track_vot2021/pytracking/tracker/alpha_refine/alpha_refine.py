import torch
import math
import time
import cv2
import numpy as np

from collections import defaultdict, OrderedDict

from pytracking.evaluation import Tracker
from pytracking.tracker.base import BaseTracker
from pytracking.utils.loading import load_network
import ltr.data.bounding_box_utils as bbutils



class TrackerAR(BaseTracker):
    # def __init__(self, tracker_name, para_name, threshold=0.65, sr=2.0, input_sz=None):
    #     if input_sz is None:
    #         input_sz = int(128 * sr)
    #     self.THRES = threshold
    #     tracker_info = Tracker(tracker_name, para_name, None)
    #     params = tracker_info.get_parameters()
    #     params.visualization = False
    #     params.debug = False
    #     params.visdom_info = {'use_visdom': False, 'server': '127.0.0.1', 'port': 8097}
    #     self.base_tracker = tracker_info.tracker_class(params)
    #
    #     self.alpha = RefineModule('SEx_beta', sr, input_sz=input_sz)


    def predicts_segmentation_mask(self):
        return self.params.get('produce_segmentation', False)

    def initialize_features(self):
        pass

    def initialize(self, image, info: dict) -> dict:
        self.frame_num = 1

        self.ths = self.params.get('threshold', 0.65)
        self.sr = self.params.get('sr', 2.0)
        self.input_sz = self.params.get('input_sz', int(128*self.sr))
        self.H, self.W, _ = image.shape

        tracker_info = Tracker(self.params.base_tracker_name, self.params.base_params_name, None)
        params = tracker_info.get_parameters()
        self.base_tracker = tracker_info.tracker_class(params)

        self.alpha = RefineModule('SEx_beta', self.sr, input_sz=self.input_sz)

        tic = time.time()

        _ = self.base_tracker.initialize(image, info)

        self.alpha.initialize(image, np.array(info['init_bbox']))

        self.logging_dict = defaultdict(list)

        out = {'time': time.time() - tic}
        return out

    def track(self, image, info: dict = None) -> dict:
        """ tracking pipeline """
        self.debug_info = {}
        self.frame_num += 1
        self.debug_info['frame_num'] = self.frame_num

        # Step0: run base tracker
        out = self.base_tracker.track(image)
        pred_bbox = out['target_bbox']

        if self.params.get('refine_bbox', False) and self.params.get('produce_segmentation', False):
            # feed through refiner with corner branch and then adapt tracker state according to new box
            # Step 1: refine tracking results with Alpha-Refine """
            refiner_out = self.alpha.refine(image, np.array(pred_bbox), branches=['corner', 'mask'])
            bbox_new = refiner_out['corner']

            # Step 2: update base tracker's state with refined result """
            self.update_base_tracker(bbox_new)

            pred_mask = refiner_out['mask']
            final_mask = (pred_mask > self.ths).astype(np.uint8)

            # Step 3: save results
            out['target_bbox'] = bbox_new.tolist()
            out['segmentation'] = final_mask
            out['segmentation_raw'] = pred_mask

        elif self.params.get('refine_bbox', False):
            # Step 1: refine tracking results with Alpha-Refine """
            refiner_out = self.alpha.refine(image, np.array(pred_bbox), branches=['corner'])
            bbox_new = refiner_out['corner']

            # Step 2: update base tracker's state with refined result """
            self.update_base_tracker(bbox_new)

            # Step 3: save results
            out['target_bbox'] = bbox_new.tolist()


        elif self.params.get('produce_segmentation', False):
            # Step1: Post-Process
            bbox_new = self.update_base_tracker(pred_bbox)

            # Step2: Mask report
            refiner_out = self.alpha.refine(image, np.array(bbox_new), branches=['mask'])
            pred_mask = refiner_out['mask']
            final_mask = (pred_mask > self.ths).astype(np.uint8)

            if self.params.get('compute_bbox_from_segmentation', False):
                bbox_new = mask2bbox(refiner_out['mask'], np.array(bbox_new)).tolist()
                self.update_base_tracker(pred_bbox)

            # Step 3: save results
            out['target_bbox'] = bbox_new
            out['segmentation'] = final_mask
            out['segmentation_raw'] = pred_mask

        else:
            raise NotImplementedError()

        if self.visdom is not None:
            if self.params.get('plot_iou', False):
                iou = torch.tensor(0.)
                if self.params.get('use_gt_box', False):
                    bbox_gth = self.frame_reader.get_bbox(self.frame_num - 1, None)
                    if np.all(np.logical_not(np.isnan(bbox_gth))) and np.all(bbox_gth >= 0) and bbox_gth is not None:
                        iou = bbutils.calc_iou(torch.FloatTensor(bbox_new), torch.from_numpy(bbox_gth))
                self.logging_dict['ious'].append(iou)
                # write debug info
                self.debug_info['IoU'] = iou
                self.debug_info['mIoU'] = np.mean(self.logging_dict['ious'])
                # plot debug data
                self.visdom.register(torch.tensor(self.logging_dict['ious']), 'lineplot', 3, 'IoU')

            self.visdom.register(self.debug_info, 'info_dict', 1, 'Status')

        return out


    def update_base_tracker(self, bbox):
        if not isinstance(bbox, list):
            bbox = bbox.tolist()

        x1, y1, w, h = bbox
        # add boundary and min size limit
        x1, y1, x2, y2 = bbox_clip(x1, y1, x1 + w, y1 + h, (self.H, self.W))
        w = x2 - x1
        h = y2 - y1
        new_pos = torch.from_numpy(np.array([y1 + h / 2, x1 + w / 2]).astype(np.float32))
        new_target_sz = torch.from_numpy(np.array([h, w]).astype(np.float32))
        new_scale = torch.sqrt(new_target_sz.prod() / self.base_tracker.base_target_sz.prod())

        # update base tracker's state
        self.base_tracker.pos = new_pos.clone()
        self.base_tracker.target_sz = new_target_sz
        self.base_tracker.target_scale = new_scale
        bbox_new = [x1, y1, w, h]
        return bbox_new


    def visdom_draw_tracking(self, image, box, segmentation=None):
        data = (image, box)

        if hasattr(self.base_tracker, 'search_area_box'):
            data += (self.base_tracker.search_area_box, )

        if self.params.get('use_gt_box', False):
            bbox_gth = self.frame_reader.get_bbox(self.frame_num - 1, None)

            if np.all(np.isnan(bbox_gth) == False):
                data += (torch.from_numpy(bbox_gth), )

        if segmentation is not None:
            data += (segmentation, )

        self.visdom.register(data, 'Tracking', 1, 'Tracking')

        # if hasattr(self.base_tracker, 'search_area_box'):
        #     if self.params.get('use_gt_box', False):
        #         bbox_gth = self.frame_reader.get_bbox(self.frame_num - 1, None)
        #
        #         if np.any(np.isnan(bbox_gth)):
        #             if segmentation is None:
        #                 self.visdom.register((image, box, self.base_tracker.search_area_box), 'Tracking', 1, 'Tracking')
        #             else:
        #                 self.visdom.register((image, box, self.base_tracker.search_area_box, segmentation), 'Tracking', 1, 'Tracking')
        #         else:
        #             if segmentation is None:
        #                 self.visdom.register((image, box, self.base_tracker.search_area_box, torch.from_numpy(bbox_gth)), 'Tracking', 1, 'Tracking')
        #             else:
        #                 self.visdom.register((image, box, self.base_tracker.search_area_box, torch.from_numpy(bbox_gth), segmentation), 'Tracking', 1, 'Tracking')
        #     else:
        #         if segmentation is None:
        #             self.visdom.register((image, box, self.base_tracker.search_area_box), 'Tracking', 1, 'Tracking')
        #         else:
        #             self.visdom.register((image, box, self.base_tracker.search_area_box, segmentation), 'Tracking', 1, 'Tracking')
        # else:
        #     if segmentation is None:
        #         self.visdom.register((image, box), 'Tracking', 1, 'Tracking')
        #     else:
        #         self.visdom.register((image, box, segmentation), 'Tracking', 1, 'Tracking')




class RefineModule(object):
    def __init__(self, refine_net_dir, search_factor=2.0, input_sz=256):
        self.refine_network = self.get_network(refine_net_dir)
        self.search_factor = search_factor
        self.input_sz = input_sz
        self.mean = np.array([0.485, 0.456, 0.406]).reshape((1, 1, 3))
        self.std = np.array([0.229, 0.224, 0.225]).reshape((1, 1, 3))

    def initialize(self, frame1, bbox1):
        """
        Args:
            frame1(np.array): cv2 iamge array with shape (H,W,3)
            bbox1(np.array): with shape(4,)
        """

        # Step1: get cropped patch(tensor)
        patch1, h_f, w_f = sample_target_SE(frame1, bbox1, self.search_factor, self.input_sz, mode=cv2.BORDER_CONSTANT)
        patch1_tensor = self.img_preprocess(patch1)

        # Step2: get GT's cooridinate on the cropped patch(tensor)
        crop_sz = torch.Tensor((self.input_sz, self.input_sz))
        bbox1_tensor = self.gt_preprocess(bbox1)  # (4,)
        bbox1_crop_tensor = transform_image_to_crop_SE(bbox1_tensor, bbox1_tensor, h_f, w_f, crop_sz).cuda()

        # Step3: forward prop (reference branch)
        with torch.no_grad():
            self.refine_network.forward_ref(patch1_tensor, bbox1_crop_tensor)

    def refine(self, Cframe, Cbbox, branches=None):
        """
        Args:
            Cframe: Current frame(cv2 array)
            Cbbox: Current bbox (ndarray) (x1,y1,w,h)
        """
        # if mode not in ['bbox', 'mask', 'corner', 'all']:
        #     raise ValueError("mode should be 'bbox' or 'mask' or 'corner' or 'all' ")
        if branches is None:
            branches = ['mask', 'corner']

        # Step1: get cropped patch (search region)
        Cpatch, h_f, w_f = sample_target_SE(Cframe, Cbbox, self.search_factor, self.input_sz, mode=cv2.BORDER_CONSTANT)
        Cpatch_tensor = self.img_preprocess(Cpatch)

        # Step2: forward prop (test branch)
        with torch.no_grad():
            output = self.refine_network.forward_test(Cpatch_tensor, mode='test', branches=branches)  # (1,1,H,W)

            if 'mask' in branches:
                Pmask_arr = self.pred2bbox(output, input_type='mask')
                mask = map_mask_back(Cframe, Cbbox, self.search_factor, Pmask_arr, mode=cv2.BORDER_CONSTANT)
                output['mask'] = mask

            if 'corner' in branches:
                Pbbox_arr = self.pred2bbox(output, input_type='corner')
                bbox = bbox_back(Pbbox_arr, Cbbox, h_f, w_f, self.search_factor)
                output['corner'] = bbox

        return output

    # def refine(self, Cframe, Cbbox, mode='all', test=False):
    #     """
    #     Args:
    #         Cframe: Current frame(cv2 array)
    #         Cbbox: Current bbox (ndarray) (x1,y1,w,h)
    #     """
    #     tic = time.time()
    #     if mode not in ['bbox', 'mask', 'corner', 'all']:
    #         raise ValueError("mode should be 'bbox' or 'mask' or 'corner' or 'all' ")
    #
    #     # Step1: get cropped patch (search region)
    #     Cpatch, h_f, w_f = sample_target_SE(Cframe, Cbbox, self.search_factor, self.input_sz, mode=cv2.BORDER_CONSTANT)
    #     Cpatch_tensor = self.img_preprocess(Cpatch)
    #
    #     # Step2: forward prop (test branch)
    #     output_dict = {}
    #     with torch.no_grad():
    #
    #         output = self.refine_network.forward_test(Cpatch_tensor, mode='test')  # (1,1,H,W)
    #
    #         if mode == 'bbox' or mode == 'corner':
    #             Pbbox_arr = self.pred2bbox(output, input_type=mode)
    #             output_dict[mode] = self.bbox_back(Pbbox_arr, Cbbox, h_f, w_f)
    #
    #         elif mode == 'mask':
    #             Pmask_arr = self.pred2bbox(output, input_type=mode)
    #             output_dict['mask'] = map_mask_back(Cframe, Cbbox, self.search_factor, Pmask_arr,
    #                                                 mode=cv2.BORDER_CONSTANT)
    #
    #         else:
    #             boxes = []
    #             box = [0, 0, 0, 0]
    #             if 'bbox' in output:
    #                 Pbbox_arr = self.pred2bbox(output, input_type='bbox')
    #                 output_dict['bbox'] = self.bbox_back(Pbbox_arr, Cbbox, h_f, w_f)
    #                 boxes.append(output_dict['bbox'])
    #                 box = output_dict['bbox']  # for mask absense
    #
    #             if 'corner' in output:
    #                 Pbbox_arr = self.pred2bbox(output, input_type='corner')
    #                 output_dict['corner'] = self.bbox_back(Pbbox_arr, Cbbox, h_f, w_f)
    #                 boxes.append(output_dict['corner'])
    #                 box = output_dict['corner']
    #
    #             if 'mask' in output:
    #                 Pmask_arr = self.pred2bbox(output, input_type='mask')
    #                 output_dict['mask'] = map_mask_back(Cframe, Cbbox, self.search_factor, Pmask_arr,
    #                                                     mode=cv2.BORDER_CONSTANT)
    #                 output_dict['mask_bbox'] = mask2bbox(output_dict['mask'], box)
    #                 boxes.append(output_dict['mask_bbox'])
    #
    #             if not isinstance(self.branch_selector, int):
    #                 branch_scores = self.branch_selector(output['feat'])
    #                 _, max_idx = torch.max(branch_scores.squeeze(), dim=0)
    #                 max_idx = max_idx.item()
    #             else:
    #                 max_idx = self.branch_selector
    #             output_dict['all'] = boxes[max_idx]
    #
    #     return output_dict if test else output_dict[mode]

    def pred2bbox(self, prediction, input_type=None):
        if input_type == 'bbox':
            Pbbox = prediction['bbox']
            Pbbox = delta2bbox(Pbbox)
            Pbbox_arr = np.array(Pbbox.squeeze().cpu())
            return Pbbox_arr

        elif input_type == 'corner':
            Pcorner = prediction['corner']  # (x1,y1,x2,y2)
            Pbbox_arr = np.array(Pcorner.squeeze().cpu())
            Pbbox_arr[2:] = Pbbox_arr[2:] - Pbbox_arr[:2]  # (x1,y1,w,h)
            return Pbbox_arr

        elif input_type == 'mask':
            Pmask = prediction['mask']
            Pmask_arr = np.array(Pmask.squeeze().cpu())  # (H,W) (0,1)
            return Pmask_arr

        else:
            raise ValueError("input_type should be 'bbox' or 'mask' or 'corner' ")

    def get_network(self, checkpoint_dir):
        network = load_network(checkpoint_dir)
        network.cuda()
        network.eval()
        return network

    def img_preprocess(self, img_arr):
        """ to torch.Tensor(RGB), normalized (minus mean, divided by std)
        Args:
            img_arr: (H,W,3)
        Return:
            (1,1,3,H,W)
        """
        norm_img = ((img_arr / 255.0) - self.mean) / (self.std)
        img_f32 = norm_img.astype(np.float32)
        img_tensor = torch.from_numpy(img_f32).cuda()
        img_tensor = img_tensor.permute((2, 0, 1))
        return img_tensor.unsqueeze(dim=0).unsqueeze(dim=0)

    def gt_preprocess(self, gt_arr):
        """
        Args:
            gt_arr: ndarray (4,)
        Return:
            `torch.Tensor` (4,)
        """
        return torch.from_numpy(gt_arr.astype(np.float32))


def rect_from_mask(mask):
    """
    create an axis-aligned rectangle from a given binary mask
    mask in created as a minimal rectangle containing all non-zero pixels
    """
    x_ = np.sum(mask, axis=0)
    y_ = np.sum(mask, axis=1)
    x0 = np.min(np.nonzero(x_))
    x1 = np.max(np.nonzero(x_))
    y0 = np.min(np.nonzero(y_))
    y1 = np.max(np.nonzero(y_))
    return [x0, y0, x1 - x0 + 1, y1 - y0 + 1]


def bbox_clip(x1, y1, x2, y2, boundary, min_sz=10):
    """boundary (H,W)"""
    x1_new = max(0, min(x1, boundary[1] - min_sz))
    y1_new = max(0, min(y1, boundary[0] - min_sz))
    x2_new = max(min_sz, min(x2, boundary[1]))
    y2_new = max(min_sz, min(y2, boundary[0]))
    return x1_new, y1_new, x2_new, y2_new

def sample_target_SE(im, target_bb, search_area_factor, output_sz=None, mode=cv2.BORDER_REPLICATE):
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

    x, y, w, h = target_bb.tolist()

    # Crop image
    ws = math.ceil(search_area_factor * w)
    hs = math.ceil(search_area_factor * h)

    if ws < 1 or hs < 1:
        raise Exception('Too small bounding box.')

    x1 = round(x + 0.5*w - ws*0.5)
    x2 = x1 + ws

    y1 = round(y + 0.5 * h - hs * 0.5)
    y2 = y1 + hs

    x1_pad = max(0, -x1)
    x2_pad = max(x2-im.shape[1]+1, 0)

    y1_pad = max(0, -y1)
    y2_pad = max(y2-im.shape[0]+1, 0)

    # Crop target
    im_crop = im[y1+y1_pad:y2-y2_pad, x1+x1_pad:x2-x2_pad, :]

    # Pad
    im_crop_padded = cv2.copyMakeBorder(im_crop, y1_pad, y2_pad, x1_pad, x2_pad, mode)

    if output_sz is not None:
        w_rsz_f = output_sz / ws
        h_rsz_f = output_sz / hs
        im_crop_padded_rsz = cv2.resize(im_crop_padded, (output_sz, output_sz))
        if len(im_crop_padded_rsz.shape)==2:
            im_crop_padded_rsz = im_crop_padded_rsz[...,np.newaxis]
        return im_crop_padded_rsz, h_rsz_f, w_rsz_f
    else:
        return im_crop_padded, 1.0, 1.0


def transform_image_to_crop_SE(box_in: torch.Tensor, box_extract: torch.Tensor, resize_factor_h: float, resize_factor_w: float,
                            crop_sz: torch.Tensor) -> torch.Tensor:
    """ Transform the box co-ordinates from the original image co-ordinates to the co-ordinates of the cropped image
    args:
        box_in - the box for which the co-ordinates are to be transformed
        box_extract - the box about which the image crop has been extracted.
        resize_factor - the ratio between the original image scale and the scale of the image crop
        crop_sz - size of the cropped image
    returns:
        torch.Tensor - transformed co-ordinates of box_in
    """
    box_extract_center = box_extract[0:2] + 0.5*box_extract[2:4]

    box_in_center = box_in[0:2] + 0.5*box_in[2:4]

    box_out_xc = (crop_sz[0] -1)/2 + (box_in_center[0] - box_extract_center[0])*resize_factor_w
    box_out_yc = (crop_sz[0] -1)/2 + (box_in_center[1] - box_extract_center[1])*resize_factor_h
    box_out_w = box_in[2] * resize_factor_w
    box_out_h = box_in[3] * resize_factor_h

    max_sz = crop_sz[0].item()
    box_out_x1 = torch.clamp(box_out_xc - 0.5 * box_out_w,0,max_sz)
    box_out_y1 = torch.clamp(box_out_yc - 0.5 * box_out_h,0,max_sz)
    box_out_x2 = torch.clamp(box_out_xc + 0.5 * box_out_w,0,max_sz)
    box_out_y2 = torch.clamp(box_out_yc + 0.5 * box_out_h,0,max_sz)
    box_out_w_new = box_out_x2 - box_out_x1
    box_out_h_new = box_out_y2 - box_out_y1
    box_out = torch.stack((box_out_x1, box_out_y1, box_out_w_new, box_out_h_new))
    return box_out


def map_mask_back(im, target_bb, search_area_factor, mask, mode=cv2.BORDER_REPLICATE):
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
    H,W = (im.shape[0], im.shape[1])
    base = np.zeros((H, W))
    x, y, w, h = target_bb.tolist()

    # Crop image
    ws = math.ceil(search_area_factor * w)
    hs = math.ceil(search_area_factor * h)

    if ws < 1 or hs < 1:
        raise Exception('Too small bounding box.')

    x1 = round(x + 0.5*w - ws*0.5)
    x2 = x1 + ws

    y1 = round(y + 0.5 * h - hs * 0.5)
    y2 = y1 + hs

    x1_pad = max(0, -x1)
    x2_pad = max(x2-im.shape[1]+1, 0)

    y1_pad = max(0, -y1)
    y2_pad = max(y2-im.shape[0]+1, 0)

    # pad base
    base_padded = cv2.copyMakeBorder(base, y1_pad, y2_pad, x1_pad, x2_pad, mode)
    # Resize mask
    mask_rsz = cv2.resize(mask,(ws,hs))
    # fill region with mask
    base_padded[y1+y1_pad:y2+y1_pad, x1+x1_pad:x2+x1_pad] = mask_rsz.copy()
    # crop base_padded to get final mask
    final_mask = base_padded[y1_pad:y1_pad+H, x1_pad:x1_pad+W]
    assert (final_mask.shape == (H,W))
    return final_mask


def delta2bbox(delta):
    bbox_cxcywh = delta.clone()
    # based on (128,128) center region
    bbox_cxcywh[:, :2] = 128.0 + delta[:, :2] * 128.0  # center offset
    bbox_cxcywh[:, 2:] = 128.0 * torch.exp(delta[:, 2:])  # wh revise
    bbox_xywh = bbox_cxcywh.clone()
    bbox_xywh[:, :2] = bbox_cxcywh[:, :2] - 0.5 * bbox_cxcywh[:, 2:]
    return bbox_xywh

def bbox_back(bbox_crop, bbox_ori, h_f, w_f, search_factor):
    """
    Args:
        bbox_crop: coordinate on (256x256) region in format (x1,y1,w,h) (4,)
        bbox_ori: origin traking result (x1,y1,w,h) (4,)
        h_f: h scale factor
        w_f: w scale factor
    Return:
        coordinate mapping back to origin image
    """
    x1_c, y1_c, w_c, h_c = bbox_crop.tolist()
    x1_o, y1_o, w_o, h_o = bbox_ori.tolist()
    x1_oo = x1_o - (search_factor - 1) / 2 * w_o
    y1_oo = y1_o - (search_factor - 1) / 2 * h_o
    delta_x1 = x1_c / w_f
    delta_y1 = y1_c / h_f
    delta_w = w_c / w_f
    delta_h = h_c / h_f
    return np.array([x1_oo + delta_x1, y1_oo + delta_y1, delta_w, delta_h])


def mask2bbox(mask, ori_bbox, MASK_THRESHOLD=0.5):
    target_mask = (mask > MASK_THRESHOLD)
    target_mask = target_mask.astype(np.uint8)
    if cv2.__version__[-5] == '4':
        contours, _ = cv2.findContours(target_mask,
                                       cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_NONE)
    else:
        _, contours, _ = cv2.findContours(target_mask,
                                          cv2.RETR_EXTERNAL,
                                          cv2.CHAIN_APPROX_NONE)
    cnt_area = [cv2.contourArea(cnt) for cnt in contours]
    if len(contours) != 0 and np.max(cnt_area) > 100:
        contour = contours[np.argmax(cnt_area)]
        polygon = contour.reshape(-1, 2)
        prbox = cv2.boundingRect(polygon)
    else:  # empty mask
        prbox = ori_bbox
    return np.array(prbox).astype(np.float)
