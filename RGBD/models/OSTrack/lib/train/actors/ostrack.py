from . import BaseActor
from lib.utils.misc import NestedTensor, adjust_keep_rate, nested_tensor_from_tensor_list
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy, box_xywh_to_cxcywh, box_xyxy_to_cxcywh
import torch
from lib.utils.merge import merge_template_search, list_template_search, merge_channel

import torch.nn.functional as F

import torch
import numpy as np

from ...utils.heapmap_utils import generate_heatmap, get_pred


def generate_bbox_mask(bbox_mask, bbox):
    b, h, w = bbox_mask.shape
    for i in range(b):
        bbox_i = bbox[i].cpu().tolist()
        bbox_mask[i, int(bbox_i[1]):int(bbox_i[1] + bbox_i[3] - 1), int(bbox_i[0]):int(bbox_i[0] + bbox_i[2] - 1)] = 1
    return bbox_mask


def generate_mask_cond(cfg, bs, device, gt_bbox):
    template_size = cfg.DATA.TEMPLATE.SIZE
    stride = cfg.MODEL.STRIDE
    template_feat_size = template_size // stride

    if cfg.MODEL.BACKBONE.TEMPLATE_RANGE == 'ALL':
        box_mask_z = None
    elif cfg.MODEL.BACKBONE.TEMPLATE_RANGE == 'CTR_POINT':
        if template_feat_size == 8:
            index = slice(3, 4)
        elif template_feat_size == 12:
            index = slice(5, 6)
        elif template_feat_size == 7:
            index = slice(3, 4)
        elif template_feat_size == 14:
            index = slice(6, 7)
        else:
            raise NotImplementedError
        box_mask_z = torch.zeros([bs, template_feat_size, template_feat_size], device=device)
        box_mask_z[:, index, index] = 1
        box_mask_z = box_mask_z.flatten(1).to(torch.bool)
    elif cfg.MODEL.BACKBONE.TEMPLATE_RANGE == 'CTR_REC':
        # use fixed 4x4 region, 3:5 for 8x8
        # use fixed 4x4 region 5:6 for 12x12
        if template_feat_size == 8:
            index = slice(3, 5)
        elif template_feat_size == 12:
            index = slice(5, 7)
        elif template_feat_size == 7:
            index = slice(3, 4)
        else:
            raise NotImplementedError
        box_mask_z = torch.zeros([bs, template_feat_size, template_feat_size], device=device)
        box_mask_z[:, index, index] = 1
        box_mask_z = box_mask_z.flatten(1).to(torch.bool)

    elif cfg.MODEL.BACKBONE.TEMPLATE_RANGE == 'GT_BOX':
        box_mask_z = torch.zeros([bs, template_size, template_size])
        # box_mask_z_ori = data['template_seg'][0].view(-1, 1, *data['template_seg'].shape[2:])  # (batch, 1, 128, 128)
        box_mask_z = generate_bbox_mask(box_mask_z, gt_bbox * template_size).unsqueeze(1).to(torch.float)  # (batch, 1, 128, 128)
        # box_mask_z_vis = box_mask_z.cpu().numpy()
        box_mask_z = F.interpolate(box_mask_z, scale_factor=1. / cfg.MODEL.STRIDE, mode='bilinear',
                                   align_corners=False)
        box_mask_z = box_mask_z.flatten(1).to(torch.bool)
        # box_mask_z_vis = box_mask_z[:, 0, ...].cpu().numpy()
        # gaussian_maps_vis = generate_heatmap(data['template_anno'], self.cfg.DATA.TEMPLATE.SIZE, self.cfg.MODEL.STRIDE)[0].cpu().numpy()
    else:
        raise NotImplementedError

    return box_mask_z


class OSTrackActor(BaseActor):
    """ Actor for training the STARK-S and STARK-ST(Stage1)"""
    def __init__(self, net, objective, loss_weight, settings, cfg=None):
        super().__init__(net, objective)
        self.loss_weight = loss_weight
        self.settings = settings
        self.bs = self.settings.batchsize  # batch size
        self.cfg = cfg
        self.train_cls = cfg.TRAIN.TRAIN_CLS
        self.train_seg = cfg.TRAIN.TRAIN_SEG

    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'template', 'search', 'gt_bbox'.
            template_images: (N_t, batch, 3, H, W)
            search_images: (N_s, batch, 3, H, W)
        returns:
            loss    - the training loss
            status  -  dict containing detailed losses
        """
        # forward pass
        out_dict, gt_score_map, pred_score_map, pred_motion, offset = self.forward_pass(data)

        # process the groundtruth
        gt_bboxes = data['search_anno']  # (Ns, batch, 4) (x1,y1,w,h)

        if self.train_cls:
            labels = data['label'].view(-1)  # (batch, ) 0 or 1
            loss, status = self.compute_losses_cls(out_dict, labels)
            return loss, status

        # get training target
        # cls_scores = out_dict['cls_pred']
        loss, status = self.compute_losses(out_dict, gt_bboxes[-1], gt_score_map, pred_score_map, pred_motion, offset,
                                           gt_offset=None)

        return loss, status

    def forward_pass(self, data):
        # gt gaussian map
        gaussian_maps = generate_heatmap(data['search_anno'], self.cfg.DATA.SEARCH.SIZE, self.cfg.MODEL.STRIDE)

        template_list = []
        for i in range(self.settings.num_template):
            template_img_i = data['template_images'][i].view(-1, *data['template_images'].shape[2:])  # (batch, 3, 128, 128)
            # template_att_i = data['template_att'][i].view(-1, *data['template_att'].shape[2:])  # (batch, 128, 128)
            template_list.append(template_img_i)

        # template_img = data['template_images'][0].view(-1, *data['template_images'].shape[2:])  # (batch, 3, 128, 128)
        # template_att = data['template_att'][0].view(-1, *data['template_att'].shape[2:])  # (batch, 128, 128)
        search_img = data['search_images'][0].view(-1, *data['search_images'].shape[2:])  # (batch, 3, 320, 320)
        # search_att = data['search_att'][0].view(-1, *data['search_att'].shape[2:])  # (batch, 320, 320)
        box_mask_z = None

        keep_rate = None
        keep_rate_template = None
        if 'prune' in self.cfg.MODEL.BACKBONE.TYPE:
            if self.cfg.MODEL.BACKBONE.KEEP_RATIO:
                box_mask_z = generate_mask_cond(self.cfg, template_list[0].shape[0], template_list[0].device, data['template_anno'][0])

                shrink_start_epoch = 20
                shrink_epochs = 80
                # shrink_epochs = 50
                keep_rate = adjust_keep_rate(data['epoch'], warmup_epochs=shrink_start_epoch,
                                             total_epochs=shrink_start_epoch + shrink_epochs,
                                             ITERS_PER_EPOCH=1, base_keep_rate=self.cfg.MODEL.BACKBONE.KEEP_RATIO[0])

            if self.cfg.MODEL.BACKBONE.KEEP_RATIO_TEMPLATE:
                shrink_start_epoch_template = 120
                shrink_epochs_template = 80
                keep_rate_template = adjust_keep_rate(data['epoch'], warmup_epochs=shrink_start_epoch_template,
                                             total_epochs=shrink_start_epoch_template + shrink_epochs_template,
                                             ITERS_PER_EPOCH=1, base_keep_rate=self.cfg.MODEL.BACKBONE.KEEP_RATIO_TEMPLATE[0])

        keep_rate = None
        keep_rate_template = None

        if len(template_list) == 1:
            template_list = template_list[0]

        if self.train_cls:
            search_bboxes = box_xywh_to_xyxy(data['search_anno'][0].clone())
            out_dict = self.net(template=template_list, search=search_img, mode='cls', gt_bbox_search=search_bboxes)
            pred_score_map, pred_motion, offset = None, None, None
        elif self.train_seg:
            out_dict = self.net(template=template_list, search=search_img, mode='seg', gt_score_map=gaussian_maps[-1].unsqueeze(1))
            pred_score_map, pred_motion, offset = None, None, None
        else:
            out_dict, _, _, pred_score_map, pred_motion, offset = self.net(template=template_list,
                                                                           search=search_img,
                                                                           gt_score_map=None,
                                                                           box_mask_z=box_mask_z,
                                                                           keep_rate_t=keep_rate_template,
                                                                           keep_rate_s=keep_rate)
                                                                           # gt_score_map=gaussian_maps[0])

        return out_dict, gaussian_maps[-1].unsqueeze(1), pred_score_map, pred_motion, offset

    def compute_losses(self, pred_dict, gt_bbox, gt_score_map, pred_score_map, pred_motion=None,
                       offset=None, gt_offset=None, return_status=True):
        # Get boxes
        pred_boxes = pred_dict['pred_boxes']
        if torch.isnan(pred_boxes).any():
            raise ValueError("Network outputs is NAN! Stop Training")
        num_queries = pred_boxes.size(1)
        pred_boxes_vec = box_cxcywh_to_xyxy(pred_boxes).view(-1, 4)  # (B,N,4) --> (BN,4) (x1,y1,x2,y2)
        gt_boxes_vec = box_xywh_to_xyxy(gt_bbox)[:, None, :].repeat((1, num_queries, 1)).view(-1, 4).clamp(min=0.0, max=1.0)  # (B,4) --> (B,1,4) --> (B,N,4)
        # compute giou and iou
        try:
            giou_loss, iou = self.objective['giou'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
        except:
            giou_loss, iou = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()
        # compute l1 loss
        l1_loss = self.objective['l1'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)

        # compute location loss
        if pred_score_map is not None:
            location_loss = self.objective['focal'](pred_score_map, gt_score_map)
        else:
            location_loss = torch.tensor(0.0, device=l1_loss.device)

        # weighted sum
        loss = self.loss_weight['giou'] * giou_loss + self.loss_weight['l1'] * l1_loss + self.loss_weight['focal'] * location_loss

        if return_status:
            # status for log
            mean_iou = iou.detach().mean()
            status = {"Loss/total": loss.item(),
                      "Loss/giou": giou_loss.item(),
                      "Loss/l1": l1_loss.item(),
                      "Loss/location": location_loss.item(),
                      # "Loss/wh": wh_loss.item(),
                      # "Loss/offset": offset_loss.item(),
                      "IoU": mean_iou.item()}
            return loss, status
        else:
            return loss

    def compute_losses_cls(self, pred_dict, labels, return_status=True):
        loss = self.loss_weight["cls"] * self.objective['cls'](pred_dict["pred_logits"].view(-1), labels)
        if return_status:
            # status for log
            status = {
                "cls_loss": loss.item()}
            return loss, status
        else:
            return loss
