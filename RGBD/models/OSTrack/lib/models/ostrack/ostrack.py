import math
import os
from typing import List

import torch
from torch import nn
from torch.nn.modules.transformer import _get_clones

from lib.models.ostrack.head import build_box_head
from lib.models.ostrack.dynamic_vit import vit_base_patch16_224_prune, vit_large_patch16_224_prune
from lib.models.ostrack.score_head import ScoreTransformer
from lib.models.ostrack.vit import vit_base_patch16_224_in21k
from lib.utils.box_ops import box_xyxy_to_cxcywh


class OSTrack(nn.Module):
    """ This is the base class for Transformer Tracking """
    def __init__(self, transformer, box_head, hidden_dim, num_queries,
                 aux_loss=False, head_type="CORNER", use_cross_attn=False, merge_layer=0, cfg=None, cls_head=None, seg_head=None):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_queries: number of object queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        # self.transformer = transformer
        self.backbone = transformer
        self.box_head = box_head
        self.num_queries = num_queries
        self.merge_layer = merge_layer

        self.aux_loss = aux_loss
        self.head_type = head_type
        # if head_type == "CORNER" or head_type == "CENTER" or head_type == 'SPARSE' or head_type == 'FCOS' or head_type == 'MLPPlus' or head_type == 'TransT':
        self.feat_sz_s = int(box_head.feat_sz)
        self.feat_len_s = int(box_head.feat_sz ** 2)
        self.use_cross_trans = use_cross_attn

        self.cls_head = cls_head
        self.seg_head = seg_head

        if self.aux_loss:
            self.box_head = _get_clones(self.box_head, 6)

    def forward_backbone(self, x, image_type='search'):
        x = self.backbone.forward_single(x.tensors, merge_layer=self.merge_layer, image_type=image_type)
        return x

    def forward_cat(self, z, x):
        return self.backbone.forward_cat(z, x, merge_layer=self.merge_layer)

    def forward(self, template: List[torch.Tensor], search: torch.Tensor, gt_score_map=None, box_mask_z=None, keep_rate_t=None, keep_rate_s=None, mode='normal', gt_bbox_search=None, mask_z=None, is_distill=False):
        x, attn = self.backbone(template, search, merge_layer=self.merge_layer, box_mask_z=box_mask_z,
                                keep_rate_t=keep_rate_t, keep_rate_s=keep_rate_s, return_last_attn=True, mask_z=mask_z)

        if mode == 'seg':
            return self.forward_seg_head(x, gt_score_map, attn)

        # Forward head
        feat_last = x
        if isinstance(x, list):
            feat_last = x[-1]
        out, outputs_coord, score_map_ctr = self.forward_head(None, feat_last, None)

        if mode == 'cls':
            return self.forward_cls_head(x, out['pred_boxes'].squeeze(1))

        out['attn'] = attn
        return out, outputs_coord, x, score_map_ctr, None, None

    def forward_cls_head(self, cat_template_search, target_bbox_search):
        # TODO: encode target_bbox_search as the class token
        cls_score = self.cls_head(cat_template_search, target_bbox_search).view(-1)
        return {'pred_logits': cls_score}

    def forward_head(self, hs, memory, gt_score_map=None):
        """
        hs: output embeddings (1, B, N, C)
        memory: encoder embeddings (HW1+HW2, B, C)"""
        if self.head_type == "CORNER":
            if self.use_cross_trans:
                enc_opt = memory.transpose(0, 1)
            else:
                enc_opt = memory[:, -self.feat_len_s:]  # encoder output for the search region (B, HW, C)
            if hs is not None:
                dec_opt = hs.squeeze(0).transpose(1, 2)  # (B, C, N)
                att = torch.matmul(enc_opt, dec_opt)  # (B, HW, N)
                opt = (enc_opt.unsqueeze(-1) * att.unsqueeze(-2)).permute(
                    (0, 3, 2, 1)).contiguous()  # (B, HW, C, N) --> (B, N, C, HW), N is the query number
            else:
                opt = (enc_opt.unsqueeze(-1)).permute((0, 3, 2, 1)).contiguous()
            bs, Nq, C, HW = opt.size()
            opt_feat = opt.view(-1, C, self.feat_sz_s, self.feat_sz_s)
            # run the corner head
            pred_box, score_map = self.box_head(opt_feat, True)
            outputs_coord = box_xyxy_to_cxcywh(pred_box)
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map,
                   }
            return out, outputs_coord_new, None

        elif self.head_type == "CENTER":

            if self.use_cross_trans:
                enc_opt = memory.transpose(0, 1)
            else:
                enc_opt = memory[:, -self.feat_len_s:]  # encoder output for the search region (B, HW, C)
            if hs is not None:
                dec_opt = hs.squeeze(0).transpose(1, 2)  # (B, C, N)
                att = torch.matmul(enc_opt, dec_opt)  # (B, HW, N)
                opt = (enc_opt.unsqueeze(-1) * att.unsqueeze(-2)).permute(
                    (0, 3, 2, 1)).contiguous()  # (B, HW, C, N) --> (B, N, C, HW), N is the query number
            else:
                opt = (enc_opt.unsqueeze(-1)).permute((0, 3, 2, 1)).contiguous()
            bs, Nq, C, HW = opt.size()
            opt_feat = opt.view(-1, C, self.feat_sz_s, self.feat_sz_s)
            # run the corner head
            score_map_ctr, bbox, size_map, offset_map = self.box_head(opt_feat, gt_score_map)
            # outputs_coord = box_xyxy_to_cxcywh(bbox)
            outputs_coord = bbox
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new}
            out['size_map'] = size_map
            out['offset_map'] = offset_map
            return out, outputs_coord_new, score_map_ctr
        else:
            raise NotImplementedError


def build_ostrack(cfg, training=True):
    current_dir = os.path.dirname(os.path.abspath(__file__))  # This is your Project Root
    pretrained_path = os.path.join(current_dir, '../../../pretrained_models')
    if cfg.MODEL.PRETRAIN_FILE and ('ViTTracker' not in cfg.MODEL.PRETRAIN_FILE) and training:
        pretrained = os.path.join(pretrained_path, cfg.MODEL.PRETRAIN_FILE)
    else:
        pretrained = ''

    if cfg.MODEL.BACKBONE.TYPE == 'vit_base_patch16_224_in21k':
        backbone = vit_base_patch16_224_in21k(pretrained, drop_path_rate=cfg.MODEL.BACKBONE.DROP_PATH_RATE)
        hidden_dim = backbone.embed_dim

    elif cfg.MODEL.BACKBONE.TYPE == 'vit_base_patch16_224_prune':
        backbone = vit_base_patch16_224_prune(pretrained, drop_path_rate=cfg.MODEL.BACKBONE.DROP_PATH_RATE,
                                              pruning_loc=cfg.MODEL.BACKBONE.PRUNING_LOC,
                                              keep_ratio=cfg.MODEL.BACKBONE.KEEP_RATIO,
                                              pruning_loc_t=cfg.MODEL.BACKBONE.PRUNING_LOC_TEMPLATE,
                                              keep_ratio_t=cfg.MODEL.BACKBONE.KEEP_RATIO_TEMPLATE)
        hidden_dim = backbone.embed_dim

    elif cfg.MODEL.BACKBONE.TYPE == 'vit_large_patch16_224_prune':
        backbone = vit_large_patch16_224_prune(pretrained, drop_path_rate=cfg.MODEL.BACKBONE.DROP_PATH_RATE,
                                               pruning_loc=cfg.MODEL.BACKBONE.PRUNING_LOC,
                                               keep_ratio=cfg.MODEL.BACKBONE.KEEP_RATIO,
                                               pruning_loc_t=cfg.MODEL.BACKBONE.PRUNING_LOC_TEMPLATE,
                                               keep_ratio_t=cfg.MODEL.BACKBONE.KEEP_RATIO_TEMPLATE)

        hidden_dim = backbone.embed_dim

    else:
        raise NotImplementedError

    search_size = cfg.DATA.SEARCH.SIZE
    template_size = cfg.DATA.TEMPLATE.SIZE

    new_patch_size = cfg.MODEL.STRIDE
    backbone.finetune_track(search_size=[search_size, search_size],
                            template_size=[template_size, template_size],
                            det_token_num=cfg.MODEL.NUM_OBJECT_QUERIES,
                            mid_pe_search_size=None,
                            mid_pe_template_size=None,
                            use_checkpoint=False,
                            add_sep_seg=cfg.MODEL.BACKBONE.SEP_SEG,
                            cat_mode=cfg.MODEL.BACKBONE.CAT_MODE,
                            new_patch_size=new_patch_size,
                            fpn=False,
                            cfg=cfg,
                            )

    if cfg.MODEL.EXTRA_MERGER:
        hidden_dim = cfg.MODEL.HIDDEN_DIM

    cls_head = None
    if cfg.TRAIN.TRAIN_CLS:
        cls_head = ScoreTransformer(
            n_cls=1,
            n_layers=cfg.MODEL.HEAD.NUM_CLS_ATTN_LAYERS,
            d_model=768,
            d_encoder=768,
            n_heads=12,
            n_mlp_layers=cfg.MODEL.HEAD.NUM_CLS_MLP_LAYERS,
        )

    box_head = build_box_head(cfg, hidden_dim)
    model = OSTrack(
        backbone,
        box_head,
        hidden_dim=hidden_dim,
        num_queries=cfg.MODEL.NUM_OBJECT_QUERIES,
        aux_loss=cfg.TRAIN.DEEP_SUPERVISION,
        head_type=cfg.MODEL.HEAD_TYPE,
        merge_layer=cfg.MODEL.BACKBONE.MERGE_LAYER,
        cfg=cfg,
        cls_head=cls_head,
        seg_head=None,
    )

    if 'OSTrack' in cfg.MODEL.PRETRAIN_FILE and training:
        checkpoint = torch.load(cfg.MODEL.PRETRAIN_FILE, map_location="cpu")
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint["net"], strict=False)
        print('Load pretrained model from: ' + cfg.MODEL.PRETRAIN_FILE)

        if cfg.TRAIN.TRAIN_CLS:
            model = copy_weights(model, checkpoint["net"])

    return model


def copy_weights(model, weight_dict):
    last_block_weights = {}
    for key, value in weight_dict.items():
        if 'blocks.11' in key:
            key_new_0 = key.replace('11', '0').replace('backbone', 'cls_head')
            key_new_1 = key.replace('11', '1').replace('backbone', 'cls_head')
            last_block_weights[key_new_0] = value
            last_block_weights[key_new_1] = value
    missing_keys, unexpected_keys = model.load_state_dict(last_block_weights, strict=False)
    return model
