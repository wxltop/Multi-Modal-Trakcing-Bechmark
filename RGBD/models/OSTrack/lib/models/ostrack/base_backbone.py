import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.models.ostrack.layers import trunc_normal_
from lib.models.ostrack.layers.patch_embed import PatchEmbed
from lib.models.ostrack.utils import combine_tokens, recover_tokens


class BaseBackbone(nn.Module):
    def __init__(self):
        super().__init__()

        self.pos_embed = None
        self.img_size = [224, 224]
        self.patch_size = 16
        self.embed_dim = 384

        self.cat_mode = 'direct'
        self.det_token_num = 0
        self.det_token = None

        self.pos_embed_z = None
        self.pos_embed_x = None
        self.pos_embed_det = None

        self.add_sep_seg = False
        self.template_segment_pos_embed = None
        self.search_segment_pos_embed = None

        self.ape = True
        self.add_fpn = False
        self.fpn_stage = [2, 5, 8, 11]

        self.multi_head = False
        self.return_inter = False

        self.msg = False

        self.add_cls_token = False

    def finetune_track(self, search_size=[320, 320], template_size=[128, 128], det_token_num=1,
                       mid_pe_search_size=None, mid_pe_template_size=None, use_checkpoint=False, add_sep_seg=False, cat_mode='direct', new_patch_size=16,fpn=False,cfg=None):
        self.fpn_stage = cfg.MODEL.FPN_STAGES
        self.cat_mode = cat_mode

        # assert new_patch_size == self.patch_size
        # resize patch embedding
        if new_patch_size != self.patch_size:
            print('Inconsistent Patch Size, Interpolate The Weight!')
            old_patch_embed = {}
            for name, param in self.patch_embed.named_parameters():
                if 'weight' in name:
                    param = nn.functional.interpolate(param, size=(new_patch_size, new_patch_size),
                                                      mode='bicubic', align_corners=False)
                    param = nn.Parameter(param)
                old_patch_embed[name] = param
            self.patch_embed = PatchEmbed(img_size=self.img_size, patch_size=new_patch_size, in_chans=3,
                                          embed_dim=self.embed_dim)
            self.patch_embed.proj.bias = old_patch_embed['proj.bias']
            self.patch_embed.proj.weight = old_patch_embed['proj.weight']

        # for det/tracking token
        self.det_token_num = det_token_num
        self.det_token = nn.Parameter(torch.zeros(1, det_token_num, self.embed_dim))
        self.det_token = trunc_normal_(self.det_token, std=.02)
        det_pos_embed = torch.zeros(1, det_token_num, self.embed_dim)
        det_pos_embed = trunc_normal_(det_pos_embed, std=.02)
        self.pos_embed_det = nn.Parameter(det_pos_embed)

        # patch embedding for search and template
        if self.ape:
            # for patch embedding
            patch_pos_embed = self.pos_embed
            patch_pos_embed = patch_pos_embed.transpose(1, 2)
            B, E, Q = patch_pos_embed.shape
            P_H, P_W = self.img_size[0] // self.patch_size, self.img_size[1] // self.patch_size
            patch_pos_embed = patch_pos_embed.view(B, E, P_H, P_W)

            # for search region
            H, W = search_size
            new_P_H, new_P_W = H // new_patch_size, W // new_patch_size
            search_patch_pos_embed = nn.functional.interpolate(patch_pos_embed, size=(new_P_H, new_P_W), mode='bicubic',
                                                        align_corners=False)
            search_patch_pos_embed = search_patch_pos_embed.flatten(2).transpose(1, 2)

            # for template region
            H, W = template_size
            new_P_H, new_P_W = H // new_patch_size, W // new_patch_size
            template_patch_pos_embed = nn.functional.interpolate(patch_pos_embed, size=(new_P_H, new_P_W), mode='bicubic',
                                                        align_corners=False)
            template_patch_pos_embed = template_patch_pos_embed.flatten(2).transpose(1, 2)

            # self.pos_embed = torch.nn.Parameter(
            #     torch.cat((cls_pos_embed, template_patch_pos_embed, search_patch_pos_embed, det_pos_embed), dim=1))

            self.pos_embed_z = nn.Parameter(template_patch_pos_embed)
            self.pos_embed_x = nn.Parameter(search_patch_pos_embed)
        # else:
        #     # force absolute position embedding, since template and search patches concated
        #     self.ape = True
        #     # for search region
        #     H, W = search_size
        #     new_P_H, new_P_W = H // 4, W // 4  # 4 for SwinT
        #     num_patches = new_P_H * new_P_W
        #     self.pos_embed_x = nn.Parameter(torch.zeros(1, num_patches, self.embed_dim))
        #     trunc_normal_(self.pos_embed_x, std=.02)
        #
        #     # for template region
        #     H, W = template_size
        #     new_P_H, new_P_W = H // 4, W // 4
        #     num_patches = new_P_H * new_P_W
        #     self.pos_embed_z = nn.Parameter(torch.zeros(1, num_patches, self.embed_dim))
        #     trunc_normal_(self.pos_embed_z, std=.02)

        # separate token and segment token
        self.add_sep_seg = add_sep_seg
        if self.add_sep_seg:
            # self.sep_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
            # self.sep_token = trunc_normal_(self.sep_token, std=.02)
            self.template_segment_pos_embed = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
            self.template_segment_pos_embed = trunc_normal_(self.template_segment_pos_embed, std=.02)
            self.search_segment_pos_embed = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
            self.search_segment_pos_embed = trunc_normal_(self.search_segment_pos_embed, std=.02)

        self.patch_size = new_patch_size

    def forward_features(self, z, x, self_layer=12, mask_z=None, mask_x=None, box_mask_z=None, keep_rate=None, return_last_attn=False):
        B, H, W = x.shape[0], x.shape[2], x.shape[3]

        x = self.patch_embed(x)
        z = self.patch_embed(z)

        # attention mask handling
        # B, H, W
        if mask_z is not None and mask_x is not None:
            mask_z = F.interpolate(mask_z[None].float(), scale_factor=1. / self.patch_size).to(torch.bool)[0]
            mask_z = mask_z.flatten(1).unsqueeze(-1)

            mask_x = F.interpolate(mask_x[None].float(), scale_factor=1. / self.patch_size).to(torch.bool)[0]
            mask_x = mask_x.flatten(1).unsqueeze(-1)

            mask_x = combine_tokens(mask_z, mask_x, mode=self.cat_mode)
            mask_x = mask_x.squeeze(-1)

        if mask_z is not None:
            # remove the surrounding tokens around the target box in the template
            pass

        if self.det_token_num > 0:
            det_token = self.det_token.expand(B, -1, -1)

        if self.add_cls_token:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            cls_tokens = cls_tokens + self.cls_pos_embed

        # add pos embedding
        # pos_embed_z = resize_pos_embed(self.pos_embed, z, num_tokens=0, gs_new=())
        # pos_embed_x = resize_pos_embed(self.pos_embed, x, num_tokens=0, gs_new=())
        # # pos_embed_z = self.pos_embed_z
        # # pos_embed_x = self.pos_embed_x
        # z += pos_embed_z
        # x += pos_embed_x

        z += self.pos_embed_z
        x += self.pos_embed_x

        if self.add_sep_seg:
            x += self.search_segment_pos_embed
            z += self.template_segment_pos_embed

        x = combine_tokens(z, x, mode=self.cat_mode)
        if self.add_cls_token:
            x = torch.cat([cls_tokens, x], dim=1)

        x = self.pos_drop(x)

        for i, blk in enumerate(self.blocks):
            if not return_last_attn:
                x = blk(x, mask_x)
            else:
                x, attn = blk(x, mask_x, True)

        lens_z = self.pos_embed_z.shape[1]
        lens_x = self.pos_embed_x.shape[1]
        x = recover_tokens(x, lens_z, lens_x, mode=self.cat_mode)

        if self.cls_mode == 'project' and self.add_cls_token:
            x = self.project_module(x)

        if return_last_attn:
            return self.norm(x), attn

        return self.norm(x), None

    def forward(self, z, x, merge_layer=0, mask_z=None, mask_x=None, box_mask_z=None, keep_rate_t=None, keep_rate_s=None, return_last_attn=False):
        x, attn = self.forward_features(z, x, mask_z=mask_z, mask_x=mask_x, box_mask_z=box_mask_z,
                                        keep_rate=keep_rate_s, return_last_attn=return_last_attn)

        return x, attn
