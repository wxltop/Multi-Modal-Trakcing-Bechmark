from pytracking.tracker.base import BaseTracker
from pytracking.features.preprocessing import numpy_to_torch
from collections import OrderedDict
import pytracking.tracker.stm.model as stm_model
from pytracking.features import augmentation
from pytracking.features.preprocessing import sample_patch_multiscale, sample_patch_transformed, sample_patch

import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torchvision import models


class STM(BaseTracker):

    multiobj_mode = 'parallel'

    def predicts_segmentation_mask(self):
        return True

    def initialize_features(self):
        if not getattr(self, 'features_initialized', False):
            self.params.device = 'cuda'
            model = nn.DataParallel(stm_model.STM())

            if torch.cuda.is_available():
                model.to(self.params.device)
            model.eval()  # turn-off BN

            pth_path = self.params.network_path
            model.load_state_dict(torch.load(pth_path))

            self.model = model

        self.features_initialized = True

    def initialize(self, image, info: dict) -> dict:
        self.frame_num = 1


        self.K = 2
        self.max_mem_sz = self.params.max_mem_sz
        num_frames = 10000       # TODO fix this

        tic = time.time()

        self.initialize_features()
        self.to_memorize = [int(i) for i in np.arange(1, num_frames, step=self.params.memory_skip_rate)]

        if 'object_ids' not in info:
            info['object_ids'] = [1, ]

        self.num_objects = len(info['object_ids'])
        self.object_ids = info['object_ids']

        state = info['init_bbox']
        self.prev_output_state = state
        self.pos = torch.Tensor([state[1] + (state[3] - 1) / 2, state[0] + (state[2] - 1) / 2])
        self.target_sz = torch.Tensor([state[3], state[2]])

        sz = self.params.image_sample_size
        self.img_sample_sz = torch.Tensor([sz, sz] if isinstance(sz, int) else sz)
        self.img_support_sz = self.img_sample_sz

        # Set search area
        search_area = torch.prod(self.target_sz * self.params.search_area_scale).item()
        self.target_scale = math.sqrt(search_area) / self.img_sample_sz.prod().sqrt()

        # Target size in base scale
        self.base_target_sz = self.target_sz / self.target_scale
        self.params.scale_factors = torch.ones(1)

        image = self._numpy_to_torch(image).to(self.params.device)
        init_mask = info['init_mask']
        init_mask = torch.from_numpy(init_mask).float()
        image, init_masks = self.generate_init_samples(image, init_mask)

        init_mask_oh = self.to_onehot(init_masks)
        init_mask_oh = init_mask_oh.float().unsqueeze(0).to(self.params.device)

        self.prev_mask = init_mask_oh.clone()
        with torch.no_grad():
            self.prev_key, self.prev_value = self.model(image, init_mask_oh, torch.tensor([self.num_objects]))

        self.feature_sz = torch.Tensor(list(self.prev_key.shape[-2:]))
        self.output_sz = self.feature_sz

        out = {'time': time.time() - tic}
        self.features_initialized = True

        return out

    def track(self, image, info: dict = None) -> dict:
        self.frame_num += 1
        image = self._numpy_to_torch(image).to(self.params.device)

        if self.frame_num == 2:  #
            this_keys, this_values = self.prev_key, self.prev_value  # only prev memory
        else:
            if self.keys.shape[3] > self.max_mem_sz:
                self.keys = torch.cat((self.keys[:, :, :, :1, :, :],
                                       self.keys[:, :, :, -self.max_mem_sz+1:, :, :]), dim=3)
                # self.keys = self.keys[:, :, :, -self.max_mem_sz:, :, :]
                # self.values = self.values[:, :, :, -self.max_mem_sz:, :, :]
                self.values = torch.cat((self.values[:, :, :, :1, :, :],
                                         self.values[:, :, :, -self.max_mem_sz + 1:, :, :]), dim=3)
            this_keys = torch.cat([self.keys, self.prev_key], dim=3)
            this_values = torch.cat([self.values, self.prev_value], dim=3)

            # this_keys = self.keys
            # this_values = self.values
        im_orig = image.clone()
        im_patches, patch_coords = sample_patch_multiscale(image, self.get_centered_sample_pos(),
                                                           self.target_scale * self.params.scale_factors,
                                                           self.img_sample_sz,
                                                           mode=self.params.get('border_mode', 'replicate'),
                                                           max_scale_change=self.params.get('patch_max_scale_change',
                                                                                            None))
        image = im_patches
        self.prev_pos = self.get_centered_sample_pos()
        self.prev_scale = self.target_scale * self.params.scale_factors[0]

        # segment
        with torch.no_grad():
            logit = self.model(image.clone(), this_keys, this_values, torch.tensor([self.num_objects]))

        pred_mask_soft = F.softmax(logit, dim=1)

        # update
        if self.frame_num - 1 in self.to_memorize:
            self.keys, self.values = this_keys, this_values

        with torch.no_grad():
            self.prev_key, self.prev_value = self.model(image.clone(), pred_mask_soft.clone(), torch.tensor([self.num_objects]))

        sample_pos, sample_scales = self.get_sample_location(patch_coords)
        pred_mask_soft = pred_mask_soft[:, 1:, :, :]

        pred_mask_im = self.convert_mask_crop_to_im(pred_mask_soft, im_orig, sample_scales, sample_pos)

        self.pos, self.target_sz = self.get_target_state(pred_mask_im.squeeze())

        new_target_scale = torch.sqrt(self.target_sz.prod() / self.base_target_sz.prod())
        if self.params.get('max_scale_change', None) is not None:
            if not isinstance(self.params.get('max_scale_change'), (tuple, list)):
                max_scale_change = (self.params.get('max_scale_change'), self.params.get('max_scale_change'))
            else:
                max_scale_change = self.params.get('max_scale_change')

            scale_change = new_target_scale / self.target_scale

            if scale_change < max_scale_change[0]:
                new_target_scale = self.target_scale * max_scale_change[0]
            elif scale_change > max_scale_change[1]:
                new_target_scale = self.target_scale * max_scale_change[1]

        # Update target scale
        self.target_scale = new_target_scale
        self.target_sz = self.base_target_sz * self.target_scale

        new_state = torch.cat((self.pos[[1, 0]] - (self.target_sz[[1, 0]] - 1) / 2, self.target_sz[[1, 0]]))

        output_state = new_state.tolist()

        # pred_mask = np.argmax(pred_mask_soft[0].cpu().numpy(), axis=0).astype(np.uint8)
        pred_mask_im = (pred_mask_im > 0.5).float()
        pred_mask_im_np = pred_mask_im.cpu().numpy()

        # pred_boxes = OrderedDict()
        # for id in self.object_ids:
        #     pred_boxes[id] = self._seg_to_box(pred_mask_im_np, object_id=int(id))

        out = {'segmentation': pred_mask_im_np, 'target_bbox': output_state}
        return out

    def get_centered_sample_pos(self):
        """Get the center position for the new sample. Make sure the target is correctly centered."""
        return self.pos # + ((self.feature_sz + self.kernel_size) % 2) * self.target_scale * self.img_support_sz / (2*self.feature_sz)

    def get_sample_location(self, sample_coord):
        """Get the location of the extracted sample."""
        sample_coord = sample_coord.float()
        sample_pos = 0.5*(sample_coord[:,:2] + sample_coord[:,2:] - 1)
        sample_scales = ((sample_coord[:,2:] - sample_coord[:,:2]) / self.img_sample_sz).prod(dim=1).sqrt()
        return sample_pos, sample_scales

    def get_target_state(self, seg_mask_im):
        if seg_mask_im.sum() < self.params.get('min_mask_area', -10):
            return self.pos, self.target_sz

        if self.params.get('seg_to_bb_mode', 'md_hack') == 'md_hack':
            prob_sum = seg_mask_im.sum()
            pos0 = torch.sum(seg_mask_im.sum(dim=-1) *
                             torch.arange(seg_mask_im.shape[-2], dtype=torch.float32)) / prob_sum
            pos1 = torch.sum(seg_mask_im.sum(dim=-2) *
                             torch.arange(seg_mask_im.shape[-1], dtype=torch.float32)) / prob_sum
            return torch.Tensor([pos0, pos1]), self.target_sz
        elif self.params.get('seg_to_bb_mode') == 'max_eigen':
            prob_sum = seg_mask_im.sum()
            e_y = torch.sum(seg_mask_im.sum(dim=-1) *
                            torch.arange(seg_mask_im.shape[-2], dtype=torch.float32)) / prob_sum
            e_x = torch.sum(seg_mask_im.sum(dim=-2) *
                            torch.arange(seg_mask_im.shape[-1], dtype=torch.float32)) / prob_sum

            e_h = torch.sum(seg_mask_im.sum(dim=-1) *
                            (torch.arange(seg_mask_im.shape[-2], dtype=torch.float32) - e_y)**2) / prob_sum
            e_w = torch.sum(seg_mask_im.sum(dim=-2) *
                            (torch.arange(seg_mask_im.shape[-1], dtype=torch.float32) - e_x)**2) / prob_sum

            e_wh = torch.sum((torch.arange(seg_mask_im.shape[-2], dtype=torch.float32).view(-1, 1) - e_y) *
                             (torch.arange(seg_mask_im.shape[-1], dtype=torch.float32).view(1, -1) - e_x) * seg_mask_im) / prob_sum

            eig = torch.eig(torch.tensor([[e_h, e_wh], [e_wh, e_w]]))

            sz = eig[0][:,0].max().sqrt().item()

            sz_factor = self.params.get('seg_to_bb_sz_factor', 4)
            return torch.Tensor([e_y, e_x]), torch.Tensor([sz* sz_factor, sz*sz_factor])
        elif self.params.get('seg_to_bb_mode') == 'var':
            prob_sum = seg_mask_im.sum()
            e_y = torch.sum(seg_mask_im.sum(dim=-1) *
                            torch.arange(seg_mask_im.shape[-2], dtype=torch.float32)) / prob_sum
            e_x = torch.sum(seg_mask_im.sum(dim=-2) *
                            torch.arange(seg_mask_im.shape[-1], dtype=torch.float32)) / prob_sum

            e_h = torch.sum(seg_mask_im.sum(dim=-1) *
                            (torch.arange(seg_mask_im.shape[-2], dtype=torch.float32) - e_y)**2) / prob_sum
            e_w = torch.sum(seg_mask_im.sum(dim=-2) *
                            (torch.arange(seg_mask_im.shape[-1], dtype=torch.float32) - e_x)**2) / prob_sum

            sz_factor = self.params.get('seg_to_bb_sz_factor', 4)
            return torch.Tensor([e_y, e_x]), torch.Tensor([e_h.sqrt()* sz_factor, e_w.sqrt()*sz_factor])
        elif self.params.get('seg_to_bb_mode') == 'area':
            prob_sum = seg_mask_im.sum()
            e_y = torch.sum(seg_mask_im.sum(dim=-1) *
                            torch.arange(seg_mask_im.shape[-2], dtype=torch.float32)) / prob_sum
            e_x = torch.sum(seg_mask_im.sum(dim=-2) *
                            torch.arange(seg_mask_im.shape[-1], dtype=torch.float32)) / prob_sum

            sz = prob_sum.sqrt()
            sz_factor = self.params.get('seg_to_bb_sz_factor', 1.5)
            return torch.Tensor([e_y, e_x]), torch.Tensor([sz* sz_factor, sz*sz_factor])


    def convert_mask_crop_to_im(self, seg_mask, im, sample_scales, sample_pos):
        seg_mask_re = F.interpolate(seg_mask, scale_factor=sample_scales[0].item(), mode='bilinear')
        seg_mask_re = seg_mask_re.view(*seg_mask_re.shape[-2:])

        # Regions outside search area get very low score
        seg_mask_im = torch.zeros(im.shape[-2:], dtype=seg_mask_re.dtype)
        r1 = int(sample_pos[0][0].item() - 0.5*seg_mask_re.shape[-2])
        c1 = int(sample_pos[0][1].item() - 0.5*seg_mask_re.shape[-1])

        r2 = r1 + seg_mask_re.shape[-2]
        c2 = c1 + seg_mask_re.shape[-1]

        r1_pad = max(0, -r1)
        c1_pad = max(0, -c1)

        r2_pad = max(r2 - im.shape[-2], 0)
        c2_pad = max(c2 - im.shape[-1], 0)
        seg_mask_im[r1 + r1_pad:r2 - r2_pad, c1 + c1_pad:c2 - c2_pad] = seg_mask_re[
                                                                        r1_pad:seg_mask_re.shape[0] - r2_pad,
                                                                        c1_pad:seg_mask_re.shape[1] - c2_pad]

        return seg_mask_im

    def _numpy_to_torch(self, im):
        return numpy_to_torch(im) / 255.0

    def to_onehot(self, mask):
        M = torch.zeros((self.K, mask.shape[0], mask.shape[1]), dtype=torch.uint8)
        for k in range(self.K):
            M[k] = (mask == k).int()
        return M

    def _seg_to_box(self, pred_mask, object_id=1):
        is_present_x = (pred_mask == object_id).sum(0) > 0
        is_present_y = (pred_mask == object_id).sum(1) > 0

        if np.nonzero(is_present_x)[0].size > 0:
            x1 = np.nonzero(is_present_x)[0][0]
            x2 = np.nonzero(is_present_x)[0][-1]
        else:
            x1 = 0
            x2 = 2

        if np.nonzero(is_present_y)[0].size > 0:
            y1 = np.nonzero(is_present_y)[0][0]
            y2 = np.nonzero(is_present_y)[0][-1]
        else:
            y1 = 0
            y2 = 2

        return [x1, y1, x2 - x1, y2 - y1]

    def generate_init_samples(self, im: torch.Tensor, init_mask):
        """Perform data augmentation to generate initial training samples."""

        mode = self.params.get('border_mode', 'replicate')
        if 'inside' in mode:
            # Get new sample size if forced inside the image
            im_sz = torch.Tensor([im.shape[2], im.shape[3]])
            sample_sz = self.target_scale * self.img_sample_sz
            shrink_factor = (sample_sz.float() / im_sz)
            if mode == 'inside':
                shrink_factor = shrink_factor.max()
            elif mode == 'inside_major':
                shrink_factor = shrink_factor.min()
            shrink_factor.clamp_(min=1, max=self.params.get('patch_max_scale_change', None))
            sample_sz = (sample_sz.float() / shrink_factor)
            self.init_sample_scale = (sample_sz / self.img_sample_sz).prod().sqrt()
            tl = self.pos - (sample_sz - 1) / 2
            br = self.pos + sample_sz / 2 + 1
            global_shift = - ((-tl).clamp(0) - (br - im_sz).clamp(0)) / self.init_sample_scale
        else:
            self.init_sample_scale = self.target_scale
            global_shift = torch.zeros(2)

        self.init_sample_pos = self.pos.round()

        # Compute augmentation size
        aug_expansion_factor = self.params.get('augmentation_expansion_factor', None)
        aug_expansion_sz = self.img_sample_sz.clone()
        aug_output_sz = None
        if aug_expansion_factor is not None and aug_expansion_factor != 1:
            aug_expansion_sz = (self.img_sample_sz * aug_expansion_factor).long()
            aug_expansion_sz += (aug_expansion_sz - self.img_sample_sz.long()) % 2
            aug_expansion_sz = aug_expansion_sz.float()
            aug_output_sz = self.img_sample_sz.long().tolist()

        # Random shift for each sample
        get_rand_shift = lambda: None
        random_shift_factor = self.params.get('random_shift_factor', 0)
        if random_shift_factor > 0:
            get_rand_shift = lambda: ((torch.rand(2) - 0.5) * self.img_sample_sz * random_shift_factor + global_shift).long().tolist()

        # Always put identity transformation first, since it is the unaugmented sample that is always used
        self.transforms = [augmentation.Identity(aug_output_sz, global_shift.long().tolist())]

        # Extract augmented image patches
        im_patches = sample_patch_transformed(im, self.init_sample_pos, self.init_sample_scale, aug_expansion_sz, self.transforms)

        init_masks = sample_patch_transformed(init_mask.unsqueeze(0).unsqueeze(0),
                                              self.init_sample_pos, self.init_sample_scale, aug_expansion_sz, self.transforms,
                                              is_mask=True)
        init_masks = init_masks.squeeze(0).squeeze(0)

        init_masks = init_masks.to(self.params.device)

        return im_patches, init_masks
