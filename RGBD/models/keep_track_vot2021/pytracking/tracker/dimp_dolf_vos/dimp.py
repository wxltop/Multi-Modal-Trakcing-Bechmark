from pytracking.tracker.base import BaseTracker
import torch
import torch.nn.functional as F
import numpy as np
import math
import time
from pytracking import dcf, TensorList
from pytracking.features.preprocessing import numpy_to_torch
from pytracking.utils.plotting import show_tensor, plot_graph
from pytracking.features.preprocessing import sample_patch_multiscale, sample_patch_transformed, sample_patch
from pytracking.features import augmentation
import ltr.data.bounding_box_utils as bbutils
from ltr.models.target_classifier.initializer import FilterInitializerZero
import cv2
import matplotlib.cm as cm
from ltr.models.layers import activation
import ltr.data.processing_utils as prutils
from ltr.data.processing_utils import gauss_2d


class DiMPDolfVos(BaseTracker):

    multiobj_mode = 'parallel'

    def initialize_features(self):
        if not getattr(self, 'features_initialized', False):
            self.params.net.initialize()
        self.features_initialized = True

        if self.params.get('run_in_train_mode', False):
            self.params.net.train(True)

    def initialize(self, image, info: dict) -> dict:
        # Initialize some stuff
        self.frame_num = 1
        if not self.params.has('device'):
            self.params.device = 'cuda' if self.params.use_gpu else 'cpu'

        # Initialize network
        self.initialize_features()

        # The DiMP network
        self.net = self.params.net

        # Time initialization
        tic = time.time()

        # Get target position and size
        state = info['init_bbox']
        init_mask = info.get('init_mask', None)

        if init_mask is not None:
            # shape 1 , 1, h, w (frames, seq, h, w)
            init_mask = torch.tensor(init_mask).unsqueeze(0).unsqueeze(0).float()

        self.pos = torch.Tensor([state[1] + (state[3] - 1)/2, state[0] + (state[2] - 1)/2])
        self.target_sz = torch.Tensor([state[3], state[2]])

        # Get object id
        self.object_id = info.get('object_ids', [None])[0]
        self.id_str = '' if self.object_id is None else ' {}'.format(self.object_id)

        # Set sizes
        sz = self.params.image_sample_size
        self.img_sample_sz = torch.Tensor([sz, sz] if isinstance(sz, int) else sz)
        self.img_support_sz = self.img_sample_sz

        # Set search area
        search_area = torch.prod(self.target_sz * self.params.search_area_scale).item()
        self.target_scale =  math.sqrt(search_area) / self.img_sample_sz.prod().sqrt()

        # Target size in base scale
        self.base_target_sz = self.target_sz / self.target_scale

        # Convert image
        im = numpy_to_torch(image)

        # Setup scale factors
        if not self.params.has('scale_factors'):
            self.params.scale_factors = torch.ones(1)
        elif isinstance(self.params.scale_factors, (list, tuple)):
            self.params.scale_factors = torch.Tensor(self.params.scale_factors)

        # Setup scale bounds
        self.image_sz = torch.Tensor([im.shape[2], im.shape[3]])
        self.min_scale_factor = torch.max(10 / self.base_target_sz)
        self.max_scale_factor = torch.min(self.image_sz / self.base_target_sz)

        # Extract and transform sample
        init_backbone_feat, init_boxes, init_masks = self.generate_init_samples(im, init_mask)

        # Initialize classifier
        self.init_seg_classifier(init_backbone_feat, init_masks)

        self.use_dimp = self.params.get('use_dimp', True)

        if self.use_dimp:
            self.init_dimp_classifier(init_backbone_feat, init_boxes)

        self.prev_test_x_seg = None
        self.prev_test_x_dimp = None
        self.prev_flag = 'init'

        out = {'time': time.time() - tic}
        return out


    def track(self, image, info: dict = None) -> dict:
        self.debug_info = {}

        # ------- UPDATE ------- #
        if self.use_dimp:
            update_flag = self.prev_flag not in ['not_found', 'uncertain']
            hard_negative = (self.prev_flag == 'hard_negative')
            learning_rate = self.params.get('hard_negative_learning_rate', None) if hard_negative else None
            if update_flag and self.params.get('update_dimp_classifier', False) and self.prev_test_x_dimp is not None:
                # Get train sample
                train_x = self.prev_test_x_dimp

                # Create target_box and label for spatial sample
                raise NotImplementedError
                # Fix sample position
                target_box = self.get_iounet_box(self.pos, self.target_sz, self.prev_pos, self.prev_scale)

                # Update the classifier model
                self.update_dimp_classifier(train_x, target_box, learning_rate)

        if self.params.get('update_seg_classifier', False) and self.prev_test_x_seg is not None and (not self.params.get('use_gt_mask_enc_for_pred', False)):
            if self.params.get('use_merged_mask_for_memory', False):
                object_pos = None
                seg_masks = []

                # Fix this
                ct = 0
                for k, v in info['previous_output'].items():
                    if k == self.object_id:
                        object_pos = ct
                    seg_masks.append(v['segmentation'])
                    ct = ct + 1
                segmentation_maps = np.stack(seg_masks)
                segmentation_maps_merged = self.merge_segmentation_results(segmentation_maps)

                if self.params.get('thresh_before_update', False):
                    seg_mask_im = segmentation_maps_merged.argmax(axis=0)
                    seg_mask_im = seg_mask_im == (object_pos + 1)
                else:
                    seg_mask_im = segmentation_maps_merged[object_pos + 1]
            else:
                if self.params.get('thresh_before_update', False):
                    raise NotImplementedError
                seg_mask_im = info['previous_output'][self.object_id]['segmentation']

            seg_mask_im = torch.from_numpy(seg_mask_im).unsqueeze(0).unsqueeze(0).float()
            seg_mask_crop, _ = sample_patch(seg_mask_im, self.prev_pos, self.prev_scale * self.img_sample_sz,
                                            self.img_sample_sz,
                                            mode=self.params.get('border_mode', 'replicate'),
                                            max_scale_change=self.params.get(
                                                'patch_max_scale_change', None), is_mask=True)

            if self.params.get('store_enc', False):
                seg_mask_enc = self.net.label_encoder(seg_mask_crop.clone(), self.prev_test_x_seg.unsqueeze(1))
                seg_mask_enc = seg_mask_enc.view(1, *seg_mask_enc.shape[-3:])
            else:
                seg_mask_enc = seg_mask_crop.clone()

            self.update_seg_classifier(self.prev_test_x_seg, seg_mask_enc, None)

        self.frame_num += 1
        self.debug_info['frame_num'] = self.frame_num

        # Convert image
        im = numpy_to_torch(image)

        # ******************************* Crop ******************************************************************
        if self.params.get('use_gt_box_for_crop', False):
            if self.gt_state is not None and self.gt_state[-1] > 0 and self.gt_state[-2] > 0:
                gt_state = self.gt_state
                self.pos = torch.Tensor([gt_state[1] + (gt_state[3] - 1) / 2, gt_state[0] + (gt_state[2] - 1) / 2])
                self.target_sz = torch.Tensor([gt_state[3], gt_state[2]])
                self.target_scale = torch.sqrt(self.target_sz.prod() / self.base_target_sz.prod())

        # Extract backbone features
        backbone_feat, sample_coords, im_patches = self.extract_backbone_features(im, self.get_centered_sample_pos(),
                                                                                  self.target_scale * self.params.scale_factors,
                                                                                  self.img_sample_sz)
        self.prev_pos = self.get_centered_sample_pos()
        self.prev_scale = self.target_scale * self.params.scale_factors[0]

        sample_pos, sample_scales = self.get_sample_location(sample_coords)

        # ******************************  DiMP *******************************************************************
        if self.use_dimp:
            # Extract classification features
            test_x_dimp = self.get_dimp_classification_features(backbone_feat)

            # Compute classification scores
            dimp_scores_raw = self.classify_target(test_x_dimp)

            # Localize the target
            translation_vec, scale_ind, s, flag = self.localize_target(dimp_scores_raw, sample_pos, sample_scales)
            new_pos = sample_pos[scale_ind, :] + translation_vec

            self.prev_flag = flag
            self.prev_test_x_dimp = test_x_dimp

        # ***************************** Segmentation ***********************************************************
        # Extract classification features
        test_x_seg = self.get_seg_classification_features(backbone_feat)

        if self.use_dimp:
            center_label = self.generate_center_label((test_x_seg.shape[-2], test_x_seg.shape[-1]), dimp_scores_raw)
        else:
            gt_state = self.gt_state
            gt_pos = torch.Tensor([gt_state[1] + (gt_state[3] - 1) / 2, gt_state[0] + (gt_state[2] - 1) / 2])
            gt_target_sz = torch.Tensor([gt_state[3], gt_state[2]])

            gt_box_in_crop = self.get_iounet_box(gt_pos, gt_target_sz, sample_pos.squeeze(),
                                                 sample_scales)
            sigma_factor = self.params.get('tc_label_sigma_factor', 1.0 / 4.0)
            gauss_sigma = sigma_factor / self.params.search_area_scale
            center_label = prutils.gaussian_label_function(gt_box_in_crop.view(-1, 4),
                                                           gauss_sigma,
                                                           self.kernel_size,
                                                           self.feature_sz_seg.long().flip(dims=(0,)).tolist(),
                                                           self.img_support_sz.flip(dims=(0,))
                                                           )
            center_label = center_label.to(self.params.device)
        seg_mask, bb_pred = self.segment_target(test_x_seg, backbone_feat, center_label)

        seg_mask_raw = seg_mask.clone()
        seg_mask = torch.sigmoid(seg_mask)

        self.prev_test_x_seg = test_x_seg

        # Compute output bounding box
        new_state = torch.cat((self.pos[[1,0]] - (self.target_sz[[1,0]]-1)/2, self.target_sz[[1,0]]))

        if self.params.get('output_not_found_box', False) and flag == 'not_found':
            output_state = [-1, -1, -1, -1]
        else:
            output_state = new_state.tolist()

        seg_mask_im = self.convert_mask_crop_to_im(seg_mask, im, sample_scales, sample_pos)

        seg_mask_im = seg_mask_im.view(*seg_mask_im.shape[-2:])

        self.pos = self.get_mask_center(seg_mask_im)

        seg_mask_im_np = seg_mask_im.cpu().numpy()

        if self.visdom is not None:
            self.visdom.register(seg_mask_raw, 'heatmap', 2, 'Seg Scores' + self.id_str)
            if self.use_dimp:
                self.visdom.register(dimp_scores_raw, 'heatmap', 2, 'Dimp Scores' + self.id_str)
            self.visdom.register(self.debug_info, 'info_dict', 1, 'Status')

        if bb_pred is not None:
            bb_pred = self.convert_box_crop_to_im(bb_pred, sample_scales, sample_pos)

            output_state = bb_pred.cpu().view(-1).tolist()
        out = {'segmentation': seg_mask_im_np, 'target_bbox': output_state}
        return out

    def get_sample_location(self, sample_coord):
        """Get the location of the extracted sample."""
        sample_coord = sample_coord.float()
        sample_pos = 0.5*(sample_coord[:,:2] + sample_coord[:,2:] - 1)
        sample_scales = ((sample_coord[:,2:] - sample_coord[:,:2]) / self.img_sample_sz).prod(dim=1).sqrt()
        return sample_pos, sample_scales

    def get_centered_sample_pos(self):
        """Get the center position for the new sample. Make sure the target is correctly centered."""
        feature_sz = getattr(self, 'feature_sz_dimp', self.feature_sz_seg)
        return self.pos + ((feature_sz + self.kernel_size) % 2) * self.target_scale * \
               self.img_support_sz / (2 * feature_sz)

    def get_mask_center(self, seg_mask_im):
        prob_sum = seg_mask_im.sum()
        pos0 = torch.sum(seg_mask_im.sum(dim=-1) *
                         torch.arange(seg_mask_im.shape[-2], dtype=torch.float32)) / prob_sum
        pos1 = torch.sum(seg_mask_im.sum(dim=-2) *
                         torch.arange(seg_mask_im.shape[-1], dtype=torch.float32)) / prob_sum
        return torch.Tensor([pos0, pos1])

    def classify_target(self, sample_x: TensorList):
        """Classify target by applying the DiMP filter."""
        with torch.no_grad():
            scores = self.net.dimp_classifier.classify(self.dimp_target_filter, sample_x)
        return scores

    def generate_center_label(self, label_sz, pred_score, sigma_factor=1/4):
        # Upsample since seg is at twice the resolution
        pred_score_up = F.interpolate(pred_score, size=label_sz, align_corners=False, mode='bilinear')

        label_sz = torch.tensor(label_sz)
        sigma = sigma_factor * label_sz.float().prod().sqrt().item() / self.params.search_area_scale

        max_score, max_loc = dcf.max2d(pred_score_up.cpu())

        label_center = max_loc.view(1, 2) - label_sz.view(1, 2)*0.5
        gauss_label = gauss_2d(label_sz.flip(dims=(0,)), sigma, label_center.flip(dims=(1,)))
        return gauss_label.to(self.params.device)

    def convert_box_crop_to_im(self, box, sample_scales, sample_pos):
        box = box.cpu()
        box[:, [1, 0]] -= (self.img_sample_sz / 2.0)
        box_sc = box*sample_scales[0].item()

        box_sc[:, [1, 0]] += sample_pos
        return box_sc

    def convert_mask_crop_to_im(self, seg_mask, im, sample_scales, sample_pos):
        seg_mask_re = F.interpolate(seg_mask, scale_factor=sample_scales[0].item(), mode='bilinear')
        seg_mask_re = seg_mask_re.view(*seg_mask_re.shape[-2:])
        # seg_mask_im = torch.ones(im.shape[-2:], dtype=seg_mask_re.dtype) * -1.0
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

    @staticmethod
    def merge_segmentation_results(segmentation_maps):
        # Soft aggregation from RGMP
        eps = 1e-7
        segmentation_maps_t = torch.from_numpy(segmentation_maps).float().clamp(eps, 1.0 - eps)

        bg_p = torch.prod(1 - segmentation_maps_t, dim=0).clamp(eps, 1.0 - eps)  # bg prob

        segm_all = torch.cat((bg_p.unsqueeze(0), segmentation_maps_t), dim=0)
        odds_t = segm_all / (1.0 - segm_all)

        norm_factor = odds_t.sum(0)

        segmentation_maps_t_agg = odds_t / norm_factor.unsqueeze(0)

        segmentation_maps_np_agg = segmentation_maps_t_agg.numpy()
        return segmentation_maps_np_agg

    def segment_target(self, sample_clf_feat, sample_x, center_label):
        """Classify target by applying the DiMP filter."""
        with torch.no_grad():
            mask, box = self.net.segment_target(self.seg_target_filter, sample_clf_feat, sample_x, center_label)
        return mask, box

    def segment_target_using_gt_enc(self, sample_clf_feat, sample_x, gt_mask):
        """Classify target by applying the DiMP filter."""
        gt_mask = gt_mask.to(self.params.device)

        with torch.no_grad():
            gt_mask_enc = self.net.label_encoder(gt_mask, sample_clf_feat.unsqueeze(1))
            if isinstance(gt_mask_enc, (tuple, list)):
                gt_mask_enc = gt_mask_enc[0]
            mask_pred, _ = self.net.decoder(gt_mask_enc, sample_x,
                                            (sample_clf_feat.shape[-2] * 16, sample_clf_feat.shape[-1] * 16))
        return mask_pred

    def localize_target(self, scores, sample_pos, sample_scales):
        """Run the target localization."""

        scores = scores.squeeze(1)

        preprocess_method = self.params.get('score_preprocess', 'none')
        if preprocess_method == 'none':
            pass
        elif preprocess_method == 'exp':
            scores = scores.exp()
        elif preprocess_method == 'softmax':
            reg_val = getattr(self.net.classifier.filter_optimizer, 'softmax_reg', None)
            scores_view = scores.view(scores.shape[0], -1)
            scores_softmax = activation.softmax_reg(scores_view, dim=-1, reg=reg_val)
            scores = scores_softmax.view(scores.shape)
        else:
            raise Exception('Unknown score_preprocess in params.')

        score_filter_ksz = self.params.get('score_filter_ksz', 1)
        if score_filter_ksz > 1:
            assert score_filter_ksz % 2 == 1
            kernel = scores.new_ones(1,1,score_filter_ksz,score_filter_ksz)
            scores = F.conv2d(scores.view(-1,1,*scores.shape[-2:]), kernel, padding=score_filter_ksz//2).view(scores.shape)

        if self.params.get('advanced_localization', False):
            return self.localize_advanced(scores, sample_pos, sample_scales)

        # Get maximum
        score_sz = torch.Tensor(list(scores.shape[-2:]))
        score_center = (score_sz - 1)/2
        max_score, max_disp = dcf.max2d(scores)
        _, scale_ind = torch.max(max_score, dim=0)
        max_disp = max_disp[scale_ind,...].float().cpu().view(-1)
        target_disp = max_disp - score_center

        # Compute translation vector and scale change factor
        output_sz = score_sz - (self.kernel_size + 1) % 2
        translation_vec = target_disp * (self.img_support_sz / output_sz) * sample_scales[scale_ind]

        return translation_vec, scale_ind, scores, None


    def localize_advanced(self, scores, sample_pos, sample_scales):
        """Run the target advanced localization (as in ATOM)."""

        sz = scores.shape[-2:]
        score_sz = torch.Tensor(list(sz))
        output_sz = score_sz - (self.kernel_size + 1) % 2
        score_center = (score_sz - 1)/2

        scores_hn = scores
        if self.output_window is not None and self.params.get('perform_hn_without_windowing', False):
            scores_hn = scores.clone()
            scores *= self.output_window

        max_score1, max_disp1 = dcf.max2d(scores)
        _, scale_ind = torch.max(max_score1, dim=0)
        sample_scale = sample_scales[scale_ind]
        max_score1 = max_score1[scale_ind]
        max_disp1 = max_disp1[scale_ind,...].float().cpu().view(-1)
        target_disp1 = max_disp1 - score_center
        translation_vec1 = target_disp1 * (self.img_support_sz / output_sz) * sample_scale

        if max_score1.item() < self.params.target_not_found_threshold:
            return translation_vec1, scale_ind, scores_hn, 'not_found'
        if max_score1.item() < self.params.get('uncertain_threshold', -float('inf')):
            return translation_vec1, scale_ind, scores_hn, 'uncertain'
        if max_score1.item() < self.params.get('hard_sample_threshold', -float('inf')):
            return translation_vec1, scale_ind, scores_hn, 'hard_negative'

        # Mask out target neighborhood
        target_neigh_sz = self.params.target_neighborhood_scale * (self.target_sz / sample_scale) * (output_sz / self.img_support_sz)

        tneigh_top = max(round(max_disp1[0].item() - target_neigh_sz[0].item() / 2), 0)
        tneigh_bottom = min(round(max_disp1[0].item() + target_neigh_sz[0].item() / 2 + 1), sz[0])
        tneigh_left = max(round(max_disp1[1].item() - target_neigh_sz[1].item() / 2), 0)
        tneigh_right = min(round(max_disp1[1].item() + target_neigh_sz[1].item() / 2 + 1), sz[1])
        scores_masked = scores_hn[scale_ind:scale_ind + 1, ...].clone()
        scores_masked[...,tneigh_top:tneigh_bottom,tneigh_left:tneigh_right] = 0

        # Find new maximum
        max_score2, max_disp2 = dcf.max2d(scores_masked)
        max_disp2 = max_disp2.float().cpu().view(-1)
        target_disp2 = max_disp2 - score_center
        translation_vec2 = target_disp2 * (self.img_support_sz / output_sz) * sample_scale

        prev_target_vec = (self.pos - sample_pos[scale_ind,:]) / ((self.img_support_sz / output_sz) * sample_scale)

        # Handle the different cases
        if max_score2 > self.params.distractor_threshold * max_score1:
            disp_norm1 = torch.sqrt(torch.sum((target_disp1-prev_target_vec)**2))
            disp_norm2 = torch.sqrt(torch.sum((target_disp2-prev_target_vec)**2))
            disp_threshold = self.params.dispalcement_scale * math.sqrt(sz[0] * sz[1]) / 2

            if disp_norm2 > disp_threshold and disp_norm1 < disp_threshold:
                return translation_vec1, scale_ind, scores_hn, 'hard_negative'
            if disp_norm2 < disp_threshold and disp_norm1 > disp_threshold:
                return translation_vec2, scale_ind, scores_hn, 'hard_negative'
            if disp_norm2 > disp_threshold and disp_norm1 > disp_threshold:
                return translation_vec1, scale_ind, scores_hn, 'uncertain'

            # If also the distractor is close, return with highest score
            return translation_vec1, scale_ind, scores_hn, 'uncertain'

        if max_score2 > self.params.hard_negative_threshold * max_score1 and max_score2 > self.params.target_not_found_threshold:
            return translation_vec1, scale_ind, scores_hn, 'hard_negative'

        return translation_vec1, scale_ind, scores_hn, 'normal'

    def extract_backbone_features(self, im: torch.Tensor, pos: torch.Tensor, scales, sz: torch.Tensor):
        im_patches, patch_coords = sample_patch_multiscale(im, pos, scales, sz,
                                                           mode=self.params.get('border_mode', 'replicate'),
                                                           max_scale_change=self.params.get('patch_max_scale_change', None))
        with torch.no_grad():
            backbone_feat = self.net.extract_backbone(im_patches)
        return backbone_feat, patch_coords, im_patches

    def get_seg_classification_features(self, backbone_feat):
        with torch.no_grad():
            return self.net.extract_seg_classification_feat(backbone_feat)

    def get_dimp_classification_features(self, backbone_feat):
        with torch.no_grad():
            return self.net.extract_dimp_classification_feat(backbone_feat)

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

        augs = self.params.augmentation if self.params.get('use_augmentation', True) else {}

        # Add all augmentations
        if 'shift' in augs:
            self.transforms.extend([augmentation.Translation(shift, aug_output_sz, global_shift.long().tolist()) for shift in augs['shift']])
        if 'relativeshift' in augs:
            get_absolute = lambda shift: (torch.Tensor(shift) * self.img_sample_sz/2).long().tolist()
            self.transforms.extend([augmentation.Translation(get_absolute(shift), aug_output_sz, global_shift.long().tolist()) for shift in augs['relativeshift']])
        if 'fliplr' in augs and augs['fliplr']:
            self.transforms.append(augmentation.FlipHorizontal(aug_output_sz, get_rand_shift()))
        if 'blur' in augs:
            self.transforms.extend([augmentation.Blur(sigma, aug_output_sz, get_rand_shift()) for sigma in augs['blur']])
        if 'scale' in augs:
            self.transforms.extend([augmentation.Scale(scale_factor, aug_output_sz, get_rand_shift()) for scale_factor in augs['scale']])
        if 'rotate' in augs:
            self.transforms.extend([augmentation.Rotate(angle, aug_output_sz, get_rand_shift()) for angle in augs['rotate']])
        if 'random_affine' in augs:
            self.transforms.extend([augmentation.RandomAffine(**augs['random_affine']['params'],
                                                              output_sz=aug_output_sz, shift=get_rand_shift())
                                    for _ in range(augs['random_affine']['num_aug'])])

        # Extract augmented image patches
        im_patches = sample_patch_transformed(im, self.init_sample_pos, self.init_sample_scale, aug_expansion_sz, self.transforms)

        init_masks = sample_patch_transformed(init_mask,
                                              self.init_sample_pos, self.init_sample_scale, aug_expansion_sz, self.transforms,
                                              is_mask=True)

        init_masks = init_masks.to(self.params.device)

        # Extract initial backbone features
        with torch.no_grad():
            init_backbone_feat = self.net.extract_backbone(im_patches)

        init_boxes = self.init_target_boxes()

        return init_backbone_feat, init_boxes, init_masks

    def init_target_boxes(self):
        """Get the target bounding boxes for the initial augmented samples."""
        self.classifier_target_box = self.get_iounet_box(self.pos, self.target_sz, self.init_sample_pos, self.init_sample_scale)
        init_target_boxes = TensorList()
        for T in self.transforms:
            init_target_boxes.append(self.classifier_target_box + torch.Tensor([T.shift[1], T.shift[0], 0, 0]))
        init_target_boxes = torch.cat(init_target_boxes.view(1, 4), 0).to(self.params.device)
        self.target_boxes = init_target_boxes.new_zeros(self.params.sample_memory_size_dimp, 4)
        self.target_boxes[:init_target_boxes.shape[0],:] = init_target_boxes
        return init_target_boxes

    def init_seg_memory(self, train_x: TensorList, masks):
        assert masks.dim() == 4

        # Initialize first-frame spatial training samples
        self.num_init_seg_samples = train_x.size(0)
        init_sample_weights_seg = TensorList([x.new_ones(1) / x.shape[0] for x in train_x])

        # Sample counters and weights for spatial
        self.num_stored_seg_samples = self.num_init_seg_samples.copy()
        self.previous_replace_ind_seg = [None] * len(self.num_stored_seg_samples)
        self.sample_weights_seg = TensorList([x.new_zeros(self.params.sample_memory_size_seg) for x in train_x])
        for sw, init_sw, num in zip(self.sample_weights_seg, init_sample_weights_seg, self.num_init_seg_samples):
            sw[:num] = init_sw

        # Initialize memory
        self.training_samples_seg = TensorList(
            [x.new_zeros(self.params.sample_memory_size_seg, x.shape[1], x.shape[2], x.shape[3]) for x in train_x])

        self.target_masks = masks.new_zeros(self.params.sample_memory_size_seg, masks.shape[-3], masks.shape[-2],
                                            masks.shape[-1])
        self.target_masks[:masks.shape[0], :, :, :] = masks

        for ts, x in zip(self.training_samples_seg, train_x):
            ts[:x.shape[0],...] = x


    def update_seg_memory(self, sample_x: TensorList, mask, learning_rate = None):
        # Update weights and get replace ind
        if learning_rate is None:
            learning_rate = self.params.learning_rate_seg
        replace_ind = self.update_sample_weights(self.sample_weights_seg,
                                                 self.previous_replace_ind_seg,
                                                 self.num_stored_seg_samples,
                                                 self.num_init_seg_samples, learning_rate)
        self.previous_replace_ind_seg = replace_ind

        # Update sample and label memory
        for train_samp, x, ind in zip(self.training_samples_seg, sample_x, replace_ind):
            train_samp[ind:ind+1,...] = x

        # Update bb memory
        # self.target_boxes[replace_ind[0],:] = target_box
        self.target_masks[replace_ind[0], :, :, :] = mask[0, ...]

        self.num_stored_seg_samples += 1


    def update_sample_weights(self, sample_weights, previous_replace_ind, num_stored_samples, num_init_samples, learning_rate):
        # Update weights and get index to replace
        replace_ind = []
        for sw, prev_ind, num_samp, num_init in zip(sample_weights, previous_replace_ind, num_stored_samples, num_init_samples):
            lr = learning_rate

            init_samp_weight = self.params.get('init_samples_minimum_weight', None)
            if init_samp_weight == 0:
                init_samp_weight = None
            s_ind = 0 if init_samp_weight is None else num_init

            if num_samp == 0 or lr == 1:
                sw[:] = 0
                sw[0] = 1
                r_ind = 0
            else:
                # Get index to replace
                if num_samp < sw.shape[0]:
                    r_ind = num_samp
                else:
                    _, r_ind = torch.min(sw[s_ind:], 0)
                    r_ind = r_ind.item() + s_ind

                # Update weights
                if prev_ind is None:
                    sw /= 1 - lr
                    sw[r_ind] = lr
                else:
                    sw[r_ind] = sw[prev_ind] / (1 - lr)

            sw /= sw.sum()
            if init_samp_weight is not None and sw[:num_init].sum() < init_samp_weight:
                sw /= init_samp_weight + sw[num_init:].sum()
                sw[:num_init] = init_samp_weight / num_init

            replace_ind.append(r_ind)

        return replace_ind

    def update_state(self, new_pos, new_scale = None):
        # Update scale
        if new_scale is not None:
            self.target_scale = new_scale.clamp(self.min_scale_factor, self.max_scale_factor)
            self.target_sz = self.base_target_sz * self.target_scale

        # Update pos
        inside_ratio = self.params.get('target_inside_ratio', 0.2)
        inside_offset = (inside_ratio - 0.5) * self.target_sz
        self.pos = torch.max(torch.min(new_pos, self.image_sz - inside_offset), inside_offset)

    def get_iounet_box(self, pos, sz, sample_pos, sample_scale):
        """All inputs in original image coordinates.
        Generates a box in the cropped image sample reference frame, in the format used by the IoUNet."""
        box_center = (pos - sample_pos) / sample_scale + (self.img_sample_sz - 1) / 2
        box_sz = sz / sample_scale
        target_ul = box_center - (box_sz - 1) / 2
        return torch.cat([target_ul.flip((0,)), box_sz.flip((0,))])

    def init_seg_classifier(self, init_backbone_feat, init_masks):
        # Get classification features
        x = self.get_seg_classification_features(init_backbone_feat)

        # Add the dropout augmentation here, since it requires extraction of the classification features
        if 'dropout' in self.params.augmentation and self.params.get('use_augmentation', True):
            raise NotImplementedError

        # Set feature size and other related sizes
        self.feature_sz_seg = torch.Tensor(list(x.shape[-2:]))
        ksz = self.net.seg_classifier.filter_size   # Assume same for seg and dimp
        self.kernel_size = torch.Tensor([ksz, ksz] if isinstance(ksz, (int, float)) else ksz)
        # self.output_sz_seg = self.feature_sz_seg + (self.kernel_size + 1) % 2

        # Set number of iterations
        num_iter = self.params.get('net_opt_iter_seg', None)

        if self.net.label_encoder is not None:
            mask_enc = self.net.label_encoder(init_masks, x.unsqueeze(1))
        else:
            mask_enc = init_masks

        # Get target filter by running the discriminative model prediction module
        with torch.no_grad():
            self.seg_target_filter, _, losses = self.net.seg_classifier.get_filter(x.unsqueeze(1), mask_enc,
                                                                                   num_iter=num_iter)

        # Init memory
        if self.params.get('update_classifier', True):
            if self.params.get('store_enc', False):
                self.init_seg_memory(TensorList([x]), masks=mask_enc.view(-1, *mask_enc.shape[-3:]))
            else:
                self.init_seg_memory(TensorList([x]), masks=init_masks.view(-1, 1, *init_masks.shape[-2:]))

    def update_seg_classifier(self, train_x, mask, learning_rate=None):
        # Set flags and learning rate
        hard_negative_flag = learning_rate is not None
        if learning_rate is None:
            learning_rate = self.params.learning_rate_seg

        # Update the tracker memory
        if hard_negative_flag or self.frame_num % self.params.get('train_sample_interval_seg', 1) == 0:
            self.update_seg_memory(TensorList([train_x]), mask, learning_rate)

        # Decide the number of iterations to run
        num_iter = 0
        if hard_negative_flag:
            num_iter = self.params.get('net_opt_hn_iter_seg', None)
        elif (self.frame_num - 1) % self.params.train_skipping_seg == 0:
            num_iter = self.params.get('net_opt_update_iter_seg', None)

        if num_iter > 0:
            # Get inputs for the DiMP filter optimizer module
            samples = self.training_samples_seg[0][:self.num_stored_seg_samples[0], ...]
            masks = self.target_masks[:self.num_stored_seg_samples[0], ...]

            if self.params.get('store_enc', False):
                mask_enc_info = masks
            else:
                mask_enc_info = self.net.label_encoder(masks, samples.unsqueeze(1))

            # target_boxes = self.target_boxes[:self.num_stored_samples[0],:].clone()
            sample_weights = self.sample_weights_seg[0][:self.num_stored_seg_samples[0]]

            if isinstance(mask_enc_info, (tuple, list)):
                mask_enc = mask_enc_info[0]
                sample_weights = mask_enc_info[1] * sample_weights.view(-1, 1, 1, 1, 1)
            else:
                mask_enc = mask_enc_info

            # Run the filter optimizer module
            with torch.no_grad():
                target_filter, _, losses = self.net.seg_classifier.filter_optimizer(
                    TensorList([self.seg_target_filter]),
                    num_iter=num_iter, feat=samples.unsqueeze(1),
                    mask=mask_enc.unsqueeze(1),
                    sample_weight=sample_weights)

            self.seg_target_filter = target_filter[0]

    def init_dimp_memory(self, train_x: TensorList, init_target_boxes):
        # Initialize first-frame spatial training samples
        self.num_init_dimp_samples = train_x.size(0)
        init_sample_weights_dimp = TensorList([x.new_ones(1) / x.shape[0] for x in train_x])

        # Sample counters and weights for spatial
        self.num_stored_dimp_samples = self.num_init_dimp_samples.copy()
        self.previous_replace_ind_dimp = [None] * len(self.num_stored_dimp_samples)
        self.sample_weights_dimp = TensorList([x.new_zeros(self.params.sample_memory_size_dimp) for x in train_x])
        for sw, init_sw, num in zip(self.sample_weights_dimp, init_sample_weights_dimp, self.num_init_dimp_samples):
            sw[:num] = init_sw

        # Initialize memory
        self.training_samples_dimp = TensorList(
            [x.new_zeros(self.params.sample_memory_size_dimp, x.shape[1], x.shape[2], x.shape[3]) for x in train_x])

        self.target_boxes = init_target_boxes.new_zeros(self.params.sample_memory_size_dimp, 4)
        self.target_boxes[:init_target_boxes.shape[0], :] = init_target_boxes

        for ts, x in zip(self.training_samples_dimp, train_x):
            ts[:x.shape[0],...] = x


    def update_dimp_memory(self, sample_x: TensorList, target_box, learning_rate = None):
        # Update weights and get replace ind
        if learning_rate is None:
            learning_rate = self.params.learning_rate_dimp
        replace_ind = self.update_sample_weights(self.sample_weights_dimp,
                                                 self.previous_replace_ind_dimp,
                                                 self.num_stored_dimp_samples,
                                                 self.num_init_dimp_samples, learning_rate)
        self.previous_replace_ind_dimp = replace_ind

        # Update sample and label memory
        for train_samp, x, ind in zip(self.training_samples_dimp, sample_x, replace_ind):
            train_samp[ind:ind+1,...] = x

        # Update bb memory
        # self.target_boxes[replace_ind[0],:] = target_box
        self.target_boxes[replace_ind[0], :] = target_box

        self.num_stored_dimp_samples += 1

    def init_dimp_classifier(self, init_backbone_feat, init_boxes):
        # Get classification features
        x = self.get_dimp_classification_features(init_backbone_feat)

        # Add the dropout augmentation here, since it requires extraction of the classification features
        if 'dropout' in self.params.augmentation and self.params.get('use_augmentation', True):
            raise Exception

        # Set feature size and other related sizes
        self.feature_sz_dimp = torch.Tensor(list(x.shape[-2:]))

        # assert self.kernel_size == self.net.dimp_classifier.filter_size
        self.output_sz_dimp = self.feature_sz_dimp + (self.kernel_size + 1)%2

        # Construct output window
        self.output_window = None
        if self.params.get('window_output', False):
            raise Exception

        num_iter = self.params.get('net_opt_iter_dimp', None)

        # Get target filter by running the discriminative model prediction module
        with torch.no_grad():
            self.dimp_target_filter, _, losses = self.net.dimp_classifier.get_filter(x, init_boxes, num_iter=num_iter)

        # Init memory
        if self.params.get('update_dimp_classifier', True):
            self.init_dimp_memory(TensorList([x]), init_boxes)

    def update_dimp_classifier(self, train_x, target_box, learning_rate=None):
        # Set flags and learning rate
        hard_negative_flag = learning_rate is not None
        if learning_rate is None:
            learning_rate = self.params.learning_rate_dimp

        # Update the tracker memory
        if hard_negative_flag or self.frame_num % self.params.get('train_sample_interval_dimp', 1) == 0:
            self.update_dimp_memory(TensorList([train_x]), target_box, learning_rate)

        # Decide the number of iterations to run
        num_iter = 0
        if hard_negative_flag:
            num_iter = self.params.get('net_opt_hn_iter_dimp', None)
        elif (self.frame_num - 1) % self.params.train_skipping_dimp == 0:
            num_iter = self.params.get('net_opt_update_iter_dimp', None)

        plot_loss = self.params.debug > 0

        if num_iter > 0:
            # Get inputs for the DiMP filter optimizer module
            samples = self.training_samples_dimp[0][:self.num_stored_dimp_samples[0],...]
            target_boxes = self.target_boxes[:self.num_stored_dimp_samples[0],:].clone()
            sample_weights = self.sample_weights_dimp[0][:self.num_stored_dimp_samples[0]]

            # Run the filter optimizer module
            with torch.no_grad():
                target_filters, _, losses = self.net.dimp_classifier.filter_optimizer(TensorList([self.dimp_target_filter]),
                                                                                      num_iter=num_iter, feat=samples,
                                                                                      bb=target_boxes,
                                                                                      sample_weight=sample_weights)
                self.dimp_target_filter = target_filters[0]
