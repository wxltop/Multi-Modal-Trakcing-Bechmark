from pytracking.tracker.base import BaseTracker
import torch
import torch.nn.functional as F
import numpy as np
import math
import time
from pytracking import dcf, TensorList
from pytracking.features.preprocessing import numpy_to_torch
from pytracking.utils.plotting import show_tensor, plot_graph
from pytracking.features.preprocessing import sample_patch_multiscale, sample_patch_transformed
from pytracking.features import augmentation
import ltr.data.bounding_box_utils as bbutils
from ltr.models.target_classifier.initializer import FilterInitializerZero
import cv2
import matplotlib.cm as cm
from ltr.models.layers import activation


class DiMPbbrVos(BaseTracker):

    multiobj_mode = 'parallel'

    def initialize_features(self):
        if not getattr(self, 'features_initialized', False):
            self.params.net.initialize()
        self.features_initialized = True

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
            self.init_masks = torch.tensor(init_mask).unsqueeze(0).unsqueeze(0).float()

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
        init_backbone_feat, init_target_boxes = self.generate_init_samples(im)

        # Initialize classifier
        self.init_classifier(init_backbone_feat, init_target_boxes)

        # Initialize IoUNet
        if self.params.get('use_iou_net', False):
            self.init_iou_net(init_backbone_feat)

        out = {'time': time.time() - tic}
        return out


    def track(self, image, info: dict = None) -> dict:
        self.debug_info = {}

        self.frame_num += 1
        self.debug_info['frame_num'] = self.frame_num

        # Convert image
        im = numpy_to_torch(image)

        # ------- LOCALIZATION ------- #

        # Extract backbone features
        backbone_feat, sample_coords, im_patches = self.extract_backbone_features(im, self.get_centered_sample_pos(),
                                                                      self.target_scale * self.params.scale_factors,
                                                                      self.img_sample_sz)

        # Location of sample
        sample_pos, sample_scales = self.get_sample_location(sample_coords)
        target_box_crop = self.get_iounet_box(self.pos, self.target_sz, sample_pos[0,:], sample_scales[0])

        search_area_bb = self.generate_search_area_bb(target_box_crop.view(1, 4)[:, 2:])
        cls_sa_pos, cls_sa_scales = self.get_cls_search_area_location(sample_pos, sample_scales,
                                                                      search_area_bb[:, [3, 2]])

        # Extract classification features
        test_x_seg = self.get_seg_classification_features(backbone_feat)

        # perform segmentation
        seg_cls_scores = self.get_seg_classification_scores(test_x_seg)

        text_x_target = self.get_target_classification_features(backbone_feat,
                                                                search_area_bb=search_area_bb.to(self.params.device))

        target_scores = self.classify_target(text_x_target)
        # TODO what happens if cls feat extractor has downsampling?
        orig_feat_sz = (backbone_feat['layer3'].shape[-1], backbone_feat['layer3'].shape[-2])
        target_scores_resamp = self.get_resampled_score(target_scores, search_area_bb, orig_feat_sz)

        decoder_in = torch.cat((seg_cls_scores, target_scores_resamp.unsqueeze(2)), dim=2)

        seg_mask, decoder_feat = self.run_mask_decoder(decoder_in, backbone_feat)
        seg_mask = torch.sigmoid(seg_mask)

        # TODO get box from segmentation
        # bbox = self.mask_to_box(seg_mask)

        # Localize the target
        # TODO handle even kernel size.
        translation_vec, scale_ind, s, flag = self.localize_target(target_scores_resamp, cls_sa_pos, cls_sa_scales,
                                                                   search_area_bb)
        new_pos = sample_pos[scale_ind,:] + translation_vec

        self.debug_info['flag' + self.id_str] = flag

        self.search_area_box = torch.cat((sample_coords[scale_ind,[1,0]], sample_coords[scale_ind,[3,2]] - sample_coords[scale_ind,[1,0]] - 1))

        # Update position and scale
        if flag != 'not_found':
            if self.params.get('use_iou_net', False):
                update_scale_flag = self.params.get('update_scale_when_uncertain', True) or flag != 'uncertain'
                if self.params.get('use_classifier', True):
                    self.update_state(new_pos)
                self.refine_target_box(decoder_feat, sample_pos[scale_ind,:], sample_scales[scale_ind], scale_ind, update_scale_flag)
            elif self.params.get('use_classifier', True):
                self.update_state(new_pos)

            self.update_state(new_pos)

        # ------- UPDATE ------- #
        update_flag = flag not in ['not_found', 'uncertain']
        hard_negative = (flag == 'hard_negative')
        learning_rate = self.params.get('hard_negative_learning_rate', None) if hard_negative else None

        if update_flag and self.params.get('update_classifier', False):
            train_x = text_x_target[scale_ind:scale_ind + 1, ...]

            # Get train sample
            # Create target_box and label for spatial sample
            # target_box = self.get_iounet_box(self.pos, self.target_sz, sample_pos[scale_ind,:], sample_scales[scale_ind])

            target_box = self.get_iounet_box(self.pos, self.target_sz, cls_sa_pos[scale_ind, :],
                                             cls_sa_scales[scale_ind])

            # TODO clean this
            norm_factor = train_x.shape[-1] / search_area_bb[0, 2]
            target_box_feat = target_box * norm_factor

            # TODO create target mask
            # Update the classifier model
            # TODO handle updates
            self.update_classifier(train_x, target_box_feat, learning_rate)

        # Set the pos of the tracker to iounet pos
        if self.params.get('use_iou_net', True) and flag != 'not_found' and hasattr(self, 'pos_iounet'):
            self.pos = self.pos_iounet.clone()
        # score_map = s[scale_ind, ...]
        # max_score = torch.max(score_map).item()
        # self.debug_info['max_score' + self.id_str] = max_score


        # TODO Return mask
        # Compute output bounding box
        new_state = torch.cat((self.pos[[1,0]] - (self.target_sz[[1,0]]-1)/2, self.target_sz[[1,0]]))

        if self.params.get('output_not_found_box', False) and flag == 'not_found':
            output_state = [-1, -1, -1, -1]
        else:
            output_state = new_state.tolist()

        seg_mask_im = self.convert_mask_crop_to_im(seg_mask, im, sample_scales, sample_pos)

        seg_mask = seg_mask.view(*seg_mask.shape[-2:])
        seg_mask_im = seg_mask_im.view(*seg_mask_im.shape[-2:])

        prob_sum = seg_mask_im.sum()
        pos0 = torch.sum(seg_mask_im.sum(dim=-1) *
                         torch.arange(seg_mask_im.shape[-2], dtype=torch.float32)) / prob_sum
        pos1 = torch.sum(seg_mask_im.sum(dim=-2) *
                         torch.arange(seg_mask_im.shape[-1], dtype=torch.float32)) / prob_sum
        # self.pos = torch.Tensor([pos0, pos1])

        # pred_mask_im = (seg_mask_im > self.params.get('segmentation_thresh', 0.5)).int().cpu().numpy()
        seg_mask_im_np = seg_mask_im.cpu().numpy()

        if self.visdom is not None:
            if self.params.debug >= 3:
                self.visdom.register(seg_cls_scores, 'featmap', 3, 'Mask enc' + self.id_str)

            self.visdom.register(F.relu(target_scores_resamp), 'heatmap', 2, 'Score Map' + self.id_str)

            self.visdom.register(seg_mask, 'heatmap', 2, 'Seg Scores' + self.id_str)
            self.visdom.register(self.debug_info, 'info_dict', 1, 'Status')


        # pred_box = self._seg_to_box(pred_mask, object_id=1)

        out = {'segmentation': seg_mask_im_np, 'target_bbox': output_state}
        return out

    def get_cls_search_area_location(self, sample_pos, sample_scales, search_area_sz):
        """Get the location of the extracted sample."""
        sa_pos = sample_pos + (search_area_sz - self.img_sample_sz.view(1, 2)) / 2.0
        sa_scales = (sample_scales * search_area_sz / self.img_sample_sz).prod(dim=1).sqrt()

        return sa_pos, sa_scales

    def get_sample_location(self, sample_coord):
        """Get the location of the extracted sample."""
        sample_coord = sample_coord.float()
        sample_pos = 0.5*(sample_coord[:,:2] + sample_coord[:,2:] - 1)
        sample_scales = ((sample_coord[:,2:] - sample_coord[:,:2]) / self.img_sample_sz).prod(dim=1).sqrt()
        return sample_pos, sample_scales

    def get_centered_sample_pos(self):
        """Get the center position for the new sample. Make sure the target is correctly centered."""
        return self.pos + ((self.feature_sz + self.kernel_size) % 2) * self.target_scale * \
               self.img_support_sz / (2*self.feature_sz)

    def generate_search_area_bb(self, target_sizes):
        if self.params.get('use_adaptive_resampling', True):
            output_sz = self.img_sample_sz[[1, 0]].view(1, 2).to(target_sizes.device)
            search_bb_sz = (output_sz * (target_sizes.prod(dim=1, keepdim=True) / output_sz.prod()).sqrt() * self.params.search_area_scale).ceil()

            search_area_bb = torch.cat([torch.zeros((target_sizes.shape[0], 2), dtype=target_sizes.dtype, device=target_sizes.device), search_bb_sz], dim=1)
            return search_area_bb
        else:
            return None

    def classify_target(self, sample_x: TensorList):
        """Classify target by applying the DiMP filter."""
        with torch.no_grad():
            scores = self.target_classifier.classify(self.target_cls_filter, sample_x)

        return scores

    def get_resampled_score(self, scores, search_area_bb, orig_feat_sz):
        # Convert last iter scores to crop co-ordinates
        scores_resamp = self.target_classifier.resample_scores(scores, search_area_bb.unsqueeze(0).to(scores.device),
                                                               orig_feat_sz)
        return scores_resamp

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

        odds_t = segmentation_maps_t / (1.0 - segmentation_maps_t)
        odds_bg = bg_p / (1.0 - bg_p)

        norm_factor = odds_t.sum(0) + odds_bg

        segmentation_maps_t_agg = odds_t / norm_factor.unsqueeze(0)

        segmentation_maps_np_agg = segmentation_maps_t_agg.numpy()
        return segmentation_maps_np_agg

    def get_seg_classification_scores(self, sample_x: TensorList):
        """Classify target by applying the DiMP filter."""
        with torch.no_grad():
            seg_cls_scores = self.seg_classifier.classify(self.seg_cls_filter, sample_x)
        return seg_cls_scores

    def run_mask_decoder(self, decoder_in, backbone_feat):
        """Classify target by applying the DiMP filter."""
        with torch.no_grad():
            mask_pred, decoder_feat = self.net.mask_decoder(decoder_in, backbone_feat, ('m3',))

        return mask_pred, decoder_feat

    def localize_target(self, scores, sample_pos, sample_scales, search_area_bb):
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
            return self.localize_advanced(scores, sample_pos, sample_scales, search_area_bb)
        else:
            raise NotImplementedError

    def localize_advanced(self, scores, sample_pos, sample_scales, search_area_bb):
        """Run the target advanced localization (as in ATOM)."""

        sz = scores.shape[-2:]
        score_sz = torch.Tensor(list(sz))
        output_sz = score_sz - (self.kernel_size + 1) % 2
        score_center = (score_sz - 1)/2

        scores_hn = scores
        if self.output_window is not None and self.params.get('perform_hn_without_windowing', False):
            raise NotImplementedError
            scores_hn = scores.clone()
            scores *= self.output_window

        max_score1, max_disp1 = dcf.max2d(scores)
        _, scale_ind = torch.max(max_score1, dim=0)
        sample_scale = sample_scales[scale_ind]
        max_score1 = max_score1[scale_ind]
        max_disp1 = max_disp1[scale_ind,...].float().cpu().view(-1)
        target_disp1 = max_disp1 - score_center
        # translation_vec1 = target_disp1 * (search_area_bb[0, [3, 2]] / output_sz) * sample_scale
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
            return self.net.extract_classification_feat(backbone_feat)

    def get_target_classification_features(self, backbone_feat, search_area_bb):
        with torch.no_grad():
            return self.net.extract_target_classification_feat(backbone_feat, search_area_bb=search_area_bb)

    def get_iou_backbone_features(self, backbone_feat):
        return self.net.get_backbone_bbreg_feat(backbone_feat)

    def get_iou_features(self, backbone_feat):
        with torch.no_grad():
            return self.net.bb_regressor.get_iou_feat(self.get_iou_backbone_features(backbone_feat))

    def get_iou_modulation(self, iou_backbone_feat, target_boxes):
        with torch.no_grad():
            return self.net.bb_regressor.get_modulation(iou_backbone_feat, target_boxes)


    def generate_init_samples(self, im: torch.Tensor) -> TensorList:
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

        # TODO clean this
        if self.init_masks is not None:
            # Add extra dimension to make it a 1 channel image
            init_masks = sample_patch_transformed(self.init_masks,
                                                  self.init_sample_pos, self.init_sample_scale, aug_expansion_sz, self.transforms,
                                                  is_mask=True)
            self.init_masks = init_masks

        self.init_masks = self.init_masks.to(self.params.device)
        # Extract initial backbone features
        with torch.no_grad():
            init_backbone_feat = self.net.extract_backbone(im_patches)

        target_boxes = self.init_target_boxes()

        return init_backbone_feat, target_boxes

    def init_target_boxes(self):
        """Get the target bounding boxes for the initial augmented samples."""
        self.classifier_target_box = self.get_iounet_box(self.pos, self.target_sz, self.init_sample_pos, self.init_sample_scale)
        init_target_boxes = TensorList()
        for T in self.transforms:
            init_target_boxes.append(self.classifier_target_box + torch.Tensor([T.shift[1], T.shift[0], 0, 0]))
        init_target_boxes = torch.cat(init_target_boxes.view(1, 4), 0).to(self.params.device)
        # self.target_boxes = init_target_boxes.new_zeros(self.params.sample_memory_size, 4)
        # self.target_boxes[:init_target_boxes.shape[0],:] = init_target_boxes
        return init_target_boxes

    def init_memory(self, train_x: TensorList, train_box):
        # Initialize first-frame spatial training samples
        self.num_init_samples = train_x.size(0)
        init_sample_weights = TensorList([x.new_ones(1) / x.shape[0] for x in train_x])

        # Sample counters and weights for spatial
        self.num_stored_samples = self.num_init_samples.copy()
        self.previous_replace_ind = [None] * len(self.num_stored_samples)
        self.sample_weights = TensorList([x.new_zeros(self.params.sample_memory_size) for x in train_x])
        for sw, init_sw, num in zip(self.sample_weights, init_sample_weights, self.num_init_samples):
            sw[:num] = init_sw

        # Initialize memory
        self.training_samples = TensorList(
            [x.new_zeros(self.params.sample_memory_size, x.shape[1], x.shape[2], x.shape[3]) for x in train_x])

        for ts, x in zip(self.training_samples, train_x):
            ts[:x.shape[0],...] = x

        self.target_boxes = train_box.new_zeros(self.params.sample_memory_size, 4)
        self.target_boxes[:train_box.shape[0], :] = train_box


    def update_memory(self, sample_x: TensorList, target_box, learning_rate = None):
        # Update weights and get replace ind
        replace_ind = self.update_sample_weights(self.sample_weights, self.previous_replace_ind, self.num_stored_samples, self.num_init_samples, learning_rate)
        self.previous_replace_ind = replace_ind

        # Update sample and label memory
        for train_samp, x, ind in zip(self.training_samples, sample_x, replace_ind):
            train_samp[ind:ind+1,...] = x

        # Update bb memory
        self.target_boxes[replace_ind[0],:] = target_box
        # self.target_masks[replace_ind[0], :, :] = mask[0, 0, :, :]

        self.num_stored_samples += 1


    def update_sample_weights(self, sample_weights, previous_replace_ind, num_stored_samples, num_init_samples, learning_rate = None):
        # Update weights and get index to replace
        replace_ind = []
        for sw, prev_ind, num_samp, num_init in zip(sample_weights, previous_replace_ind, num_stored_samples, num_init_samples):
            lr = learning_rate
            if lr is None:
                lr = self.params.learning_rate

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


    def init_iou_net(self, backbone_feat):
        # Setup IoU net and objective
        for p in self.net.bb_regressor.parameters():
            p.requires_grad = False

        # Get target boxes for the different augmentations
        self.classifier_target_box = self.get_iounet_box(self.pos, self.target_sz, self.init_sample_pos, self.init_sample_scale)

        # Get iou features
        iou_backbone_feat = self.get_iou_backbone_features(backbone_feat)
        iou_backbone_feat = [f[0:1, :, :, :] for f in iou_backbone_feat]

        bb = self.classifier_target_box.view(1, 4).to(self.params.device)
        self.bbr_modulation_vector = self.net.bb_regressor.get_modulation(iou_backbone_feat, bb)

    def init_classifier(self, init_backbone_feat, init_target_boxes):
        # Get classification features
        search_area_bb = self.generate_search_area_bb(init_target_boxes[:, 2:])

        x_seg = self.get_seg_classification_features(init_backbone_feat)
        x_target = self.get_target_classification_features(init_backbone_feat, search_area_bb)

        self.seg_classifier = self.net.seg_classifier
        self.target_classifier = self.net.target_classifier

        # Set number of iterations
        plot_loss = self.params.debug > 0
        num_iter_seg = self.params.get('net_opt_iter_seg', None)

        # Get target filter by running the discriminative model prediction module
        with torch.no_grad():
            mask_enc = self.net.encode_masks(self.init_masks)
            self.seg_cls_filter, _, _ = self.seg_classifier.get_filter(x_seg.unsqueeze(1), mask_enc,
                                                                                num_iter=num_iter_seg)
        num_iter_target = self.params.get('net_opt_iter_target', None)

        norm_factor = x_target.shape[-1] / search_area_bb[:, 2:3]
        init_target_boxes_norm = init_target_boxes * norm_factor

        # Get target filter by running the discriminative model prediction module
        with torch.no_grad():
            self.target_cls_filter, _, losses = self.target_classifier.get_filter(x_target.unsqueeze(1), init_target_boxes_norm,
                                                                                num_iter=num_iter_target)

        # Init memory (Only dimp currently)
        if self.params.get('update_classifier', True):
            self.init_memory(TensorList([x_target]), init_target_boxes_norm)

        # Note: Currently using seg classifier info for feature sz, kernel sz etc
        # Set feature size and other related sizes
        self.feature_sz = torch.Tensor(list(x_target.shape[-2:]))
        ksz = self.target_classifier.filter_size
        self.kernel_size = torch.Tensor([ksz, ksz] if isinstance(ksz, (int, float)) else ksz)
        self.output_sz = self.feature_sz + (self.kernel_size + 1) % 2

        # Construct output window
        self.output_window = None
        if self.params.get('window_output', False):
            if self.params.get('use_clipped_window', False):
                self.output_window = dcf.hann2d_clipped(self.output_sz.long(), (
                        self.output_sz * self.params.effective_search_area / self.params.search_area_scale).long(),
                                                        centered=True).to(self.params.device)
            else:
                self.output_window = dcf.hann2d(self.output_sz.long(), centered=True).to(self.params.device)
            self.output_window = self.output_window.squeeze(0)

        #
        # if plot_loss:
        #     if isinstance(losses, dict):
        #         losses = losses['train']
        #     self.losses = torch.cat(losses)
        #     if self.visdom is not None:
        #         self.visdom.register((self.losses, torch.arange(self.losses.numel())), 'lineplot', 3, 'Training Loss' + self.id_str)
        #     elif self.params.debug >= 3:
        #         plot_graph(self.losses, 10, title='Training Loss' + self.id_str)


    def update_classifier(self, train_x, target_box, learning_rate=None):
        # Set flags and learning rate
        hard_negative_flag = learning_rate is not None
        if learning_rate is None:
            learning_rate = self.params.learning_rate

        # Update the tracker memory
        if hard_negative_flag or self.frame_num % self.params.get('train_sample_interval', 1) == 0:
            self.update_memory(TensorList([train_x]), target_box, learning_rate)

        # Decide the number of iterations to run
        num_iter = 0
        low_score_th = self.params.get('low_score_opt_threshold', None)
        if hard_negative_flag:
            num_iter = self.params.get('net_opt_hn_iter', None)
        elif (self.frame_num - 1) % self.params.train_skipping == 0:
            num_iter = self.params.get('net_opt_update_iter', None)

        plot_loss = self.params.debug > 0

        if num_iter > 0:
            # Get inputs for the DiMP filter optimizer module
            samples = self.training_samples[0][:self.num_stored_samples[0],...]
            # masks = self.target_masks[:self.num_stored_samples[0], ...]
            target_boxes = self.target_boxes[:self.num_stored_samples[0],:].clone()
            sample_weights = self.sample_weights[0][:self.num_stored_samples[0]]

            # Run the filter optimizer module
            with torch.no_grad():
                target_filter, _, losses = self.target_classifier.filter_optimizer(TensorList([self.target_cls_filter]),
                                                                                num_iter=num_iter, feat=samples,
                                                                                bb=target_boxes,
                                                                                sample_weight=sample_weights)

            self.target_cls_filter = target_filter[0]
            # if plot_loss:
            #     if isinstance(losses, dict):
            #         losses = losses['train']
            #     self.losses = torch.cat((self.losses, torch.cat(losses)))
            #     if self.visdom is not None:
            #         self.visdom.register((self.losses, torch.arange(self.losses.numel())), 'lineplot', 3, 'Training Loss' + self.id_str)
            #     elif self.params.debug >= 3:
            #         plot_graph(self.losses, 10, title='Training Loss' + self.id_str)

    def refine_target_box(self, backbone_feat, sample_pos, sample_scale, scale_ind, update_scale = True):
        """Run the ATOM IoUNet to refine the target bounding box."""

        # Initial box for refinement
        init_box = self.get_iounet_box(self.pos, self.target_sz, sample_pos, sample_scale)

        # Generate random initial boxes
        init_boxes = init_box.view(1, 1, 4).clone().to(self.params.device)

        # Optimize the boxes
        iou_feat = self.net.bb_regressor.get_iou_feat([backbone_feat['m3'], ])
        predicted_box = self.net.bb_regressor.predict_bb(self.bbr_modulation_vector, iou_feat, init_boxes)
        predicted_box = predicted_box.squeeze().cpu()
        # Get new position and size
        new_pos = predicted_box[:2] + predicted_box[2:] / 2
        new_pos = (new_pos.flip((0,)) - (self.img_sample_sz - 1) / 2) * sample_scale + sample_pos
        new_target_sz = predicted_box[2:].flip((0,)) * sample_scale
        new_scale = torch.sqrt(new_target_sz.prod() / self.base_target_sz.prod())

        self.pos_iounet = new_pos.clone()

        if self.params.get('use_iounet_pos_for_learning', True):
            self.pos = new_pos.clone()

        self.target_sz = new_target_sz

        if update_scale:
            self.target_scale = new_scale

        # self.visualize_iou_pred(iou_features, predicted_box)
