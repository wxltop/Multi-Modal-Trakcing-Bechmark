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
from collections import OrderedDict


class DiMPVos(BaseTracker):

    multiobj_mode = 'parallel'

    def predicts_segmentation_mask(self):
        return True

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

        self.prev_output_state = state
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
        self.min_scale_factor = 1.0#torch.max(10 / self.base_target_sz)
        self.max_scale_factor = torch.min(self.image_sz / self.base_target_sz)

        # Extract and transform sample
        init_backbone_feat, init_masks = self.generate_init_samples(im, init_mask)

        # Initialize classifier
        self.init_classifier(init_backbone_feat, init_masks)

        self.prev_flag = 'init'
        self.prev_test_x = None

        out = {'time': time.time() - tic}
        return out


    def track(self, image, info: dict = None) -> dict:
        self.debug_info = {}

        # ------- UPDATE ------- #
        update_flag = self.prev_flag not in ['not_found', 'uncertain']
        hard_negative = (self.prev_flag == 'hard_negative')
        learning_rate = self.params.get('hard_negative_learning_rate', None) if hard_negative else None

        if update_flag and self.params.get('update_classifier', False) and self.prev_test_x is not None:
            if self.object_id is None:
                seg_mask_im = info['previous_output']['segmentation_raw']
            else:
                seg_mask_im = info['previous_output']['segmentation_raw'][self.object_id]

            seg_mask_im = torch.from_numpy(seg_mask_im).unsqueeze(0).unsqueeze(0).float()

            self.pos, self.target_sz = self.get_target_state(seg_mask_im.squeeze())

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

            seg_mask_crop, _ = sample_patch(seg_mask_im, self.prev_pos, self.prev_scale * self.img_sample_sz,
                                            self.img_sample_sz,
                                            mode=self.params.get('border_mode', 'replicate'),
                                            max_scale_change=self.params.get(
                                                'patch_max_scale_change', None), is_mask=True)

            if self.visdom is not None:
                self.visdom.register(seg_mask_im.squeeze()  , 'heatmap', 3, 'Seg Scores Merge' + self.id_str)

            if self.params.get('store_enc', False):
                seg_mask_enc = self.net.label_encoder(seg_mask_crop.clone(), self.prev_test_x.unsqueeze(1))
                seg_mask_enc = seg_mask_enc.view(1, *seg_mask_enc.shape[-3:])
            else:
                seg_mask_enc = seg_mask_crop.clone()

            self.update_classifier(self.prev_test_x, seg_mask_enc, learning_rate)

        self.frame_num += 1
        self.debug_info['frame_num'] = self.frame_num

        # Convert image
        im = numpy_to_torch(image)

        # ------- LOCALIZATION ------- #
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

        # Extract classification features
        test_x = self.get_classification_features(backbone_feat)

        # Location of sample
        sample_pos, sample_scales = self.get_sample_location(sample_coords)

        seg_mask, bb_pred = self.segment_target(test_x, backbone_feat)

        scale_ind = 0
        flag = 'normal'
        self.search_area_box = torch.cat((sample_coords[scale_ind,[1,0]], sample_coords[scale_ind,[3,2]] - sample_coords[scale_ind,[1,0]] - 1))

        self.prev_flag = flag
        self.prev_test_x = test_x

        # Compute output bounding box
        new_state = torch.cat((self.pos[[1,0]] - (self.target_sz[[1,0]]-1)/2, self.target_sz[[1,0]]))

        if self.params.get('output_not_found_box', False) and flag == 'not_found':
            output_state = [-1, -1, -1, -1]
        else:
            output_state = new_state.tolist()

        seg_mask_im = self.convert_mask_crop_to_im(seg_mask, im, sample_scales, sample_pos)

        seg_mask_im_raw = seg_mask_im.clone()
        seg_mask_im = (seg_mask_im > 0.0).float()

        if self.object_id is None:
            # Single object mode, no merge called
            seg_mask_im_raw = torch.sigmoid(seg_mask_im_raw)

        # seg_mask_im = torch.sigmoid(seg_mask_im)

        seg_mask_im = seg_mask_im.view(*seg_mask_im.shape[-2:])

        # pred_mask_im = (seg_mask_im > self.params.get('segmentation_thresh', 0.5)).int().cpu().numpy()
        seg_mask_im_np = seg_mask_im.cpu().numpy()
        seg_mask_im_raw_np = seg_mask_im_raw.cpu().numpy()

        if self.visdom is not None:
            self.visdom.register(seg_mask, 'heatmap', 2, 'Seg Scores' + self.id_str)
            self.visdom.register(self.debug_info, 'info_dict', 1, 'Status')

        if bb_pred is not None:
            bb_pred = self.convert_box_crop_to_im(bb_pred, sample_scales, sample_pos)

            output_state = bb_pred.cpu().view(-1).tolist()

            if self.params.get('use_estimated_bb_for_crop', False) and output_state[-1] > 10 and output_state[-2] > 10:
                self.pos = torch.Tensor([output_state[1] + (output_state[3] - 1) / 2, output_state[0] + (output_state[2] - 1) / 2])
                self.target_sz = torch.Tensor([output_state[3], output_state[2]])
                self.target_scale = torch.sqrt(self.target_sz.prod() / self.base_target_sz.prod())

        out = {'segmentation': seg_mask_im_np, 'target_bbox': output_state, 'segmentation_raw': seg_mask_im_raw_np}
        return out

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

    def classify_target(self, sample_x: TensorList):
        """Classify target by applying the DiMP filter."""
        with torch.no_grad():
            scores = self.net.classifier.classify(self.target_filter, sample_x)
        return scores

    def convert_box_crop_to_im(self, box, sample_scales, sample_pos):
        box = box.cpu()
        box[:, [1, 0]] -= (self.img_sample_sz / 2.0)
        box_sc = box*sample_scales[0].item()

        box_sc[:, [1, 0]] += sample_pos
        return box_sc

    def convert_mask_crop_to_im(self, seg_mask, im, sample_scales, sample_pos):
        seg_mask_re = F.interpolate(seg_mask, scale_factor=sample_scales[0].item(), mode='bilinear')
        seg_mask_re = seg_mask_re.view(*seg_mask_re.shape[-2:])

        # Regions outside search area get very low score
        seg_mask_im = torch.ones(im.shape[-2:], dtype=seg_mask_re.dtype) * (-100.0)
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
    def merge_results(out_all):
        out_merged = OrderedDict()

        obj_ids = list(out_all.keys())

        raw_scores = []
        for id in obj_ids:
            if 'segmentation_raw' in out_all[id].keys():
                raw_scores.append(out_all[id]['segmentation_raw'])
            else:
                raw_scores.append((out_all[id]['segmentation'] - 0.5) * 200.0)

        raw_scores = np.stack(raw_scores)

        raw_scores = torch.from_numpy(raw_scores).float()
        raw_scores_prob = torch.sigmoid(raw_scores)

        eps = 1e-7
        bg_p = torch.prod(1 - raw_scores_prob, dim=0).clamp(eps, 1.0 - eps)  # bg prob
        bg_score = (bg_p / (1.0 - bg_p)).log()

        raw_scores_all = torch.cat((bg_score.unsqueeze(0), raw_scores), dim=0)

        out = []
        for s in raw_scores_all:
            s_out = 1.0 / (raw_scores_all - s.unsqueeze(0)).exp().sum(dim=0)
            out.append(s_out)

        segmentation_maps_t_agg = torch.stack(out, dim=0)
        segmentation_maps_np_agg = segmentation_maps_t_agg.numpy()

        obj_ids_all = np.array([0, *map(int, obj_ids)], dtype=np.uint8)
        merged_segmentation = obj_ids_all[segmentation_maps_np_agg.argmax(axis=0)]

        out_merged['segmentation'] = merged_segmentation
        out_merged['segmentation_raw'] = OrderedDict({key: segmentation_maps_np_agg[i + 1]
                                                      for i, key in enumerate(obj_ids)})

        out_first = list(out_all.values())[0]
        out_types = out_first.keys()

        for key in out_types:
            if 'segmentation' in key:
                pass
            else:
                out_merged[key] = {obj_id: out[key] for obj_id, out in out_all.items()}

        return out_merged

    def segment_target(self, sample_clf_feat, sample_x):
        """Classify target by applying the DiMP filter."""
        with torch.no_grad():
            mask, box, aux_mask_pred = self.net.segment_target(self.target_filter, sample_clf_feat, sample_x)

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

    def extract_backbone_features(self, im: torch.Tensor, pos: torch.Tensor, scales, sz: torch.Tensor):
        im_patches, patch_coords = sample_patch_multiscale(im, pos, scales, sz,
                                                           mode=self.params.get('border_mode', 'replicate'),
                                                           max_scale_change=self.params.get('patch_max_scale_change', None))
        with torch.no_grad():
            backbone_feat = self.net.extract_backbone(im_patches)
        return backbone_feat, patch_coords, im_patches

    def get_classification_features(self, backbone_feat):
        with torch.no_grad():
            return self.net.extract_classification_feat(backbone_feat)

    def generate_init_samples(self, im: torch.Tensor, init_mask) -> TensorList:
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

        return init_backbone_feat, init_masks

    def init_target_boxes(self):
        """Get the target bounding boxes for the initial augmented samples."""
        self.classifier_target_box = self.get_iounet_box(self.pos, self.target_sz, self.init_sample_pos, self.init_sample_scale)
        init_target_boxes = TensorList()
        for T in self.transforms:
            init_target_boxes.append(self.classifier_target_box + torch.Tensor([T.shift[1], T.shift[0], 0, 0]))
        init_target_boxes = torch.cat(init_target_boxes.view(1, 4), 0).to(self.params.device)
        self.target_boxes = init_target_boxes.new_zeros(self.params.sample_memory_size, 4)
        self.target_boxes[:init_target_boxes.shape[0],:] = init_target_boxes
        return init_target_boxes

    def init_memory(self, train_x: TensorList, masks):
        assert masks.dim() == 4

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

        self.target_masks = masks.new_zeros(self.params.sample_memory_size, masks.shape[-3], masks.shape[-2],
                                            masks.shape[-1])
        self.target_masks[:masks.shape[0], :, :, :] = masks

        for ts, x in zip(self.training_samples, train_x):
            ts[:x.shape[0],...] = x


    def update_memory(self, sample_x: TensorList, mask, learning_rate = None):
        # Update weights and get replace ind
        replace_ind = self.update_sample_weights(self.sample_weights, self.previous_replace_ind, self.num_stored_samples, self.num_init_samples, learning_rate)
        self.previous_replace_ind = replace_ind

        # Update sample and label memory
        for train_samp, x, ind in zip(self.training_samples, sample_x, replace_ind):
            train_samp[ind:ind+1,...] = x

        # Update bb memory
        # self.target_boxes[replace_ind[0],:] = target_box
        self.target_masks[replace_ind[0], :, :, :] = mask[0, ...]

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
                    if self.params.get('lower_init_weight', False):
                        sw[r_ind] = 1
                    else:
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

    def init_classifier(self, init_backbone_feat, init_masks):
        # Get classification features
        x = self.get_classification_features(init_backbone_feat)

        # Add the dropout augmentation here, since it requires extraction of the classification features
        if 'dropout' in self.params.augmentation and self.params.get('use_augmentation', True):
            raise NotImplementedError

        # Set feature size and other related sizes
        self.feature_sz = torch.Tensor(list(x.shape[-2:]))
        ksz = self.net.classifier.filter_size
        self.kernel_size = torch.Tensor([ksz, ksz] if isinstance(ksz, (int, float)) else ksz)
        self.output_sz = self.feature_sz + (self.kernel_size + 1)%2

        # Set number of iterations
        plot_loss = self.params.debug > 0
        num_iter = self.params.get('net_opt_iter', None)

        if self.net.label_encoder is not None:
            with torch.no_grad():
                mask_enc = self.net.label_encoder(init_masks, x.unsqueeze(1))
        else:
            mask_enc = init_masks

        # Get target filter by running the discriminative model prediction module
        with torch.no_grad():
            self.target_filter, _, losses = self.net.classifier.get_filter(x.unsqueeze(1), mask_enc,
                                                                           num_iter=num_iter)

        # Init memory
        if self.params.get('update_classifier', True):
            if self.params.get('store_enc', False):
                self.init_memory(TensorList([x]), masks=mask_enc.view(-1, *mask_enc.shape[-3:]))
            else:
                self.init_memory(TensorList([x]), masks=init_masks.view(-1, 1, *init_masks.shape[-2:]))

    def update_classifier(self, train_x, mask, learning_rate=None):
        # Set flags and learning rate
        hard_negative_flag = learning_rate is not None
        if learning_rate is None:
            learning_rate = self.params.learning_rate

        # Update the tracker memory
        if hard_negative_flag or self.frame_num % self.params.get('train_sample_interval', 1) == 0:
            self.update_memory(TensorList([train_x]), mask, learning_rate)

        # Decide the number of iterations to run
        num_iter = 0
        if hard_negative_flag:
            num_iter = self.params.get('net_opt_hn_iter', None)
        elif (self.frame_num - 1) % self.params.train_skipping == 0:
            num_iter = self.params.get('net_opt_update_iter', None)

        plot_loss = self.params.debug > 0

        if num_iter > 0:
            # Get inputs for the DiMP filter optimizer module
            samples = self.training_samples[0][:self.num_stored_samples[0],...]
            masks = self.target_masks[:self.num_stored_samples[0], ...]

            if self.params.get('store_enc', False):
                mask_enc_info = masks
            else:
                with torch.no_grad():
                    mask_enc_info = self.net.label_encoder(masks, samples.unsqueeze(1))

            # target_boxes = self.target_boxes[:self.num_stored_samples[0],:].clone()
            sample_weights = self.sample_weights[0][:self.num_stored_samples[0]]

            if isinstance(mask_enc_info, (tuple, list)):
                mask_enc = mask_enc_info[0]
                sample_weights = mask_enc_info[1] * sample_weights.view(-1, 1, 1, 1, 1)
                pass
            else:
                mask_enc = mask_enc_info

            # Run the filter optimizer module
            with torch.no_grad():
                target_filter, _, losses = self.net.classifier.filter_optimizer(TensorList([self.target_filter]),
                                                                                num_iter=num_iter, feat=samples.unsqueeze(1),
                                                                                mask=mask_enc.unsqueeze(1),
                                                                                sample_weight=sample_weights)

            self.target_filter = target_filter[0]

    def visdom_draw_tracking(self, image, box, segmentation=None):
        box = (box,)
        if segmentation is None:
            self.visdom.register((image, *box), 'Tracking', 1, 'Tracking')
        else:
            self.visdom.register((image, *box, self.search_area_box, segmentation), 'Tracking', 1, 'Tracking')
