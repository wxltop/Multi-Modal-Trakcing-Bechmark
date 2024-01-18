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


class DiMPVos(BaseTracker):

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

        if update_flag and self.params.get('update_classifier', False) and self.prev_test_x is not None and (not self.params.get('use_gt_mask_enc_for_pred', False)):
            if self.params.get('use_merged_mask_for_memory', False):
                object_pos = None
                seg_masks = []

                # Fix this
                ct = 0
                for k, v in info['previous_output'].items():
                    if k == self.object_id:
                        object_pos = ct

                    if self.params.get('return_raw_scores', False):
                        seg_masks.append(v['segmentation_raw'])
                    else:
                        seg_masks.append(v['segmentation'])
                    ct = ct + 1
                segmentation_maps = np.stack(seg_masks)
                segmentation_maps_merged = self.merge_segmentation_results(segmentation_maps,
                                                                           raw_scores=self.params.get('return_raw_scores', False))

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

            if self.params.get('update_state_using_merged_mask', True):
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

            else:
                raise Exception

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

        # TODO perform segmentation
        if not self.params.get('use_gt_mask_enc_for_pred', False):
            seg_mask, bb_pred = self.segment_target(test_x, backbone_feat)
        else:
            gt_mask_crop, _ = sample_patch(torch.from_numpy(self.gt_mask).unsqueeze(0).unsqueeze(0).float(),
                                           self.prev_pos, self.prev_scale * self.img_sample_sz,
                                           self.img_sample_sz,
                                           mode=self.params.get('border_mode', 'replicate'),
                                           max_scale_change=self.params.get('patch_max_scale_change', None),
                                           is_mask=True)
            seg_mask, bb_pred = self.segment_target_using_gt_enc(test_x, backbone_feat, gt_mask_crop)

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
        seg_mask_im = torch.sigmoid(seg_mask_im)

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
    def merge_segmentation_results(segmentation_maps, raw_scores=False):
        # Soft aggregation from RGMP
        eps = 1e-7

        if not raw_scores:
            segmentation_maps_t = torch.from_numpy(segmentation_maps).float().clamp(eps, 1.0 - eps)

            bg_p = torch.prod(1 - segmentation_maps_t, dim=0).clamp(eps, 1.0 - eps)  # bg prob

            segm_all = torch.cat((bg_p.unsqueeze(0), segmentation_maps_t), dim=0)
            odds_t = segm_all / (1.0 - segm_all)

            #odds_t = segmentation_maps_t / (1.0 - segmentation_maps_t)
            #odds_bg = bg_p / (1.0 - bg_p)

            #norm_factor = odds_t.sum(0) + odds_bg
            norm_factor = odds_t.sum(0)

            segmentation_maps_t_agg = odds_t / norm_factor.unsqueeze(0)

            segmentation_maps_np_agg = segmentation_maps_t_agg.numpy()
            return segmentation_maps_np_agg
        else:
            segmentation_maps_t = torch.from_numpy(segmentation_maps).float()
            segmentation_maps_t_prob = torch.sigmoid(segmentation_maps_t)

            bg_p = torch.prod(1 - segmentation_maps_t_prob, dim=0).clamp(eps, 1.0 - eps)  # bg prob
            bg_score = (bg_p / (1.0 - bg_p)).log()

            scores_all = torch.cat((bg_score.unsqueeze(0), segmentation_maps_t), dim=0)

            out = []
            for s in scores_all:
                s_out = 1.0 / (scores_all - s.unsqueeze(0)).exp().sum(dim=0)
                out.append(s_out)

            segmentation_maps_t_agg = torch.stack(out, dim=0)
            segmentation_maps_np_agg = segmentation_maps_t_agg.numpy()
            return segmentation_maps_np_agg

    def segment_target(self, sample_clf_feat, sample_x):
        """Classify target by applying the DiMP filter."""
        with torch.no_grad():
            mask, box, aux_mask_pred = self.net.segment_target(self.target_filter, sample_clf_feat, sample_x)
        # if 'mask_enc_iter' in aux_mask_pred.keys():
        #     aux_pred = aux_mask_pred['mask_enc_iter']
        #
        #     if self.visdom is not None:
        #         self.visdom.register(aux_pred.squeeze(), 'heatmap', 2, 'Aux Seg Scores' + self.id_str)
        #
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

    def get_classification_features(self, backbone_feat):
        with torch.no_grad():
            return self.net.extract_classification_feat(backbone_feat)

    def get_iou_backbone_features(self, backbone_feat):
        return self.net.get_backbone_bbreg_feat(backbone_feat)

    def get_iou_features(self, backbone_feat):
        with torch.no_grad():
            return self.net.bb_regressor.get_iou_feat(self.get_iou_backbone_features(backbone_feat))

    def get_iou_modulation(self, iou_backbone_feat, target_boxes):
        with torch.no_grad():
            return self.net.bb_regressor.get_modulation(iou_backbone_feat, target_boxes)


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
        target_boxes = TensorList()
        if self.params.iounet_augmentation:
            for T in self.transforms:
                if not isinstance(T, (augmentation.Identity, augmentation.Translation, augmentation.FlipHorizontal, augmentation.FlipVertical, augmentation.Blur)):
                    break
                target_boxes.append(self.classifier_target_box + torch.Tensor([T.shift[1], T.shift[0], 0, 0]))
        else:
            target_boxes.append(self.classifier_target_box + torch.Tensor([self.transforms[0].shift[1], self.transforms[0].shift[0], 0, 0]))
        target_boxes = torch.cat(target_boxes.view(1,4), 0).to(self.params.device)

        # Get iou features
        iou_backbone_feat = self.get_iou_backbone_features(backbone_feat)

        # Remove other augmentations such as rotation
        iou_backbone_feat = TensorList([x[:target_boxes.shape[0],...] for x in iou_backbone_feat])

        # Get modulation vector
        self.iou_modulation = self.get_iou_modulation(iou_backbone_feat, target_boxes)
        self.iou_modulation = TensorList([x.detach().mean(0) for x in self.iou_modulation])


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
            mask_enc = self.net.label_encoder(init_masks, x.unsqueeze(1))
        else:
            mask_enc = init_masks

        # init_masks_d = F.interpolate(init_masks, (30, 52), mode='bilinear') > 0
        # mask_enc[1][0, 0, :, init_masks_d.squeeze()] = mask_enc[1][0, 0, :, init_masks_d.squeeze()] * 2.0

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

        # Extract features from the relevant scale
        iou_features = self.get_iou_features(backbone_feat)
        iou_features = TensorList([x[scale_ind:scale_ind+1,...] for x in iou_features])

        # Generate random initial boxes
        init_boxes = init_box.view(1,4).clone()
        if self.params.num_init_random_boxes > 0:
            square_box_sz = init_box[2:].prod().sqrt()
            rand_factor = square_box_sz * torch.cat([self.params.box_jitter_pos * torch.ones(2), self.params.box_jitter_sz * torch.ones(2)])

            minimal_edge_size = init_box[2:].min()/3
            rand_bb = (torch.rand(self.params.num_init_random_boxes, 4) - 0.5) * rand_factor
            new_sz = (init_box[2:] + rand_bb[:,2:]).clamp(minimal_edge_size)
            new_center = (init_box[:2] + init_box[2:]/2) + rand_bb[:,:2]
            init_boxes = torch.cat([new_center - new_sz/2, new_sz], 1)
            init_boxes = torch.cat([init_box.view(1,4), init_boxes])

        # Optimize the boxes
        output_boxes, output_iou = self.optimize_boxes(iou_features, init_boxes)

        # Remove weird boxes
        output_boxes[:, 2:].clamp_(1)
        aspect_ratio = output_boxes[:,2] / output_boxes[:,3]
        keep_ind = (aspect_ratio < self.params.maximal_aspect_ratio) * (aspect_ratio > 1/self.params.maximal_aspect_ratio)
        output_boxes = output_boxes[keep_ind,:]
        output_iou = output_iou[keep_ind]

        # If no box found
        if output_boxes.shape[0] == 0:
            return

        # Predict box
        k = self.params.get('iounet_k', 5)
        topk = min(k, output_boxes.shape[0])
        _, inds = torch.topk(output_iou, topk)
        predicted_box = output_boxes[inds, :].mean(0)
        predicted_iou = output_iou.view(-1, 1)[inds, :].mean(0)

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


    def optimize_boxes(self, iou_features, init_boxes):
        box_refinement_space = self.params.get('box_refinement_space', 'default')
        if box_refinement_space == 'default':
            return self.optimize_boxes_default(iou_features, init_boxes)
        if box_refinement_space == 'relative':
            return self.optimize_boxes_relative(iou_features, init_boxes)
        raise ValueError('Unknown box_refinement_space {}'.format(box_refinement_space))


    def optimize_boxes_default(self, iou_features, init_boxes):
        # Optimize iounet boxes
        output_boxes = init_boxes.view(1, -1, 4).to(self.params.device)
        step_length = self.params.box_refinement_step_length
        if isinstance(step_length, (tuple, list)):
            step_length = torch.Tensor([step_length[0], step_length[0], step_length[1], step_length[1]], device=self.params.device).view(1,1,4)

        for i_ in range(self.params.box_refinement_iter):
            # forward pass
            bb_init = output_boxes.clone().detach()
            bb_init.requires_grad = True

            outputs = self.net.bb_regressor.predict_iou(self.iou_modulation, iou_features, bb_init)

            if isinstance(outputs, (list, tuple)):
                outputs = outputs[0]

            outputs.backward(gradient = torch.ones_like(outputs))

            # Update proposal
            output_boxes = bb_init + step_length * bb_init.grad * bb_init[:, :, 2:].repeat(1, 1, 2)
            output_boxes.detach_()

            step_length *= self.params.box_refinement_step_decay

        return output_boxes.view(-1,4).cpu(), outputs.detach().view(-1).cpu()


    def optimize_boxes_relative(self, iou_features, init_boxes):
        # Optimize iounet boxes
        output_boxes = init_boxes.view(1, -1, 4).to(self.params.device)
        step_length = self.params.box_refinement_step_length
        if isinstance(step_length, (tuple, list)):
            step_length = torch.Tensor([step_length[0], step_length[0], step_length[1], step_length[1]]).to(self.params.device).view(1,1,4)

        sz_norm = output_boxes[:,:1,2:].clone()
        output_boxes_rel = bbutils.rect_to_rel(output_boxes, sz_norm)
        for i_ in range(self.params.box_refinement_iter):
            # forward pass
            bb_init_rel = output_boxes_rel.clone().detach()
            bb_init_rel.requires_grad = True

            bb_init = bbutils.rel_to_rect(bb_init_rel, sz_norm)
            outputs = self.net.bb_regressor.predict_iou(self.iou_modulation, iou_features, bb_init)

            if isinstance(outputs, (list, tuple)):
                outputs = outputs[0]

            outputs.backward(gradient = torch.ones_like(outputs))

            # Update proposal
            output_boxes_rel = bb_init_rel + step_length * bb_init_rel.grad
            output_boxes_rel.detach_()

            step_length *= self.params.box_refinement_step_decay

        #     for s in outputs.view(-1):
        #         print('{:.2f}  '.format(s.item()), end='')
        #     print('')
        # print('')

        output_boxes = bbutils.rel_to_rect(output_boxes_rel, sz_norm)

        return output_boxes.view(-1,4).cpu(), outputs.detach().view(-1).cpu()


    def visualize_iou_pred(self, iou_features, center_box):
        center_box = center_box.view(1,1,4)
        sz_norm = center_box[...,2:].clone()
        center_box_rel = bbutils.rect_to_rel(center_box, sz_norm)

        pos_dist = 1.0
        sz_dist = math.log(3.0)
        pos_step = 0.01
        sz_step = 0.01

        pos_scale = torch.arange(-pos_dist, pos_dist+pos_step, step=pos_step)
        sz_scale = torch.arange(-sz_dist, sz_dist+sz_step, step=sz_step)

        bbx = torch.zeros(1, pos_scale.numel(), 4)
        bbx[0,:,0] = pos_scale.clone()
        bby = torch.zeros(pos_scale.numel(), 1, 4)
        bby[:,0,1] = pos_scale.clone()
        bbw = torch.zeros(1, sz_scale.numel(), 4)
        bbw[0,:,2] = sz_scale.clone()
        bbh = torch.zeros(sz_scale.numel(), 1, 4)
        bbh[:,0,3] = sz_scale.clone()

        pos_boxes = bbutils.rel_to_rect((center_box_rel + bbx) + bby, sz_norm).view(1,-1,4).to(self.params.device)
        sz_boxes = bbutils.rel_to_rect((center_box_rel + bbw) + bbh, sz_norm).view(1,-1,4).to(self.params.device)

        pos_scores = self.net.bb_regressor.predict_iou(self.iou_modulation, iou_features, pos_boxes).exp()
        sz_scores = self.net.bb_regressor.predict_iou(self.iou_modulation, iou_features, sz_boxes).exp()

        show_tensor(pos_scores.view(pos_scale.numel(),-1), title='Position scores', fig_num=21)
        show_tensor(sz_scores.view(sz_scale.numel(),-1), title='Size scores', fig_num=22)


    def visdom_draw_tracking(self, image, box, segmentation=None):
        box = (box,)
        if segmentation is None:
            self.visdom.register((image, *box), 'Tracking', 1, 'Tracking')
        else:
            self.visdom.register((image, *box, self.search_area_box, segmentation), 'Tracking', 1, 'Tracking')
