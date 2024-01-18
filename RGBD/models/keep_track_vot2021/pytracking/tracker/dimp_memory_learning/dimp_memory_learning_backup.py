from pytracking.tracker.base import BaseTracker
import torch
import torch.nn.functional as F
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

# from ltr.models.weight_predictor.features import compute_cosine_similarity, compute_proj_test_score_on_memory_frames_feature
from ltr.models.memory_learning.attention import compute_cosine_similarity, proj_test_to_mem, proj_mem_to_test

from collections import defaultdict

import numpy as np


def confusion_matrix(gth, pred):
    tp = np.sum(np.logical_and(pred, gth))
    fp = np.sum(np.logical_and(pred, np.logical_not(gth)))
    fn = np.sum(np.logical_and(np.logical_not(pred), gth))
    tn = np.sum(np.logical_and(np.logical_not(pred), np.logical_not(gth)))
    return tp, fp, fn, tn

def precision(tp, fp): return tp / float(tp + fp)
def recall(tp, fn): return tp / float(tp + fn)
def TNR(tn, fp): return  1.0 if (tn + fp) == 0 else tn / float(tn + fp)
def f1_score(tp, fp, fn): return 2 * tp / float(2 * tp + fp + fn)
def accuracy(tp, tn, fp, fn): return (tp + tn) / float(tp + fp + fn + tn)

def cross_correlation(a, b):
    a, b = np.array(a), np.array(b)
    a_norm = (a - np.mean(a)) / np.std(a)
    b_norm = (b - np.mean(b)) / np.std(b)
    return np.corrcoef(a_norm, b_norm)[0, 1]

def find_local_maxima(scores, th=0.):
    peaks = dict()
    scores = scores.clone()
    scores[scores<=0] = 0

    ths = torch.sort(scores.view(-1), descending=True)[0]
    ths = ths[ths>th]

    for j, t in enumerate(ths):
        mask = torch.zeros_like(scores, dtype=torch.bool)
        mask[scores>=t] = True

        components = torch.arange(1, 23*23+1).reshape(1, 1, 23, 23).float().to(scores.device)
        components[~mask] = 0

        components_old = components.clone()
        while True:
            components[mask] = F.max_pool2d(components, kernel_size=3, stride=1, padding=1)[mask]
            if torch.allclose(components, components_old):
                break
            else:
                components_old = components.clone()

        peak_maps = components[0,0]

        peak_ids = torch.unique(peak_maps.view(-1))
        peak_ids = peak_ids[peak_ids > 0]
        num_peaks = peak_ids.shape[0]

        s = scores[0, 0].clone()
        for _ in range(0, num_peaks):
            peak_val, peak_coord = dcf.max2d(s)
            peak_id = peak_maps[peak_coord[0], peak_coord[1]]
            mask = (peak_id == peak_maps)
            s[mask] = 0
            peak_coord = (peak_coord[0].item(), peak_coord[1].item())
            if peak_val > 0:
                if peak_coord not in peaks:
                    peaks[peak_coord] = {'score': peak_val, 'prom': peak_val - t}
                else:
                    peaks[peak_coord]['prom'] = peak_val - t

    # peaks = {key: val for key, val in peaks.items() if val['prom'] > 0.01}

    num_peaks = len(peaks)
    peak_scores = torch.zeros(num_peaks).to(scores.device)
    peak_proms = torch.zeros(num_peaks).to(scores.device)
    peak_coords = torch.zeros((num_peaks, 2), dtype=torch.int).to(scores.device)

    for k, (coord, vals) in enumerate(peaks.items()):
        peak_coords[k] = torch.tensor(coord, dtype=torch.int).to(scores.device)
        peak_scores[k] = vals['score']
        peak_proms[k] = vals['prom']

    return peak_coords, peak_scores, peak_proms


def find_local_maxima_v2(scores, th=0.01, ks=5):
    scores_max = F.max_pool2d(scores.reshape(1,1,scores.shape[-2], scores.shape[-1]),
                              kernel_size=ks, stride=1, padding=ks//2)
    scores_max = scores_max.reshape(scores.shape)

    peak_mask = (scores == scores_max) & (scores > th)
    coord = np.nonzero(peak_mask)
    intensities = scores[coord[:,0], coord[:,1]]

    # Highest peak first
    idx_maxsort = torch.argsort(-intensities)
    coord = coord[idx_maxsort]
    intensities = intensities[idx_maxsort]
    return coord, intensities


class DiMPMemoryLearning(BaseTracker):

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

        # Convert image
        im = numpy_to_torch(image)

        # Get target position and size
        state = info['init_bbox']
        self.pos = torch.Tensor([state[1] + (state[3] - 1)/2, state[0] + (state[2] - 1)/2])
        self.target_sz = torch.Tensor([state[3], state[2]])

        # Get object id
        self.object_id = info.get('object_ids', [None])[0]
        self.id_str = '' if self.object_id is None else ' {}'.format(self.object_id)

        # Set sizes
        self.image_sz = torch.Tensor([im.shape[2], im.shape[3]])
        sz = self.params.image_sample_size
        sz = torch.Tensor([sz, sz] if isinstance(sz, int) else sz)
        if self.params.get('use_image_aspect_ratio', False):
            sz = self.image_sz * sz.prod().sqrt() / self.image_sz.prod().sqrt()
            stride = self.params.get('feature_stride', 32)
            sz = torch.round(sz / stride) * stride
        self.img_sample_sz = sz
        self.img_support_sz = self.img_sample_sz

        # Set search area
        search_area = torch.prod(self.target_sz * self.params.search_area_scale).item()
        self.target_scale =  math.sqrt(search_area) / self.img_sample_sz.prod().sqrt()

        # Target size in base scale
        self.base_target_sz = self.target_sz / self.target_scale

        # Setup scale factors
        if not self.params.has('scale_factors'):
            self.params.scale_factors = torch.ones(1)
        elif isinstance(self.params.scale_factors, (list, tuple)):
            self.params.scale_factors = torch.Tensor(self.params.scale_factors)

        # Setup scale bounds
        self.min_scale_factor = torch.max(10 / self.base_target_sz)
        self.max_scale_factor = torch.min(self.image_sz / self.base_target_sz)

        # Extract and transform sample
        init_backbone_feat = self.generate_init_samples(im)

        # Initialize classifier
        self.init_classifier(init_backbone_feat)

        # Initialize IoUNet
        if self.params.get('use_iou_net', True):
            self.init_iou_net(init_backbone_feat)

        self.previous_state = None
        self.logging_dict = defaultdict(list)
        self.previous_score_map = None

        self.cp_net = None
        if hasattr(self.params, 'cp_net'):
            self.params.cp_net.initialize()
            self.cp_net = self.params.cp_net

        self.peak_pred_net = None
        self.score_map_prev = None
        self.train_y_prev = None

        if hasattr(self.params, 'peak_pred_net'):
            self.params.peak_pred_net.initialize()
            self.peak_pred_net = self.params.peak_pred_net

        self.weights = None

        self.target_scales = []
        self.target_not_found_counter = 0

        self.mem_sort_indices = torch.arange(0, self.num_init_samples[0])

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
        # Extract classification features
        test_x = self.get_classification_features(backbone_feat)

        # Location of sample
        sample_pos, sample_scales = self.get_sample_location(sample_coords)

        # Compute classification scores
        scores_raw, scores_unfused, sim = self.classify_target(test_x)

        # Localize the target
        translation_vec, scale_ind, s, flag = self.localize_target(scores_raw, sample_pos, sample_scales)

        new_pos = sample_pos[scale_ind,:] + translation_vec

        self.debug_info['flag' + self.id_str] = flag

        self.search_area_box = torch.cat((sample_coords[scale_ind,[1,0]], sample_coords[scale_ind,[3,2]] - sample_coords[scale_ind,[1,0]] - 1))

        # Update position and scale
        if flag != 'not_found':
            self.target_not_found_counter = 0
            self.target_scales.append(self.target_scale)

            if self.params.get('use_iou_net', True):
                update_scale_flag = self.params.get('update_scale_when_uncertain', True) or flag != 'uncertain'
                if self.params.get('use_classifier', True):
                    self.update_state(new_pos)
                self.refine_target_box(backbone_feat, sample_pos[scale_ind,:], sample_scales[scale_ind], scale_ind, update_scale_flag)
            elif self.params.get('use_classifier', True):
                self.update_state(new_pos, sample_scales[scale_ind])

        else:
            if self.params.get('enable_search_area_rescaling_at_occlusion', False):
                # if self.target_not_found_counter == 0:
                #     breakpoint()
                self.target_not_found_counter += 1
                num_scales = max(2, min(30, self.target_not_found_counter))
                target_scales = torch.tensor(self.target_scales)

                # ensures that target scale doesn't get smaller.
                target_scales = target_scales[-60:] # max history
                target_scales = target_scales[target_scales>=target_scales[-1]] # only boxes that are bigger than the not found
                target_scales = target_scales[-num_scales:] # look as many samples into past as not found endures.

                self.target_scale = torch.mean(target_scales)
                print(self.frame_num, self.target_scale, target_scales[-1])


        # Compute Iou etc.
        # Set the pos of the tracker to iounet pos
        if self.params.get('use_iou_net', True) and flag != 'not_found' and hasattr(self, 'pos_iounet'):
            self.pos = self.pos_iounet.clone()

        score_map = s[scale_ind, ...]
        max_score = torch.max(score_map).item()
        self.debug_info['max_score' + self.id_str] = max_score

        # ------- Compute target certainty ------ #
        target_label_certainty = self.compute_target_label_certainty(score_map, sim, test_x)

        # ------- UPDATE ------- #
        self.score_map_prev = s.clone()
        self.train_y_prev = None

        update_flag = flag not in ['not_found', 'uncertain']
        hard_negative = (flag == 'hard_negative')
        learning_rate = self.params.get('hard_negative_learning_rate', None) if hard_negative else None

        if update_flag and self.params.get('update_classifier', False):
            # Get train sample
            train_x = test_x[scale_ind:scale_ind+1, ...]

            # Create target_box and label for spatial sample
            target_box = self.get_iounet_box(self.pos, self.target_sz, sample_pos[scale_ind,:], sample_scales[scale_ind])
            train_y = self.get_label_function(self.pos, sample_pos[scale_ind,:], sample_scales[scale_ind]).to(self.params.device)

            self.update_classifier(train_x, train_y, target_box, learning_rate, s[scale_ind,...], target_label_certainty)
            self.train_y_prev = train_y.clone()

        # Compute output bounding box
        new_state = torch.cat((self.pos[[1,0]] - (self.target_sz[[1,0]]-1)/2, self.target_sz[[1,0]]))

        if self.params.get('output_not_found_box', False) and flag == 'not_found':
            output_state = [-1, -1, -1, -1]
        else:
            output_state = new_state.tolist()

        out = {'target_bbox': output_state}

        if self.visdom is not None:
            cache = self.prepare_logging_data(test_x, score_map, sim)
            self.visualize_debugging_plots(new_state, image, score_map, cache)
            self.visualize_debugging_heatmaps(score_map, cache)
            self.visualize_graph(score_map, cache)
        elif self.params.debug >= 2:
            show_tensor(score_map, 5, title='Max score = {:.2f}'.format(max_score))

        return out

    def visualize_graph(self, score_map, cache):
        if self.previous_score_map is None:
            self.previous_score_map = score_map.clone()
            return

        peak_coords_old, peak_scores_old = find_local_maxima_v2(self.previous_score_map, th=0.05, ks=5)
        peak_coords_cur, peak_scores_cur = find_local_maxima_v2(score_map, th=0.05, ks=5)

        peaks_old_raw = torch.zeros_like(score_map)
        peaks_cur_raw = torch.zeros_like(score_map)
        peaks_old_raw[peak_coords_old[:, 0].long(), peak_coords_old[:, 1].long()] = peak_scores_old
        peaks_cur_raw[peak_coords_cur[:, 0].long(), peak_coords_cur[:, 1].long()] = peak_scores_cur

        peaks_old = torch.zeros_like(peaks_old_raw)
        peaks_cur = torch.zeros_like(peaks_old_raw)

        peaks_old[peaks_old_raw>0] = 1
        peaks_cur[peaks_cur_raw > 0] = 1

        if peak_coords_cur.shape[0] > 0 and peak_coords_old.shape[0] > 0:
            v_old = peak_coords_old[0] - peak_coords_old
            f_old = torch.sum(peak_scores_old.view(-1, 1) * v_old, dim=0)

            v_cur = (peak_coords_cur[:, None, :] - peak_coords_cur[None, :, :])
            f_cur = torch.sum(peak_scores_cur.view(1, -1, 1) * v_cur, dim=1)

            err = torch.sqrt(torch.sum((f_old - f_cur) ** 2, dim=1))
            coord = peak_coords_cur[torch.argmin(err)]

            peaks_old[peak_coords_old[0, 0].long(), peak_coords_old[0, 1].long()] = 2
            peaks_cur[coord[0].long(), coord[1].long()] = 2

        self.visdom.register(self.previous_score_map, 'heatmap', 2, 'previous_score_map' + self.id_str)
        # self.visdom.register(peaks_old_raw, 'heatmap', 2, 'peaks_raw_old' + self.id_str)
        # self.visdom.register(peaks_cur_raw, 'heatmap', 2, 'peaks_raw_cur' + self.id_str)
        self.visdom.register(peaks_old, 'heatmap', 2, 'peaks_old' + self.id_str)
        self.visdom.register(peaks_cur, 'heatmap', 2, 'peaks_cur' + self.id_str)

        self.previous_score_map = score_map.clone()

        # breakpoint()
        for i in range(0, 4):
            train_x_ = self.training_samples[0][:self.num_stored_samples[0]].unsqueeze(1)
            train_y_ = self.target_labels[0][:self.num_stored_samples[0]].reshape(-1, 1, score_map.shape[0],
                                                                                  score_map.shape[1])
            if i < len(peak_scores_cur):
                coord = peak_coords_cur[i]
                ksz_even = torch.Tensor([(self.kernel_size[0] + 1) % 2, (self.kernel_size[1] + 1) % 2])
                label = dcf.label_function_spatial(self.feature_sz, self.sigma, coord.cpu() - self.feature_sz//2, end_pad=ksz_even)
                label = label.to(score_map.device)
                scalar = peak_scores_cur[i]/label.max()
                label = scalar*label

                ptm = self.proj_test_score_to_train_labels(cache['sim'], train_x_, train_y_, label)

                val = self.compute_test_certainty(train_y_, ptm=ptm, proj_score_computation_type='average_all_proj_scores')

                self.logging_dict['ptm_peak_{}'.format(i)].append(val)

            else:
                self.logging_dict['ptm_peak_{}'.format(i)].append(0.)

        x = torch.arange(0, len(self.logging_dict['ptm_peak_0']))
        d = {'peak_{}'.format(i): (torch.tensor(self.logging_dict['ptm_peak_{}'.format(i)]), x) for i in range(0, 4)}
        self.visdom.register(d, 'lineplot', 3, 'ptm_per_peak')


    def prepare_logging_data(self, test_x, score_map, sim):
        cache = dict()

        test_x_ = test_x.unsqueeze(1)
        train_x_ = self.training_samples[0][:self.num_stored_samples[0]].unsqueeze(1)
        train_y_ = self.target_labels[0][:self.num_stored_samples[0]].reshape(-1, 1, score_map.shape[0], score_map.shape[1])
        if 'sim' not in cache:
            cache['sim'] = sim if sim is not None else compute_cosine_similarity(train_feat=train_x_, test_feat=test_x_)
        if 'ptm' not in cache:
            cache['ptm'] = self.proj_test_score_to_train_labels(cache['sim'], train_x_, train_y_, score_map)
        if 'pmt' not in cache:
            cache['pmt'] = self.proj_train_labels_to_test_score(cache['sim'], train_x_, train_y_)

        return cache

    def visualize_debugging_plots(self, state, img, score_map, cache):
        train_y_ = self.target_labels[0][:self.num_stored_samples[0]].reshape(-1, 1, score_map.shape[0], score_map.shape[1])

        debugging_plots_list = self.params.get('debugging_plots_list', list())

        if 'iou' in debugging_plots_list:
            iou = torch.zeros(1)
            if self.params.get('use_gt_box', False):
                bbox_gth = self.frame_reader.get_bbox(self.frame_num - 1, None)
                if np.all(np.logical_not(np.isnan(bbox_gth))) and np.all(bbox_gth >= 0) and bbox_gth is not None:
                    iou = bbutils.calc_iou(state, torch.from_numpy(bbox_gth))
            self.logging_dict['ious'].append(iou)
            # write debug info
            self.debug_info['IoU'] = iou
            self.debug_info['mIoU'] = np.mean(self.logging_dict['ious'])
            # plot debug data
            self.visdom.register(torch.tensor(self.logging_dict['ious']), 'lineplot', 3, 'IoU')

        if 'center_dist' in debugging_plots_list and self.previous_state is not None:
            h, w = img.shape[:2]
            new_center_norm = state[:2] / torch.tensor([w, h])
            old_center_norm = self.previous_state[:2] / torch.tensor([w, h])
            new_center_gth_norm = torch.from_numpy(self.frame_reader.get_bbox(self.frame_num - 1, None)[:2]) / torch.tensor([w, h])
            old_center_gth_norm = torch.from_numpy(self.frame_reader.get_bbox(self.frame_num - 2, None)[:2]) / torch.tensor([w, h])

            self.logging_dict['pred_center_dist_l2'].append(torch.sqrt(torch.sum((new_center_norm - old_center_norm)**2)))
            self.logging_dict['gth_center_dist_l2'].append(torch.sqrt(torch.sum((new_center_gth_norm - old_center_gth_norm)**2)))

            pred = torch.tensor(self.logging_dict['pred_center_dist_l2'])
            gth = torch.tensor(self.logging_dict['gth_center_dist_l2'])
            x = torch.arange(0, pred.shape[0])
            # plot debug data
            self.visdom.register({'raw': (pred, x), 'mean': (torch.cumsum(pred, 0)/torch.cumsum(torch.ones_like(pred), 0), x)},
                                 'lineplot', 3, 'Pred: center_dist_l2')
            self.visdom.register({'raw': (gth, x), 'mean': (torch.cumsum(gth, 0)/torch.cumsum(torch.ones_like(gth), 0),x)},
                                 'lineplot', 3, 'Gth: center_dist_l2')

        if 'bbox_size_diff' in debugging_plots_list and self.previous_state is not None:
            h, w = img.shape[:2]
            new_center_norm = state[2:] / torch.tensor([w, h])
            old_center_norm = self.previous_state[2:] / torch.tensor([w, h])
            new_center_gth_norm = torch.from_numpy(self.frame_reader.get_bbox(self.frame_num - 1, None)[2:]) / torch.tensor([w, h])
            old_center_gth_norm = torch.from_numpy(self.frame_reader.get_bbox(self.frame_num - 2, None)[2:]) / torch.tensor([w, h])

            self.logging_dict['pred_bbox_diff_l1'].append(torch.sum(torch.abs(new_center_norm - old_center_norm)))
            self.logging_dict['gth_bbox_diff_l1'].append(torch.sum(torch.abs(new_center_gth_norm - old_center_gth_norm)))

            pred = torch.tensor(self.logging_dict['pred_bbox_diff_l1'])
            gth = torch.tensor(self.logging_dict['gth_bbox_diff_l1'])
            x = torch.arange(0, pred.shape[0])
            # plot debug data
            self.visdom.register({'raw': (pred, x), 'mean': (torch.cumsum(pred, 0)/torch.cumsum(torch.ones_like(pred), 0), x)},
                                 'lineplot', 3, 'Pred: bbox_size_diff')
            self.visdom.register({'raw': (gth, x), 'mean': (torch.cumsum(gth, 0)/torch.cumsum(torch.ones_like(gth), 0), x)},
                                 'lineplot', 3, 'Gth: bbox_size_diff')

        if 'max_score_map' in debugging_plots_list:
            self.logging_dict['max_score_map'].append(torch.max(score_map).item())
            # plot debug data
            self.visdom.register(torch.tensor(self.logging_dict['max_score_map']), 'lineplot', 3, 'max_score_map')

        if 'pred_bbox_size' in debugging_plots_list:
            h, w = img.shape[:2]
            self.logging_dict['pred_bbox_size'].append((state[2]/w)*(state[3]/h))
            # plot debug data
            self.visdom.register(torch.tensor(self.logging_dict['pred_bbox_size']), 'lineplot', 3, 'pred_bbox_size')

        if 'num_iters' in debugging_plots_list and len(self.logging_dict['num_iters']) > 0:
            self.visdom.register(torch.tensor(self.logging_dict['num_iters']), 'lineplot', 3, 'num_iters')

        if 'certainties_mem' in debugging_plots_list:
            self.visdom.register(self.target_label_certainties[self.mem_sort_indices].view(-1), 'lineplot', 3, 'certainties_mem')

        if 'sample_weights' in debugging_plots_list:
            self.visdom.register(self.sample_weights[0][self.mem_sort_indices].view(-1), 'lineplot', 3, 'sample_weights')

        if 'scaled_sample_weights' in debugging_plots_list:
            certainties = self.target_label_certainties.view(-1)*self.sample_weights[0].view(-1)
            self.visdom.register(certainties[self.mem_sort_indices].view(-1), 'lineplot', 3, 'scaled_sample_weights')

        if 'mean_max_ptm_all' in debugging_plots_list:
            self.logging_dict['mean_max_ptm_all'].append(
                self.compute_test_certainty(train_y_, ptm=cache['ptm'], proj_score_computation_type='average_all_proj_scores'))

            self.visdom.register(torch.tensor(self.logging_dict['mean_max_ptm_all']), 'lineplot', 3, 'mean_max_ptm_all')

        if 'mean_max_ptm_gth' in debugging_plots_list:
            self.logging_dict['mean_max_ptm_gth'].append(
                self.compute_test_certainty(train_y_, ptm=cache['ptm'], proj_score_computation_type='average_gth_proj_scores'))

            self.visdom.register(torch.tensor(self.logging_dict['mean_max_ptm_gth']), 'lineplot', 3, 'mean_max_ptm_gth')

        if 'mean_max_scaled_ptm_all' in debugging_plots_list:
            scaled_ptm = train_y_ * cache['ptm'].reshape(train_y_.shape[0], 1, score_map.shape[0], score_map.shape[1])
            self.logging_dict['mean_max_scaled_ptm_all'].append(
                self.compute_test_certainty(train_y_, ptm=scaled_ptm, proj_score_computation_type='average_all_proj_scores'))

            self.visdom.register(torch.tensor(self.logging_dict['mean_max_scaled_ptm_all']), 'lineplot', 3, 'mean_max_scaled_ptm_all')

        if 'mean_max_scaled_ptm_gth' in debugging_plots_list:
            scaled_ptm = train_y_ * cache['ptm'].reshape(train_y_.shape[0], 1, score_map.shape[0], score_map.shape[1])
            self.logging_dict['mean_max_scaled_ptm_gth'].append(
                self.compute_test_certainty(train_y_, ptm=scaled_ptm, proj_score_computation_type='average_gth_proj_scores'))

            self.visdom.register(torch.tensor(self.logging_dict['mean_max_scaled_ptm_gth']), 'lineplot', 3, 'mean_max_scaled_ptm_gth')

        if 'predicted_certainties' in debugging_plots_list:
            self.visdom.register(torch.tensor(self.logging_dict['predicted_certainties']), 'lineplot', 3, 'predicted_certainties')

        if 'peak_prob' in debugging_plots_list:
            # plot debug data
            if len(self.logging_dict['peak_prob']) < self.frame_num - 1:
                if self.frame_num <= 2:
                    self.logging_dict['peak_prob'].append(1.)
                else:
                    self.logging_dict['peak_prob'].append(0.)
            self.logging_dict['running_mean_peak_prob'].append(torch.mean(torch.tensor(self.logging_dict['peak_prob'][-10:])))

            if len(self.logging_dict['peak_prob']) > 0:
                self.visdom.register(torch.tensor(self.logging_dict['peak_prob']), 'lineplot', 3, 'peak_prob')

            self.visdom.register(torch.tensor(self.logging_dict['running_mean_peak_prob']), 'lineplot', 3, 'running_mean_peak_prob')

        self.visdom.register(self.debug_info, 'info_dict', 1, 'Status')

        self.previous_state = state

        return cache

    def visualize_debugging_heatmaps(self, score_map, cache, score_map_unfused=None):
        debugging_heat_map_list = self.params.get('debugging_heat_map_list', list())

        if 'score_map' in debugging_heat_map_list:
            self.visdom.register(score_map, 'heatmap', 2, 'score_map' + self.id_str)

        if 'ptm_gth' in debugging_heat_map_list:
            self.visdom.register(cache['ptm'][0, 0], 'heatmap', 2, 'ptm_gth' + self.id_str)

        if 'scaled_ptm_gth' in debugging_heat_map_list:
            train_y_ = self.target_labels[0][:self.num_stored_samples[0]].reshape(-1, 1, score_map.shape[0], score_map.shape[1])
            self.visdom.register(train_y_[0, 0]*cache['ptm'][0, 0], 'heatmap', 2, 'scaled_ptm_gth' + self.id_str)

        if 'ptm_other' in debugging_heat_map_list:
            self.visdom.register(cache['ptm'][self.mem_sort_indices[15], 0], 'heatmap', 2, 'ptm_other' + self.id_str)

        if 'pmt_mean_all' in debugging_heat_map_list:
            self.visdom.register(torch.mean(cache['pmt'][:, 0], dim=0), 'heatmap', 2, 'pmt_mean_all' + self.id_str)
        if 'pmt_std_all' in debugging_heat_map_list:
            self.visdom.register(torch.std(cache['pmt'][:, 0], dim=0), 'heatmap', 2, 'pmt_std_all' + self.id_str)
        if 'pmt_mean_gth' in debugging_heat_map_list:
            self.visdom.register(torch.mean(cache['pmt'][:15, 0], dim=0), 'heatmap', 2, 'pmt_mean_gth' + self.id_str)
        if 'pmt_std_gth' in debugging_heat_map_list:
            self.visdom.register(torch.std(cache['pmt'][:15, 0], dim=0), 'heatmap', 2, 'pmt_std_gth' + self.id_str)
        if 'pmt_mean_other' in debugging_heat_map_list:
            self.visdom.register(torch.mean(cache['pmt'][15:, 0], dim=0), 'heatmap', 2, 'pmt_mean_other' + self.id_str)
        if 'pmt_std_other' in debugging_heat_map_list:
            self.visdom.register(torch.std(cache['pmt'][15:, 0], dim=0), 'heatmap', 2, 'pmt_std_other' + self.id_str)

        if 'score_map_unfused' in debugging_heat_map_list and hasattr(self.net.classifier, 'fusion_module'):
            self.visdom.register(score_map_unfused, 'heatmap', 2, 'score_map_unfused' + self.id_str)
        if 'score_map_diff' in debugging_heat_map_list and hasattr(self.net.classifier, 'fusion_module'):
            self.visdom.register(score_map - score_map_unfused, 'heatmap', 2, 'score_map_diff' + self.id_str)


        if 'score_peak_map' in debugging_heat_map_list:
            def f(scores):
                scores = scores.reshape(1, 1, scores.shape[0], scores.shape[1])
                feats = []
                max_score = scores.max().item()
                min_score = max(max_score/4, 0.1)
                ths = torch.linspace(min_score, max_score, 10)
                for th in ths:
                    components = torch.arange(scores.shape[2]*scores.shape[3]).reshape(scores.shape).float()
                    # components = scores.clone()
                    mask = torch.zeros_like(scores, dtype=torch.bool)
                    mask[scores >= th] = True
                    components[~mask] = 0

                    for i in range(0, 23):
                        components[mask] = F.max_pool2d(components, kernel_size=3, stride=1, padding=1)[mask]

                    components[components > 0] = scores.cpu()[components > 0]
                    feats.append(components[0])

                feats = torch.cat(feats, dim=0)

                # num_peaks = [torch.unique(f).shape[0] for f in feats.reshape(feats.shape[0], -1)]
                # print(num_peaks)

                return feats

            peak_maps_pmt = f(torch.mean(cache['pmt'][:, 0], dim=0))
            peak_maps_score = f(score_map)
            # print(peak_maps.shape)
            train_y_ = self.target_labels[0][:self.num_stored_samples[0]].reshape(-1, 1, score_map.shape[0],
                                                                                  score_map.shape[1])
            self.visdom.register(peak_maps_pmt[0], 'heatmap', 2, 'peak_maps_pmt' + self.id_str)
            self.visdom.register(peak_maps_score[0], 'heatmap', 2, 'peak_maps_score' + self.id_str)
            self.visdom.register(train_y_[self.mem_sort_indices.view(-1)][-2][0], 'heatmap', 2, 'last peak' + self.id_str)



    def compute_target_label_certainty(self, score_map, sim=None, test_x=None):
        target_label_certainty = None
        if self.params.get('use_gt_box', False) and self.params.get('target_label_certainty_type', None) == 'iou':
            iou = torch.zeros(1)
            if self.params.get('use_gt_box', False):
                new_state_ = torch.cat((self.pos[[1, 0]] - (self.target_sz[[1, 0]] - 1) / 2, self.target_sz[[1, 0]]))
                bbox_gth = self.frame_reader.get_bbox(self.frame_num - 1, None)
                if np.all(np.logical_not(np.isnan(bbox_gth))) and np.all(bbox_gth >= 0):
                    iou = bbutils.calc_iou(new_state_, torch.from_numpy(bbox_gth))

            target_label_certainty = iou.view(1, 1, 1, 1)

        elif self.params.get('target_label_certainty_type', None) == 'max_proj_test_score_on_gth':
            train_bb = self.target_boxes[:self.num_stored_samples[0]]
            test_x = test_x.unsqueeze(1)
            train_x_ = self.training_samples[0][:self.num_stored_samples[0]].unsqueeze(1)
            train_y_ = self.target_labels[0][:self.num_stored_samples[0]].reshape(-1, 1, score_map.shape[0], score_map.shape[1])

            if sim is None and test_x is not None:
                sim = compute_cosine_similarity(train_feat=train_x_, test_feat=test_x)

            target_label_certainty = self.compute_test_certainty(train_y_, score_map, train_x_, train_bb, sim,
                                                                 proj_score_computation_type=self.params.get('proj_score_computation_type', None))

        elif self.params.get('target_label_certainty_type', None) == 'learned_predicted_certainty':
            # required: ptm, pmt, target_scores, predicted_iou
            test_x = test_x.unsqueeze(1)
            train_x_ = self.training_samples[0][:self.num_stored_samples[0]].unsqueeze(1)
            train_y_ = self.target_labels[0][:self.num_stored_samples[0]].reshape(-1, 1, score_map.shape[0], score_map.shape[1])

            if sim is None and test_x is not None:
                sim = compute_cosine_similarity(train_feat=train_x_, test_feat=test_x)

            pmt = self.proj_train_labels_to_test_score(sim, train_x_, train_y_)
            ptm = self.proj_test_score_to_train_labels(sim, train_x_, train_y_, score_map)

            predicted_iou = self.predicted_iou if self.predicted_iou is not None else torch.zeros(1)
            predicted_iou = predicted_iou.cuda(pmt.get_device())

            with torch.no_grad():
                target_label_certainty = self.cp_net.predict_certainties(target_scores=score_map, ptm=ptm, pmt=pmt,
                                                                         predicted_iou=predicted_iou)
            self.logging_dict['predicted_certainties'].append(target_label_certainty)

        return target_label_certainty

    def proj_train_labels_to_test_score(self, sim, train_x, train_y):
        wf, hf = train_x.shape[-2:]
        wl, hl = train_y.shape[-2:]
        n_mem = train_y.shape[0]
        nseq = train_x.shape[1]
        sim = sim.reshape(nseq, wf * hf, n_mem, wf * hf).permute(2, 0, 1, 3)  # (M, N, W*H, W*H)
        train_y_down = F.interpolate(train_y, size=(hf, wf), mode='bilinear')  # (22,22)

        pmt = proj_mem_to_test(sim, train_y_down).view(n_mem, 1, hf, wf)  # (M, 1, H, W)
        pmt = F.interpolate(pmt, size=(hl, wl), mode='bilinear')  # (22,22)

        return pmt

    def proj_test_score_to_train_labels(self, sim, train_x, train_y, score_map):
        wf, hf = train_x.shape[-2:]
        wl, hl = train_y.shape[-2:]
        n_mem = train_y.shape[0]
        nseq = train_x.shape[1]
        score_map = score_map.reshape(1, 1, hl, wl)
        sim = sim.reshape(nseq, wf * hf, n_mem, wf * hf).permute(2, 0, 1, 3)  # (M, N, W*H, W*H)
        score_map_down = F.interpolate(score_map, size=(hf, wf), mode='bilinear')  # (22,22)

        ptm = proj_test_to_mem(sim, score_map_down).view(n_mem, 1, hf, wf)  # (M, 1, W, H)
        ptm = F.interpolate(ptm, size=(hl, wl), mode='bilinear')  # (23,23)

        return ptm

    def compute_test_certainty(self, train_y, score_map=None, train_x=None, train_bb=None, sim=None, ptm=None, proj_score_computation_type=None):
        n_mem = train_y.shape[0]
        wl, hl = train_y.shape[-2:]

        if ptm is None:
            ptm = self.proj_test_score_to_train_labels(sim, train_x, train_y, score_map)

        max_vals, _ = torch.max(ptm.view(n_mem, wl*hl), dim=1)

        if proj_score_computation_type == 'average_gth_proj_scores':
            target_label_certainty = torch.mean(max_vals[:self.num_init_samples[0]]).view(1, 1, 1, 1)
        elif proj_score_computation_type == 'average_all_proj_scores':
            target_label_certainty = torch.mean(max_vals).view(1, 1, 1, 1)
        elif proj_score_computation_type == 'single_gth_proj_score':
            target_label_certainty = max_vals[0].view(1, 1, 1, 1)
        elif proj_score_computation_type == 'scaled_average_all_proj_scores':
            if self.weights is None:
                target_label_certainty = torch.mean(max_vals).view(1, 1, 1, 1)
            else:
                weights = self.weights.view(-1) / torch.sum(self.weights.view(-1))
                target_label_certainty = torch.sum(weights * max_vals.view(-1)).view(1, 1, 1, 1)
        elif proj_score_computation_type == 'average_all_proj_score_inside_bbox':
            x = torch.round(train_bb[..., 0]/16).clamp(0., 22.).type(torch.int)
            y = torch.round(train_bb[..., 1]/16).clamp(0., 22.).type(torch.int)
            w = torch.ceil(train_bb[..., 2]/16).clamp(1., 22.).type(torch.int)
            h = torch.ceil(train_bb[..., 3]/16).clamp(1., 22.).type(torch.int)

            max_vals = [
                torch.max(ptm[i, 0, y[i].item():y[i].item() + h[i].item(), x[i].item():x[i].item() + w[i].item()])
                for i in torch.arange(0, train_bb.shape[0])]

            max_vals = torch.tensor(max_vals)
            target_label_certainty = torch.mean(max_vals)
        else:
            raise NotImplementedError()

        return target_label_certainty

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
        if hasattr(self.net.classifier, 'fusion_module'):
            test_x = sample_x.unsqueeze(1)
            train_x = self.training_samples[0][:self.num_stored_samples[0]].unsqueeze(1)
            train_y = self.target_labels[0][:self.num_stored_samples[0]]

            sim = compute_cosine_similarity(train_x, test_x)

            with torch.no_grad():
                scores_unfused = self.net.classifier.raw_scores(self.target_filter, sample_x)
                scores_fused = self.net.classifier.fuse_scores(sim, scores_unfused, train_y, test_x, train_x)
            return scores_fused, scores_unfused, sim

        else:
            with torch.no_grad():
                scores_raw = self.net.classifier.classify(self.target_filter, sample_x)
            return scores_raw, None, None



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

        if self.params.get('enable_peak_localization', False) and self.frame_num > 2 and self.train_y_prev is not None:
                return self.localize_peak(scores, sample_pos, sample_scales)

        elif self.params.get('advanced_localization', False):
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

    def localize_peak(self, scores, sample_pos, sample_scales):
        sz = scores.shape[-2:]
        score_sz = torch.Tensor(list(sz))
        output_sz = score_sz - (self.kernel_size + 1) % 2
        score_center = (score_sz - 1) / 2

        max_score1, max_disp1 = dcf.max2d(scores)
        _, scale_ind = torch.max(max_score1, dim=0)
        sample_scale = sample_scales[scale_ind]

        # If no peaks are found skip and return not found
        if max_score1 < 0.05:
            target_disp = max_disp1.cpu() - score_center
            translation_vec = target_disp * (self.img_support_sz / output_sz) * sample_scale
            return translation_vec, scale_ind, scores, 'not_found'

        with torch.no_grad():
            peak_probs, peak_coords, peak_scores = self.peak_pred_net.compute_peak_probabilities(self.score_map_prev, scores, self.train_y_prev)

        selected_peak_idx = torch.argmax(peak_probs)
        peak_prob = peak_probs[selected_peak_idx]
        peak_coord = peak_coords[selected_peak_idx].cpu()

        target_disp = peak_coord - score_center
        translation_vec = target_disp * (self.img_support_sz / output_sz) * sample_scale

        self.logging_dict['peak_prob'].append(peak_prob)
        # print(self.frame_num, peak_prob.cpu().item(), peak_probs.view(-1).cpu().numpy(), peak_scores.view(-1).cpu().numpy())

        if self.visdom is not None:
            score_map = torch.zeros((23, 23))
            score_mask = torch.zeros((23, 23))
            for p, c in zip(peak_probs, peak_coords):
                score_map[c[0], c[1]] = p
                score_mask[c[0], c[1]] = 1
            self.visdom.register(score_map, 'heatmap', 2, 'peak_pred_map' + self.id_str)
            self.visdom.register(score_mask, 'heatmap', 2, 'peak_pred_mask' + self.id_str)


        num_peaks = peak_probs.shape[0]

        if num_peaks == 0:
            return translation_vec, scale_ind, scores, 'not_found'

        elif num_peaks == 1:
            selected_peak_idx = torch.argmax(peak_probs)
            selected_peak_score = peak_scores[selected_peak_idx]
            selected_peak_prob = peak_probs[selected_peak_idx]

            # if ((selected_peak_score < 0.15) or (selected_peak_prob < 0.5) or
            #         (0.15 < selected_peak_score < 0.25 and selected_peak_prob < 0.75)):
            if ((selected_peak_score < 0.15) or ((selected_peak_prob < 0.5) and (selected_peak_score < 0.25))):
                return translation_vec, scale_ind, scores, 'not_found'

            # if (0.15 < selected_peak_score < 0.25 and 0.75 < selected_peak_prob):
            #     return translation_vec, scale_ind, scores, 'hard_negative'

            return translation_vec, scale_ind, scores, 'normal'

        else:
            peak_scores_sort_ids = torch.argsort(peak_scores, descending=True)
            peak_probs_sort_ids = torch.argsort(peak_probs, descending=True)

            selected_peak_score = peak_scores[peak_probs_sort_ids[0]]
            highest_peak_score = peak_scores[peak_scores_sort_ids[0]]
            second_highest_peak_score = peak_scores[peak_scores_sort_ids[1]]

            selected_peak_prob = peak_probs[peak_probs_sort_ids[0]]
            runnerup_peak_prob = peak_probs[peak_probs_sort_ids[1]]

            if ((highest_peak_score < 0.15) or ((selected_peak_prob < 0.5) and (selected_peak_score < 0.25))):
                return translation_vec, scale_ind, scores, 'not_found'

            # if (selected_peak_score < highest_peak_score) and (selected_peak_prob < 0.75) and (selected_peak_score < 0.25):
            #     return translation_vec, scale_ind, scores, 'not_found'

            # if (0.75 < selected_peak_prob) and (0.75 < runnerup_peak_prob):
            #     return translation_vec, scale_ind, scores, 'uncertain'

            # if (selected_peak_score < highest_peak_score) and (0.75 < selected_peak_prob):
            #     # print('HN', self.frame_num, peak_probs.view(-1).cpu().numpy(), peak_scores.view(-1).cpu().numpy())
            #     return translation_vec, scale_ind, scores, 'hard_negative'

            # if ((selected_peak_score == highest_peak_score) and
            #         (0.8*selected_peak_score < second_highest_peak_score) and (0.75 < selected_peak_prob) and (runnerup_peak_prob < 0.75)):
            #     # print('HN', self.frame_num, peak_probs.view(-1).cpu().numpy(), peak_scores.view(-1).cpu().numpy())
            #     return translation_vec, scale_ind, scores, 'hard_negative'


            # if ((highest_peak_score == selected_peak_score) and (0.75 < selected_peak_prob) and
            #         (0.5 < runnerup_peak_prob) and (0.25 < highest_peak_score) and (0.25 < second_highest_peak_score)):
            #     # print('Unlikely Scenario', peak_probs, peak_scores, selected_peak_score, highest_peak_score, selected_peak_score, selected_peak_prob, runnerup_peak_prob)
            #     return translation_vec, scale_ind, scores, 'normal'

            return translation_vec, scale_ind, scores, 'normal'


        # if probs.shape[0] == 1 and max_score1 < 0.3 and max_score1 > 0.15 and peak_prob > 0.80:
        #     return translation_vec, scale_ind, scores, 'hard_negative'
        #
        # if max_score1 < 0.15 or peak_prob < 0.5 or probs.shape[0] == 1 and max_score1 < 0.25:
        #     return translation_vec, scale_ind, scores, 'not_found'
        #
        # if probs.shape[0] > 1:
        #     score_sort_ids = torch.argsort(peak_scores[0], descending=True)
        #
        #     selected_peak_score = peak_scores[0][selected_peak_idx]
        #     highest_peak_score = peak_scores[0][score_sort_ids[0]]
        #     second_highest_peak_score = peak_scores[0][score_sort_ids[1]]
        #
        #     if highest_peak_score > selected_peak_score and peak_prob > 0.9:
        #         return translation_vec, scale_ind, scores, 'hard_negative'
        #
        #     if highest_peak_score == selected_peak_score and peak_prob > 0.9 and highest_peak_score*0.8 < second_highest_peak_score:
        #         return translation_vec, scale_ind, scores, 'hard_negative'
        #
        #
        # return translation_vec, scale_ind, scores, 'normal'

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

        if self.params.get('ideal_peak_localization', False):
            gth_box = self.frame_reader.get_bbox(self.frame_num - 1, None)
            if gth_box is not None:
                gth_center = torch.tensor(gth_box[:2] + (gth_box[2:] - 1) / 2)[[1, 0]]
                anno_y = self.get_label_function(gth_center, sample_pos[scale_ind, :], sample_scales[scale_ind])
                max_score_anno, max_disp_anno = dcf.max2d(anno_y[0][0])
                max_disp_anno = max_disp_anno.float().cpu().view(-1)
                target_disp_anno = max_disp_anno - score_center
                disp_norm_anno_1 = torch.sqrt(torch.sum((target_disp1 - target_disp_anno) ** 2))
                disp_norm_anno_2 = torch.sqrt(torch.sum((target_disp2 - target_disp_anno) ** 2))

                if disp_norm_anno_1 < disp_norm_anno_2:
                    if max_score1.item() < self.params.target_not_found_threshold:
                        return translation_vec1, scale_ind, scores_hn, 'not_found'
                    elif max_score2 > self.params.distractor_threshold * max_score1:
                        return translation_vec1, scale_ind, scores_hn, 'hard_negative'
                    else:
                        return translation_vec1, scale_ind, scores_hn, 'normal'

                if disp_norm_anno_1 > disp_norm_anno_2:
                    if max_score2.item() < self.params.target_not_found_threshold:
                        return translation_vec2, scale_ind, scores_hn, 'not_found'
                    elif max_score1 > self.params.distractor_threshold * max_score2:
                        return translation_vec2, scale_ind, scores_hn, 'hard_negative'
                    else:
                        return translation_vec2, scale_ind, scores_hn, 'normal'

                if disp_norm_anno_2 == disp_norm_anno_1:
                    return translation_vec1, scale_ind, scores_hn, 'uncertain'

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

    def generate_init_samples(self, im: torch.Tensor) -> TensorList:
        """Perform data augmentation to generate initial training samples."""

        mode = self.params.get('border_mode', 'replicate')
        if mode == 'inside':
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

        # Extract augmented image patches
        im_patches = sample_patch_transformed(im, self.init_sample_pos, self.init_sample_scale, aug_expansion_sz, self.transforms)

        # Extract initial backbone features
        with torch.no_grad():
            init_backbone_feat = self.net.extract_backbone(im_patches)

        return init_backbone_feat

    def init_target_label_certainties(self, train_x: TensorList):
        num_train_samples = train_x[0].shape[0]
        self.target_label_certainties = train_x[0].new_zeros(self.params.sample_memory_size, 1, 1, 1)
        self.target_label_certainties[:num_train_samples] = 1.

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

    def init_target_labels(self, train_x: TensorList):
        self.target_labels = TensorList([x.new_zeros(self.params.sample_memory_size, 1,
                                                     x.shape[2] + (int(self.kernel_size[0].item()) + 1) % 2,
                                                     x.shape[3] + (int(self.kernel_size[1].item()) + 1) % 2)
                                         for x in train_x])
        # Output sigma factor
        output_sigma_factor = self.params.get('output_sigma_factor', 1/4)
        self.sigma = (self.feature_sz / self.img_support_sz * self.base_target_sz).prod().sqrt() * output_sigma_factor * torch.ones(2)

        # Center pos in normalized img_coords
        target_center_norm = (self.pos - self.init_sample_pos) / (self.init_sample_scale * self.img_support_sz)

        for target, x in zip(self.target_labels, train_x):
            ksz_even = torch.Tensor([(self.kernel_size[0] + 1) % 2, (self.kernel_size[1] + 1) % 2])
            center_pos = self.feature_sz * target_center_norm + 0.5 * ksz_even
            for i, T in enumerate(self.transforms[:x.shape[0]]):
                sample_center = center_pos + torch.Tensor(T.shift) / self.img_support_sz * self.feature_sz
                target[i, 0, ...] = dcf.label_function_spatial(self.feature_sz, self.sigma, sample_center, end_pad=ksz_even)

        return self.target_labels[0][:train_x[0].shape[0]]

    def init_memory(self, train_x: TensorList):
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

    def update_memory(self, sample_x: TensorList, sample_y: TensorList, target_box, learning_rate=None, target_label_certainty=None):
        # Update weights and get replace ind
        # replace_ind = self.update_sample_weights(self.sample_weights, self.previous_replace_ind, self.num_stored_samples, self.num_init_samples, learning_rate)
        certainties = [self.target_label_certainties.view(-1)*self.sample_weights[0].view(-1)]

        replace_ind = self.update_sample_weights_based_on_certainty(certainties, self.sample_weights,
                                                                    self.previous_replace_ind, self.num_stored_samples,
                                                                    self.num_init_samples, learning_rate)

        self.previous_replace_ind = replace_ind

        # Update sample and label memory
        for train_samp, x, ind in zip(self.training_samples, sample_x, replace_ind):
            train_samp[ind:ind+1,...] = x

        for y_memory, y, ind in zip(self.target_labels, sample_y, replace_ind):
            y_memory[ind:ind+1,...] = y

        # Update target label certainties memory
        self.target_label_certainties[replace_ind[0]] = target_label_certainty

        # Update bb memory
        self.target_boxes[replace_ind[0],:] = target_box

        if replace_ind[0] >= len(self.mem_sort_indices):
            self.mem_sort_indices = torch.cat([self.mem_sort_indices, torch.zeros(1, dtype=torch.long)])
            self.mem_sort_indices[replace_ind[0]] = torch.max(self.mem_sort_indices) + 1
        else:
            idx = torch.nonzero(self.mem_sort_indices == replace_ind[0])
            mem_temp = self.mem_sort_indices.clone()
            mem_temp[idx:-1] = self.mem_sort_indices[idx+1:]
            mem_temp[-1] = replace_ind[0]
            self.mem_sort_indices = mem_temp

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

    def update_sample_weights_based_on_certainty(self, certainties, sample_weights, previous_replace_ind, num_stored_samples, num_init_samples, learning_rate=None):
        # Update weights and get index to replace
        replace_ind = []
        for cert, sw, prev_ind, num_samp, num_init in zip(certainties, sample_weights, previous_replace_ind, num_stored_samples, num_init_samples):
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
                    _, r_ind = torch.min(cert[s_ind:], 0)
                    r_ind = r_ind.item() + s_ind

                # Update weights
                if prev_ind is None:
                    sw /= 1 - lr
                    sw[r_ind] = lr
                elif r_ind == prev_ind:
                    pass
                else:
                    sw[r_ind] = sw[prev_ind] / (1 - lr)

            sw /= sw.sum()
            if init_samp_weight is not None and sw[:num_init].sum() < init_samp_weight:
                sw /= init_samp_weight + sw[num_init:].sum()
                sw[:num_init] = init_samp_weight / num_init

            replace_ind.append(r_ind)

        return replace_ind

    def get_label_function(self, pos, sample_pos, sample_scale):
        train_y = TensorList()
        target_center_norm = (pos - sample_pos) / (sample_scale * self.img_support_sz)

        for sig, sz, ksz in zip([self.sigma], [self.feature_sz], [self.kernel_size]):
            ksz_even = torch.Tensor([(self.kernel_size[0] + 1) % 2, (self.kernel_size[1] + 1) % 2])
            center = sz * target_center_norm + 0.5*ksz_even
            train_y.append(dcf.label_function_spatial(sz, sig, center, end_pad=ksz_even))

        return train_y

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
        if torch.is_tensor(self.iou_modulation[0]):
            self.iou_modulation = TensorList([x.detach().mean(0) for x in self.iou_modulation])

    def init_classifier(self, init_backbone_feat):
        # Get classification features
        x = self.get_classification_features(init_backbone_feat)

        # Set regularization weight and initializer
        if hasattr(self.net, 'classifier'):
            pred_module = getattr(self.net.classifier.filter_optimizer, 'score_predictor', self.net.classifier.filter_optimizer)
        elif hasattr(self.net, 'dimp_classifier'):
            self.net.classifier = self.net.dimp_classifier
            pred_module = getattr(self.net.dimp_classifier.filter_optimizer, 'score_predictor',
                                  self.net.dimp_classifier.filter_optimizer)
        else:
            raise NotImplementedError

        if self.params.get('label_threshold', None) is not None:
            self.net.classifier.filter_optimizer.label_threshold = self.params.label_threshold
        if self.params.get('label_shrink', None) is not None:
            self.net.classifier.filter_optimizer.label_shrink = self.params.label_shrink
        if self.params.get('softmax_reg', None) is not None:
            self.net.classifier.filter_optimizer.softmax_reg = self.params.softmax_reg
        if self.params.get('filter_reg', None) is not None:
            pred_module.filter_reg[0] = self.params.filter_reg
            pred_module.min_filter_reg = self.params.filter_reg
        if self.params.get('filter_init_zero', False):
            self.net.classifier.filter_initializer = FilterInitializerZero(self.net.classifier.filter_size, x.shape[-3])

        # Add the dropout augmentation here, since it requires extraction of the classification features
        if 'dropout' in self.params.augmentation and self.params.get('use_augmentation', True):
            num, prob = self.params.augmentation['dropout']
            self.transforms.extend(self.transforms[:1]*num)
            x = torch.cat([x, F.dropout2d(x[0:1,...].expand(num,-1,-1,-1), p=prob, training=True)])

        # Set feature size and other related sizes
        self.feature_sz = torch.Tensor(list(x.shape[-2:]))
        ksz = self.net.classifier.filter_size
        self.kernel_size = torch.Tensor([ksz, ksz] if isinstance(ksz, (int, float)) else ksz)
        self.output_sz = self.feature_sz + (self.kernel_size + 1)%2

        # Construct output window
        self.output_window = None
        if self.params.get('window_output', False):
            if self.params.get('use_clipped_window', False):
                self.output_window = dcf.hann2d_clipped(self.output_sz.long(), (self.output_sz*self.params.effective_search_area / self.params.search_area_scale).long(), centered=True).to(self.params.device)
            else:
                self.output_window = dcf.hann2d(self.output_sz.long(), centered=True).to(self.params.device)
            self.output_window = self.output_window.squeeze(0)

        # Get target boxes for the different augmentations
        target_boxes = self.init_target_boxes()

        # Get target boxes for the different augmentations
        target_labels = self.init_target_labels(TensorList([x]))

        # Init target label certainties, init gth samples as 1.0
        self.init_target_label_certainties(TensorList([x]))

        # Set number of iterations
        plot_loss = self.params.debug > 0
        num_iter = self.params.get('net_opt_iter', None)

        self.net.classifier.compute_losses = plot_loss

        # Get target filter by running the discriminative model prediction module
        with torch.no_grad():
            self.target_filter, _, losses = self.net.classifier.get_filter(x, target_boxes,
                                                                           train_label=target_labels,
                                                                           num_iter=num_iter)

        # Init memory
        if self.params.get('update_classifier', True):
            self.init_memory(TensorList([x]))

        if plot_loss:
            if isinstance(losses, dict):
                losses = losses['train']
            self.losses = torch.stack(losses)
            if self.visdom is not None:
                self.visdom.register((self.losses, torch.arange(self.losses.numel())), 'lineplot', 3, 'Training Loss' + self.id_str)
            elif self.params.debug >= 3:
                plot_graph(self.losses, 10, title='Training Loss' + self.id_str)

    def update_classifier(self, train_x, train_y, target_box, learning_rate=None, scores=None, target_label_certainty=None):
        if target_label_certainty is None:
            target_label_certainty = 1.

        # Set flags and learning rate
        hard_negative_flag = learning_rate is not None
        if learning_rate is None:
            learning_rate = self.params.learning_rate

        if self.params.get('skip_low_certainty_frames', False):
            if target_label_certainty < self.params.get('low_certainty_th'):
                self.logging_dict['num_iters'].append(0)
                return
            else:
                target_label_certainty = 1

        # Update the tracker memory
        if hard_negative_flag or self.frame_num % self.params.get('train_sample_interval', 1) == 0:
            self.update_memory(TensorList([train_x]), train_y, target_box, learning_rate, target_label_certainty)

        # Decide the number of iterations to run
        num_iter = 0
        low_score_th = self.params.get('low_score_opt_threshold', None)
        if hard_negative_flag:
            num_iter = self.params.get('net_opt_hn_iter', None)

            # do not update if certainty of hn_sample is lower than ths it won't be considered during update anyway.
            ths_cert = self.params.get('certainty_for_weight_computation_ths', 0.5)
            ths_hn = self.params.get('certainty_for_weight_computation_hn_skip_ths', ths_cert)

            if (self.params.get('use_certainty_for_weight_computation', False) and ths_hn > target_label_certainty):
                num_iter = 0

        elif low_score_th is not None and low_score_th > scores.max().item():
            num_iter = self.params.get('net_opt_low_iter', None)
        elif (self.frame_num - 1) % self.params.train_skipping == 0:
            num_iter = self.params.get('net_opt_update_iter', None)

        if self.params.get('net_opt_every_frame', False):
            num_iter = self.params.get('net_opt_every_frame_iter', 1)

        plot_loss = self.params.debug > 0

        self.logging_dict['num_iters'].append(num_iter)



        # Compute sample weights either fully on age or mix with correctness certainty of target lables.
        # Supress memory sample if certainty is below certain threshold.

        sample_weights = self.sample_weights[0][:self.num_stored_samples[0]].view(-1, 1, 1, 1)

        if self.params.get('use_certainty_for_weight_computation', False):
            target_label_certainties = self.target_label_certainties[:self.num_stored_samples[0]].view(-1, 1, 1, 1)
            ths = self.params.get('certainty_for_weight_computation_ths', 0.5)
            weights = target_label_certainties
            weights[weights < ths] = 0.0
            weights = weights*sample_weights
        else:
            weights = sample_weights.clone()

        self.weights = weights


        if num_iter > 0:
            # Get inputs for the DiMP filter optimizer module
            samples = self.training_samples[0][:self.num_stored_samples[0],...]
            target_labels = self.target_labels[0][:self.num_stored_samples[0],...]
            target_boxes = self.target_boxes[:self.num_stored_samples[0],:].clone()

            self.net.classifier.compute_losses = plot_loss


            # Run the filter optimizer module
            with torch.no_grad():
                target_filter, _, losses = self.net.classifier.filter_optimizer(TensorList([self.target_filter]),
                                                                                num_iter=num_iter, feat=samples,
                                                                                bb=target_boxes, train_label=target_labels,
                                                                                sample_weight=weights)
                self.target_filter = target_filter[0]


            if plot_loss:
                if isinstance(losses, dict):
                    losses = losses['train']

                self.losses = torch.cat((self.losses, torch.stack(losses)))
                if self.visdom is not None:
                    self.visdom.register((self.losses, torch.arange(self.losses.numel())), 'lineplot', 3, 'Training Loss' + self.id_str)
                elif self.params.debug >= 3:
                    plot_graph(self.losses, 10, title='Training Loss' + self.id_str)


    def refine_target_box(self, backbone_feat, sample_pos, sample_scale, scale_ind, update_scale = True):
        """Run the ATOM IoUNet to refine the target bounding box."""

        if hasattr(self.net.bb_regressor, 'predict_bb'):
            return self.direct_box_regression(backbone_feat, sample_pos, sample_scale, scale_ind, update_scale)

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
        self.predicted_iou = output_iou.view(-1, 1)[inds, :].mean(0)

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

    def direct_box_regression(self, backbone_feat, sample_pos, sample_scale, scale_ind, update_scale = True):
        """Run the ATOM IoUNet to refine the target bounding box."""

        # Initial box for refinement
        init_box = self.get_iounet_box(self.pos, self.target_sz, sample_pos, sample_scale)

        # Extract features from the relevant scale
        iou_features = self.get_iou_features(backbone_feat)
        iou_features = TensorList([x[scale_ind:scale_ind+1,...] for x in iou_features])

        # Generate random initial boxes
        init_boxes = init_box.view(1, 1, 4).clone().to(self.params.device)

        # Optimize the boxes
        output_boxes = self.net.bb_regressor.predict_bb(self.iou_modulation, iou_features, init_boxes).view(-1,4).cpu()

        # Remove weird boxes
        output_boxes[:, 2:].clamp_(1)

        predicted_box = output_boxes[0, :]

        # Get new position and size
        new_pos = predicted_box[:2] + predicted_box[2:] / 2
        new_pos = (new_pos.flip((0,)) - (self.img_sample_sz - 1) / 2) * sample_scale + sample_pos
        new_target_sz = predicted_box[2:].flip((0,)) * sample_scale
        new_scale_bbr = torch.sqrt(new_target_sz.prod() / self.base_target_sz.prod())
        new_scale = new_scale_bbr

        self.pos_iounet = new_pos.clone()

        if self.params.get('use_iounet_pos_for_learning', True):
            self.pos = new_pos.clone()

        self.target_sz = new_target_sz

        if update_scale:
            self.target_scale = new_scale

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
        if hasattr(self, 'search_area_box'):
            if self.params.get('use_gt_box', False):
                bbox_gth = self.frame_reader.get_bbox(self.frame_num - 1, None)
                if np.any(np.isnan(bbox_gth)):
                    self.visdom.register((image, box, self.search_area_box), 'Tracking', 1, 'Tracking')
                else:
                    self.visdom.register((image, box, self.search_area_box,torch.from_numpy(bbox_gth)),
                                          'Tracking', 1, 'Tracking')
            else:
                self.visdom.register((image, box, self.search_area_box), 'Tracking', 1, 'Tracking')
        else:
            self.visdom.register((image, box), 'Tracking', 1, 'Tracking')