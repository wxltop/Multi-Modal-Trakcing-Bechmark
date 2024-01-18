from pytracking.utils import TrackerParams
from pytracking.features.net_wrappers import NetWithBackbone, NetWrapper

def parameters():
    params = TrackerParams()

    params.debug = 0
    params.visualization = False

    params.use_gpu = True

    params.image_sample_size = 30*16
    params.search_area_scale = 8
    params.border_mode = 'inside_major'
    params.patch_max_scale_change = 1.5

    # Learning parameters
    params.sample_memory_size = 50
    params.learning_rate = 0.01
    params.init_samples_minimum_weight = 0.25
    params.train_skipping = 20

    # Net optimization params
    params.update_classifier = True
    params.net_opt_iter = 10
    params.net_opt_update_iter = 2
    params.net_opt_hn_iter = 1

    # Detection parameters
    params.window_output = False

    # Init augmentation parameters
    params.use_augmentation = True
    params.augmentation = {'fliplr': True,
                           'rotate': [10, -10, 45, -45],
                           'blur': [(3,1), (1, 3), (2, 2)],
                           'relativeshift': [(0.6, 0.6), (-0.6, 0.6), (0.6, -0.6), (-0.6,-0.6)],
                           'dropout': (2, 0.2)}

    params.augmentation_expansion_factor = 2
    params.random_shift_factor = 1/3

    # Advanced localization parameters
    params.advanced_localization = True
    params.target_not_found_threshold = 0.25
    params.distractor_threshold = 0.8
    params.hard_negative_threshold = 0.5
    params.target_neighborhood_scale = 2.2
    params.dispalcement_scale = 0.8
    params.hard_negative_learning_rate = 0.02
    params.update_scale_when_uncertain = True

    # IoUnet parameters
    params.box_refinement_space = 'relative'
    params.iounet_augmentation = False      # Use the augmented samples to compute the modulation vector
    params.iounet_k = 3                     # Top-k average to estimate final box
    params.num_init_random_boxes = 9        # Num extra random boxes in addition to the classifier prediction
    params.box_jitter_pos = 0.1             # How much to jitter the translation for random boxes
    params.box_jitter_sz = 0.5              # How much to jitter the scale for random boxes
    params.maximal_aspect_ratio = 6         # Limit on the aspect ratio
    params.box_refinement_iter = 10          # Number of iterations for refining the boxes
    params.box_refinement_step_length = 2.5e-3 # 1   # Gradient step length in the bounding box refinement
    params.box_refinement_step_decay = 1    # Multiplicative step length decay (1 means no decay)

    params.use_gt_box = True
    params.net = NetWithBackbone(net_path='super_dimp_hinge', use_gpu=params.use_gpu)
    # params.cp_net = NetWrapper(net_path='certainty_prediction_v1', use_gpu=params.use_gpu)
    # params.peak_pred_net = NetWrapper(net_path='peak_prediction_v3', use_gpu=params.use_gpu)
    params.peak_match_net = NetWrapper(net_path='peak_matching_v1_mixed', use_gpu=params.use_gpu)

    params.vot_anno_conversion_type = 'preserve_area'

    # params.target_label_certainty_type = 'learned_predicted_certainty'
    params.target_label_certainty_type = None
    # params.target_label_certainty_type = 'max_score_map'
    # params.proj_score_computation_type = 'average_all_proj_scores'
    # params.certainty_for_weight_computation_ths = 0.0
    # params.certainty_for_weight_computation_hn_skip_ths = 0.5 # not used at the moment
    params.use_certainty_for_weight_computation = False

    params.enable_search_area_rescaling_at_occlusion = False

    params.enable_peak_localization_by_matching = True
    params.disable_chronological_occlusion_redetection_logic = True

    params.skip_running_matching_network_for_single_peak_cases = True
    params.use_image_coordinates_for_matching = True

    params.id0_weight_increase = False

    params.drop_low_assignment_prob = False

    params.debugging_plots_list = [
        'iou',
        # 'max_score_map',
        # 'bbox_size_diff',
        # 'center_dist',
        # 'num_iters',
        # 'mean_max_ptm_all',
        # 'mean_max_ptm_gth',
        # 'mean_max_scaled_ptm_all',
        # 'mean_max_scaled_ptm_gth',
        # 'predicted_certainties',
        # 'certainties_mem',
        # 'sample_weights',
        # 'scaled_sample_weights',
        # 'peak_prob',
        # 'pred_bbox_size',
        # 'assignment_prob',
        # 'certainties_mem'
    ]

    params.debugging_heat_map_list = [
        # 'score_map',
        # 'ptm_other',
        # 'ptm_gth',
        # 'pmt_mean_all',
        # 'pmt_std_all',
        # 'pmt_mean_gth',
        # 'pmt_std_gth',
        # 'pmt_mean_other',
        # 'pmt_std_other',
        # 'scaled_ptm_gth',
        # 'score_peak_map'
    ]

    return params
