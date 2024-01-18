from pytracking.evaluation import get_dataset, trackerlist


def mem1():
    trackers = trackerlist('dimp', 'dimp18_memory_most_recent', range(0, 3))

    dataset = get_dataset('lasot')
    return trackers, dataset


def mem2():
    trackers = trackerlist('dimp', 'dimp18_memory_two_most_recent', range(0, 3))

    dataset = get_dataset('lasot')
    return trackers, dataset


def mem5():
    trackers = trackerlist('dimp', 'dimp18_memory_five_most_recent', range(0, 3))

    dataset = get_dataset('lasot')
    return trackers, dataset


def mem50():
    trackers = trackerlist('dimp', 'dimp18', range(0, 5))

    dataset = get_dataset('lasot')
    return trackers, dataset


def no_aug_lasot():
    trackers = trackerlist('dimp', 'dimp18_no_augmentation', range(0, 5))

    dataset = get_dataset('lasot')
    return trackers, dataset


def no_aug_triple():
    trackers = trackerlist('dimp', 'dimp18_no_augmentation', range(0, 5))

    dataset = get_dataset('nfs', 'uav', 'otb')
    return trackers, dataset


def baseline():
    trackers = trackerlist('dimp_weight_learning', 'dimp18_baseline', range(0, 5))
    dataset = get_dataset('lasot')
    return trackers, dataset


def patch_filter():
    trackers = trackerlist('dimp_weight_learning', 'dimp18_patch_filter', range(0, 5))
    dataset = get_dataset('lasot')
    return trackers, dataset

def baseline_super_dimp_hinge_uav():
    trackers = trackerlist('dimp_weight_learning', 'super_dimp_hinge', range(0, 5))
    dataset = get_dataset('uav')
    return trackers, dataset


def patch_similarity_learning():
    trackers = trackerlist('dimp_patch_similarity_learning', 'dimp18_patch_similarity_learning', range(0, 5))
    dataset = get_dataset('otb')
    return trackers, dataset


def transformer_baseline_otb():
    trackers = trackerlist('dimp_patch_similarity_learning', 'super_dimp_hinge', range(0, 5))
    dataset = get_dataset('otb')
    return trackers, dataset

def transformer_baseline_lasot():
    trackers = trackerlist('dimp_weight_prediction', 'super_dimp_weight_prediction_baseline', range(2, 3))
    dataset = get_dataset('lasot')
    return trackers, dataset

# def transformer_baseline_lasot():
#     trackers = trackerlist('dimp_patch_similarity_learning', 'super_dimp_hinge_baseline', range(2, 5))
#     dataset = get_dataset('lasot')
#     return trackers, dataset

def transformer_proj_only_lasot():
    trackers = trackerlist('dimp_weight_prediction', 'super_dimp_weight_prediction_proj_only', range(1,2))
    dataset = get_dataset('lasot')
    return trackers, dataset



def transformer_baseline_fix_lasot_0():
    trackers = trackerlist('dimp_weight_prediction', 'super_dimp_weight_prediction_baseline_update_flag_fix_always_optimize', range(0, 1))
    dataset = get_dataset('lasot')
    return trackers, dataset

def transformer_baseline_fix_lasot_1():
    trackers = trackerlist('dimp_weight_prediction', 'super_dimp_weight_prediction_baseline_update_flag_fix_always_optimize', range(1, 2))
    dataset = get_dataset('lasot')
    return trackers, dataset

def transformer_baseline_fix_lasot_2():
    trackers = trackerlist('dimp_weight_prediction', 'super_dimp_weight_prediction_baseline_update_flag_fix_always_optimize', range(2, 3))
    dataset = get_dataset('lasot')
    return trackers, dataset

def transformer_baseline_fix_lasot_3():
    trackers = trackerlist('dimp_weight_prediction', 'super_dimp_weight_prediction_baseline_update_flag_fix_always_optimize', range(3, 4))
    dataset = get_dataset('lasot')
    return trackers, dataset

def transformer_baseline_fix_lasot_4():
    trackers = trackerlist('dimp_weight_prediction', 'super_dimp_weight_prediction_baseline_update_flag_fix_always_optimize', range(4, 5))
    dataset = get_dataset('lasot')
    return trackers, dataset


def transformer_continuous_proj_only_baseline_lasot_critical_0():
    trackers = trackerlist('dimp_weight_prediction', 'super_dimp_weight_prediction_continuous_proj_only_baseline', range(0, 1))
    dataset = get_dataset('lasot_critical')
    return trackers, dataset

def transformer_continuous_proj_only_baseline_lasot_critical_1():
    trackers = trackerlist('dimp_weight_prediction', 'super_dimp_weight_prediction_continuous_proj_only_baseline', range(1, 2))
    dataset = get_dataset('lasot_critical')
    return trackers, dataset

def transformer_continuous_proj_only_baseline_lasot_critical_2():
    trackers = trackerlist('dimp_weight_prediction', 'super_dimp_weight_prediction_continuous_proj_only_baseline', range(2, 3))
    dataset = get_dataset('lasot_critical')
    return trackers, dataset

def transformer_continuous_proj_only_baseline_lasot_critical_3():
    trackers = trackerlist('dimp_weight_prediction', 'super_dimp_weight_prediction_continuous_proj_only_baseline', range(3, 4))
    dataset = get_dataset('lasot_critical')
    return trackers, dataset

def transformer_continuous_proj_only_baseline_lasot_critical_4():
    trackers = trackerlist('dimp_weight_prediction', 'super_dimp_weight_prediction_continuous_proj_only_baseline', range(4, 5))
    dataset = get_dataset('lasot_critical')
    return trackers, dataset

def transformer_continuous_proj_only_baseline_lasot_0():
    trackers = trackerlist('dimp_weight_prediction', 'super_dimp_weight_prediction_continuous_proj_only_baseline', range(0, 1))
    dataset = get_dataset('lasot')
    return trackers, dataset

def transformer_continuous_proj_only_baseline_lasot_1():
    trackers = trackerlist('dimp_weight_prediction', 'super_dimp_weight_prediction_continuous_proj_only_baseline', range(1, 2))
    dataset = get_dataset('lasot')
    return trackers, dataset

def transformer_continuous_proj_only_baseline_lasot_2():
    trackers = trackerlist('dimp_weight_prediction', 'super_dimp_weight_prediction_continuous_proj_only_baseline', range(2, 3))
    dataset = get_dataset('lasot')
    return trackers, dataset

def transformer_continuous_proj_only_baseline_lasot_3():
    trackers = trackerlist('dimp_weight_prediction', 'super_dimp_weight_prediction_continuous_proj_only_baseline', range(3, 4))
    dataset = get_dataset('lasot')
    return trackers, dataset

def transformer_continuous_proj_only_baseline_lasot_4():
    trackers = trackerlist('dimp_weight_prediction', 'super_dimp_weight_prediction_continuous_proj_only_baseline', range(4, 5))
    dataset = get_dataset('lasot')
    return trackers, dataset


def transformer_continuous_proj_dist_map_cls_feat_lasot_critical_0():
    trackers = trackerlist('dimp_weight_prediction', 'super_dimp_weight_prediction_continuous_proj_dist_map_cls_feat', range(0, 1))
    dataset = get_dataset('lasot_critical')
    return trackers, dataset

def transformer_continuous_proj_dist_map_cls_feat_lasot_critical_1():
    trackers = trackerlist('dimp_weight_prediction', 'super_dimp_weight_prediction_continuous_proj_dist_map_cls_feat', range(1, 2))
    dataset = get_dataset('lasot_critical')
    return trackers, dataset

def transformer_continuous_proj_dist_map_cls_feat_lasot_critical_2():
    trackers = trackerlist('dimp_weight_prediction', 'super_dimp_weight_prediction_continuous_proj_dist_map_cls_feat', range(2, 3))
    dataset = get_dataset('lasot_critical')
    return trackers, dataset

def transformer_continuous_proj_dist_map_cls_feat_lasot_critical_3():
    trackers = trackerlist('dimp_weight_prediction', 'super_dimp_weight_prediction_continuous_proj_dist_map_cls_feat', range(3, 4))
    dataset = get_dataset('lasot_critical')
    return trackers, dataset

def transformer_continuous_proj_dist_map_cls_feat_lasot_critical_4():
    trackers = trackerlist('dimp_weight_prediction', 'super_dimp_weight_prediction_continuous_proj_dist_map_cls_feat', range(4, 5))
    dataset = get_dataset('lasot_critical')
    return trackers, dataset



def transformer_cont_multi_test_random_gap_lasot_critical_0():
    trackers = trackerlist('dimp_weight_prediction', 'super_dimp_weight_prediction_cont_multi_test_random_gap', range(0, 1))
    dataset = get_dataset('lasot_critical')
    return trackers, dataset

def transformer_cont_multi_test_random_gap_lasot_critical_1():
    trackers = trackerlist('dimp_weight_prediction', 'super_dimp_weight_prediction_cont_multi_test_random_gap', range(1, 2))
    dataset = get_dataset('lasot_critical')
    return trackers, dataset

def transformer_cont_multi_test_random_gap_lasot_critical_2():
    trackers = trackerlist('dimp_weight_prediction', 'super_dimp_weight_prediction_cont_multi_test_random_gap', range(2, 3))
    dataset = get_dataset('lasot_critical')
    return trackers, dataset

def transformer_cont_multi_test_random_gap_lasot_critical_3():
    trackers = trackerlist('dimp_weight_prediction', 'super_dimp_weight_prediction_cont_multi_test_random_gap', range(3, 4))
    dataset = get_dataset('lasot_critical')
    return trackers, dataset

def transformer_cont_multi_test_random_gap_lasot_critical_4():
    trackers = trackerlist('dimp_weight_prediction', 'super_dimp_weight_prediction_cont_multi_test_random_gap', range(4, 5))
    dataset = get_dataset('lasot_critical')
    return trackers, dataset

def transformer_cont_multi_test_lasot_0():
    trackers = trackerlist('dimp_weight_prediction', 'super_dimp_weight_prediction_cont_multi_test', range(0, 1))
    dataset = get_dataset('lasot')
    return trackers, dataset

def transformer_cont_multi_test_lasot_1():
    trackers = trackerlist('dimp_weight_prediction', 'super_dimp_weight_prediction_cont_multi_test', range(1, 2))
    dataset = get_dataset('lasot')
    return trackers, dataset

def transformer_cont_multi_test_lasot_2():
    trackers = trackerlist('dimp_weight_prediction', 'super_dimp_weight_prediction_cont_multi_test', range(2, 3))
    dataset = get_dataset('lasot')
    return trackers, dataset

def transformer_cont_multi_test_lasot_3():
    trackers = trackerlist('dimp_weight_prediction', 'super_dimp_weight_prediction_cont_multi_test', range(3, 4))
    dataset = get_dataset('lasot')
    return trackers, dataset

def transformer_cont_multi_test_lasot_4():
    trackers = trackerlist('dimp_weight_prediction', 'super_dimp_weight_prediction_cont_multi_test', range(4, 5))
    dataset = get_dataset('lasot')
    return trackers, dataset


def transformer_cont_dummy_lasot_critical_0():
    trackers = trackerlist('dimp_weight_prediction', 'super_dimp_weight_prediction_cont_dummy', range(0, 1))
    dataset = get_dataset('lasot_critical')
    return trackers, dataset

def transformer_cont_dummy_lasot_critical_1():
    trackers = trackerlist('dimp_weight_prediction', 'super_dimp_weight_prediction_cont_dummy', range(1, 2))
    dataset = get_dataset('lasot_critical')
    return trackers, dataset

def transformer_cont_dummy_lasot_critical_2():
    trackers = trackerlist('dimp_weight_prediction', 'super_dimp_weight_prediction_cont_dummy', range(2, 3))
    dataset = get_dataset('lasot_critical')
    return trackers, dataset

def transformer_cont_dummy_lasot_critical_3():
    trackers = trackerlist('dimp_weight_prediction', 'super_dimp_weight_prediction_cont_dummy', range(3, 4))
    dataset = get_dataset('lasot_critical')
    return trackers, dataset

def transformer_cont_dummy_lasot_critical_4():
    trackers = trackerlist('dimp_weight_prediction', 'super_dimp_weight_prediction_cont_dummy', range(4, 5))
    dataset = get_dataset('lasot_critical')
    return trackers, dataset


def transformer_cont_multi_test_long_test_lasot_critical_0():
    trackers = trackerlist('dimp_weight_prediction', 'super_dimp_weight_prediction_cont_multi_test_long_test', range(0, 1))
    dataset = get_dataset('lasot_critical')
    return trackers, dataset

def transformer_cont_multi_test_long_test_lasot_critical_1():
    trackers = trackerlist('dimp_weight_prediction', 'super_dimp_weight_prediction_cont_multi_test_long_test', range(1, 2))
    dataset = get_dataset('lasot_critical')
    return trackers, dataset

def transformer_cont_multi_test_long_test_lasot_critical_2():
    trackers = trackerlist('dimp_weight_prediction', 'super_dimp_weight_prediction_cont_multi_test_long_test', range(2, 3))
    dataset = get_dataset('lasot_critical')
    return trackers, dataset

def transformer_cont_multi_test_long_test_lasot_critical_3():
    trackers = trackerlist('dimp_weight_prediction', 'super_dimp_weight_prediction_cont_multi_test_long_test', range(3, 4))
    dataset = get_dataset('lasot_critical')
    return trackers, dataset

def transformer_cont_multi_test_long_test_lasot_critical_4():
    trackers = trackerlist('dimp_weight_prediction', 'super_dimp_weight_prediction_cont_multi_test_long_test', range(4, 5))
    dataset = get_dataset('lasot_critical')
    return trackers, dataset


def transformer_cont_spread_certain_dist_proj_raw_lasot_0():
    trackers = trackerlist('dimp_weight_prediction', 'super_dimp_weight_prediction_cont_spread_certain_dist_proj_raw', range(0, 1))
    dataset = get_dataset('lasot')
    return trackers, dataset

def transformer_cont_spread_certain_dist_proj_raw_lasot_1():
    trackers = trackerlist('dimp_weight_prediction', 'super_dimp_weight_prediction_cont_spread_certain_dist_proj_raw', range(1, 2))
    dataset = get_dataset('lasot')
    return trackers, dataset

def transformer_cont_spread_certain_dist_proj_raw_lasot_2():
    trackers = trackerlist('dimp_weight_prediction', 'super_dimp_weight_prediction_cont_spread_certain_dist_proj_raw', range(2, 3))
    dataset = get_dataset('lasot')
    return trackers, dataset

def transformer_cont_spread_certain_dist_proj_raw_lasot_3():
    trackers = trackerlist('dimp_weight_prediction', 'super_dimp_weight_prediction_cont_spread_certain_dist_proj_raw', range(3, 4))
    dataset = get_dataset('lasot')
    return trackers, dataset

def transformer_cont_spread_certain_dist_proj_raw_lasot_4():
    trackers = trackerlist('dimp_weight_prediction', 'super_dimp_weight_prediction_cont_spread_certain_dist_proj_raw', range(4, 5))
    dataset = get_dataset('lasot')
    return trackers, dataset

def transformer_cont_spread_certain_dist_proj_raw_update_logic_lasot_0():
    trackers = trackerlist('dimp_weight_prediction', 'super_dimp_weight_prediction_cont_spread_certain_dist_proj_raw_update_logic', range(0, 1))
    dataset = get_dataset('lasot')
    return trackers, dataset

def transformer_cont_spread_certain_dist_proj_raw_update_logic_lasot_1():
    trackers = trackerlist('dimp_weight_prediction', 'super_dimp_weight_prediction_cont_spread_certain_dist_proj_raw_update_logic', range(1, 2))
    dataset = get_dataset('lasot')
    return trackers, dataset

def transformer_cont_spread_certain_dist_proj_raw_update_logic_lasot_2():
    trackers = trackerlist('dimp_weight_prediction', 'super_dimp_weight_prediction_cont_spread_certain_dist_proj_raw_update_logic', range(2, 3))
    dataset = get_dataset('lasot')
    return trackers, dataset

def transformer_cont_spread_certain_dist_proj_raw_update_logic_lasot_3():
    trackers = trackerlist('dimp_weight_prediction', 'super_dimp_weight_prediction_cont_spread_certain_dist_proj_raw_update_logic', range(3, 4))
    dataset = get_dataset('lasot')
    return trackers, dataset

def transformer_cont_spread_certain_dist_proj_raw_update_logic_lasot_4():
    trackers = trackerlist('dimp_weight_prediction', 'super_dimp_weight_prediction_cont_spread_certain_dist_proj_raw_update_logic', range(4, 5))
    dataset = get_dataset('lasot')
    return trackers, dataset


def transformer_cont_spread_certain_raw_lasot_critical_0():
    trackers = trackerlist('dimp_weight_prediction', 'super_dimp_weight_prediction_cont_spread_certain_raw', range(0, 1))
    dataset = get_dataset('lasot_critical')
    return trackers, dataset

def transformer_cont_spread_certain_raw_lasot_critical_1():
    trackers = trackerlist('dimp_weight_prediction', 'super_dimp_weight_prediction_cont_spread_certain_raw', range(1, 2))
    dataset = get_dataset('lasot_critical')
    return trackers, dataset

def transformer_cont_spread_certain_raw_lasot_critical_2():
    trackers = trackerlist('dimp_weight_prediction', 'super_dimp_weight_prediction_cont_spread_certain_raw', range(2, 3))
    dataset = get_dataset('lasot_critical')
    return trackers, dataset

def transformer_cont_spread_certain_raw_lasot_critical_3():
    trackers = trackerlist('dimp_weight_prediction', 'super_dimp_weight_prediction_cont_spread_certain_raw', range(3, 4))
    dataset = get_dataset('lasot_critical')
    return trackers, dataset

def transformer_cont_spread_certain_raw_lasot_critical_4():
    trackers = trackerlist('dimp_weight_prediction', 'super_dimp_weight_prediction_cont_spread_certain_raw', range(4, 5))
    dataset = get_dataset('lasot_critical')
    return trackers, dataset

def transformer_cont_spread_certain_update_logic_lasot_critical_0():
    trackers = trackerlist('dimp_weight_prediction', 'super_dimp_weight_prediction_cont_spread_certain_update_logic', range(0, 1))
    dataset = get_dataset('lasot_critical')
    return trackers, dataset

def transformer_cont_spread_certain_update_logic_lasot_critical_1():
    trackers = trackerlist('dimp_weight_prediction', 'super_dimp_weight_prediction_cont_spread_certain_update_logic', range(1, 2))
    dataset = get_dataset('lasot_critical')
    return trackers, dataset

def transformer_cont_spread_certain_update_logic_lasot_critical_2():
    trackers = trackerlist('dimp_weight_prediction', 'super_dimp_weight_prediction_cont_spread_certain_update_logic', range(2, 3))
    dataset = get_dataset('lasot_critical')
    return trackers, dataset

def transformer_cont_spread_certain_update_logic_lasot_critical_3():
    trackers = trackerlist('dimp_weight_prediction', 'super_dimp_weight_prediction_cont_spread_certain_update_logic', range(3, 4))
    dataset = get_dataset('lasot_critical')
    return trackers, dataset

def transformer_cont_spread_certain_update_logic_lasot_critical_4():
    trackers = trackerlist('dimp_weight_prediction', 'super_dimp_weight_prediction_cont_spread_certain_update_logic', range(4, 5))
    dataset = get_dataset('lasot_critical')
    return trackers, dataset

def transformer_cont_spread_certain_update_logic_lasot_0():
    trackers = trackerlist('dimp_weight_prediction', 'super_dimp_weight_prediction_cont_spread_certain_update_logic', range(0, 1))
    dataset = get_dataset('lasot')
    return trackers, dataset

def transformer_cont_spread_certain_update_logic_lasot_1():
    trackers = trackerlist('dimp_weight_prediction', 'super_dimp_weight_prediction_cont_spread_certain_update_logic', range(1, 2))
    dataset = get_dataset('lasot')
    return trackers, dataset

def transformer_cont_spread_certain_update_logic_lasot_2():
    trackers = trackerlist('dimp_weight_prediction', 'super_dimp_weight_prediction_cont_spread_certain_update_logic', range(2, 3))
    dataset = get_dataset('lasot')
    return trackers, dataset

def transformer_cont_spread_certain_update_logic_lasot_3():
    trackers = trackerlist('dimp_weight_prediction', 'super_dimp_weight_prediction_cont_spread_certain_update_logic', range(3, 4))
    dataset = get_dataset('lasot')
    return trackers, dataset

def transformer_cont_spread_certain_update_logic_lasot_4():
    trackers = trackerlist('dimp_weight_prediction', 'super_dimp_weight_prediction_cont_spread_certain_update_logic', range(4, 5))
    dataset = get_dataset('lasot')
    return trackers, dataset

def transformer_cont_spread_cont_spread_certain_dist_proj_raw_every_2_update_logic_lasot_critical_0():
    trackers = trackerlist('dimp_weight_prediction', 'super_dimp_weight_prediction_cont_spread_certain_dist_proj_raw_every_2_update_logic', range(0, 1))
    dataset = get_dataset('lasot_critical')
    return trackers, dataset

def transformer_cont_spread_cont_spread_certain_dist_proj_raw_every_2_update_logic_lasot_critical_1():
    trackers = trackerlist('dimp_weight_prediction', 'super_dimp_weight_prediction_cont_spread_certain_dist_proj_raw_every_2_update_logic', range(1, 2))
    dataset = get_dataset('lasot_critical')
    return trackers, dataset

def transformer_cont_spread_cont_spread_certain_dist_proj_raw_every_2_update_logic_lasot_critical_2():
    trackers = trackerlist('dimp_weight_prediction', 'super_dimp_weight_prediction_cont_spread_certain_dist_proj_raw_every_2_update_logic', range(2, 3))
    dataset = get_dataset('lasot_critical')
    return trackers, dataset

def transformer_cont_spread_cont_spread_certain_dist_proj_raw_every_2_update_logic_lasot_critical_3():
    trackers = trackerlist('dimp_weight_prediction', 'super_dimp_weight_prediction_cont_spread_certain_dist_proj_raw_every_2_update_logic', range(3, 4))
    dataset = get_dataset('lasot_critical')
    return trackers, dataset

def transformer_cont_spread_cont_spread_certain_dist_proj_raw_every_2_update_logic_lasot_critical_4():
    trackers = trackerlist('dimp_weight_prediction', 'super_dimp_weight_prediction_cont_spread_certain_dist_proj_raw_every_2_update_logic', range(4, 5))
    dataset = get_dataset('lasot_critical')
    return trackers, dataset

def transformer_cont_spread_cont_spread_certain_proj_lasot_critical_0():
    trackers = trackerlist('dimp_weight_prediction', 'super_dimp_weight_prediction_cont_spread_certain_proj', range(0, 1))
    dataset = get_dataset('lasot_critical')
    return trackers, dataset

def transformer_cont_spread_cont_spread_certain_proj_lasot_critical_1():
    trackers = trackerlist('dimp_weight_prediction', 'super_dimp_weight_prediction_cont_spread_certain_proj', range(1, 2))
    dataset = get_dataset('lasot_critical')
    return trackers, dataset

def transformer_cont_spread_cont_spread_certain_proj_lasot_critical_2():
    trackers = trackerlist('dimp_weight_prediction', 'super_dimp_weight_prediction_cont_spread_certain_proj', range(2, 3))
    dataset = get_dataset('lasot_critical')
    return trackers, dataset

def transformer_cont_spread_cont_spread_certain_proj_lasot_critical_3():
    trackers = trackerlist('dimp_weight_prediction', 'super_dimp_weight_prediction_cont_spread_certain_proj', range(3, 4))
    dataset = get_dataset('lasot_critical')
    return trackers, dataset

def transformer_cont_spread_cont_spread_certain_proj_lasot_critical_4():
    trackers = trackerlist('dimp_weight_prediction', 'super_dimp_weight_prediction_cont_spread_certain_proj', range(4, 5))
    dataset = get_dataset('lasot_critical')
    return trackers, dataset


# Baselines

def baseline_every_10_lasot_critical_0():
    trackers = trackerlist('dimp_weight_prediction', 'super_dimp_weight_prediction_baseline_cls_every_10', range(0, 1))
    dataset = get_dataset('lasot_critical')
    return trackers, dataset

def baseline_every_10_lasot_critical_1():
    trackers = trackerlist('dimp_weight_prediction', 'super_dimp_weight_prediction_baseline_cls_every_10', range(1, 2))
    dataset = get_dataset('lasot_critical')
    return trackers, dataset

def baseline_every_10_lasot_critical_2():
    trackers = trackerlist('dimp_weight_prediction', 'super_dimp_weight_prediction_baseline_cls_every_10', range(2, 3))
    dataset = get_dataset('lasot_critical')
    return trackers, dataset

def baseline_every_20_lasot_critical_0():
    trackers = trackerlist('dimp_weight_prediction', 'super_dimp_weight_prediction_baseline_cls_every_20', range(0, 1))
    dataset = get_dataset('lasot_critical')
    return trackers, dataset

def baseline_every_20_lasot_critical_1():
    trackers = trackerlist('dimp_weight_prediction', 'super_dimp_weight_prediction_baseline_cls_every_20', range(1, 2))
    dataset = get_dataset('lasot_critical')
    return trackers, dataset

def baseline_every_20_lasot_critical_2():
    trackers = trackerlist('dimp_weight_prediction', 'super_dimp_weight_prediction_baseline_cls_every_20', range(2, 3))
    dataset = get_dataset('lasot_critical')
    return trackers, dataset


def baseline_every_10_mem_update_logic_lasot_critical_0():
    trackers = trackerlist('dimp_weight_prediction', 'super_dimp_weight_prediction_baseline_cls_every_10_mem_update_logic', range(0, 1))
    dataset = get_dataset('lasot_critical')
    return trackers, dataset

def baseline_every_10_mem_update_logic_lasot_critical_1():
    trackers = trackerlist('dimp_weight_prediction', 'super_dimp_weight_prediction_baseline_cls_every_10_mem_update_logic', range(1, 2))
    dataset = get_dataset('lasot_critical')
    return trackers, dataset

def baseline_every_10_mem_update_logic_lasot_critical_2():
    trackers = trackerlist('dimp_weight_prediction', 'super_dimp_weight_prediction_baseline_cls_every_10_mem_update_logic', range(2, 3))
    dataset = get_dataset('lasot_critical')
    return trackers, dataset

def baseline_every_20_mem_update_logic_lasot_critical_0():
    trackers = trackerlist('dimp_weight_prediction', 'super_dimp_weight_prediction_baseline_cls_every_20_mem_update_logic', range(0, 1))
    dataset = get_dataset('lasot_critical')
    return trackers, dataset

def baseline_every_20_mem_update_logic_lasot_critical_1():
    trackers = trackerlist('dimp_weight_prediction', 'super_dimp_weight_prediction_baseline_cls_every_20_mem_update_logic', range(1, 2))
    dataset = get_dataset('lasot_critical')
    return trackers, dataset

def baseline_every_20_mem_update_logic_lasot_critical_2():
    trackers = trackerlist('dimp_weight_prediction', 'super_dimp_weight_prediction_baseline_cls_every_20_mem_update_logic', range(2, 3))
    dataset = get_dataset('lasot_critical')
    return trackers, dataset

def baseline_every_20_mem_update_logic_lasot_0():
    trackers = trackerlist('dimp_weight_prediction', 'super_dimp_weight_prediction_baseline_cls_every_20_mem_update_logic', range(0, 1))
    dataset = get_dataset('lasot')
    return trackers, dataset

def baseline_every_20_mem_update_logic_lasot_1():
    trackers = trackerlist('dimp_weight_prediction', 'super_dimp_weight_prediction_baseline_cls_every_20_mem_update_logic', range(1, 2))
    dataset = get_dataset('lasot')
    return trackers, dataset

def baseline_every_20_mem_update_logic_lasot_2():
    trackers = trackerlist('dimp_weight_prediction', 'super_dimp_weight_prediction_baseline_cls_every_20_mem_update_logic', range(2, 3))
    dataset = get_dataset('lasot')
    return trackers, dataset

def baseline_every_20_mem_update_logic_lasot_3():
    trackers = trackerlist('dimp_weight_prediction', 'super_dimp_weight_prediction_baseline_cls_every_20_mem_update_logic', range(3, 4))
    dataset = get_dataset('lasot')
    return trackers, dataset

def baseline_every_20_mem_update_logic_lasot_4():
    trackers = trackerlist('dimp_weight_prediction', 'super_dimp_weight_prediction_baseline_cls_every_20_mem_update_logic', range(4, 5))
    dataset = get_dataset('lasot')
    return trackers, dataset


def baseline_every_20_lasot_0():
    trackers = trackerlist('dimp_weight_prediction', 'super_dimp_weight_prediction_baseline_cls_every_20', range(0, 1))
    dataset = get_dataset('lasot')
    return trackers, dataset

def baseline_every_20_lasot_1():
    trackers = trackerlist('dimp_weight_prediction', 'super_dimp_weight_prediction_baseline_cls_every_20', range(1, 2))
    dataset = get_dataset('lasot')
    return trackers, dataset

def baseline_every_20_lasot_2():
    trackers = trackerlist('dimp_weight_prediction', 'super_dimp_weight_prediction_baseline_cls_every_20', range(2, 3))
    dataset = get_dataset('lasot')
    return trackers, dataset


def dimp_iter_every_1_lasot_0():
    trackers = trackerlist('dimp', 'super_dimp_update_cls_every_frame', range(0, 1))
    dataset = get_dataset('lasot')
    return trackers, dataset

def dimp_iter_every_1_lasot_1():
    trackers = trackerlist('dimp', 'super_dimp_update_cls_every_frame', range(1, 2))
    dataset = get_dataset('lasot')
    return trackers, dataset

def dimp_iter_every_1_lasot_2():
    trackers = trackerlist('dimp', 'super_dimp_update_cls_every_frame', range(2, 3))
    dataset = get_dataset('lasot')
    return trackers, dataset


def certainty_mem_control_lasot_0():
    trackers = trackerlist('dimp_weight_prediction', 'super_dimp_weight_prediction_cont_spread_certainty_mem_control', range(0, 1))
    dataset = get_dataset('lasot')
    return trackers, dataset

def certainty_mem_control_lasot_1():
    trackers = trackerlist('dimp_weight_prediction', 'super_dimp_weight_prediction_cont_spread_certainty_mem_control', range(1, 2))
    dataset = get_dataset('lasot')
    return trackers, dataset

def certainty_mem_control_lasot_2():
    trackers = trackerlist('dimp_weight_prediction', 'super_dimp_weight_prediction_cont_spread_certainty_mem_control', range(2, 3))
    dataset = get_dataset('lasot')
    return trackers, dataset

def certainty_mem_control_lasot_3():
    trackers = trackerlist('dimp_weight_prediction', 'super_dimp_weight_prediction_cont_spread_certainty_mem_control', range(3, 4))
    dataset = get_dataset('lasot')
    return trackers, dataset

def certainty_mem_control_lasot_4():
    trackers = trackerlist('dimp_weight_prediction', 'super_dimp_weight_prediction_cont_spread_certainty_mem_control', range(4, 5))
    dataset = get_dataset('lasot')
    return trackers, dataset



def memory_learning_baseline_iou_certainty_lasot_critical_0():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_baseline_iou_certainty', range(0, 1))
    dataset = get_dataset('lasot_critical')
    return trackers, dataset

def memory_learning_baseline_iou_certainty_lasot_critical_1():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_baseline_iou_certainty', range(1, 2))
    dataset = get_dataset('lasot_critical')
    return trackers, dataset

def memory_learning_baseline_iou_certainty_lasot_critical_2():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_baseline_iou_certainty', range(2, 3))
    dataset = get_dataset('lasot_critical')
    return trackers, dataset

def memory_learning_baseline_iou_certainty_lasot_critical_3():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_baseline_iou_certainty', range(3, 4))
    dataset = get_dataset('lasot_critical')
    return trackers, dataset

def memory_learning_baseline_iou_certainty_lasot_critical_4():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_baseline_iou_certainty', range(4, 5))
    dataset = get_dataset('lasot_critical')
    return trackers, dataset


def memory_learning_baseline_iou_certainty_lasot_0():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_baseline_iou_certainty', range(0, 1))
    dataset = get_dataset('lasot')
    return trackers, dataset

def memory_learning_baseline_iou_certainty_lasot_1():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_baseline_iou_certainty', range(1, 2))
    dataset = get_dataset('lasot')
    return trackers, dataset

def memory_learning_baseline_iou_certainty_lasot_2():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_baseline_iou_certainty', range(2, 3))
    dataset = get_dataset('lasot')
    return trackers, dataset

def memory_learning_baseline_iou_certainty_lasot_3():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_baseline_iou_certainty', range(3, 4))
    dataset = get_dataset('lasot')
    return trackers, dataset

def memory_learning_baseline_iou_certainty_lasot_4():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_baseline_iou_certainty', range(4, 5))
    dataset = get_dataset('lasot')
    return trackers, dataset


def memory_learning_max_proj_test_score_merge_certainty_lasot_critical_0():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_max_proj_test_score_on_gth_merge', range(0, 1))
    dataset = get_dataset('lasot_critical')
    return trackers, dataset

def memory_learning_max_proj_test_score_merge_certainty_lasot_critical_1():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_max_proj_test_score_on_gth_merge', range(1, 2))
    dataset = get_dataset('lasot_critical')
    return trackers, dataset

def memory_learning_max_proj_test_score_merge_certainty_lasot_critical_2():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_max_proj_test_score_on_gth_merge', range(2, 3))
    dataset = get_dataset('lasot_critical')
    return trackers, dataset


def memory_learning_max_proj_test_score_merge_certainty_lasot_0():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_max_proj_test_score_on_gth_merge', range(0, 1))
    dataset = get_dataset('lasot')
    return trackers, dataset

def memory_learning_max_proj_test_score_merge_certainty_lasot_1():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_max_proj_test_score_on_gth_merge', range(1, 2))
    dataset = get_dataset('lasot')
    return trackers, dataset

def memory_learning_max_proj_test_score_merge_certainty_lasot_2():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_max_proj_test_score_on_gth_merge', range(2, 3))
    dataset = get_dataset('lasot')
    return trackers, dataset

def memory_learning_max_proj_test_score_merge_certainty_lasot_3():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_max_proj_test_score_on_gth_merge', range(3, 4))
    dataset = get_dataset('lasot')
    return trackers, dataset

def memory_learning_max_proj_test_score_merge_certainty_lasot_4():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_max_proj_test_score_on_gth_merge', range(4, 5))
    dataset = get_dataset('lasot')
    return trackers, dataset


def memory_learning_max_proj_test_score_single_certainty_lasot_critical_0():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_max_proj_test_score_on_gth_single', range(0, 1))
    dataset = get_dataset('lasot_critical')
    return trackers, dataset

def memory_learning_max_proj_test_score_single_certainty_lasot_critical_1():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_max_proj_test_score_on_gth_single', range(1, 2))
    dataset = get_dataset('lasot_critical')
    return trackers, dataset

def memory_learning_max_proj_test_score_single_certainty_lasot_critical_2():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_max_proj_test_score_on_gth_single', range(2, 3))
    dataset = get_dataset('lasot_critical')
    return trackers, dataset


def memory_learning_max_proj_test_score_average_all_certainty_lasot_critical_0():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_max_proj_test_score_on_average_all', range(0, 1))
    dataset = get_dataset('lasot_critical')
    return trackers, dataset

def memory_learning_max_proj_test_score_average_all_certainty_lasot_critical_1():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_max_proj_test_score_on_average_all', range(1, 2))
    dataset = get_dataset('lasot_critical')
    return trackers, dataset

def memory_learning_max_proj_test_score_average_all_certainty_lasot_critical_2():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_max_proj_test_score_on_average_all', range(2, 3))
    dataset = get_dataset('lasot_critical')
    return trackers, dataset


def memory_learning_max_proj_test_score_average_all_certainty_lasot_0():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_max_proj_test_score_on_average_all', range(0, 1))
    dataset = get_dataset('lasot')
    return trackers, dataset

def memory_learning_max_proj_test_score_average_all_certainty_lasot_1():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_max_proj_test_score_on_average_all', range(1, 2))
    dataset = get_dataset('lasot')
    return trackers, dataset

def memory_learning_max_proj_test_score_average_all_certainty_lasot_2():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_max_proj_test_score_on_average_all', range(2, 3))
    dataset = get_dataset('lasot')
    return trackers, dataset

def memory_learning_max_proj_test_score_average_all_certainty_lasot_3():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_max_proj_test_score_on_average_all', range(3, 4))
    dataset = get_dataset('lasot')
    return trackers, dataset

def memory_learning_max_proj_test_score_average_all_certainty_lasot_4():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_max_proj_test_score_on_average_all', range(4, 5))
    dataset = get_dataset('lasot')
    return trackers, dataset

def memory_learning_max_proj_test_score_average_all_certainty_otb():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_max_proj_test_score_on_average_all', range(0, 5))
    dataset = get_dataset('otb')
    return trackers, dataset

def memory_learning_max_proj_test_score_average_all_certainty_uav():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_max_proj_test_score_on_average_all', range(0, 5))
    dataset = get_dataset('uav')
    return trackers, dataset

def memory_learning_max_proj_test_score_average_all_certainty_nfs():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_max_proj_test_score_on_average_all', range(0, 5))
    dataset = get_dataset('nfs')
    return trackers, dataset


def memory_learning_baseline_iou_certainty_otb():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_baseline_iou_certainty', range(0, 5))
    dataset = get_dataset('otb')
    return trackers, dataset

def memory_learning_baseline_iou_certainty_uav():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_baseline_iou_certainty', range(0, 5))
    dataset = get_dataset('uav')
    return trackers, dataset

def memory_learning_baseline_iou_certainty_nfs():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_baseline_iou_certainty', range(0, 5))
    dataset = get_dataset('nfs')
    return trackers, dataset

def memory_learning_max_proj_test_score_scaled_average_all_certainty_lasot_0():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_max_proj_test_score_on_scaled_average_all', range(0, 1))
    dataset = get_dataset('lasot')
    return trackers, dataset

def memory_learning_max_proj_test_score_scaled_average_all_certainty_lasot_1():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_max_proj_test_score_on_scaled_average_all', range(1, 2))
    dataset = get_dataset('lasot')
    return trackers, dataset

def memory_learning_max_proj_test_score_scaled_average_all_certainty_lasot_2():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_max_proj_test_score_on_scaled_average_all', range(2, 3))
    dataset = get_dataset('lasot')
    return trackers, dataset

def memory_learning_max_proj_test_score_average_all_inside_bbox_certainty_lasot_0():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_max_proj_test_score_on_average_all_inside_bbox', range(0, 1))
    dataset = get_dataset('lasot')
    return trackers, dataset

def memory_learning_max_proj_test_score_average_all_inside_bbox_certainty_lasot_1():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_max_proj_test_score_on_average_all_inside_bbox', range(1, 2))
    dataset = get_dataset('lasot')
    return trackers, dataset

def memory_learning_max_proj_test_score_average_all_inside_bbox_certainty_lasot_2():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_max_proj_test_score_on_average_all_inside_bbox', range(2, 3))
    dataset = get_dataset('lasot')
    return trackers, dataset

def memory_learning_max_proj_test_score_average_all_inside_bbox_certainty_lasot_3():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_max_proj_test_score_on_average_all_inside_bbox', range(3, 4))
    dataset = get_dataset('lasot')
    return trackers, dataset

def memory_learning_max_proj_test_score_average_all_inside_bbox_certainty_lasot_4():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_max_proj_test_score_on_average_all_inside_bbox', range(4, 5))
    dataset = get_dataset('lasot')
    return trackers, dataset

def memory_learning_max_proj_test_score_average_all_015_certainty_lasot_0():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_max_proj_test_score_on_average_all_015', range(0, 1))
    dataset = get_dataset('lasot')
    return trackers, dataset

def memory_learning_max_proj_test_score_average_all_015_certainty_lasot_1():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_max_proj_test_score_on_average_all_015', range(1, 2))
    dataset = get_dataset('lasot')
    return trackers, dataset

def memory_learning_max_proj_test_score_average_all_015_certainty_lasot_2():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_max_proj_test_score_on_average_all_015', range(2, 3))
    dataset = get_dataset('lasot')
    return trackers, dataset

def memory_learning_max_proj_test_score_average_all_015_certainty_lasot_3():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_max_proj_test_score_on_average_all_015', range(3, 4))
    dataset = get_dataset('lasot')
    return trackers, dataset

def memory_learning_max_proj_test_score_average_all_015_certainty_lasot_4():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_max_proj_test_score_on_average_all_015', range(4, 5))
    dataset = get_dataset('lasot')
    return trackers, dataset

def memory_learning_max_proj_test_score_average_all_015_certainty_lasot_5():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_max_proj_test_score_on_average_all_015', range(5, 6))
    dataset = get_dataset('lasot')
    return trackers, dataset

def memory_learning_max_proj_test_score_average_all_015_certainty_lasot_6():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_max_proj_test_score_on_average_all_015', range(6, 7))
    dataset = get_dataset('lasot')
    return trackers, dataset

def memory_learning_max_proj_test_score_average_all_015_certainty_lasot_7():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_max_proj_test_score_on_average_all_015', range(7, 8))
    dataset = get_dataset('lasot')
    return trackers, dataset

def memory_learning_max_proj_test_score_average_all_025_certainty_lasot_0():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_max_proj_test_score_on_average_all_025', range(0, 1))
    dataset = get_dataset('lasot')
    return trackers, dataset

def memory_learning_max_proj_test_score_average_all_025_certainty_lasot_1():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_max_proj_test_score_on_average_all_025', range(1, 2))
    dataset = get_dataset('lasot')
    return trackers, dataset


def memory_learning_attention_only_lasot_critical_0():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_attention_only', range(0, 1))
    dataset = get_dataset('lasot_critical')
    return trackers, dataset

def memory_learning_attention_only_lasot_critical_1():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_attention_only', range(1, 2))
    dataset = get_dataset('lasot_critical')
    return trackers, dataset

def memory_learning_attention_only_lasot_critical_2():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_attention_only', range(2, 3))
    dataset = get_dataset('lasot_critical')
    return trackers, dataset

def memory_learning_attention_only_lasot_0():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_attention_only', range(0, 1))
    dataset = get_dataset('lasot')
    return trackers, dataset

def memory_learning_attention_only_lasot_1():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_attention_only', range(1, 2))
    dataset = get_dataset('lasot')
    return trackers, dataset



def memory_learning_attention_average_lasot_critical_0():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_attention_average', range(0, 1))
    dataset = get_dataset('lasot_critical')
    return trackers, dataset

def memory_learning_attention_average_lasot_critical_1():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_attention_average', range(1, 2))
    dataset = get_dataset('lasot_critical')
    return trackers, dataset

def memory_learning_attention_average_lasot_critical_2():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_attention_average', range(2, 3))
    dataset = get_dataset('lasot_critical')
    return trackers, dataset

def memory_learning_attention_average_lasot_0():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_attention_average', range(0, 1))
    dataset = get_dataset('lasot')
    return trackers, dataset

def memory_learning_attention_average_lasot_1():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_attention_average', range(1, 2))
    dataset = get_dataset('lasot')
    return trackers, dataset




def memory_learning_attention_average_std_scaling_lasot_critical_0():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_attention_average_std_scaling', range(0, 1))
    dataset = get_dataset('lasot_critical')
    return trackers, dataset

def memory_learning_attention_average_std_scaling_lasot_critical_1():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_attention_average_std_scaling', range(1, 2))
    dataset = get_dataset('lasot_critical')
    return trackers, dataset

def memory_learning_attention_average_std_scaling_lasot_critical_2():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_attention_average_std_scaling', range(2, 3))
    dataset = get_dataset('lasot_critical')
    return trackers, dataset

def memory_learning_attention_average_std_scaling_lasot_critical_3():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_attention_average_std_scaling', range(3, 4))
    dataset = get_dataset('lasot_critical')
    return trackers, dataset

def memory_learning_attention_average_std_scaling_lasot_critical_4():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_attention_average_std_scaling', range(4, 5))
    dataset = get_dataset('lasot_critical')
    return trackers, dataset

def memory_learning_attention_average_std_scaling_lasot_0():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_attention_average_std_scaling', range(0, 1))
    dataset = get_dataset('lasot')
    return trackers, dataset

def memory_learning_attention_average_std_scaling_lasot_1():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_attention_average_std_scaling', range(1, 2))
    dataset = get_dataset('lasot')
    return trackers, dataset

def memory_learning_attention_average_std_scaling_lasot_2():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_attention_average_std_scaling', range(2, 3))
    dataset = get_dataset('lasot')
    return trackers, dataset

def memory_learning_attention_average_std_scaling_lasot_3():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_attention_average_std_scaling', range(3, 4))
    dataset = get_dataset('lasot')
    return trackers, dataset

def memory_learning_attention_average_std_scaling_lasot_4():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_attention_average_std_scaling', range(4, 5))
    dataset = get_dataset('lasot')
    return trackers, dataset





def memory_learning_attention_std_scaling_lasot_critical_0():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_attention_std_scaling', range(0, 1))
    dataset = get_dataset('lasot_critical')
    return trackers, dataset

def memory_learning_attention_std_scaling_lasot_critical_1():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_attention_std_scaling', range(1, 2))
    dataset = get_dataset('lasot_critical')
    return trackers, dataset

def memory_learning_attention_std_scaling_lasot_critical_2():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_attention_std_scaling', range(2, 3))
    dataset = get_dataset('lasot_critical')
    return trackers, dataset

def memory_learning_attention_std_scaling_lasot_0():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_attention_std_scaling', range(0, 1))
    dataset = get_dataset('lasot')
    return trackers, dataset

def memory_learning_attention_std_scaling_lasot_1():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_attention_std_scaling', range(1, 2))
    dataset = get_dataset('lasot')
    return trackers, dataset





def memory_learning_attention_averate_std_scaling_rescale_lasot_critical_0():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_attention_average_std_scaling_rescale', range(0, 1))
    dataset = get_dataset('lasot_critical')
    return trackers, dataset

def memory_learning_attention_averate_std_scaling_rescale_lasot_critical_1():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_attention_average_std_scaling_rescale', range(1, 2))
    dataset = get_dataset('lasot_critical')
    return trackers, dataset

def memory_learning_attention_averate_std_scaling_rescale_lasot_critical_2():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_attention_average_std_scaling_rescale', range(2, 3))
    dataset = get_dataset('lasot_critical')
    return trackers, dataset

def memory_learning_attention_averate_std_scaling_rescale_lasot_critical_3():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_attention_average_std_scaling_rescale', range(3, 4))
    dataset = get_dataset('lasot_critical')
    return trackers, dataset

def memory_learning_attention_averate_std_scaling_rescale_lasot_critical_4():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_attention_average_std_scaling_rescale', range(4, 5))
    dataset = get_dataset('lasot_critical')
    return trackers, dataset



def memory_learning_attention_averate_std_scaling_rescale_lasot_0():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_attention_average_std_scaling_rescale', range(0, 1))
    dataset = get_dataset('lasot')
    return trackers, dataset

def memory_learning_attention_averate_std_scaling_rescale_lasot_1():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_attention_average_std_scaling_rescale', range(1, 2))
    dataset = get_dataset('lasot')
    return trackers, dataset

def memory_learning_attention_averate_std_scaling_rescale_lasot_2():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_attention_average_std_scaling_rescale', range(2, 3))
    dataset = get_dataset('lasot')
    return trackers, dataset

def memory_learning_attention_averate_std_scaling_rescale_lasot_3():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_attention_average_std_scaling_rescale', range(3, 4))
    dataset = get_dataset('lasot')
    return trackers, dataset

def memory_learning_attention_averate_std_scaling_rescale_lasot_4():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_attention_average_std_scaling_rescale', range(4, 5))
    dataset = get_dataset('lasot')
    return trackers, dataset







def memory_learning_attention_learn_fusion_scaling_lasot_critical_0():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_attention_learn_fusion_scaling', range(0, 1))
    dataset = get_dataset('lasot_critical')
    return trackers, dataset

def memory_learning_attention_learn_fusion_scaling_lasot_critical_1():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_attention_learn_fusion_scaling', range(1, 2))
    dataset = get_dataset('lasot_critical')
    return trackers, dataset

def memory_learning_attention_learn_fusion_scaling_lasot_critical_2():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_attention_learn_fusion_scaling', range(2, 3))
    dataset = get_dataset('lasot_critical')
    return trackers, dataset

def memory_learning_attention_learn_fusion_scaling_lasot_critical_3():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_attention_learn_fusion_scaling', range(3, 4))
    dataset = get_dataset('lasot_critical')
    return trackers, dataset

def memory_learning_attention_learn_fusion_scaling_lasot_critical_4():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_attention_learn_fusion_scaling', range(4, 5))
    dataset = get_dataset('lasot_critical')
    return trackers, dataset


def memory_learning_attention_learn_fusion_scaling_lasot_0():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_attention_learn_fusion_scaling', range(0, 1))
    dataset = get_dataset('lasot')
    return trackers, dataset

def memory_learning_attention_learn_fusion_scaling_lasot_1():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_attention_learn_fusion_scaling', range(1, 2))
    dataset = get_dataset('lasot')
    return trackers, dataset

def memory_learning_attention_learn_fusion_scaling_lasot_2():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_attention_learn_fusion_scaling', range(2, 3))
    dataset = get_dataset('lasot')
    return trackers, dataset

def memory_learning_attention_learn_fusion_scaling_lasot_3():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_attention_learn_fusion_scaling', range(3, 4))
    dataset = get_dataset('lasot')
    return trackers, dataset

def memory_learning_attention_learn_fusion_scaling_lasot_4():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_attention_learn_fusion_scaling', range(4, 5))
    dataset = get_dataset('lasot')
    return trackers, dataset


def memory_learning_attention_average_std_scaling_frozen_bn_lasot_small_0():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_attention_average_std_scaling_frozen_bn', range(0, 1))
    dataset = get_dataset('lasot_small')
    return trackers, dataset

def memory_learning_attention_average_std_scaling_frozen_bn_lasot_small_1():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_attention_average_std_scaling_frozen_bn', range(1, 2))
    dataset = get_dataset('lasot_small')
    return trackers, dataset

def memory_learning_attention_average_std_scaling_frozen_bn_lasot_small_2():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_attention_average_std_scaling_frozen_bn', range(2, 3))
    dataset = get_dataset('lasot_small')
    return trackers, dataset

def memory_learning_attention_average_std_scaling_frozen_bn_lasot_small_3():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_attention_average_std_scaling_frozen_bn', range(3, 4))
    dataset = get_dataset('lasot_small')
    return trackers, dataset

def memory_learning_attention_average_std_scaling_frozen_bn_lasot_small_4():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_attention_average_std_scaling_frozen_bn', range(4, 5))
    dataset = get_dataset('lasot_small')
    return trackers, dataset


def memory_learning_attention_average_std_scaling_frozen_bn_lasot_0():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_attention_average_std_scaling_frozen_bn', range(0, 1))
    dataset = get_dataset('lasot')
    return trackers, dataset

def memory_learning_attention_average_std_scaling_frozen_bn_lasot_1():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_attention_average_std_scaling_frozen_bn', range(1, 2))
    dataset = get_dataset('lasot')
    return trackers, dataset

def memory_learning_attention_average_std_scaling_frozen_bn_lasot_2():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_attention_average_std_scaling_frozen_bn', range(2, 3))
    dataset = get_dataset('lasot')
    return trackers, dataset

def memory_learning_attention_average_std_scaling_frozen_bn_lasot_3():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_attention_average_std_scaling_frozen_bn', range(3, 4))
    dataset = get_dataset('lasot')
    return trackers, dataset

def memory_learning_attention_average_std_scaling_frozen_bn_lasot_4():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_attention_average_std_scaling_frozen_bn', range(4, 5))
    dataset = get_dataset('lasot')
    return trackers, dataset


def memory_learning_attention_average_std_scaling_frozen_bn_lasot_critical_all():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_attention_average_std_scaling_frozen_bn', range(0, 5))
    dataset = get_dataset('lasot_critical')
    return trackers, dataset


def memory_learning_attention_average_std_scaling_rescale_frozen_bn_lasot_small_0():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_attention_average_std_scaling_rescale_frozen_bn', range(0, 1))
    dataset = get_dataset('lasot_small')
    return trackers, dataset

def memory_learning_attention_average_std_scaling_rescale_frozen_bn_lasot_small_1():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_attention_average_std_scaling_rescale_frozen_bn', range(1, 2))
    dataset = get_dataset('lasot_small')
    return trackers, dataset

def memory_learning_attention_average_std_scaling_rescale_frozen_bn_lasot_small_2():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_attention_average_std_scaling_rescale_frozen_bn', range(2, 3))
    dataset = get_dataset('lasot_small')
    return trackers, dataset

def memory_learning_attention_average_std_scaling_rescale_frozen_bn_lasot_small_3():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_attention_average_std_scaling_rescale_frozen_bn', range(3, 4))
    dataset = get_dataset('lasot_small')
    return trackers, dataset

def memory_learning_attention_average_std_scaling_rescale_frozen_bn_lasot_small_4():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_attention_average_std_scaling_rescale_frozen_bn', range(4, 5))
    dataset = get_dataset('lasot_small')
    return trackers, dataset


def memory_learning_attention_average_std_scaling_rescale_frozen_bn_lasot_0():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_attention_average_std_scaling_rescale_frozen_bn', range(0, 1))
    dataset = get_dataset('lasot')
    return trackers, dataset

def memory_learning_attention_average_std_scaling_rescale_frozen_bn_lasot_1():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_attention_average_std_scaling_rescale_frozen_bn', range(1, 2))
    dataset = get_dataset('lasot')
    return trackers, dataset

def memory_learning_attention_average_std_scaling_rescale_frozen_bn_lasot_2():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_attention_average_std_scaling_rescale_frozen_bn', range(2, 3))
    dataset = get_dataset('lasot')
    return trackers, dataset

def memory_learning_attention_average_std_scaling_rescale_frozen_bn_lasot_3():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_attention_average_std_scaling_rescale_frozen_bn', range(3, 4))
    dataset = get_dataset('lasot')
    return trackers, dataset

def memory_learning_attention_average_std_scaling_rescale_frozen_bn_lasot_4():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_attention_average_std_scaling_rescale_frozen_bn', range(4, 5))
    dataset = get_dataset('lasot')
    return trackers, dataset

def memory_learning_attention_average_std_scaling_rescale_frozen_bn_lasot_5():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_attention_average_std_scaling_rescale_frozen_bn', range(5, 6))
    dataset = get_dataset('lasot')
    return trackers, dataset

def memory_learning_attention_average_std_scaling_rescale_frozen_bn_lasot_6():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_attention_average_std_scaling_rescale_frozen_bn', range(6, 7))
    dataset = get_dataset('lasot')
    return trackers, dataset

def memory_learning_attention_average_std_scaling_rescale_frozen_bn_lasot_7():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_attention_average_std_scaling_rescale_frozen_bn', range(7, 8))
    dataset = get_dataset('lasot')
    return trackers, dataset

def memory_learning_attention_average_std_scaling_rescale_frozen_bn_lasot_8():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_attention_average_std_scaling_rescale_frozen_bn', range(8, 9))
    dataset = get_dataset('lasot')
    return trackers, dataset


def memory_learning_attention_average_std_scaling_rescale_frozen_bn_iou_lasot_0():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_attention_average_std_scaling_rescale_frozen_bn_iou_certainty', range(0, 1))
    dataset = get_dataset('lasot')
    return trackers, dataset

def memory_learning_attention_average_std_scaling_rescale_frozen_bn_iou_lasot_1():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_attention_average_std_scaling_rescale_frozen_bn_iou_certainty', range(1, 2))
    dataset = get_dataset('lasot')
    return trackers, dataset

def memory_learning_attention_average_std_scaling_rescale_frozen_bn_iou_lasot_2():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_attention_average_std_scaling_rescale_frozen_bn_iou_certainty', range(2, 3))
    dataset = get_dataset('lasot')
    return trackers, dataset




def memory_learning_max_proj_test_score_on_average_all_disable_certainty_scaling_lasot_0():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_max_proj_test_score_on_average_all_disable_certainty_scaling', range(0, 1))
    dataset = get_dataset('lasot')
    return trackers, dataset

def memory_learning_max_proj_test_score_on_average_all_disable_certainty_scaling_lasot_1():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_max_proj_test_score_on_average_all_disable_certainty_scaling', range(1, 2))
    dataset = get_dataset('lasot')
    return trackers, dataset

def memory_learning_max_proj_test_score_on_average_all_disable_certainty_scaling_lasot_2():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_max_proj_test_score_on_average_all_disable_certainty_scaling', range(2, 3))
    dataset = get_dataset('lasot')
    return trackers, dataset



def memory_learning_max_proj_test_score_on_average_all_no_ths_lasot_0():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_max_proj_test_score_on_average_all_no_ths', range(0, 1))
    dataset = get_dataset('lasot')
    return trackers, dataset

def memory_learning_max_proj_test_score_on_average_all_no_ths_lasot_1():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_max_proj_test_score_on_average_all_no_ths', range(1, 2))
    dataset = get_dataset('lasot')
    return trackers, dataset

def memory_learning_max_proj_test_score_on_average_all_no_ths_lasot_2():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_max_proj_test_score_on_average_all_no_ths', range(2, 3))
    dataset = get_dataset('lasot')
    return trackers, dataset


def baseline_super_dimp_hinge_lasot_extension_0():
    trackers = trackerlist('dimp_weight_learning', 'super_dimp_hinge', range(0, 1))
    dataset = get_dataset('lasot_extension_subset')
    return trackers, dataset

def baseline_super_dimp_hinge_lasot_extension_1():
    trackers = trackerlist('dimp_weight_learning', 'super_dimp_hinge', range(1, 2))
    dataset = get_dataset('lasot_extension_subset')
    return trackers, dataset

def baseline_super_dimp_hinge_lasot_extension_2():
    trackers = trackerlist('dimp_weight_learning', 'super_dimp_hinge', range(2, 3))
    dataset = get_dataset('lasot_extension_subset')
    return trackers, dataset

def baseline_super_dimp_hinge_lasot_extension_3():
    trackers = trackerlist('dimp_weight_learning', 'super_dimp_hinge', range(3, 4))
    dataset = get_dataset('lasot_extension_subset')
    return trackers, dataset

def baseline_super_dimp_hinge_lasot_extension_4():
    trackers = trackerlist('dimp_weight_learning', 'super_dimp_hinge', range(3, 4))
    dataset = get_dataset('lasot_extension_subset')
    return trackers, dataset

def baseline_super_dimp_lasot_extension_0():
    trackers = trackerlist('dimp', 'super_dimp', range(0, 1))
    dataset = get_dataset('lasot_extension_subset')
    return trackers, dataset

def baseline_super_dimp_lasot_extension_1():
    trackers = trackerlist('dimp', 'super_dimp', range(1, 2))
    dataset = get_dataset('lasot_extension_subset')
    return trackers, dataset

def baseline_super_dimp_lasot_extension_2():
    trackers = trackerlist('dimp', 'super_dimp', range(2, 3))
    dataset = get_dataset('lasot_extension_subset')
    return trackers, dataset


def memory_learning_baseline_iou_certainty_lasot_extension_0():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_baseline_iou_certainty', range(0, 1))
    dataset = get_dataset('lasot_extension_subset')
    return trackers, dataset

def memory_learning_baseline_iou_certainty_lasot_extension_1():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_baseline_iou_certainty', range(1, 2))
    dataset = get_dataset('lasot_extension_subset')
    return trackers, dataset

def memory_learning_baseline_iou_certainty_lasot_extension_2():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_baseline_iou_certainty', range(2, 3))
    dataset = get_dataset('lasot_extension_subset')
    return trackers, dataset

def memory_learning_baseline_iou_certainty_lasot_extension_3():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_baseline_iou_certainty', range(3, 4))
    dataset = get_dataset('lasot_extension_subset')
    return trackers, dataset

def memory_learning_baseline_iou_certainty_lasot_extension_4():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_baseline_iou_certainty', range(4, 5))
    dataset = get_dataset('lasot_extension_subset')
    return trackers, dataset


def memory_learning_max_proj_test_score_average_all_015_certainty_lasot_extension_0():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_max_proj_test_score_on_average_all_015', range(0, 1))
    dataset = get_dataset('lasot_extension_subset')
    return trackers, dataset

def memory_learning_max_proj_test_score_average_all_015_certainty_lasot_extension_1():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_max_proj_test_score_on_average_all_015', range(1, 2))
    dataset = get_dataset('lasot_extension_subset')
    return trackers, dataset

def memory_learning_max_proj_test_score_average_all_015_certainty_lasot_extension_2():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_max_proj_test_score_on_average_all_015', range(2, 3))
    dataset = get_dataset('lasot_extension_subset')
    return trackers, dataset

def memory_learning_max_proj_test_score_average_all_015_certainty_lasot_extension_3():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_max_proj_test_score_on_average_all_015', range(3, 4))
    dataset = get_dataset('lasot_extension_subset')
    return trackers, dataset

def memory_learning_max_proj_test_score_average_all_015_certainty_lasot_extension_4():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_max_proj_test_score_on_average_all_015', range(4, 5))
    dataset = get_dataset('lasot_extension_subset')
    return trackers, dataset


def memory_learning_attention_average_std_scaling_rescale_frozen_bn_lasot_extension_0():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_attention_average_std_scaling_rescale_frozen_bn', range(0, 1))
    dataset = get_dataset('lasot_extension_subset')
    return trackers, dataset

def memory_learning_attention_average_std_scaling_rescale_frozen_bn_lasot_extension_1():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_attention_average_std_scaling_rescale_frozen_bn', range(1, 2))
    dataset = get_dataset('lasot_extension_subset')
    return trackers, dataset

def memory_learning_attention_average_std_scaling_rescale_frozen_bn_lasot_extension_2():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_attention_average_std_scaling_rescale_frozen_bn', range(2, 3))
    dataset = get_dataset('lasot_extension_subset')
    return trackers, dataset

def memory_learning_attention_average_std_scaling_rescale_frozen_bn_lasot_extension_3():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_attention_average_std_scaling_rescale_frozen_bn', range(3, 4))
    dataset = get_dataset('lasot_extension_subset')
    return trackers, dataset

def memory_learning_attention_average_std_scaling_rescale_frozen_bn_lasot_extension_4():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_attention_average_std_scaling_rescale_frozen_bn', range(4, 5))
    dataset = get_dataset('lasot_extension_subset')
    return trackers, dataset


def memory_learning_attention_average_std_scaling_rescale_frozen_bn_iou_lasot_extension_0():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_attention_average_std_scaling_rescale_frozen_bn_iou_certainty', range(0, 1))
    dataset = get_dataset('lasot_extension_subset')
    return trackers, dataset

def memory_learning_attention_average_std_scaling_rescale_frozen_bn_iou_lasot_extension_1():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_attention_average_std_scaling_rescale_frozen_bn_iou_certainty', range(1, 2))
    dataset = get_dataset('lasot_extension_subset')
    return trackers, dataset

def memory_learning_attention_average_std_scaling_rescale_frozen_bn_iou_lasot_extension_2():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_attention_average_std_scaling_rescale_frozen_bn_iou_certainty', range(2, 3))
    dataset = get_dataset('lasot_extension_subset')
    return trackers, dataset



def memory_learning_baseline_iou_certainty_disable_certainty_scaling_lasot_0():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_baseline_iou_certainty_disable_certainty_scaling', range(0, 1))
    dataset = get_dataset('lasot')
    return trackers, dataset

def memory_learning_baseline_iou_certainty_disable_certainty_scaling_lasot_1():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_baseline_iou_certainty_disable_certainty_scaling', range(1, 2))
    dataset = get_dataset('lasot')
    return trackers, dataset

def memory_learning_baseline_iou_certainty_disable_certainty_scaling_lasot_2():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_baseline_iou_certainty_disable_certainty_scaling', range(2, 3))
    dataset = get_dataset('lasot')
    return trackers, dataset



def memory_learning_baseline_iou_certainty_no_ths_lasot_0():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_baseline_iou_certainty_no_ths', range(0, 1))
    dataset = get_dataset('lasot')
    return trackers, dataset

def memory_learning_baseline_iou_certainty_no_ths_lasot_1():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_baseline_iou_certainty_no_ths', range(1, 2))
    dataset = get_dataset('lasot')
    return trackers, dataset

def memory_learning_baseline_iou_certainty_no_ths_lasot_2():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_baseline_iou_certainty_no_ths', range(2, 3))
    dataset = get_dataset('lasot')
    return trackers, dataset

def memory_learning_baseline_iou_certainty_no_ths_lasot_3():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_baseline_iou_certainty_no_ths', range(3, 4))
    dataset = get_dataset('lasot')
    return trackers, dataset

def memory_learning_baseline_iou_certainty_no_ths_lasot_4():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_baseline_iou_certainty_no_ths', range(4, 5))
    dataset = get_dataset('lasot')
    return trackers, dataset


def memory_learning_baseline_iou_certainty_no_ths_lasot_extension_0():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_baseline_iou_certainty_no_ths', range(0, 1))
    dataset = get_dataset('lasot_extension_subset')
    return trackers, dataset

def memory_learning_baseline_iou_certainty_no_ths_lasot_extension_1():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_baseline_iou_certainty_no_ths', range(1, 2))
    dataset = get_dataset('lasot_extension_subset')
    return trackers, dataset

def memory_learning_baseline_iou_certainty_no_ths_lasot_extension_2():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_baseline_iou_certainty_no_ths', range(2, 3))
    dataset = get_dataset('lasot_extension_subset')
    return trackers, dataset



def dump_results_for_training_lasot_0_to_200():
    trackers = trackerlist('dimp_weight_learning', 'super_dimp_hinge_save_tracking_data', range(0, 1))
    dataset = get_dataset('lasot_train')
    return trackers, dataset

def dump_results_for_training_lasot_200_to_400():
    trackers = trackerlist('dimp_weight_learning', 'super_dimp_hinge_save_tracking_data', range(0, 1))
    dataset = get_dataset('lasot_train')
    return trackers, dataset

def dump_results_for_training_lasot_400_to_600():
    trackers = trackerlist('dimp_weight_learning', 'super_dimp_hinge_save_tracking_data', range(0, 1))
    dataset = get_dataset('lasot_train')
    return trackers, dataset

def dump_results_for_training_lasot_600_to_800():
    trackers = trackerlist('dimp_weight_learning', 'super_dimp_hinge_save_tracking_data', range(0, 1))
    dataset = get_dataset('lasot_train')
    return trackers, dataset

def dump_results_for_training_lasot_800_to_1000():
    trackers = trackerlist('dimp_weight_learning', 'super_dimp_hinge_save_tracking_data', range(0, 1))
    dataset = get_dataset('lasot_train')
    return trackers, dataset

def dump_results_for_training_lasot_1000_to_1120():
    trackers = trackerlist('dimp_weight_learning', 'super_dimp_hinge_save_tracking_data', range(0, 1))
    dataset = get_dataset('lasot_train')
    return trackers, dataset

def dump_results_for_training_lasot():
    trackers = trackerlist('dimp_weight_learning', 'super_dimp_hinge_save_tracking_data', range(0, 1))
    dataset = get_dataset('lasot')
    return trackers, dataset


def memory_learning_learned_certainty_prediction_v1_lasot_0():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_learned_certainty_prediction_v1', range(0, 1))
    dataset = get_dataset('lasot')
    return trackers, dataset

def memory_learning_learned_certainty_prediction_v1_lasot_1():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_learned_certainty_prediction_v1', range(1, 2))
    dataset = get_dataset('lasot')
    return trackers, dataset

def memory_learning_learned_certainty_prediction_v1_lasot_2():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_learned_certainty_prediction_v1', range(2, 3))
    dataset = get_dataset('lasot')
    return trackers, dataset


def memory_learning_learned_certainty_prediction_v1_lasot_extension_0():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_learned_certainty_prediction_v1', range(0, 1))
    dataset = get_dataset('lasot_extension_subset')
    return trackers, dataset

def memory_learning_learned_certainty_prediction_v1_lasot_extension_1():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_learned_certainty_prediction_v1', range(1, 2))
    dataset = get_dataset('lasot_extension_subset')
    return trackers, dataset

def memory_learning_learned_certainty_prediction_v1_lasot_extension_2():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_learned_certainty_prediction_v1', range(2, 3))
    dataset = get_dataset('lasot_extension_subset')
    return trackers, dataset


def memory_learning_learned_certainty_prediction_v1_ths_080_lasot_0():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_learned_certainty_prediction_v1_ths_080', range(0, 1))
    dataset = get_dataset('lasot')
    return trackers, dataset

def memory_learning_learned_certainty_prediction_v1_ths_080_lasot_1():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_learned_certainty_prediction_v1_ths_080', range(1, 2))
    dataset = get_dataset('lasot')
    return trackers, dataset

def memory_learning_learned_certainty_prediction_v1_ths_080_lasot_2():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_learned_certainty_prediction_v1_ths_080', range(2, 3))
    dataset = get_dataset('lasot')
    return trackers, dataset

def memory_learning_learned_certainty_prediction_v1_ths_070_lasot_0():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_learned_certainty_prediction_v1_ths_070', range(0, 1))
    dataset = get_dataset('lasot')
    return trackers, dataset

def memory_learning_learned_certainty_prediction_v1_ths_070_lasot_1():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_learned_certainty_prediction_v1_ths_070', range(1, 2))
    dataset = get_dataset('lasot')
    return trackers, dataset

def memory_learning_learned_certainty_prediction_v1_ths_070_lasot_2():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_learned_certainty_prediction_v1_ths_070', range(2, 3))
    dataset = get_dataset('lasot')
    return trackers, dataset


def memory_learning_select_gth_peak_lasot_0():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_select_gth_peak', range(0, 1))
    dataset = get_dataset('lasot')
    return trackers, dataset

def memory_learning_select_gth_peak_lasot_1():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_select_gth_peak', range(1, 2))
    dataset = get_dataset('lasot')
    return trackers, dataset

def memory_learning_select_gth_peak_lasot_2():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_select_gth_peak', range(2, 3))
    dataset = get_dataset('lasot')
    return trackers, dataset

def memory_learning_select_gth_peak_lasot_3():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_select_gth_peak', range(3, 4))
    dataset = get_dataset('lasot')
    return trackers, dataset


def memory_learning_select_gth_peak_lasot_extension_0():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_select_gth_peak', range(0, 1))
    dataset = get_dataset('lasot_extension_subset')
    return trackers, dataset

def memory_learning_select_gth_peak_lasot_extension_1():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_select_gth_peak', range(1, 2))
    dataset = get_dataset('lasot_extension_subset')
    return trackers, dataset

def memory_learning_select_gth_peak_lasot_extension_2():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_select_gth_peak', range(2, 3))
    dataset = get_dataset('lasot_extension_subset')
    return trackers, dataset

def memory_learning_select_gth_peak_lasot_extension_3():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_select_gth_peak', range(3, 4))
    dataset = get_dataset('lasot_extension_subset')
    return trackers, dataset


def memory_learning_max_proj_test_score_on_average_all_015_select_gth_peak_lasot_0():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_max_proj_test_score_on_average_all_015_select_gth_peak', range(0, 1))
    dataset = get_dataset('lasot')
    return trackers, dataset

def memory_learning_max_proj_test_score_on_average_all_015_select_gth_peak_lasot_1():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_max_proj_test_score_on_average_all_015_select_gth_peak', range(1, 2))
    dataset = get_dataset('lasot')
    return trackers, dataset

def memory_learning_max_proj_test_score_on_average_all_015_select_gth_peak_lasot_2():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_max_proj_test_score_on_average_all_015_select_gth_peak', range(2, 3))
    dataset = get_dataset('lasot')
    return trackers, dataset

def memory_learning_max_proj_test_score_on_average_all_015_select_gth_peak_lasot_3():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_max_proj_test_score_on_average_all_015_select_gth_peak', range(3, 4))
    dataset = get_dataset('lasot')
    return trackers, dataset


def memory_learning_baseline_iou_certainty_select_gth_peak_lasot_0():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_baseline_iou_certainty_select_gth_peak', range(0, 1))
    dataset = get_dataset('lasot')
    return trackers, dataset

def memory_learning_baseline_iou_certainty_select_gth_peak_lasot_1():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_baseline_iou_certainty_select_gth_peak', range(1, 2))
    dataset = get_dataset('lasot')
    return trackers, dataset

def memory_learning_baseline_iou_certainty_select_gth_peak_lasot_2():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_baseline_iou_certainty_select_gth_peak', range(2, 3))
    dataset = get_dataset('lasot')
    return trackers, dataset

def memory_learning_baseline_iou_certainty_select_gth_peak_lasot_3():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_baseline_iou_certainty_select_gth_peak', range(3, 4))
    dataset = get_dataset('lasot')
    return trackers, dataset


def memory_learning_max_proj_test_score_on_average_all_015_skip_uncertain_frames_lasot_0():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_max_proj_test_score_on_average_all_015_skip_uncertain_frames', range(0, 1))
    dataset = get_dataset('lasot')
    return trackers, dataset

def memory_learning_max_proj_test_score_on_average_all_015_skip_uncertain_frames_lasot_1():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_max_proj_test_score_on_average_all_015_skip_uncertain_frames', range(1, 2))
    dataset = get_dataset('lasot')
    return trackers, dataset

def memory_learning_max_proj_test_score_on_average_all_015_skip_uncertain_frames_lasot_2():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_max_proj_test_score_on_average_all_015_skip_uncertain_frames', range(2, 3))
    dataset = get_dataset('lasot')
    return trackers, dataset

def memory_learning_max_proj_test_score_on_average_all_015_skip_uncertain_frames_lasot_3():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_max_proj_test_score_on_average_all_015_skip_uncertain_frames', range(3, 4))
    dataset = get_dataset('lasot')
    return trackers, dataset


def memory_learning_peak_prediction_v3_lasot_0():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_peak_prediction_v3', range(0, 1))
    dataset = get_dataset('lasot')
    return trackers, dataset

def memory_learning_peak_prediction_v3_lasot_1():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_peak_prediction_v3', range(1, 2))
    dataset = get_dataset('lasot')
    return trackers, dataset

def memory_learning_peak_prediction_v3_lasot_2():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_peak_prediction_v3', range(2, 3))
    dataset = get_dataset('lasot')
    return trackers, dataset

def memory_learning_peak_prediction_v3_lasot_3():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_peak_prediction_v3', range(3, 4))
    dataset = get_dataset('lasot')
    return trackers, dataset


def memory_learning_max_proj_test_score_on_average_all_015_peak_prediction_v3_lasot_0():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_max_proj_test_score_on_average_all_015_peak_prediction_v3_simple', range(0, 1))
    dataset = get_dataset('lasot')
    return trackers, dataset

def memory_learning_max_proj_test_score_on_average_all_015_peak_prediction_v3_lasot_1():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_max_proj_test_score_on_average_all_015_peak_prediction_v3_simple', range(1, 2))
    dataset = get_dataset('lasot')
    return trackers, dataset

def memory_learning_max_proj_test_score_on_average_all_015_peak_prediction_v3_lasot_2():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_max_proj_test_score_on_average_all_015_peak_prediction_v3_simple', range(2, 3))
    dataset = get_dataset('lasot')
    return trackers, dataset


def memory_learning_max_proj_test_score_on_average_all_015_occlusion_rescale_lasot_0():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_max_proj_test_score_on_average_all_015_occlusion_rescale', range(0, 1))
    dataset = get_dataset('lasot')
    return trackers, dataset

def memory_learning_max_proj_test_score_on_average_all_015_occlusion_rescale_lasot_1():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_max_proj_test_score_on_average_all_015_occlusion_rescale', range(1, 2))
    dataset = get_dataset('lasot')
    return trackers, dataset

def memory_learning_max_proj_test_score_on_average_all_015_occlusion_rescale_lasot_2():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_max_proj_test_score_on_average_all_015_occlusion_rescale', range(2, 3))
    dataset = get_dataset('lasot')
    return trackers, dataset


def memory_learning_max_proj_test_score_on_average_all_015_occlusion_rescale_lasot_extension_0():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_max_proj_test_score_on_average_all_015_occlusion_rescale', range(0, 1))
    dataset = get_dataset('lasot_extension_subset')
    return trackers, dataset

def memory_learning_max_proj_test_score_on_average_all_015_occlusion_rescale_lasot_extension_1():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_max_proj_test_score_on_average_all_015_occlusion_rescale', range(1, 2))
    dataset = get_dataset('lasot_extension_subset')
    return trackers, dataset

def memory_learning_max_proj_test_score_on_average_all_015_occlusion_rescale_lasot_extension_2():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_max_proj_test_score_on_average_all_015_occlusion_rescale', range(2, 3))
    dataset = get_dataset('lasot_extension_subset')
    return trackers, dataset


def memory_learning_peak_prediction_v3_lasot_extension_0():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_peak_prediction_v3', range(0, 1))
    dataset = get_dataset('lasot_extension_subset')
    return trackers, dataset

def memory_learning_peak_prediction_v3_lasot_extension_1():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_peak_prediction_v3', range(1, 2))
    dataset = get_dataset('lasot_extension_subset')
    return trackers, dataset

def memory_learning_peak_prediction_v3_lasot_extension_2():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_peak_prediction_v3', range(2, 3))
    dataset = get_dataset('lasot_extension_subset')
    return trackers, dataset

def memory_learning_peak_prediction_v3_lasot_extension_3():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_peak_prediction_v3', range(3, 4))
    dataset = get_dataset('lasot_extension_subset')
    return trackers, dataset




def memory_learning_peak_matching_v1_mixed_with_lasot_0():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_peak_matching_v1_mixed_with_cor_logic', range(0, 1))
    dataset = get_dataset('lasot')
    return trackers, dataset

def memory_learning_peak_matching_v1_mixed_with_lasot_1():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_peak_matching_v1_mixed_with_cor_logic', range(1, 2))
    dataset = get_dataset('lasot')
    return trackers, dataset

def memory_learning_peak_matching_v1_mixed_wo_lasot_0():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_peak_matching_v1_mixed_wo_cor_logic', range(2, 3))
    dataset = get_dataset('lasot')
    return trackers, dataset

def memory_learning_peak_matching_v1_mixed_wo_lasot_1():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_peak_matching_v1_mixed_wo_cor_logic', range(3, 4))
    dataset = get_dataset('lasot')
    return trackers, dataset


def memory_learning_peak_matching_v1_mixed_wo_speed_lasot_0():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_peak_matching_v1_mixed_wo_cor_logic_speedup', range(0, 1))
    dataset = get_dataset('lasot')
    return trackers, dataset

def memory_learning_peak_matching_v1_mixed_wo_speed_lasot_1():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_peak_matching_v1_mixed_wo_cor_logic_speedup', range(1, 2))
    dataset = get_dataset('lasot')
    return trackers, dataset

def memory_learning_peak_matching_v1_mixed_wo_speed_lasot_2():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_peak_matching_v1_mixed_wo_cor_logic_speedup', range(2, 3))
    dataset = get_dataset('lasot')
    return trackers, dataset

def memory_learning_peak_matching_v1_mixed_wo_speed_lasot_3():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_peak_matching_v1_mixed_wo_cor_logic_speedup', range(3, 4))
    dataset = get_dataset('lasot')
    return trackers, dataset

def memory_learning_peak_matching_v1_mixed_wo_speed_lasot_4():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_peak_matching_v1_mixed_wo_cor_logic_speedup', range(4, 5))
    dataset = get_dataset('lasot')
    return trackers, dataset

def memory_learning_peak_matching_v1_mixed_wo_speed_rescale_lasot_0():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_peak_matching_v1_mixed_wo_cor_logic_speedup_rescale', range(0, 1))
    dataset = get_dataset('lasot')
    return trackers, dataset

def memory_learning_peak_matching_v1_mixed_wo_speed_rescale_lasot_1():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_peak_matching_v1_mixed_wo_cor_logic_speedup_rescale', range(1, 2))
    dataset = get_dataset('lasot')
    return trackers, dataset

def memory_learning_peak_matching_v1_mixed_wo_speed_rescale_lasot_2():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_peak_matching_v1_mixed_wo_cor_logic_speedup_rescale', range(2, 3))
    dataset = get_dataset('lasot')
    return trackers, dataset

def memory_learning_peak_matching_v1_mixed_wo_speed_rescale_lasot_3():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_peak_matching_v1_mixed_wo_cor_logic_speedup_rescale', range(3, 4))
    dataset = get_dataset('lasot')
    return trackers, dataset

def memory_learning_peak_matching_v1_mixed_wo_speed_rescale_lasot_4():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_peak_matching_v1_mixed_wo_cor_logic_speedup_rescale', range(4, 5))
    dataset = get_dataset('lasot')
    return trackers, dataset

def memory_learning_max_score_no_ths_lasot_0():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_max_score_no_ths', range(0, 1))
    dataset = get_dataset('lasot')
    return trackers, dataset

def memory_learning_max_score_no_ths_lasot_1():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_max_score_no_ths', range(1, 2))
    dataset = get_dataset('lasot')
    return trackers, dataset

def memory_learning_max_score_no_ths_lasot_2():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_max_score_no_ths', range(2, 3))
    dataset = get_dataset('lasot')
    return trackers, dataset

def memory_learning_max_score_no_ths_lasot_3():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_max_score_no_ths', range(3, 4))
    dataset = get_dataset('lasot')
    return trackers, dataset

def memory_learning_max_score_no_ths_lasot_4():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_max_score_no_ths', range(4, 5))
    dataset = get_dataset('lasot')
    return trackers, dataset


def kys_super_dimp_lasot_0():
    trackers = trackerlist('kys', 'eccv2020_rel_super_dimp', range(0, 1))
    dataset = get_dataset('lasot')
    return trackers, dataset

def kys_super_dimp_lasot_1():
    trackers = trackerlist('kys', 'eccv2020_rel_super_dimp', range(1, 2))
    dataset = get_dataset('lasot')
    return trackers, dataset

def kys_super_dimp_lasot_2():
    trackers = trackerlist('kys', 'eccv2020_rel_super_dimp', range(2, 3))
    dataset = get_dataset('lasot')
    return trackers, dataset

def kys_super_dimp_lasot_3():
    trackers = trackerlist('kys', 'eccv2020_rel_super_dimp', range(3, 4))
    dataset = get_dataset('lasot')
    return trackers, dataset

def kys_super_dimp_lasot_4():
    trackers = trackerlist('kys', 'eccv2020_rel_super_dimp', range(4, 5))
    dataset = get_dataset('lasot')
    return trackers, dataset


def memory_learning_peak_matching_v1_mixed_wo_speed_rescale_trackingnet_0():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_max_score_no_ths_peak_matching_v1_mixed_wo_cor_logic_sa_scale_8_fsize_30_rescale_speedup_id0_weight_increase', range(0, 1))
    dataset = get_dataset('trackingnet')
    return trackers, dataset

def memory_learning_peak_matching_v1_mixed_wo_speed_rescale_got10kval_0():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_max_score_no_ths_peak_matching_v1_mixed_wo_cor_logic_sa_scale_8_fsize_30_rescale_speedup_id0_weight_increase', range(0, 1))
    dataset = get_dataset('got10k_val')
    return trackers, dataset

def memory_learning_peak_matching_v1_mixed_wo_speed_rescale_got10kval_1():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_max_score_no_ths_peak_matching_v1_mixed_wo_cor_logic_sa_scale_8_fsize_30_rescale_speedup_id0_weight_increase', range(1, 2))
    dataset = get_dataset('got10k_val')
    return trackers, dataset

def memory_learning_peak_matching_v1_mixed_wo_speed_rescale_got10kval_2():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_max_score_no_ths_peak_matching_v1_mixed_wo_cor_logic_sa_scale_8_fsize_30_rescale_speedup_id0_weight_increase', range(2, 3))
    dataset = get_dataset('got10k_val')
    return trackers, dataset

def memory_learning_peak_matching_v1_mixed_wo_speed_rescale_got10kval_3():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_max_score_no_ths_peak_matching_v1_mixed_wo_cor_logic_sa_scale_8_fsize_30_rescale_speedup_id0_weight_increase', range(3, 4))
    dataset = get_dataset('got10k_val')
    return trackers, dataset

def memory_learning_peak_matching_v1_mixed_wo_speed_rescale_got10kval_4():
    trackers = trackerlist('dimp_memory_learning', 'super_dimp_memory_learning_max_score_no_ths_peak_matching_v1_mixed_wo_cor_logic_sa_scale_8_fsize_30_rescale_speedup_id0_weight_increase', range(4, 5))
    dataset = get_dataset('got10k_val')
    return trackers, dataset


def dimp_super_dimp_got10kval_2():
    trackers = trackerlist('dimp', 'super_dimp', [2])
    dataset = get_dataset('got10k_val')
    return trackers, dataset

def dimp_super_dimp_got10kval_3():
    trackers = trackerlist('dimp', 'super_dimp', [3])
    dataset = get_dataset('got10k_val')
    return trackers, dataset

def dimp_super_dimp_got10kval_4():
    trackers = trackerlist('dimp', 'super_dimp', [4])
    dataset = get_dataset('got10k_val')
    return trackers, dataset