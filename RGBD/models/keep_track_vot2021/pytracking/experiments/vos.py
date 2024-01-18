from pytracking.evaluation import Tracker, get_dataset, trackerlist


def evaluate_ytvos():
    trackers = []
    trackers.extend(trackerlist('dimp_vos',
                                'dolf_test1_lr2_nfilt32_ytvosjj_3_3_sw_nobn_res50_final_conv_mem_lr01',
                                range(0, 1)))
    trackers.extend(trackerlist('dimp_vos',
                                'dolf_test1_lr2_nfilt32_ytvosjj_3_3_sw_nobn_res50_final_conv_mem_lr01_sk1_it1',
                                range(0, 1)))
    trackers.extend(trackerlist('dimp_vos',
                                'dolf_test1_lr2_nfilt32_ytvosjj_3_3_sw_mem_r3_det_fix_nobn_init_res50_final_conv_mem_lr01',
                                range(0, 1)))
    trackers.extend(trackerlist('dimp_vos',
                                'dolf_test1_lr2_nfilt32_ytvosjj_3_3_sw_mem_r3_det_fix_nobn_init_res50_final_conv_mem_lr01_sk1_it1',
                                range(0, 1)))
    trackers.extend(trackerlist('dimp_vos',
                                'dolf_test1_lr2_nfilt32_ytvosjj_3_3_sw_mem_r3_det_fix_nobn_init_res50_final_conv_mem_lr01_sk1_it3',
                                range(0, 1)))
    trackers.extend(trackerlist('dimp_vos',
                                'dolf_test1_lr2_nfilt32_ytvosjj_3_3_sw_mem_r3_det_fix_nobn_init_res50_final_conv_mem_lr01_sk1_it3_sa100',
                                range(0, 1)))
    trackers.extend(trackerlist('dimp_vos',
                                'dolf_test1_lr2_nfilt32_ytvosjj_1_5_sw_mem_r2_det_fix_nobn_init_res50_final_conv_mem_lr01_sk1_it3',
                                range(0, 1)))
    trackers.extend(trackerlist('dimp_vos',
                                'dolf_test1_lr2_nfilt32_ytvosjj_1_5_sw_mem_r2_det_fix_nobn_init_res50_final_conv_mem_lr01_sk1_it3_hires1',
                                range(0, 1)))
    trackers.extend(trackerlist('dimp_vos',
                                'dolf_test1_lr2_nfilt32_ytvosjj_1_5_sw_mem_r2_det_fix_nobn_init_res50_final_conv_mem_lr01_sk1_it3_rawmerge',
                                range(0, 1)))
    trackers.extend(
        trackerlist('dimp_vos', 'default_v1_3_3_res50_final_conv_full_im_flip_aug_stdjit_s1_i3_rawmerge', range(0, 1)))
    trackers.extend(
        trackerlist('dimp_vos', 'default_v1_3_3_res50_final_conv_full_im_flip_aug_lessjit_occ_s1_i3_rawmerge',
                    range(0, 1)))
    trackers.extend(
        trackerlist('dimp_vos', 'default_v1_3_3_res50_final_conv_full_im_flip_aug_lessjit_s1_i3_rawmerge', range(0, 1)))
    trackers.extend(
        trackerlist('dimp_vos', 'default_v1_3_3_res50_final_conv_sa100_mem_lr01_sk1_it3_rawmerge_hr', range(0, 1)))

    trackers.extend(trackerlist('dimp_vos',
                                'dolf_test1_lr2_nfilt32_ytvosjj_1_5_sw_mem_r2_det_fix_nobn_init_res50_final_conv_flip_aug_dv_s1_i3_rawmerge',
                                range(0, 1)))

    trackers.extend(trackerlist('dimp_vos',
                                'dolf_test1_lr2_nfilt32_ytvosjj_1_5_sw_mem_r2_det_fix_nobn_init_res50_final_conv_flip_aug_dv_s1_i3_rawmerge_ep100',
                                range(0, 1)))
    trackers.extend(trackerlist('dimp_vos',
                                'dolf_test1_lr2_nfilt32_ytvosjj_1_5_sw_mem_r2_det_fix_nobn_init_res50_final_conv_flip_aug_dv_s1_i3_rawmerge_ep70',
                                range(0, 1)))
    trackers.extend(trackerlist('dimp_vos',
                                'dolf_test1_lr2_nfilt32_ytvosjj_1_5_sw_mem_r2_det_fix_nobn_init_res50_final_conv_flip_aug_dv_s1_i3_rawmerge_ep90_sa100',
                                range(0, 1)))
    trackers.extend(trackerlist('dimp_vos',
                                'dolf_test1_lr2_nfilt32_ytvosjj_1_5_sw_mem_r2_det_fix_nobn_init_res50_final_conv_flip_aug_dv_s1_i3_rawmerge_ep90_sa100_hr',
                                range(0, 1)))
    trackers.extend(trackerlist('dimp_vos',
                                'default_v1_3_3_res50_final_conv_full_im_flip_aug_lessjit_occ_s1_i3_rawmerge_hr',
                                range(0, 1)))
    trackers.extend(trackerlist('dimp_vos',
                                'default_v1_3_3_res50_final_conv_sa35_mem_lr01_sk1_it3_rawmerge_fix',
                                range(0, 1)))
    trackers.extend(trackerlist('dimp_vos',
                                'dolf_test1_lr2_nfilt32_ytvosjj_1_5_sw_mem_r2_det_fix_nobn_res50_final_conv_flip_aug_dv_lovashz_s1_i3_rawmerge_ep90',
                                range(0, 1)))
    trackers.extend(trackerlist('dimp_vos',
                                'dolf_test1_lr2_nfilt32_ytvosjj_1_5_sw_mem_r2_det_fix_nobn_res50_final_conv_flip_aug_dv_focal_s1_i3_rawmerge_ep90',
                                range(0, 1)))
    trackers.extend(trackerlist('dimp_vos',
                                'default_v1_1_5_r2_init_res50_final_conv_full_im_flip_aug_lessjit_occ_s1_i3_rawmerge_sa100',
                                range(0, 1)))
    trackers.extend(trackerlist('dimp_vos',
                                'dolf_test1_lr2_nfilt32_ytvosjj_1_5_sw_mem_r2_det_fix_nobn_init_res50_final_conv_flip_aug_dv_s1_i3_rawmerge_ep100_sa1000',
                                range(0, 1)))
    trackers.extend(trackerlist('dimp_vos',
                                'dolf_test1_lr2_nfilt32_ytvosjj_1_4_sw_mem_r2_det_fix_nobn_res50_final_conv_flip_aug_dv_lovashz_bbft_lockbn_s1_i3_rawmerge_ep40',
                                range(0, 1)))
    trackers.extend(trackerlist('dimp_vos',
                                'dolf_test1_lr2_nfilt32_ytvosjj_1_4_sw_mem_r2_det_fix_nobn_res50_final_conv_flip_aug_dv_lovashz_bbft_lockbn_s1_i3_rawmerge_ep80',
                                range(0, 1)))
    trackers.extend(trackerlist('dimp_vos',
                                'dolf_test1_lr2_nfilt32_ytvosjj_1_5_sw_mem_r2_det_fix_nobn_res50_final_conv_flip_aug_dv_lovashz_fullim_s1_i3_rawmerge_ep90_sa1000',
                                range(0, 1)))
    trackers.extend(trackerlist('dimp_vos',
                                'dolf_test1_lr2_nfilt32_ytvosjj_1_4_sw_mem_r2_det_fix_nobn_res50_final_conv_flip_aug_dv_lovashz_bbft_s1_i3_rawmerge_ep40',
                                range(0, 1)))
    trackers.extend(trackerlist('dimp_vos',
                                'dolf_test1_lr2_nfilt32_ytvosjj_1_4_sw_mem_r2_det_fix_nobn_res50_final_conv_flip_aug_dv_lovashz_bbft_s1_i3_rawmerge_ep80',
                                range(0, 1)))
    trackers.extend(trackerlist('dimp_vos_multi',
                                'default_v2_1_4_r2_res50_dv_lovasz_mobj2_mgpu_fix_s1_i3',
                                range(0, 1)))
    trackers.extend(trackerlist('dimp_vos_multi',
                                'default_v2_1_3_r2_res50_dv_lovasz_mobj3_mgpu_fix_s1_i3',
                                range(0, 1)))
    # trackers.extend(trackerlist('dimp_vos',
    #                            'dolf_test1_lr2_nfilt32_ytvosjj_1_3_sw_mem_r2_det_fix_nobn_res50_final_conv_flip_aug_dv_lovashz_s1_i3',
    #                             range(0, 1)))
    trackers.extend(trackerlist('dimp_vos',
                                'dolf_test1_lr2_nfilt32_ytvosjj_1_5_sw_mem_r2_det_fix_nobn_res50_final_conv_flip_aug_dv_lovashz_sa1000_s1_i3_sa1000',
                                range(0, 1)))

    return trackers, 'yt2019_jjval'


def evaluate_dv():
    trackers = []
    trackers.extend(trackerlist('dimp_vos',
                                'dolf_test1_lr2_nfilt32_ytvosjj_1_5_sw_mem_r2_det_fix_nobn_init_res50_final_conv_flip_aug_dv_s1_i3_rawmerge',
                                range(0, 1)))
    trackers.extend(
        trackerlist('dimp_vos',
                    'dolf_test1_lr2_nfilt32_ytvosjj_1_5_sw_mem_r2_det_fix_nobn_init_res50_final_conv_flip_aug_dv_s1_i3_rawmerge_aug',
                    range(0, 1)))
    trackers.extend(trackerlist('dimp_vos',
                                'dolf_test1_lr2_nfilt32_ytvosjj_1_5_sw_mem_r2_det_fix_nobn_init_res50_final_conv_flip_aug_dv_s1_i3_rawmerge_ep100_sa100',
                                range(0, 1)))
    trackers.extend(trackerlist('dimp_vos',
                                'dolf_test1_lr2_nfilt32_ytvosjj_1_5_sw_mem_r2_det_fix_nobn_init_res50_final_conv_flip_aug_dv_s1_i3_rawmerge_ep100',
                                range(0, 1)))
    trackers.extend(trackerlist('dimp_vos',
                                'dolf_test1_lr2_nfilt32_ytvosjj_1_5_sw_mem_r2_det_fix_nobn_init_res50_final_conv_flip_aug_dv_s1_i1_rawmerge_ep100',
                                range(0, 1)))
    trackers.extend(trackerlist('dimp_vos',
                                'dolf_test1_lr2_nfilt32_ytvosjj_1_5_sw_mem_r2_det_fix_nobn_init_res50_final_conv_flip_aug_dv_s1_i3_ii10_rawmerge_ep100',
                                range(0, 1)))
    trackers.extend(trackerlist('dimp_vos',
                                'dolf_test1_lr2_nfilt32_ytvosjj_1_5_sw_mem_r2_det_fix_nobn_init_res50_final_conv_flip_aug_dv_s1_i3_ii40_rawmerge_ep100',
                                range(0, 1)))
    trackers.extend(trackerlist('dimp_vos',
                                'dolf_test1_lr2_nfilt32_ytvosjj_1_5_sw_mem_r2_det_fix_nobn_init_res50_final_conv_flip_aug_dv_s1_i3_iw1_rawmerge_ep100',
                                range(0, 1)))
    trackers.extend(trackerlist('dimp_vos',
                                'dolf_test1_lr2_nfilt32_ytvosjj_1_5_sw_mem_r2_det_fix_nobn_init_res50_final_conv_flip_aug_dv_s1_i3_lr005_rawmerge_ep100',
                                range(0, 1)))
    trackers.extend(trackerlist('dimp_vos',
                                'dolf_test1_lr2_nfilt32_ytvosjj_1_5_sw_mem_r2_det_fix_nobn_init_res50_final_conv_flip_aug_dv_s1_i3_lr02_rawmerge_ep100',
                                range(0, 1)))
    trackers.extend(trackerlist('dimp_vos',
                                'dolf_test1_lr2_nfilt32_ytvosjj_1_5_sw_mem_r2_det_fix_nobn_init_res50_final_conv_flip_aug_dv_s1_i5_ii40_m50_rawmerge_ep100',
                                range(0, 1)))
    trackers.extend(trackerlist('dimp_vos',
                                'dolf_test1_lr2_nfilt32_ytvosjj_1_5_sw_mem_r2_det_fix_nobn_init_res50_final_conv_flip_aug_dv_s1_i5_m50_rawmerge_ep100',
                                range(0, 1)))
    trackers.extend(trackerlist('dimp_vos',
                                'dolf_test1_lr2_nfilt32_ytvosjj_1_5_sw_mem_r2_det_fix_nobn_init_res50_final_conv_flip_aug_dv_s1_i5_rawmerge_ep100',
                                range(0, 1)))
    trackers.extend(trackerlist('dimp_vos',
                                'dolf_test1_lr2_nfilt32_ytvosjj_1_5_sw_mem_r2_det_fix_nobn_init_res50_final_conv_flip_aug_dv_s1_i3_lr001_rawmerge_ep100',
                                range(0, 1)))
    trackers.extend(trackerlist('dimp_vos',
                                'default_v1_1_5_r2_init_res50_final_conv_full_im_flip_aug_lessjit_occ_s1_i3_rawmerge_sa100',
                                range(0, 1)))
    trackers.extend(trackerlist('dimp_vos',
                                'dolf_test1_lr2_nfilt32_ytvosjj_1_4_sw_mem_r2_det_fix_nobn_res50_final_conv_flip_aug_dv_lovashz_bbft_lockbn_s1_i3_rawmerge_ep40',
                                range(0, 1)))
    trackers.extend(trackerlist('dimp_vos',
                                'dolf_test1_lr2_nfilt32_ytvosjj_1_4_sw_mem_r2_det_fix_nobn_res50_final_conv_flip_aug_dv_lovashz_bbft_lockbn_s1_i3_rawmerge_ep80',
                                range(0, 1)))
    trackers.extend(trackerlist('dimp_vos',
                                'dolf_test1_lr2_nfilt32_ytvosjj_1_4_sw_mem_r2_det_fix_nobn_res50_final_conv_flip_aug_dv_lovashz_bbft_s1_i3_rawmerge_ep40',
                                range(0, 1)))
    trackers.extend(trackerlist('dimp_vos',
                                'dolf_test1_lr2_nfilt32_ytvosjj_1_4_sw_mem_r2_det_fix_nobn_res50_final_conv_flip_aug_dv_lovashz_bbft_s1_i3_rawmerge_ep80',
                                range(0, 1)))
    trackers.extend(trackerlist('dimp_vos_multi',
                                'default_v2_1_4_r2_res50_dv_lovasz_mobj2_mgpu_fix_s1_i3',
                                range(0, 1)))
    trackers.extend(trackerlist('dimp_vos_multi',
                                'default_v2_1_3_r2_res50_dv_lovasz_mobj3_mgpu_fix_s1_i3',
                                range(0, 1)))

    return trackers, 'dv2017_val'
