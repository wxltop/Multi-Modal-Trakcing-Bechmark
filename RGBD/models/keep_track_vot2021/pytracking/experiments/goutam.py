from pytracking.evaluation import Tracker, get_dataset, trackerlist


def get_scale_change_sequences():
    return ['f7e0c9bb83', 'a3f51855c3', '90c7a87887', 'df59cfd91d', '5110dc72c0', 'a9c9c1517e', 'b274456ce1',
            'd6917db4be',
            '84044f37f3', 'ec4186ce12', '0d2fcc0dcd', '72a810a799', 'd5b6c6d94a', 'c98b6fe013', '53af427bb2',
            '1f4ec0563d',
            'ae93214fe6', 'c557b69fbf', 'b6e9ec577f', '7a5f46198d', 'c4571bedc8', '390c51b987', 'a263ce8a87',
            'fad633fbe1',
            '15617297cc', '4019231330', '2bbde474ef', '01e64dd36a', '11ce6f452e', '9533fc037c', 'c2a35c1cda',
            'd301ca58cc',
            'b24fe36b2a', 'ef45ce3035', 'f8b4ac12f1', '0f2ab8b1ff', '927647fe08', '2df005b843', '2f5b0c89b1',
            'c1c830a735',
            '41dab05200', '6cccc985e0', '7a8b5456ca', 'bfd8f6e6c9', 'b2ce7699e3', '4122aba5f9', '1b8680f8cd']


def test():
    trackers = trackerlist('dimp', 'dimp18', 0)

    dataset = get_dataset('got10k_test')

    return trackers, dataset


def aws1():
    trackers = []
    trackers.extend(trackerlist('dimp_motion',
                                'res50_rel_motion_trdt05_occ_fix_win_confentr_tnf05_offset_winbu_aws_up_sc25',
                                range(0, 5)))

    dataset = get_dataset('lasot')

    return trackers, dataset


def aws2():
    trackers = []
    trackers.extend(trackerlist('dimp_motion',
                                'res50_rel_motion_trdt05_occ_fix_win_confentr_tnf05_offset_winbu_aws_up_sc25_r120',
                                range(0, 5)))

    dataset = get_dataset('lasot')

    return trackers, dataset


def aws3():
    trackers = []
    trackers.extend(trackerlist('dimp_motion',
                                'res50_rel_motion_trdt05_occ_fix_win_confentr_tnf05_offset_winbu_aws_up_sc25_r300',
                                range(0, 3)))

    dataset = get_dataset('lasot')[
        7, 14, 25, 31, 32, 44, 47, 55, 57, 64, 68, 77, 92, 95, 97, 103, 137, 149, 150, 154, 159, 169, 176, 192,
        195, 202, 210, 211, 218, 219, 220, 232, 243, 245, 253, 255, 262, 268, 275, 277]

    return trackers, dataset


def aws4():
    trackers = []
    trackers.extend(trackerlist('dimp_motion', 'res50_rel_motion_trdt05_occ_fix_win_confentr_tnf05_offset_winbu_aws_up_sc25_c95', range(0, 3)))

    dataset = get_dataset('lasot')

    return trackers, dataset


def aws5():
    trackers = []
    trackers.extend(trackerlist('dimp_motion', 'res50_rel_motion_trdt05_occ_fix_win_confentr_tnf05_offset_winbu_aws_up_sc25_c95', range(0, 3)))

    dataset = get_dataset('lasot')[::-1]

    return trackers, dataset


def caramba0():
    trackers = []
    trackers.extend(trackerlist('kys', 'eccv2020_rel', range(0, 5)))

    dataset = get_dataset('nfs', 'otb')

    return trackers, dataset


def caramba1():
    trackers = []
    trackers.extend(trackerlist('kys', 'eccv2020_rel', range(0, 5)))

    dataset = get_dataset('nfs', 'otb')[::-1]

    return trackers, dataset


def caramba2():
    trackers = []
    trackers = []
    trackers.extend(trackerlist('dimp_vos_clean', 'eccv2020_base_clean_convweight_ep70_dvft_lr02_aug_dec_ep2', range(0, 1)))
    trackers.extend(trackerlist('dimp_vos_clean', 'eccv2020_base_clean_convweight_ep70_dvft_lr02_aug_dec_ep3', range(0, 1)))
    trackers.extend(trackerlist('dimp_vos_clean', 'eccv2020_base_clean_convweight_ep70_dvft_lr02_aug_dec_ep4', range(0, 1)))
    trackers.extend(trackerlist('dimp_vos_clean', 'eccv2020_base_clean_convweight_ep70_dvft_lr02_aug_dec_ep5', range(0, 1)))

    dataset = get_dataset('dv2017_val')

    return trackers, dataset


def caramba3():
    trackers = []
    trackers.extend(trackerlist('dimp_vos_clean', 'eccv2020_base_clean_retrain1_ft_ep100', range(0, 1)))

    dataset = get_dataset('dv2017_val')

    return trackers, dataset


def caramba4():
    trackers = []
    trackers.extend(trackerlist('dimp_vos',
                                'dolf_test1_lr2_nfilt32_ytvosjj_1_3_sw_mem_r2_det_fix_nobn_res50_final_conv_flip_aug_dv_lovashz_bugflip_mrcnnweights_nobn_nfilt16_bsz20_dv6_ft_ep100_var_it1_sk4',
                                range(0, 1)))

    dataset = get_dataset('yt2019_jjval_all')[::-1]

    return trackers, dataset


def caramba5():
    trackers = []
    trackers.extend(trackerlist('dimp_vos',
                                'dolf_test1_lr2_nfilt32_ytvosjj_1_3_sw_mem_r2_det_fix_nobn_res50_final_conv_flip_aug_dv_lovashz_bugflip_mrcnnweights_ftvid_l234_mgpu_ft3_onlydv_lr2_ep15_var_sa35_tdev',
                                range(0, 1)))

    dataset = get_dataset('dv2017_test_dev')
    return trackers, dataset


def caramba6():
    trackers = []
    trackers.extend(trackerlist('dimp_vos',
                                'dolf_test1_lr2_nfilt32_ytvosjj_1_3_sw_mem_r2_det_fix_nobn_res50_final_conv_flip_aug_dv_lovashz_bugflip_mrcnnweights_nobn_nfilt16_bsz20_dv6_ep56_var',
                                range(0, 1)))
    trackers.extend(trackerlist('dimp_vos',
                                'dolf_test1_lr2_nfilt32_ytvosjj_1_5_sw_mem_r2_det_fix_nobn_res50_final_conv_flip_aug_dv_lovashz_bugflip_mrcnnweights_nfilt16_bsz20_dv6_ep60_var',
                                range(0, 1)))

    dataset = get_dataset('yt2019_jjval_all')
    return trackers, dataset



def supra1():
    trackers = []
    trackers.extend(trackerlist('dimp_vos',
                                'dolf_test1_lr2_nfilt32_ytvosjj_1_3_sw_mem_r2_det_fix_nobn_res50_final_conv_flip_aug_dv_lovashz_bugflip_mrcnnweights_ftvid_l234_mgpu_ft3_onlydv_lr2_longer_var_ep10',
                                range(0, 1)))

    dataset = get_dataset('dv2017_val')[::-1]

    return trackers, dataset


def supra2():
    trackers = []
    trackers.extend(trackerlist('dimp_vos',
                                'dolf_test1_lr2_nfilt32_ytvosjj_1_3_sw_mem_r2_det_fix_nobn_res50_final_conv_flip_aug_dv_lovashz_bugflip_mrcnnweights_ftvid_l234_mgpu_ft3_onlydv_lr2_longer_var_ep20',
                                range(0, 1)))

    dataset = get_dataset('dv2017_val')

    return trackers, dataset


def supra3():
    trackers = []
    trackers.extend(trackerlist('dimp_vos',
                                'dolf_test1_lr2_nfilt32_ytvosjj_1_3_sw_mem_r2_det_fix_nobn_res50_final_conv_flip_aug_dv_lovashz_bugflip_mrcnnweights_ftvid_l234_mgpu_ft3_onlydv_lr2_longer_var_ep30',
                                range(0, 1)))

    dataset = get_dataset('dv2017_val')

    return trackers, dataset


def supra4():
    trackers = []
    trackers.extend(trackerlist('dimp_vos',
                                'dolf_test1_lr2_nfilt32_ytvosjj_1_3_sw_mem_r2_det_fix_nobn_res50_final_conv_flip_aug_dv_lovashz_bugflip_mrcnnweights_ftvid_l234_mgpu_ft3_onlydv_lr2_longer_var_ep40',
                                range(0, 1)))

    dataset = get_dataset('dv2017_val')

    return trackers, dataset


def supra5():
    trackers = []
    trackers.extend(trackerlist('dimp_vos',
                                'dolf_test1_lr2_nfilt32_ytvosjj_1_3_sw_mem_r2_det_fix_nobn_res50_final_conv_flip_aug_dv_lovashz_bugflip_mrcnnweights_ftvid_l234_mgpu_ft3_onlydv_lr2_longer_var_ep50',
                                range(0, 1)))


    dataset = get_dataset('dv2017_val')

    return trackers, dataset


def supra6():
    trackers = []
    trackers.extend(trackerlist('dimp_vos',
                                'dolf_test1_lr2_nfilt32_ytvosjj_1_3_sw_mem_r2_det_fix_nobn_res50_final_conv_flip_aug_dv_lovashz_bugflip_mrcnnweights_ftvid_l234_mgpu_ft3_onlydv_lr2_longer_var_ep60',
                                range(0, 1)))


    dataset = get_dataset('dv2017_val')

    return trackers, dataset


def supra7():
    trackers = trackerlist('etcom', 'sdlearn_300_onlytestloss_lr_causal_mg30_iou_coco_gd_it_40_8_4', range(0, 5))

    dataset = get_dataset('nfs', 'otb', 'uav')

    return trackers, dataset


def supra8():
    trackers = trackerlist('etcom', 'sdlearn_300_onlytestloss_lr_causal_mg30_iou_coco_gd_it40_it_10_2_1', range(0, 5))

    dataset = get_dataset('nfs', 'otb', 'uav')

    return trackers, dataset


def supra9():
    trackers = trackerlist('etcom', 'sdlearn_300_onlytestloss_lr_causal_mg30_iou_coco_gd_it40_it_20_4_2', range(0, 5))

    dataset = get_dataset('nfs', 'otb', 'uav')

    return trackers, dataset
