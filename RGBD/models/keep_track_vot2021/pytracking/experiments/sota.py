from pytracking.evaluation import Tracker, get_dataset, trackerlist


def dimp_nfs():
    trackers = trackerlist('MDNet', 'default', None, 'MDNet') + \
               trackerlist('CCOT', 'OTB_deepHOG_75', None, 'CCOT') + \
               trackerlist('ECOgpu', 'OTB_DEEP_settings', None, 'ECO') + \
               trackerlist('UPDT', 'updt_rand', range(0, 10), 'UPDT') + \
               trackerlist('atom_cvpr19', 'default', range(0, 5), 'ATOM') + \
               trackerlist('etcom', 'sdlearn_300_onlytestloss_lr_causal_mg30_iou_coco_hnfix', range(0, 5), 'DiMP-18') + \
               trackerlist('etcom', 'sdlearn_300_onlytestloss_lr_causal_mg30_iou_nocf_res50_lfilt512_coco_hnfix', range(0, 5), 'DiMP-50')

    dataset = get_dataset('nfs')

    return trackers, dataset


def dimp_uav():
    trackers = trackerlist('CCOT', 'OTB_deepHOG_75', None, 'CCOT') + \
               trackerlist('ECOgpu', 'OTB_HC_settings', None, 'ECOhc') + \
               trackerlist('ECOgpu', 'OTB_DEEP_settings', None, 'ECO') + \
               trackerlist('DaSiamRPN', 'default', None, 'DaSiamRPN') + \
               trackerlist('UPDT', 'updt_rand', range(0, 10), 'UPDT') + \
               trackerlist('atom_cvpr19', 'default', range(0, 5), 'ATOM') + \
               trackerlist('etcom', 'sdlearn_300_onlytestloss_lr_causal_mg30_iou_coco_hnfix', range(0, 5), 'DiMP-18') + \
               trackerlist('etcom', 'sdlearn_300_onlytestloss_lr_causal_mg30_iou_nocf_res50_lfilt512_coco_hnfix', range(0, 5), 'DiMP-50') # + \
               # trackerlist('Staple', 'default', None, 'Staple') + \
               # trackerlist('SRDCF', 'default', None, 'SRDCF')

    dataset = get_dataset('uav')

    return trackers, dataset


def dimp_otb():
    trackers = trackerlist('MDNet', 'default', None, 'MDNet') + \
               trackerlist('CCOT', 'OTB_deepHOG_75', None, 'CCOT') + \
               trackerlist('ECOgpu', 'OTB_HC_settings', None, 'ECOhc') + \
               trackerlist('ECOgpu', 'OTB_DEEP_settings', None, 'ECO') + \
               trackerlist('DaSiamRPN', 'default', None, 'DaSiamRPN') + \
               trackerlist('SiamRPN++', 'default', None, 'SiamRPN++') + \
               trackerlist('UPDT', 'updt_rand', range(0, 10), 'UPDT') + \
               trackerlist('atom_cvpr19', 'default', range(0, 5), 'ATOM') + \
               trackerlist('etcom', 'sdlearn_300_onlytestloss_lr_causal_mg30_iou_coco_hnfix', range(0, 5), 'DiMP-18') + \
               trackerlist('etcom', 'sdlearn_300_onlytestloss_lr_causal_mg30_iou_nocf_res50_lfilt512_coco_hnfix', range(0, 5), 'DiMP-50')

    dataset = get_dataset('otb')

    return trackers, dataset


def dimp_lasot():
    trackers = trackerlist('MDNet', 'default', None, 'MDNet') + \
               trackerlist('VITAL', 'default', None, 'VITAL') + \
               trackerlist('DSiam', 'default', None, 'DSiam') + \
               trackerlist('etcom', 'sdlearn_300_onlytestloss_lr_causal_mg30_iou_nocf_res50_lfilt512_coco_hnfix', range(0, 5), 'DiMP-50') + \
               trackerlist('StructSiam', 'default', None, 'StructSiam') + \
               trackerlist('ECOgpu', 'default', None, 'ECO') + \
               trackerlist('SiameseFC', 'default', None, 'SiamFC') + \
               trackerlist('SiamRPN++', 'default', None, 'SiamRPN++') + \
               trackerlist('atom_cvpr19', 'default', range(0, 5), 'ATOM') + \
               trackerlist('etcom', 'sdlearn_300_onlytestloss_lr_causal_mg30_iou_coco_hnfix', range(0, 5), 'DiMP-18') + \
               trackerlist('etcom', 'sdlearn_300_onlytestloss_lr_causal_mg30_iou_nocf_res50_lfilt512_coco_hnfix', range(0, 5), 'DiMP-50')

    dataset = get_dataset('lasot')

    return trackers, dataset


def dimp_got():
    trackers = trackerlist('MDNet', '', None, 'MDNet') + \
               trackerlist('CF2', '', None, 'CF2') + \
               trackerlist('ECO', '', None, 'ECO') + \
               trackerlist('CCOT', '', None, 'CCOT') + \
               trackerlist('GOTURN', '', None, 'GOTURN') + \
               trackerlist('SiamFC', '', None, 'SiamFC') + \
               trackerlist('SiamFCv2', '', None, 'SiamFCv2') + \
               trackerlist('ATOM', '', None, 'ATOM') + \
               trackerlist('DiMP18', '', None, 'DiMP-18') + \
               trackerlist('DiMP50', '', None, 'DiMP-50')

    dataset = get_dataset('got10k_test')

    return trackers, dataset