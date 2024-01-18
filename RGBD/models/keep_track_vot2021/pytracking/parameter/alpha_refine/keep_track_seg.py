from pytracking.utils import TrackerParams


def parameters():
    params = TrackerParams()

    params.debug = 0
    params.visualization = False

    params.use_gpu = True

    params.ths = 0.65
    params.sr = 2.0
    params.input_sz = int(128*params.sr)

    params.base_tracker_name = 'keep_track'
    params.base_params_name = 'vot2021_30'

    params.refine_bbox = False
    params.produce_segmentation = True
    params.compute_bbox_from_segmentation = False

    params.vot_anno_conversion_type = 'preserve_area'

    params.use_gt_box = True
    params.plot_iou = True

    return params
