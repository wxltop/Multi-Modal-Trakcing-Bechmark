import os
from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.work_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))

    settings.davis_dir = ''
    settings.got10k_lmdb_path = './data/got10k_lmdb'
    settings.got10k_path = './data/got10k'
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.lasot_lmdb_path = './data/lasot_lmdb'
    settings.lasot_path = './data/lasot'
    settings.network_path = './test/networks'    # Where tracking networks are stored.
    settings.nfs_path = './data/nfs'
    settings.otb_path = './data/OTB2015'
    settings.prj_dir = settings.work_dir
    settings.result_plot_path = './test/result_plots'
    settings.results_path = './test/tracking_results'    # Where to store tracking results
    settings.save_dir = settings.work_dir
    settings.segmentation_path = './test/segmentation_results'
    settings.tc128_path = './data/TC128'
    settings.tn_packed_results_path = ''
    settings.tpl_path = ''
    settings.trackingnet_path = './data/trackingNet'
    settings.uav_path = './data/UAV123'
    settings.vot_path = './data/VOT2019'
    settings.youtubevos_dir = ''
    settings.coco_rgbd_dir = "./data/COCO2017_RGBD"
    settings.lasot_rgbd_dir = "./data/LaSOT_RGBD"
    settings.depthtrack_dir = "./data/DepthTrack"

    return settings

