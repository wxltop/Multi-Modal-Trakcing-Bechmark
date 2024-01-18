from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.davis_dir = ''
    settings.got10k_lmdb_path = '/home/wangxiaolong/data/vot/got10k_lmdb'
    settings.got10k_path = '/home/wangxiaolong/data/vot/got10k'
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.lasot_lmdb_path = '/home/wangxiaolong/data/vot/lasot_lmdb'
    settings.lasot_path = '/home/wangxiaolong/data/vot/lasot'
    settings.network_path = '/home/wangxiaolong/workspace/projects/Multi-Modal-Trakcing-Bechmark/RGBD/models/MixFormer_RGBD/output/test/networks'    # Where tracking networks are stored.
    settings.nfs_path = '/home/wangxiaolong/data/vot/nfs'
    settings.otb_path = '/home/wangxiaolong/data/vot/OTB2015'
    settings.prj_dir = '/home/wangxiaolong/workspace/projects/Multi-Modal-Trakcing-Bechmark/RGBD/models/MixFormer_RGBD'
    settings.result_plot_path = '/home/wangxiaolong/workspace/projects/Multi-Modal-Trakcing-Bechmark/RGBD/models/MixFormer_RGBD/output/test/result_plots'
    settings.results_path = '/home/wangxiaolong/workspace/projects/Multi-Modal-Trakcing-Bechmark/RGBD/models/MixFormer_RGBD/output/test/tracking_results'    # Where to store tracking results
    settings.save_dir = '/home/wangxiaolong/workspace/projects/Multi-Modal-Trakcing-Bechmark/RGBD/models/MixFormer_RGBD/output'
    settings.segmentation_path = '/home/wangxiaolong/workspace/projects/Multi-Modal-Trakcing-Bechmark/RGBD/models/MixFormer_RGBD/output/test/segmentation_results'
    settings.tc128_path = '/home/wangxiaolong/data/vot/TC128'
    settings.tn_packed_results_path = ''
    settings.tpl_path = ''
    settings.trackingnet_path = '/home/wangxiaolong/data/vot/trackingNet'
    settings.uav_path = '/home/wangxiaolong/data/vot/UAV123'
    settings.vot_path = '/home/wangxiaolong/data/vot/VOT2019'
    settings.youtubevos_dir = ''

    return settings

