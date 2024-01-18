from pytracking.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()
    # Download and compress weights from here https://polybox.ethz.ch/index.php/s/WrpeaU3kZ1ZgjKU
    settings.network_path = ['please change path to checkpoints folder /checkpoints/alpha_refine/',
                             'please change path to checkpoints folder /checkpoints/dimp/',
                             'please change path to checkpoints folder /checkpoints/memory_learning/',
                             ]
    settings.nfs_path = ''
    settings.otb_path = ''
    settings.results_path = 'tracking_results/'
    settings.segmentation_path = 'segmentation_results/'
    settings.tpl_path = ''
    settings.trackingnet_path = ''
    settings.uav_path = ''
    settings.vot_path = '' # vot2018
    settings.lasot_path = ''
    settings.lasot_extension_subset_path = ''
    settings.got10k_path = ''
    settings.oxuva_path = ''

    settings.perf_mat_path = 'data'
    settings.result_plot_path = 'data'
    settings.tn_packed_results_path = 'data'

    settings.vot2020_ws_path = 'please change path to 2021 workspace vot2021'
    # vot2020_path only required when testing the tracker without the vot toolkit.
    settings.vot2020_path = 'please change path to 2021 sequences vot2021/sequences'

    return settings
