class EnvironmentSettings():
    def __init__(self):
        self.set_default()

    def set_default(self):
        # self.workspace_dir = '/srv/glusterfs/damartin/train_ws/'
        self.workspace_dir = '/srv/beegfs02/scratch/visobt4s/data/train_ws/'
        self.tensorboard_dir = '{}/tensorboard/'.format(self.workspace_dir)
        self.base_model_dir = '/home/damartin/scratch/networks/tracking/'
        self.base_data_dir = '/scratch_net/supra/common/data/tracking/'
        # self.lasot_dir = '{}/LaSOTBenchmark'.format(self.base_data_dir)
        self.got10k_dir = '{}/got10k/train'.format(self.base_data_dir)
        # self.lasot_dir = '/scratch_net/caramba/common/data/tracking/LaSOTBenchmark/'
        self.lasot_dir = '/srv/beegfs02/scratch/visobt4s/data/tracking_datasets/LaSOTBenchmark/'
        self.lasot_dumped_dir = '/srv/beegfs02/scratch/visobt4s/data/tracking_datasets/LaSOTBenchmark_super_dimp_hinge_dumped_data/'
        # self.got10k_dir = '/scratch_net/knurrhahn_second/common/data/tracking/got10k/train/'
        self.trackingnet_dir = '{}/trackingnet'.format(self.base_data_dir)
        # self.trackingnet_dir = '/srv/glusterfs/damartin/data/tracking/trackingnet/'
        self.imagenet_dir = '/scratch_net/minnow_second/data/tracking/ILSVRC2015/'
        self.imagenetdet_dir = '/scratch_net/caramba_second/common/data/tracking/ImageNet-DET/'
        # self.coco_dir = '/scratch_net/minnow_second/data/tracking/coco'
        self.coco_dir = '/srv/beegfs02/scratch/visobt4s/data/tracking_datasets/coco'
        self.ytbb_dir = '/srv/beegfs02/scratch/visobt4s/data/tracking_datasets/ytbb/videos'
        self.davis_dir = '/scratch_net/supra/common/data/tracking/davis'
        self.youtubevos_dir = '/scratch_net/supra/common/data/tracking/youtubevos'
        self.lvis_dir = '/srv/beegfs02/scratch/visobt4s/data/tracking_datasets/lvis2'


