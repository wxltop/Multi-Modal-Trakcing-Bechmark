class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = '/data/Disk_B/xuefeng/VOT2022/SPT_FT'    # Base directory for saving network checkpoints.
        self.tensorboard_dir = '/data/Disk_B/xuefeng/VOT2022/SPT_FT/tensorboard'    # Directory for tensorboard files.
        self.pretrained_networks = '/data/Disk_B/xuefeng/VOT2022/SPT_FT/pretrained_network'

        self.rgbd_dir = '/data/Disk_A/RGBD_dataset'
        self.depthtrack_dir = '/data/Disk_A/DepthTrack/Train'

