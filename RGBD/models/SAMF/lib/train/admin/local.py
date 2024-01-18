class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = '.'    # Base directory for saving network checkpoints.
        self.tensorboard_dir = './tensorboard'    # Directory for tensorboard files.
        self.pretrained_networks = './pretrained_networks'
        self.lasot_dir = './data/lasot'
        self.got10k_dir = './data/got10k/train'
        self.lasot_lmdb_dir = './data/lasot_lmdb'
        self.got10k_lmdb_dir = './data/got10k_lmdb'
        self.trackingnet_dir = './data/trackingnet'
        self.trackingnet_lmdb_dir = './data/trackingnet_lmdb'
        self.coco_dir = './data/coco'
        self.coco_lmdb_dir = './data/coco_lmdb'
        self.lvis_dir = ''
        self.sbd_dir = ''
        self.imagenet_dir = './data/vid'
        self.imagenet_lmdb_dir = './data/vid_lmdb'
        self.imagenetdet_dir = ''
        self.ecssd_dir = ''
        self.hkuis_dir = ''
        self.msra10k_dir = ''
        self.davis_dir = ''
        self.youtubevos_dir = ''
        self.coco_rgbd_dir = "./data/COCO2017_RGBD"
        self.lasot_rgbd_dir = "./data/LaSOT_RGBD"
        self.depthtrack_dir = "./data/DepthTrack"
