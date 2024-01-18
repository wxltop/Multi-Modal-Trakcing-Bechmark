class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = '/home/wangxiaolong/workspace/projects/Multi-Modal-Trakcing-Bechmark/RGBD/models/MixFormer_RGBD'    # Base directory for saving network checkpoints.
        self.tensorboard_dir = '/home/wangxiaolong/workspace/projects/Multi-Modal-Trakcing-Bechmark/RGBD/models/MixFormer_RGBD/tensorboard'    # Directory for tensorboard files.
        self.pretrained_networks = '/home/wangxiaolong/workspace/projects/Multi-Modal-Trakcing-Bechmark/RGBD/models/MixFormer_RGBD/pretrained_networks'
        self.lasot_dir = '/home/wangxiaolong/data/vot/lasot'
        self.got10k_dir = '/home/wangxiaolong/data/vot/got10k/train'
        self.lasot_lmdb_dir = '/home/wangxiaolong/data/vot/lasot_lmdb'
        self.got10k_lmdb_dir = '/home/wangxiaolong/data/vot/got10k_lmdb'
        self.trackingnet_dir = '/home/wangxiaolong/data/vot/trackingnet'
        self.trackingnet_lmdb_dir = '/home/wangxiaolong/data/vot/trackingnet_lmdb'
        self.coco_dir = '/home/wangxiaolong/data/vot/coco'
        self.coco_lmdb_dir = '/home/wangxiaolong/data/vot/coco_lmdb'
        self.lvis_dir = ''
        self.sbd_dir = ''
        self.imagenet_dir = '/home/wangxiaolong/data/vot/vid'
        self.imagenet_lmdb_dir = '/home/wangxiaolong/data/vot/vid_lmdb'
        self.imagenetdet_dir = ''
        self.ecssd_dir = ''
        self.hkuis_dir = ''
        self.msra10k_dir = ''
        self.davis_dir = ''
        self.youtubevos_dir = ''
