class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = '/home/wangxiaolong/workspace/projects/Multi-Modal-Trakcing-Bechmark/ViPT'    # Base directory for saving network checkpoints.
        self.tensorboard_dir = '/home/wangxiaolong/workspace/projects/Multi-Modal-Trakcing-Bechmark/ViPT/tensorboard'    # Directory for tensorboard files.
        self.pretrained_networks = '/home/wangxiaolong/workspace/projects/Multi-Modal-Trakcing-Bechmark/ViPT/pretrained_networks'
        self.got10k_val_dir = '/home/wangxiaolong/workspace/projects/Multi-Modal-Trakcing-Bechmark/data/got10k/val'
        self.lasot_lmdb_dir = '/home/wangxiaolong/workspace/projects/Multi-Modal-Trakcing-Bechmark/data/lasot_lmdb'
        self.got10k_lmdb_dir = '/home/wangxiaolong/workspace/projects/Multi-Modal-Trakcing-Bechmark/data/got10k_lmdb'
        self.trackingnet_lmdb_dir = '/home/wangxiaolong/workspace/projects/Multi-Modal-Trakcing-Bechmark/data/trackingnet_lmdb'
        self.coco_lmdb_dir = '/home/wangxiaolong/workspace/projects/Multi-Modal-Trakcing-Bechmark/data/coco_lmdb'
        self.coco_dir = '/home/wangxiaolong/workspace/projects/Multi-Modal-Trakcing-Bechmark/data/coco'
        self.lasot_dir = '/home/wangxiaolong/workspace/projects/Multi-Modal-Trakcing-Bechmark/data/lasot'
        self.got10k_dir = '/home/wangxiaolong/workspace/projects/Multi-Modal-Trakcing-Bechmark/data/got10k/train'
        self.trackingnet_dir = '/home/wangxiaolong/workspace/projects/Multi-Modal-Trakcing-Bechmark/data/trackingnet'
        self.depthtrack_dir = '/home/wangxiaolong/workspace/projects/Multi-Modal-Trakcing-Bechmark/data/depthtrack/train'
        self.lasher_dir = '/home/wangxiaolong/workspace/projects/Multi-Modal-Trakcing-Bechmark/data/lasher/trainingset'
        self.visevent_dir = '/home/wangxiaolong/workspace/projects/Multi-Modal-Trakcing-Bechmark/data/visevent/train'
