from easydict import EasyDict as edict
import yaml

"""
Add default config for OSTrack.
"""
cfg = edict()

# MODEL
cfg.MODEL = edict()
cfg.MODEL.PRETRAIN_FILE = "deit_small_patch16_224-cd65a155.pth"
cfg.MODEL.STRIDE = 16
cfg.MODEL.FPN = False
cfg.MODEL.EXTRA_MERGER = False
cfg.MODEL.HEAD_TYPE = "CORNER"
cfg.MODEL.HIDDEN_DIM = 256
cfg.MODEL.NUM_OBJECT_QUERIES = 1
cfg.MODEL.POSITION_EMBEDDING = 'sine'  # sine or learned
cfg.MODEL.PREDICT_MASK = False
cfg.MODEL.USE_CROSS_ATTN = False
cfg.MODEL.UP_SAMPLE = False
cfg.MODEL.RETURN_INTER = False
cfg.MODEL.FPN_STAGES = [2, 5, 8, 11]
# MODEL.BACKBONE
cfg.MODEL.BACKBONE = edict()
cfg.MODEL.BACKBONE.TYPE = "resnet50"  # resnet50, resnext101_32x8d
cfg.MODEL.BACKBONE.OUTPUT_LAYERS = ["layer3"]
cfg.MODEL.BACKBONE.DILATION = False
cfg.MODEL.BACKBONE.MID_PE = True
cfg.MODEL.BACKBONE.MERGE_LAYER = 0
cfg.MODEL.BACKBONE.SEP_SEG = False
cfg.MODEL.BACKBONE.CAT_MODE = 'direct'

cfg.MODEL.BACKBONE.ADD_CLS_TOKEN = False
cfg.MODEL.BACKBONE.CLS_MODE = 'ignore'
cfg.MODEL.BACKBONE.DROP_PATH_RATE = 0.

# only work when using dynamicViT backbone
cfg.MODEL.BACKBONE.PRUNING_LOC = [3, 6, 9]
cfg.MODEL.BACKBONE.KEEP_RATIO = [0.7, 0.7, 0.7]
cfg.MODEL.BACKBONE.TEMPLATE_RANGE = 'ALL'  # choose between ALL, CTR_POINT, CTR_REC, GT_BOX

cfg.MODEL.BACKBONE.PRUNING_LOC_TEMPLATE = []
cfg.MODEL.BACKBONE.KEEP_RATIO_TEMPLATE = []

# MODEL.TRANSFORMER
cfg.MODEL.TRANSFORMER = edict()
cfg.MODEL.TRANSFORMER.NHEADS = 8
cfg.MODEL.TRANSFORMER.DROPOUT = 0.1
cfg.MODEL.TRANSFORMER.DIM_FEEDFORWARD = 2048
cfg.MODEL.TRANSFORMER.ENC_LAYERS = 6
cfg.MODEL.TRANSFORMER.DEC_LAYERS = 6
cfg.MODEL.TRANSFORMER.PRE_NORM = False
cfg.MODEL.TRANSFORMER.DIVIDE_NORM = False
cfg.MODEL.TRANSFORMER.DROP_PATH_RATE = 0.1
# MODEL.HEAD
cfg.MODEL.HEAD = edict()
cfg.MODEL.HEAD.DIM_DYNAMIC = 64
cfg.MODEL.HEAD.NUM_DYNAMIC = 2
cfg.MODEL.HEAD.NUM_REG = 3
cfg.MODEL.HEAD.NUM_CLS = 1
cfg.MODEL.HEAD.CENTERNESS = False
cfg.MODEL.HEAD.NORM_ON_BBOX = False
cfg.MODEL.HEAD.CENTER_SAMPLING = False
cfg.MODEL.HEAD.CENTER_SAMPLE_RADIUS = 1.5
cfg.MODEL.HEAD.LOSS_TYPE = 'FCOS'

cfg.MODEL.HEAD.NUM_CLS_ATTN_LAYERS = 2
cfg.MODEL.HEAD.NUM_CLS_MLP_LAYERS = 12

# MODEL.DISTILL
cfg.MODEL.DISTILL = edict()
cfg.MODEL.DISTILL.LOSS_TYPE = 'KL'

# TRAIN
cfg.TRAIN = edict()
cfg.TRAIN.LR = 0.0001
cfg.TRAIN.WEIGHT_DECAY = 0.0001
cfg.TRAIN.EPOCH = 500
cfg.TRAIN.LR_DROP_EPOCH = 400
cfg.TRAIN.BATCH_SIZE = 16
cfg.TRAIN.NUM_WORKER = 8
cfg.TRAIN.OPTIMIZER = "ADAMW"
cfg.TRAIN.BACKBONE_MULTIPLIER = 0.1
cfg.TRAIN.GIOU_WEIGHT = 2.0
cfg.TRAIN.L1_WEIGHT = 5.0
cfg.TRAIN.DEEP_SUPERVISION = False
cfg.TRAIN.FREEZE_BACKBONE_BN = True
cfg.TRAIN.FREEZE_LAYERS = ['conv1', 'layer1']
cfg.TRAIN.PRINT_INTERVAL = 50
cfg.TRAIN.VAL_EPOCH_INTERVAL = 20
cfg.TRAIN.GRAD_CLIP_NORM = 0.1
cfg.TRAIN.EXTRA_LOSS = False
cfg.TRAIN.TRAIN_CLS = False
cfg.TRAIN.TRAIN_SEG = False
# TRAIN.SCHEDULER
cfg.TRAIN.SCHEDULER = edict()
cfg.TRAIN.SCHEDULER.TYPE = "step"
cfg.TRAIN.SCHEDULER.DECAY_RATE = 0.1

# DATA
cfg.DATA = edict()
cfg.DATA.SAMPLER_MODE = "causal"  # sampling methods
cfg.DATA.MEAN = [0.485, 0.456, 0.406]
cfg.DATA.STD = [0.229, 0.224, 0.225]
cfg.DATA.MAX_SAMPLE_INTERVAL = 200
cfg.DATA.USE_SEG = False
# DATA.TRAIN
cfg.DATA.TRAIN = edict()
cfg.DATA.TRAIN.DATASETS_NAME = ["LASOT", "GOT10K_vottrain"]
cfg.DATA.TRAIN.DATASETS_RATIO = [1, 1]
cfg.DATA.TRAIN.SAMPLE_PER_EPOCH = 60000
# DATA.VAL
cfg.DATA.VAL = edict()
cfg.DATA.VAL.DATASETS_NAME = ["GOT10K_votval"]
cfg.DATA.VAL.DATASETS_RATIO = [1]
cfg.DATA.VAL.SAMPLE_PER_EPOCH = 10000
# DATA.SEARCH
cfg.DATA.SEARCH = edict()
cfg.DATA.SEARCH.SIZE = 320
cfg.DATA.SEARCH.FACTOR = 5.0
cfg.DATA.SEARCH.CENTER_JITTER = 4.5
cfg.DATA.SEARCH.SCALE_JITTER = 0.5
cfg.DATA.SEARCH.NUMBER = 1
# DATA.TEMPLATE
cfg.DATA.TEMPLATE = edict()
cfg.DATA.TEMPLATE.NUMBER = 1
cfg.DATA.TEMPLATE.SIZE = 128
cfg.DATA.TEMPLATE.FACTOR = 2.0
cfg.DATA.TEMPLATE.CENTER_JITTER = 0
cfg.DATA.TEMPLATE.SCALE_JITTER = 0

# DISTILL
cfg.DISTILL = edict()
cfg.DISTILL.DISTILL_LAYERS = [2, 5, 8, 11]
cfg.DISTILL.TEMPERATURE = 1.0  # the temperature for distillation loss
cfg.DISTILL.ALPHA = 0.0  # the weight to balance the soft label loss and ground-truth label loss

cfg.DISTILL.ATTN_LOSS = False
cfg.DISTILL.QKV_LOSS_WEIGHT = 1
cfg.DISTILL.AR = 1  # The number of relative heads

cfg.DISTILL.HIDDEN_LOSS = True
cfg.DISTILL.HIDDEN_RELATION = True
cfg.DISTILL.HIDDEN_LOSS_WEIGHT = 1

# TEST
cfg.TEST = edict()
cfg.TEST.TEMPLATE_FACTOR = 2.0
cfg.TEST.TEMPLATE_SIZE = 128
cfg.TEST.SEARCH_FACTOR = 5.0
cfg.TEST.SEARCH_SIZE = 320
cfg.TEST.EPOCH = 500
cfg.TEST.MAIN_LOB_AREA_THR = 10
cfg.TEST.REDETECT = False
cfg.TEST.REDE_THRESH = 0.05
cfg.TEST.SEG_THRESH = 0.65


def _edict2dict(dest_dict, src_edict):
    if isinstance(dest_dict, dict) and isinstance(src_edict, dict):
        for k, v in src_edict.items():
            if not isinstance(v, edict):
                dest_dict[k] = v
            else:
                dest_dict[k] = {}
                _edict2dict(dest_dict[k], v)
    else:
        return


def gen_config(config_file):
    cfg_dict = {}
    _edict2dict(cfg_dict, cfg)
    with open(config_file, 'w') as f:
        yaml.dump(cfg_dict, f, default_flow_style=False)


def _update_config(base_cfg, exp_cfg):
    if isinstance(base_cfg, dict) and isinstance(exp_cfg, edict):
        for k, v in exp_cfg.items():
            if k in base_cfg:
                if not isinstance(v, dict):
                    base_cfg[k] = v
                else:
                    _update_config(base_cfg[k], v)
            else:
                raise ValueError("{} not exist in config.py".format(k))
    else:
        return


def update_config_from_file(filename, base_cfg=None):
    exp_config = None
    with open(filename) as f:
        exp_config = edict(yaml.safe_load(f))
        if base_cfg is not None:
            _update_config(base_cfg, exp_config)
        else:
            _update_config(cfg, exp_config)
