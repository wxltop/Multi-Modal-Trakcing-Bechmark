from lib.test.utils import TrackerParams
import os
from lib.test.evaluation.environment import env_settings
from lib.config.mixformerrgbd_online.config import cfg, update_config_from_file # todo


def parameters(yaml_name: str, model=None, search_area_scale=None):
    params = TrackerParams()
    prj_dir = env_settings().prj_dir
    save_dir = env_settings().save_dir
    # update default config from yaml file
    yaml_file = os.path.join(prj_dir, 'experiments/mixformerrgbd_online/{}.yaml'.format(yaml_name))
    update_config_from_file(yaml_file)
    params.cfg = cfg
    print("test config: ", cfg)

    # template and search region
    params.template_factor = cfg.TEST.TEMPLATE_FACTOR
    params.template_size = cfg.TEST.TEMPLATE_SIZE
    if search_area_scale is not None:
        params.search_factor = search_area_scale
    else:
        params.search_factor = cfg.TEST.SEARCH_FACTOR
    print("search_area_scale: {}".format(params.search_factor))
    # params.search_factor = cfg.TEST.SEARCH_FACTOR
    params.search_size = cfg.TEST.SEARCH_SIZE
    params.max_score_decay = cfg.TEST.MAX_SCORE_DECAY

    # Network checkpoint path
    # params.checkpoint = os.path.join(save_dir, "models/%s.pth.tar" % yaml_name)
    # params.checkpoint = os.path.join(save_dir, "models/mixformer_online_got_1k_ep35.pth.tar")
    if model is None:
        raise NotImplementedError("Please set proper model to test.")
    else:
        # params.checkpoint = os.path.join(save_dir, "models/%s" % model)
        params.checkpoint = os.path.join(
            save_dir, "{}".format(model))

    # whether to save boxes from all queries
    params.save_all_boxes = False

    return params