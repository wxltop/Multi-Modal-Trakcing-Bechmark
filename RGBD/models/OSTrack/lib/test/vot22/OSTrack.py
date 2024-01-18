import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from lib.test.vot22.rgbd_tracker import run_vot_exp

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
run_vot_exp('ostrack_online', 'ostrack320_elimination_cls_t2m12_ep50', vis=False)
