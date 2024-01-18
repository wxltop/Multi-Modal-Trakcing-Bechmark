from lib.test.vot22.vot22_rgbd import run_vot_exp
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
run_vot_exp('stark_s', 'rgbd', vis=False)