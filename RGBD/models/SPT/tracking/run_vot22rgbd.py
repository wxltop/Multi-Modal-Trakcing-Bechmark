from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


import cv2
import torch

import sys
import time
import os
os.chdir('/home/wangxiaolong/workspace/projects/Multi-Modal-Trakcing-Bechmark/RGBD/models/SPT')
print('---> current dir: ', os.getcwd())
sys.path.insert(0, os.getcwd())
import numpy as np
from lib.test.evaluation import Tracker
from lib.test.vot22 import vot
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
tracker_name = 'stark_s'
para_name = 'rgbd'
# create tracker
tracker_info = Tracker(tracker_name, para_name, "vot22rgbd", None)
params = tracker_info.get_parameters()
params.visualization = False
params.debug = False
tracker = tracker_info.create_tracker(params)

handle = vot.VOT("rectangle", 'rgbd')
selection = handle.region()
imagefile = handle.frame()
init_box = [int(selection.x), int(selection.y), int(selection.width), int(selection.height)]
image_rgb = cv2.cvtColor(cv2.imread(imagefile[0]), cv2.COLOR_BGR2RGB)  # Right
image_depth = cv2.imread(imagefile[1], -1)

colormap = cv2.normalize(image_depth, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
colormap = np.asarray(colormap, dtype=np.uint8)
colormap = cv2.applyColorMap(colormap, cv2.COLORMAP_JET)
img = cv2.merge((image_rgb, colormap))

init_info = {'init_bbox': init_box}
_ = tracker.initialize(img, init_info)
while True:

    imagefile = handle.frame()
    image_rgb = cv2.cvtColor(cv2.imread(imagefile[0]), cv2.COLOR_BGR2RGB)  # Right
    image_depth = cv2.imread(imagefile[1], -1)

    colormap = cv2.normalize(image_depth, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    colormap = np.asarray(colormap, dtype=np.uint8)
    colormap = cv2.applyColorMap(colormap, cv2.COLORMAP_JET)

    img = cv2.merge((image_rgb, colormap))
    outputs = tracker.track(img)
    pred_bbox = outputs['target_bbox']
    selection = vot.Rectangle(int(pred_bbox[0]), int(pred_bbox[1]), int(pred_bbox[2]), int(pred_bbox[3]))
    handle.report(selection)
    time.sleep(0.01)
