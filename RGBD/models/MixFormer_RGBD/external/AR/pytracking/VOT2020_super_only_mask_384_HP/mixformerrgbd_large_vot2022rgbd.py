from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import cv2
import torch
import vot
import sys
import time
import os
import numpy as np
from vot import Rectangle
from lib.test.tracker.mixformerrgbd_online import MixFormerRGBDOnline
import lib.test.parameter.mixformerrgbd_online as vot_params

# filepath = os.path.abspath(__file__)
# AR_dir = os.path.abspath(os.path.join(os.path.dirname(filepath), "../../"))
# sys.path.append(AR_dir)
def get_rgbcolormap(color_path, depth_path):
    rgb = cv2.imread(color_path)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    dp = cv2.imread(depth_path, -1)
    max_depth = min(np.median(dp) * 3, 10000)
    dp[dp>max_depth] = max_depth
    dp = cv2.normalize(dp, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    dp = np.asarray(dp, dtype=np.uint8)
    colormap = cv2.applyColorMap(dp, cv2.COLORMAP_JET)  # (h,w) -> (h,w,3)
    img = cv2.merge((rgb, colormap)) # (h,w,6)
    return img

class MIXFORMERRGBD_RGBD(object):
    def __init__(self, tracker):
        self.tracker = tracker

    def initialize(self, image, region):
        self.H, self.W, _ = image.shape
        gt_bbox_np = np.array(region).astype(np.float32)
        '''Initialize STARK for specific video'''
        init_info = {'init_bbox': list(gt_bbox_np)}  # input must be (x,y,w,h)
        self.tracker.initialize(image, init_info)

    def track(self, img_RGB):
        '''TRACK'''
        '''base tracker'''
        outputs = self.tracker.track(img_RGB)
        pred_bbox = outputs['target_bbox']
        pred_score = outputs['target_score']
        return pred_bbox, pred_score

params = vot_params.parameters("baseline_large", model="MixFormerRGBDOnlineScore_ep0030.pth.tar")
mixformer = MixFormerRGBDOnline(params, "VOT2022RGBD")
tracker = MIXFORMERRGBD_RGBD(tracker=mixformer)
handle = vot.VOT("rectangle", 'rgbd')
selection = handle.region()
imagefile = handle.frame()
init_box = [selection.x, selection.y, selection.width, selection.height]
if not imagefile:
    sys.exit(0)
imagefile_rgb = imagefile[0]
imagefile_depth = imagefile[1]
rgbcolormap = get_rgbcolormap(imagefile_rgb, imagefile_depth)  # Right

tracker.H = rgbcolormap.shape[0]
tracker.W = rgbcolormap.shape[1]

tracker.initialize(rgbcolormap, init_box)

while True:
    imagefile = handle.frame()
    if not imagefile:
        break
    imagefile_rgb = imagefile[0]
    imagefile_depth = imagefile[1]
    rgbcolormap = get_rgbcolormap(imagefile_rgb, imagefile_depth)  # Right
    region, confidence = tracker.track(rgbcolormap)
    selection = Rectangle(region[0], region[1], region[2], region[3])
    handle.report(selection, confidence)
