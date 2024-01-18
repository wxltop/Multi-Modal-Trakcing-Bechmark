from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import cv2
import external.vot_utils.vot as vot
import sys
import numpy as np
import torch
torch.set_num_threads(4)
from external.vot_utils.vot20_utils import *
from lib.test.tracker.mixformer_online import MixFormerOnline

import lib.test.parameter.mixformer_online as vot_params
from lib.train.dataset.depth_utils import get_rgbd_frame


class SAMixFormer(object):
    def __init__(self, tracker, threshold=0.65):
        self.THRES = threshold
        self.tracker = tracker
        '''create tracker'''

    def initialize(self, image, init_bbox):
        r"""
        init_bbox: [x1, y1, w, h]
        """
        self.H, self.W, _ = image.shape
        gt_bbox_np = np.array(init_bbox).astype(np.float32)
        init_info = {'init_bbox': list(gt_bbox_np)}
        self.tracker.initialize(image, init_info)

    def track(self, img_RGB):
        '''TRACK'''
        outputs = self.tracker.track(img_RGB)
        pred_bbox = outputs['target_bbox']
        return pred_bbox, None


def make_full_size(x, output_sz):
    '''
    zero-pad input x (right and down) to match output_sz
    x: numpy array e.g., binary mask
    output_sz: size of the output [width, height]
    '''
    if x.shape[0] == output_sz[1] and x.shape[1] == output_sz[0]:
        return x
    pad_x = output_sz[0] - x.shape[1]
    if pad_x < 0:
        x = x[:, :x.shape[1] + pad_x]
        # padding has to be set to zero, otherwise pad function fails
        pad_x = 0
    pad_y = output_sz[1] - x.shape[0]
    if pad_y < 0:
        x = x[:x.shape[0] + pad_y, :]
        # padding has to be set to zero, otherwise pad function fails
        pad_y = 0
    return np.pad(x, ((0, pad_y), (0, pad_x)), 'constant', constant_values=0)


# params = vot_params.parameters("baseline_large")
params = vot_params.parameters("baseline_large", model="SAMixFormerOnlineScore_ep0040.pth.tar")
mixformer = MixFormerOnline(params, "VOT22")
tracker = SAMixFormer(tracker=mixformer)
handle = vot.VOT("rectangle", "rgbd")
selection = handle.region()
color_image_path, depth_image_path = handle.frame()

if not color_image_path or not depth_image_path:
    sys.exit(0)

image = get_rgbd_frame(color_image_path, depth_image_path, dtype=params.cfg.DATA.INPUT_TYPE, depth_clip=True)
init_box = [selection.x, selection.y, selection.width, selection.height]

tracker.H = image.shape[0]
tracker.W = image.shape[1]

tracker.initialize(image, init_box)

while True:
    color_image_path, depth_image_path = handle.frame()
    if not color_image_path or not depth_image_path:
        break
    image = get_rgbd_frame(color_image_path, depth_image_path, dtype=params.cfg.DATA.INPUT_TYPE, depth_clip=True)
    rectangle, confidence = tracker.track(image)
    handle.report(vot.Rectangle(*rectangle), confidence)
