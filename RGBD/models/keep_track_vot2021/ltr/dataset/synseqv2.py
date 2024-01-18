import os
import os.path
import numpy as np
import torch
import csv
import pandas
import random
from collections import OrderedDict
from .base_video_dataset import BaseVideoDataset
from ltr.data.image_loader import jpeg4py_loader
from ltr.admin.environment import env_settings


class SynSeqv2(BaseVideoDataset):

    def __init__(self, root=None, image_loader=jpeg4py_loader, use_dist=True, use_occ=True):

        root = env_settings().synseqv2_dir if root is None else root
        super().__init__('SynSeqv2', root, image_loader)

        self.use_dist = use_dist
        self.use_occ = use_occ

        # all folders inside the root
        self.sequence_list = self._get_sequence_list(root)

    def get_name(self):
        return 'synseqv2'

    def has_class_info(self):
        return False

    def has_occlusion_info(self):
        return True

    def _get_sequence_list(self, root):
        sequence_list = []

        if self.use_occ:
            dir_list_occ = os.listdir(root + '/occ')
            dir_list_occ = [('occ', d) for d in dir_list_occ]
            sequence_list = sequence_list + dir_list_occ

        if self.use_dist:
            dir_list_dist = os.listdir(root + '/dist')
            dir_list_dist = [('dist', d) for d in dir_list_dist]
            sequence_list = sequence_list + dir_list_dist
        return sequence_list

    def _read_bb_anno(self, seq_path, mode):
        bb_anno_file = os.path.join(seq_path, mode + "_groundtruth.txt")
        gt = pandas.read_csv(bb_anno_file, delimiter=',', header=None, dtype=np.float32, na_filter=False, low_memory=False).values
        return torch.tensor(gt)

    def _read_test_target_visible(self, seq_path):
        # Read full occlusion and out_of_view
        # only applicable for test frames
        cover_file = os.path.join(seq_path, "test_occlusion.txt")

        if not os.path.exists(cover_file):
            return None, None

        with open(cover_file, 'r', newline='') as f:
            cover_ratio = torch.tensor([float(v[0]) for v in csv.reader(f)])

        visible_ratio = 1.0 - cover_ratio
        target_visible = (visible_ratio > 0.5)

        return target_visible, visible_ratio

    def _get_sequence_path(self, seq_id):
        return os.path.join(self.root + '/' + self.sequence_list[seq_id][0], self.sequence_list[seq_id][1])

    def get_train_sequence_info(self, seq_id):
        seq_path = self._get_sequence_path(seq_id)
        bbox = self._read_bb_anno(seq_path, 'train')

        valid = (bbox[:, 2] > 0) & (bbox[:, 3] > 0)

        return {'bbox': bbox, 'valid': valid}

    def get_test_sequence_info(self, seq_id):
        seq_path = self._get_sequence_path(seq_id)
        bbox = self._read_bb_anno(seq_path, 'test')

        valid = (bbox[:, 2] > 0) & (bbox[:, 3] > 0)
        visible, visible_ratio = self._read_test_target_visible(seq_path)

        if visible is None:
            visible = torch.ones(bbox.shape[0], dtype=torch.uint8)
            visible_ratio = torch.ones(bbox.shape[0])
        visible = visible & valid

        return {'bbox': bbox, 'valid': valid, 'visible': visible, 'visible_ratio': visible_ratio}

    def _get_frame_path(self, seq_path, frame_id, mode):
        return os.path.join(seq_path, '{}_{:03}.jpg'.format(mode, frame_id))

    def _get_frame(self, seq_path, frame_id, mode):
        return self.image_loader(self._get_frame_path(seq_path, frame_id, mode))

    def get_train_frames(self, seq_id, frame_ids, anno=None):
        seq_path = self._get_sequence_path(seq_id)

        frame_list = [self._get_frame(seq_path, f_id, 'train') for f_id in frame_ids]

        if anno is None:
            anno = self.get_train_sequence_info(seq_id)

        anno_frames = {}
        for key, value in anno.items():
            anno_frames[key] = [value[f_id, ...].clone() for f_id in frame_ids]

        return frame_list, anno_frames, None

    def get_test_frames(self, seq_id, frame_ids, anno=None):
        seq_path = self._get_sequence_path(seq_id)

        frame_list = [self._get_frame(seq_path, f_id, 'test') for f_id in frame_ids]

        if anno is None:
            anno = self.get_test_sequence_info(seq_id)

        anno_frames = {}
        for key, value in anno.items():
            anno_frames[key] = [value[f_id, ...].clone() for f_id in frame_ids]

        return frame_list, anno_frames, None