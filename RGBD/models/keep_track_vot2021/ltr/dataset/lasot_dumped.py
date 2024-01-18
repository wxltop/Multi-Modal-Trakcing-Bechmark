import os
import os.path
import torch
import numpy as np
import pandas
import csv
import random
from tqdm import tqdm
from collections import OrderedDict, defaultdict
if __name__ == '__main__':
    from base_video_dataset import BaseVideoDataset
else:
    from .base_video_dataset import BaseVideoDataset
from ltr.data.image_loader import jpeg4py_loader
from ltr.admin.environment import env_settings

from ltr.data.processing_utils import sample_target_from_crop_region


def _clone(tensor, attr_name='to_tensor_list'):
    tensor_new = tensor.clone()
    if hasattr(tensor, attr_name):
        setattr(tensor_new, attr_name, getattr(tensor, attr_name))
    return tensor_new


class LasotDumped(BaseVideoDataset):
    """ LaSOT dataset dumped results during tracking super_dimp_hinge.
    """

    def __init__(self, root=None, image_loader=jpeg4py_loader, vid_ids=None, split=None, data_fraction=None,
                 gth_mem_size=15, full_mem_size=50, load_img=False, img_crop_output_sz=22*6):
        """
        args:
            root - path to the lasot dataset.
            image_loader (jpeg4py_loader) -  The function to read the images. jpeg4py (https://github.com/ajkxyz/jpeg4py)
                                            is used by default.
            vid_ids - List containing the ids of the videos (1 - 20) used for training. If vid_ids = [1, 3, 5], then the
                    videos with subscripts -1, -3, and -5 from each class will be used for training.
            split - If split='train', the official train split (protocol-II) is used for training. Note: Only one of
                    vid_ids or split option can be used at a time.
            data_fraction - Fraction of dataset to be used. The complete dataset is used by default
        """
        root = env_settings().lasot_dumped_dir if root is None else root
        super().__init__('LaSOTDumped', root, image_loader)

        self.gth_mem_size = gth_mem_size
        self.full_mem_size = full_mem_size
        self.load_img = load_img
        self.img_crop_output_sz = img_crop_output_sz

        self.sequence_info_cache = {}

        # Keep a list of all classes
        self.class_list = [f for f in os.listdir(self.root)]
        self.class_to_id = {cls_name: cls_id for cls_id, cls_name in enumerate(self.class_list)}

        self.sequence_list = self._build_sequence_list(vid_ids, split)

        if data_fraction is not None:
            self.sequence_list = random.sample(self.sequence_list, int(len(self.sequence_list)*data_fraction))

        self.seq_per_class = self._build_class_list()

    def _build_sequence_list(self, vid_ids=None, split=None):
        if split is not None:
            if vid_ids is not None:
                raise ValueError('Cannot set both split_name and vid_ids.')
            ltr_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
            if split == 'train':
                file_path = os.path.join(ltr_path, 'data_specs', 'lasot_train_split.txt')
            elif split == 'val':
                file_path = os.path.join(ltr_path, 'data_specs', 'lasot_val_split.txt')
            elif split == 'train-train':
                file_path = os.path.join(ltr_path, 'data_specs', 'lasot_train_train_split.txt')
            elif split == 'train-val':
                file_path = os.path.join(ltr_path, 'data_specs', 'lasot_train_val_split.txt')
            else:
                raise ValueError('Unknown split name.')
            sequence_list = pandas.read_csv(file_path, header=None, squeeze=True).values.tolist()
        elif vid_ids is not None:
            sequence_list = [c+'-'+str(v) for c in self.class_list for v in vid_ids]
        else:
            raise ValueError('Set either split_name or vid_ids.')

        return sequence_list

    def _build_class_list(self):
        seq_per_class = {}
        for seq_id, seq_name in enumerate(self.sequence_list):
            class_name = seq_name.split('-')[0]
            if class_name in seq_per_class:
                seq_per_class[class_name].append(seq_id)
            else:
                seq_per_class[class_name] = [seq_id]

        return seq_per_class

    def get_name(self):
        return 'lasot_dumped'

    def has_class_info(self):
        return True

    def has_occlusion_info(self):
        return True

    def get_num_sequences(self):
        return len(self.sequence_list)

    def get_num_classes(self):
        return len(self.class_list)

    def get_sequences_in_class(self, class_name):
        return self.seq_per_class[class_name]

    def _read_bb_anno(self, seq_path):
        bb_anno_file = os.path.join(seq_path, "groundtruth.txt")
        gt = pandas.read_csv(bb_anno_file, delimiter=',', header=None, dtype=np.float32, na_filter=False, low_memory=False).values
        return torch.tensor(gt)

    def _read_target_visible(self, seq_path):
        # Read full occlusion and out_of_view
        occlusion_file = os.path.join(seq_path, "full_occlusion.txt")
        out_of_view_file = os.path.join(seq_path, "out_of_view.txt")

        with open(occlusion_file, 'r', newline='') as f:
            occlusion = torch.ByteTensor([int(v) for v in list(csv.reader(f))[0]])
        with open(out_of_view_file, 'r') as f:
            out_of_view = torch.ByteTensor([int(v) for v in list(csv.reader(f))[0]])

        target_visible = ~occlusion & ~out_of_view

        return target_visible

    def _read_overlap(self, seq_path):
        # Read iou of dumped data
        overlap_file = os.path.join(seq_path, 'iou_per_frame.txt')
        overlap = pandas.read_csv(overlap_file, delimiter=',', header=None, dtype=np.float32, na_filter=False, low_memory=False).values
        return torch.tensor(overlap).reshape(-1)

    def _read_update_flag(self, seq_path):
        update_flag_file = os.path.join(seq_path, 'update_flag_per_frame.txt')
        update_flags_numpy = np.loadtxt(update_flag_file)
        update_flags = torch.ByteTensor(len(update_flags_numpy) + 1)
        update_flags[0] = 1 # First frame is not dumped so add it here artificially.
        update_flags[1:] = torch.ByteTensor(update_flags_numpy)
        return update_flags

    def _read_peak_dist_pred_anno(self, seq_path):
        peak_dist_pred_anno_file = os.path.join(seq_path, 'peak_dist_pred_anno.txt')
        peak_dist_pred_anno = np.loadtxt(peak_dist_pred_anno_file)
        return torch.tensor(peak_dist_pred_anno)

    def _read_peak_dist_anno_2nd_closest_peak(self, seq_path):
        peak_dist_anno_2nd_closest_peak_file = os.path.join(seq_path, 'peak_dist_anno_2nd_closest_peak.txt')
        peak_dist_anno_2nd_closest_peak = np.loadtxt(peak_dist_anno_2nd_closest_peak_file)
        return torch.tensor(peak_dist_anno_2nd_closest_peak)

    def _read_num_peaks(self, seq_path):
        num_peaks_file = os.path.join(seq_path, 'num_peaks.txt')
        num_peaks = np.loadtxt(num_peaks_file)
        return torch.tensor(num_peaks)

    def _read_max_peak_score(self, seq_path):
        max_peak_score_file = os.path.join(seq_path, 'max_peak_score.txt')
        max_peak_score = np.loadtxt(max_peak_score_file)
        return torch.tensor(max_peak_score)

    def _read_correct_peak_score(self, seq_path):
        correct_peak_score_file = os.path.join(seq_path, 'correct_peak_score.txt')
        correct_peak_score = np.loadtxt(correct_peak_score_file)
        return torch.tensor(correct_peak_score)

    def _read_sortindex_coorect_peak_score(self, seq_path):
        sortindex_coorect_peak_score_file = os.path.join(seq_path, 'sortindex_coorect_peak_score.txt')
        sortindex_coorect_peak_score = np.loadtxt(sortindex_coorect_peak_score_file)
        return torch.ByteTensor(sortindex_coorect_peak_score)

    def _read_peak_detected(self, seq_path):
        peak_detected_file = os.path.join(seq_path, 'peak_detected.txt')
        peak_detected = np.loadtxt(peak_detected_file)
        return torch.tensor(peak_detected)

    def _read_search_area_boxes(self, seq_path):
        search_area_bb_file = os.path.join(seq_path, 'search_area_boxes.txt')
        search_area_bbs = pandas.read_csv(search_area_bb_file, delimiter=',', header=None, dtype=np.float32,
                                          na_filter=False, low_memory=False).values
        return torch.tensor(search_area_bbs)

    def _read_subsequence_states(self, seq_path):
        sub_sequence_states_file = os.path.join(seq_path, 'subsequences.csv')
        df = pandas.read_csv(sub_sequence_states_file, delimiter=',', dtype=np.int, index_col=0)
        d = {}
        for state_name in df.columns:
            d[state_name] = torch.tensor(df[state_name].values)
        return d

    def _read_frame_states(self, seq_path):
        frame_states_file = os.path.join(seq_path, 'frame_states.csv')
        df = pandas.read_csv(frame_states_file, delimiter=',', dtype=np.int, index_col=0)
        d = {}
        for state_name in df.columns:
            d[state_name] = torch.tensor(df[state_name].values)
        return d

    def _compute_critical_frames(self, overlaps, visible, update_flag):
        overlaps = overlaps

        mask_down = torch.zeros(overlaps.shape[0])
        mask_up = torch.zeros(overlaps.shape[0])
        up_fail_high_ids = {}
        down_high_fail_ids = {}

        width_sample = 25
        transition_window = 50

        min_fail = 5
        min_high = 5

        th_high = 0.65
        th_fail = 0.1

        is_high = False
        has_failed = False

        for i in range(0, overlaps.shape[0]):
            o = overlaps[i]

            if o >= th_high:
                is_high = True

            if is_high and o < th_high:
                # potential failure might come soon.
                for ii in range(i, min(overlaps.shape[0], i + transition_window)):
                    if overlaps[ii] <= th_fail:
                        # failure_detected:

                        # get more high samples before failure max 'width_samples'
                        num_high = 0
                        max_high = dict(num=0, idx=0)
                        for kk in range(i, max(0, i - width_sample), -1):
                            if overlaps[kk] >= th_high:
                                num_high += 1
                                if num_high > max_high['num']:
                                    max_high = dict(num=num_high, idx=kk)
                            else:
                                num_high = 0

                        # get more low samples after failure max 'with_samples'
                        num_fail = 0
                        max_fail = dict(num=0, idx=0)
                        for jj in range(ii, min(overlaps.shape[0], ii+width_sample), 1):
                            if overlaps[jj] <= th_fail:
                                num_fail += 1
                                if num_fail > max_fail['num']:
                                    max_fail = dict(num=num_fail, idx=jj)
                            else:
                                num_fail = 0

                        # check if detected failure is usable.
                        if (max_fail['num'] > min_fail and max_high['num'] > min_high):
                            length = max_fail['idx'] - max_high['idx']
                            # keep the shortest subsequence around failure if multiple are found.
                            if ii not in down_high_fail_ids:
                                down_high_fail_ids[ii] = dict(start=max_high['idx'], end=max_fail['idx'], length=length)
                            else:
                                if length < down_high_fail_ids[ii]['length']:
                                    down_high_fail_ids[ii] = dict(start=max_high['idx'], end=max_fail['idx'], length=length)

                        break

                is_high = False


            if o <= th_fail:
                has_failed = True

            if has_failed and o > th_fail:
                # potential redetection might come soon.
                for ii in range(i, min(overlaps.shape[0], i + transition_window)):
                    if overlaps[ii] >= th_high:
                        # redetection detected

                        # get more failed samples before redetection max 'width_samples'
                        num_fail = 0
                        max_fail = dict(num=0, idx=0)
                        for kk in range(i, max(0, i - width_sample), -1):
                            if overlaps[kk] <= th_fail:
                                num_fail += 1
                                if num_fail > max_fail['num']:
                                    max_fail = dict(num=num_fail, idx=kk)
                            else:
                                num_fail = 0

                        # get more high samples after redetection max 'with_samples'
                        num_high = 0
                        max_high = dict(num=0, idx=0)
                        for jj in range(ii, min(overlaps.shape[0], ii + width_sample), 1):
                            if overlaps[jj] >= th_high:
                                num_high += 1
                                if num_high > max_high['num']:
                                    max_high = dict(num=num_high, idx=jj)
                            else:
                                num_high = 0

                        # check if detected redetection is usable
                        if (max_fail['num'] > min_fail and max_high['num'] > min_high):
                            length = max_high['idx'] - max_fail['idx']
                            # keep the shortest subsequence around redetection if multiple are found.
                            if ii not in up_fail_high_ids:
                                up_fail_high_ids[ii] = dict(start=max_fail['idx'], end=max_high['idx'], length=length)
                            else:
                                if length < up_fail_high_ids[ii]['length']:
                                    up_fail_high_ids[ii] = dict(start=max_fail['idx'], end=max_high['idx'], length=length)

                        break

                has_failed = False

        for mask, ids in zip([mask_up, mask_down], [up_fail_high_ids, down_high_fail_ids]):
            for val in ids.values():
                mask[val['start']:val['end']] = 1

        return (mask_up == 1) | (mask_down == 1), up_fail_high_ids, down_high_fail_ids

    def build_subsequence_states(self):
        subseq_states_all = defaultdict(list)
        for seq_id, seqname in enumerate(tqdm(self.sequence_list)):
            seq_path_dumped = self._get_sequence_path(self.root, seq_id)
            subseq_states = self._read_subsequence_states(seq_path_dumped)

            for state_name, states in subseq_states.items():
                start_idx = torch.nonzero(states)
                seq_ids = seq_id*torch.ones_like(start_idx)
                data = torch.cat([seq_ids, start_idx], dim=1)
                subseq_states_all[state_name].append(data)

        for state_name in subseq_states_all.keys():
            subseq_states_all[state_name] = torch.cat(subseq_states_all[state_name], dim=0)

        return subseq_states_all

    def build_frame_states(self):
        frame_states_all = defaultdict(list)
        for seq_id, seqname in enumerate(tqdm(self.sequence_list)):
            seq_path_dumped = self._get_sequence_path(self.root, seq_id)
            frame_states = self._read_frame_states(seq_path_dumped)

            for state_name, states in frame_states.items():
                frame_idx = torch.nonzero(states)
                seq_ids = seq_id*torch.ones_like(frame_idx)
                data = torch.cat([seq_ids, frame_idx], dim=1)
                frame_states_all[state_name].append(data)

        for state_name in frame_states_all.keys():
            frame_states_all[state_name] = torch.cat(frame_states_all[state_name], dim=0)

        return frame_states_all

    def _get_sequence_path(self, root, seq_id):
        seq_name = self.sequence_list[seq_id]
        class_name = seq_name.split('-')[0]
        vid_id = seq_name.split('-')[1]

        return os.path.join(root, class_name, class_name + '-' + vid_id)

    def get_sequence_info(self, seq_id):
        if seq_id not in self.sequence_info_cache:
            seq_path_img = self._get_sequence_path(env_settings().lasot_dir, seq_id)
            seq_path_dumped = self._get_sequence_path(self.root, seq_id)
            bbox = self._read_bb_anno(seq_path_img)

            valid = (bbox[:, 2] > 0) & (bbox[:, 3] > 0)
            visible = self._read_target_visible(seq_path_img) & valid.byte()
            overlap = self._read_overlap(seq_path_dumped)
            update_flag = self._read_update_flag(seq_path_dumped)

            num_peaks = self._read_num_peaks(seq_path_dumped)
            peak_dist_pred_anno = self._read_peak_dist_pred_anno(seq_path_dumped)
            peak_dist_anno_2nd_closest_peak = self._read_peak_dist_anno_2nd_closest_peak(seq_path_dumped)
            max_peak_score = self._read_max_peak_score(seq_path_dumped)
            correct_peak_score = self._read_correct_peak_score(seq_path_dumped)
            sortindex_coorect_peak_score = self._read_sortindex_coorect_peak_score(seq_path_dumped)
            peak_detected = self._read_peak_detected(seq_path_dumped)
            search_area_boxes = self._read_search_area_boxes(seq_path_dumped)
            sub_sequence_states = self._read_subsequence_states(seq_path_dumped)

            critical_frames, _, _ = self._compute_critical_frames(overlap, visible, update_flag)

            sequence_info = {
                'bbox': bbox,
                'valid': valid,
                'visible': visible,
                'overlap': overlap,
                'update_flag': update_flag,
                'critical_frames': critical_frames,
                'num_peaks': num_peaks,
                'peak_dist_pred_anno': peak_dist_pred_anno,
                'peak_dist_anno_2nd_closest_peak': peak_dist_anno_2nd_closest_peak,
                'max_peak_score': max_peak_score,
                'correct_peak_score': correct_peak_score,
                'sortindex_coorect_peak_score': sortindex_coorect_peak_score,
                'peak_detected': peak_detected,
                'search_area_boxes': search_area_boxes,
                'sub_sequence_states': sub_sequence_states
            }
            self.sequence_info_cache[seq_id] = sequence_info

        return self.sequence_info_cache[seq_id]

    def _get_frame_path(self, seq_path, frame_id):
        return os.path.join(seq_path, 'img', '{:08}.jpg'.format(frame_id+1))    # frames start from 1

    def _get_dumped_data_path(self, seq_path, frame_id):
        return os.path.join(seq_path, 'data', '{:08}.npz'.format(frame_id+1))    # frames start from 1

    def _get_frame(self, seq_path, frame_id):
        return self.image_loader(self._get_frame_path(seq_path, frame_id))

    def _get_class(self, seq_path):
        raw_class = seq_path.split('/')[-2]
        return raw_class

    def _get_dumped_data(self, seq_path, seq_img_path, frame_id):
        data_numpy = np.load(self._get_dumped_data_path(seq_path, frame_id))

        data_torch = {}
        for key, val in data_numpy.items():
            if key not in ('seq_name', 'frame_number'):
                val = torch.tensor(val.astype(np.float32))

                # Check if loaded data contains memory dependent tensors where the size may differ
                if len(val.shape) == 4 and self.gth_mem_size <= val.shape[0]:
                    mem_size = val.shape[0]
                    mem_mask = torch.zeros(self.full_mem_size, dtype=torch.bool)
                    mem_mask[:mem_size] = True

                    # pad tensor to full_mem_size
                    val_pad = torch.zeros((self.full_mem_size,) + val.shape[1:])
                    val_pad[mem_mask] = val
                    val = val_pad
                    if 'mem_mask' not in data_torch:
                        data_torch['mem_mask'] = mem_mask
                    else:
                        if np.allclose(mem_mask, data_torch['mem_mask']) == False:
                            raise RuntimeError(frame_id)
                            pass

                data_torch[key] = val

        if self.load_img:
            img = self.image_loader(self._get_frame_path(seq_img_path, frame_id))
            data_torch['img'] = img


        return data_torch

    def get_class_name(self, seq_id):
        seq_path = self._get_sequence_path(self.root, seq_id)
        obj_class = self._get_class(seq_path)

        return obj_class

    def get_sequence_name(self, seq_id):
        return self.sequence_list[seq_id]

    def get_frames(self, seq_id, frame_ids, anno=None):
        frames_dict = dict()

        if anno is None:
            anno = self.get_sequence_info(seq_id)

        seq_path_img = self._get_sequence_path(env_settings().lasot_dir, seq_id)
        seq_path_dumped = self._get_sequence_path(self.root, seq_id)

        obj_class = self._get_class(seq_path_dumped)

        # frame_list = [self._get_frame(seq_path_img, f_id) for f_id in frame_ids]
        dumped_data_frame_list = [self._get_dumped_data(seq_path_dumped, seq_path_img, f_id) for f_id in frame_ids]

        for key in dumped_data_frame_list[0].keys():
            frames_dict[key] = [data[key] for data in dumped_data_frame_list] # is cloning needed here?

        for key, value in anno.items():
            if key != 'update_flag' and key != 'sub_sequence_states':
                frames_dict[key] = [value[f_id, ...].clone() for f_id in frame_ids]

        object_meta = OrderedDict({'object_class_name': obj_class,
                                   'motion_class': None,
                                   'major_class': None,
                                   'root_class': None,
                                   'motion_adverb': None})

        return frames_dict, object_meta


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    from tqdm import tqdm
    import pandas as pd
    from ltr.data.processing_utils import sample_target_adaptive, sample_target_from_crop_region

    ROOT_PATH = '/srv/beegfs02/scratch/visobt4s/data/tracking_datasets/LaSOTBenchmark_super_dimp_hinge_dumped_data'

    dataset = LasotDumped(split='train', load_img=True)

    th = 0.25
    # seq_id = 30
    # frames_dict, _ = dataset.get_frames(seq_id, range(2, 12))
    #
    # print(dataset.get_sequence_name(seq_id))
    #
    # i = 6
    # img = frames_dict['img_crop'][i]
    # pred_bbox = frames_dict['pred_bbox'][i+1]
    # search_area_box = frames_dict['search_area_box'][i]
    #
    # # im_out, crop_box = sample_target_from_crop_region(img.numpy(), search_area_box, output_sz=22*16)
    #
    #
    #
    # fig, axes = plt.subplots(2, 1, figsize=(5, 10))
    # axes[0].imshow(img)
    # # axes[1].imshow(im_out)
    # axes[1].imshow(frames_dict['target_scores'][i][0,0])
    #
    # plt.show()
    # print(search_area_box)



    # stats = defaultdict(int)
    #
    # bins = np.zeros((len(dataset.sequence_list), 20))
    # for i in tqdm(range(0, len(dataset.sequence_list))):
    #     # subseq = dict()
    #
    #     anno = dataset.get_sequence_info(i)
    #     seqname = dataset.get_sequence_name(i)
    #
    #     modeA = ((anno['num_peaks'] == 0) & (anno['visible'] == 1))
    #     modeA[:2] = False
    #     modeB = (
    #         (anno['num_peaks'] == 1) &
    #         (anno['peak_dist_pred_anno'] > 4) &
    #         (anno['max_peak_score'] < th) &
    #         (anno['visible'] == 1)
    #     )
    #     modeC = (
    #         (anno['num_peaks'] == 1) &
    #         (anno['peak_dist_pred_anno'] <= 2) &
    #         (anno['sortindex_coorect_peak_score'] == 0) &
    #         (anno['max_peak_score'] < th) &
    #         (anno['visible'] == 1)
    #     )
    #     modeD = (
    #         (anno['num_peaks'] == 1) &
    #         (anno['peak_dist_pred_anno'] <= 2) &
    #         (anno['sortindex_coorect_peak_score'] == 0) &
    #         (anno['max_peak_score'] >= th) &
    #         (anno['visible'] == 1)
    #     )
    #     modeE = (
    #         (anno['num_peaks'] == 1) &
    #         (anno['peak_dist_pred_anno'] > 4) &
    #         (anno['max_peak_score'] >= th) &
    #         (anno['visible'] == 1)
    #     )
    #     modeF = (
    #         (anno['num_peaks'] > 1) &
    #         (anno['peak_dist_pred_anno'] > 4) &
    #         (anno['max_peak_score'] < th) &
    #         (anno['visible'] == 1)
    #     )
    #     modeG = (
    #         (anno['num_peaks'] > 1) &
    #         (anno['peak_dist_pred_anno'] <= 2) &
    #         (anno['peak_dist_anno_2nd_closest_peak'] > 4) &
    #         (anno['sortindex_coorect_peak_score'] == 0) &
    #         (anno['max_peak_score'] < th) &
    #         (anno['visible'] == 1)
    #     )
    #     modeH = (
    #         (anno['num_peaks'] > 1) &
    #         (anno['peak_dist_pred_anno'] <= 2) &
    #         (anno['peak_dist_anno_2nd_closest_peak'] > 4) &
    #         (anno['sortindex_coorect_peak_score'] == 0) &
    #         (anno['max_peak_score'] >= th) &
    #         (anno['visible'] == 1)
    #     )
    #     modeI = (
    #         (anno['num_peaks'] > 1) &
    #         (anno['peak_dist_pred_anno'] <= 2) &
    #         (anno['peak_dist_anno_2nd_closest_peak'] > 4) &
    #         (anno['sortindex_coorect_peak_score'] > 0) &
    #         (anno['max_peak_score'] < th) &
    #         (anno['visible'] == 1)
    #     )
    #     modeJ = (
    #         (anno['num_peaks'] > 1) &
    #         (anno['peak_dist_pred_anno'] > 4) &
    #         (anno['max_peak_score'] >= th) &
    #         (anno['visible'] == 1)
    #     )
    #     modeK = (
    #         (anno['num_peaks'] > 1) &
    #         (anno['peak_dist_pred_anno'] <= 2) &
    #         (anno['peak_dist_anno_2nd_closest_peak'] > 4) &
    #         (anno['sortindex_coorect_peak_score'] > 0) &
    #         (anno['max_peak_score'] >= th) &
    #         (anno['visible'] == 1)
    #     )
    #     modeL = ((anno['num_peaks'] == 0) & (anno['visible'] == 0))
    #     modeM = (
    #         (anno['num_peaks'] == 1) &
    #         (anno['peak_dist_pred_anno'] > 4) &
    #         (anno['max_peak_score'] < th) &
    #         (anno['visible'] == 0)
    #     )
    #     modeN = (
    #         (anno['num_peaks'] == 1) &
    #         (anno['peak_dist_pred_anno'] > 4) &
    #         (anno['max_peak_score'] >= th) &
    #         (anno['visible'] == 0)
    #     )
    #     modeO = (
    #         (anno['num_peaks'] > 1) &
    #         (anno['peak_dist_pred_anno'] > 4) &
    #         (anno['max_peak_score'] < th) &
    #         (anno['visible'] == 0)
    #     )
    #
    #     modeP = (
    #         (anno['num_peaks'] > 1) &
    #         (anno['peak_dist_pred_anno'] > 4) &
    #         (anno['max_peak_score'] >= th) &
    #         (anno['visible'] == 0)
    #     )
    #
    #     # startC_idx = torch.nonzero(modeC[:-1])
    #     # startD_idx = torch.nonzero(modeD[:-1])
    #     # startG_idx = torch.nonzero(modeG[:-1])
    #     # startH_idx = torch.nonzero(modeH[:-1])
    #     # startI_idx = torch.nonzero(modeI[:-1])
    #     # startK_idx = torch.nonzero(modeK[:-1])
    #     #
    #     # start_ids = {
    #     #     'C': startC_idx, 'D': startD_idx, 'G': startG_idx, 'H': startH_idx, 'I': startI_idx, 'K': startK_idx,
    #     # }
    #
    #     modes = {
    #         'A' : modeA, 'B' : modeB, 'C' : modeC, 'D' : modeD, 'E' : modeE, 'F' : modeF, 'G' : modeG, 'H' : modeH,
    #         'I' : modeI, 'J' : modeJ, 'K' : modeK, 'L' : modeL, 'M' : modeM, 'N' : modeN, 'O' : modeO, 'P' : modeP
    #     }
    #
    #     for mode in modes:
    #         stats[mode] += torch.sum(modes[mode]).item()
    #
    #     modes_np = {key: val.numpy().astype(np.int) for key, val in modes.items()}
    #     df_modes = pd.DataFrame.from_dict(modes_np)
    #
    #     # for start in start_ids.keys():
    #     #     for stop in modes.keys():
    #     #         start_idx = start_ids[start]
    #     #         mode = modes[stop]
    #     #         mask = torch.zeros(mode.shape[0])
    #     #         mask[start_idx[mode[start_idx + 1]]] = 1
    #     #         subseq[start+stop] = mask.numpy().astype(np.int)
    #     #         stats[start+stop] += int(torch.sum(mask).item())
    #     #
    #     # df = pd.DataFrame.from_dict(subseq)
    #
    #     classname = seqname.split('-')[0]
    #     save_path = os.path.join(ROOT_PATH, classname, seqname)
    #
    #     # df.to_csv(os.path.join(save_path, 'subsequences.csv'))
    #     df_modes.to_csv(os.path.join(save_path, 'frame_states.csv'))
    #
    # for key, val in stats.items():
    #     print(key, val)
