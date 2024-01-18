import numpy as np
import cv2 as cv
from pytracking.evaluation.data import Sequence, BaseDataset, SequenceList
from pytracking.evaluation.vot2020dataset import create_mask_from_string, get_array, mask2bbox


class VOT2021Dataset(BaseDataset):
    """
    """
    def __init__(self):
        super().__init__()
        self.base_path = self.env_settings.vot2020_path
        self.sequence_list = self._get_sequence_list()

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_list])

    def parse_groundtruth_file(self, string, frame_size):
        # input is a mask - decode it
        m_, offset_ = create_mask_from_string(string[1:].split(','))
        mask = get_array(m_, offset_, frame_size)
        return mask

    def _load_groundtruth_masks(self, groundtruth_file, frame_size):
        masks = []
        with open(groundtruth_file, 'r') as groundtruth:
            for region in groundtruth.readlines():
                masks.append(self.parse_groundtruth_file(region, frame_size))

        return masks

    def _construct_sequence(self, sequence_name):
        sequence_path = sequence_name
        nz = 8
        ext = 'jpg'
        start_frame = 1

        first_frame = '{base_path}/{sequence_path}/color/{frame:0{nz}}.{ext}'.format(base_path=self.base_path,
                                                                                     sequence_path=sequence_path,
                                                                                     frame=1, nz=nz, ext=ext)
        first_frame = cv.imread(first_frame)

        frame_size = [first_frame.shape[1], first_frame.shape[0]]
        anno_path = '{}/{}/groundtruth.txt'.format(self.base_path, sequence_name)
        masks = self._load_groundtruth_masks(anno_path, frame_size)

        boxes = [mask2bbox(m) for m in masks]
        ground_truth_rect = np.array(boxes)
        end_frame = len(masks)

        frames = ['{base_path}/{sequence_path}/color/{frame:0{nz}}.{ext}'.format(base_path=self.base_path,
                  sequence_path=sequence_path, frame=frame_num, nz=nz, ext=ext)
                  for frame_num in range(start_frame, end_frame+1)]

        return Sequence(name=sequence_name, frames=frames, dataset='VOT2021', ground_truth_rect=ground_truth_rect,
                        ground_truth_seg=masks, object_ids=None,
                        multiobj_mode=False)

    def __len__(self):
        return len(self.sequence_list)

    def _get_sequence_list(self):
        with open('{base_path}/list.txt'.format(base_path=self.base_path), 'r') as f:
            sequence_list = f.read().splitlines()

        # sequence_list = ['agility',
        #                  'animal',
        #                  'ants1',
        #                  'bag',
        #                  'ball2',
        #                  'ball3',
        #                  'basketball',
        #                  'birds1',
        #                  'birds2',
        #                  'bolt1',
        #                  'book',
        #                  'butterfly',
        #                  'car1',
        #                  'conduction1',
        #                  'crabs1',
        #                  'dinosaur',
        #                  'diver',
        #                  'drone1',
        #                  'drone_across',
        #                  'fernando',
        #                  'fish1',
        #                  'fish2',
        #                  'flamingo1',
        #                  'frisbee',
        #                  'girl',
        #                  'graduate',
        #                  'gymnastics1',
        #                  'gymnastics2',
        #                  'gymnastics3',
        #                  'hand',
        #                  'hand2',
        #                  'handball1',
        #                  'handball2',
        #                  'helicopter',
        #                  'iceskater1',
        #                  'iceskater2',
        #                  'kangaroo',
        #                  'lamb',
        #                  'leaves',
        #                  'marathon',
        #                  'matrix',
        #                  'monkey',
        #                  'motocross1',
        #                  'nature',
        #                  'polo',
        #                  'rabbit',
        #                  'rabbit2',
        #                  'rowing',
        #                  'shaking',
        #                  'singer2',
        #                  'singer3',
        #                  'snake',
        #                  'soccer1',
        #                  'soccer2',
        #                  'soldier',
        #                  'surfing',
        #                  'tiger',
        #                  'wheel',
        #                  'wiper',
        #                  'zebrafish1']

        return sequence_list