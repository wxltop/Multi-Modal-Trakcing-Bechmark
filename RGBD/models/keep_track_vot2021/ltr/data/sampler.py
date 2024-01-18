import random
import itertools
import torch.utils.data
from pytracking import TensorDict
import numpy as np
import math
from torch._six import int_classes as _int_classes


def no_processing(data):
    return data


class RandomSequenceWithDistractors(torch.utils.data.Dataset):
    """
    """

    def __init__(self, datasets, p_datasets, samples_per_epoch, max_gap,
                 num_seq_test_frames, num_class_distractor_frames=0, num_random_distractor_frames=0,
                 num_seq_train_frames=1, num_class_distractor_train_frames=0, num_random_distractor_train_frames=0,
                 processing=no_processing, sample_mode='sequence',
                 frame_sample_mode='default', max_distractor_gap=9999999):

        self.use_class_info = num_class_distractor_train_frames > 0 or num_class_distractor_frames > 0
        if self.use_class_info:
            for d in datasets:
                assert d.has_class_info(), 'Dataset must have class info'

        assert num_class_distractor_frames >= num_class_distractor_train_frames, 'Cannot have >1 train frame per distractor'
        assert num_random_distractor_frames >= num_random_distractor_train_frames, 'Cannot have >1 train frame per distractor'

        self.datasets = datasets

        # If p not provided, sample uniformly from all videos
        if p_datasets is None:
            p_datasets = [len(d) for d in self.datasets]

        # Normalize
        p_total = sum(p_datasets)
        self.p_datasets = [x/p_total for x in p_datasets]

        self.samples_per_epoch = samples_per_epoch
        self.max_gap = max_gap
        self.num_seq_test_frames = num_seq_test_frames
        self.num_class_distractor_frames = num_class_distractor_frames
        self.num_random_distractor_frames = num_random_distractor_frames
        self.num_seq_train_frames = num_seq_train_frames
        self.num_class_distractor_train_frames = num_class_distractor_train_frames
        self.num_random_distractor_train_frames = num_random_distractor_train_frames
        self.processing = processing
        self.sample_mode = sample_mode
        self.frame_sample_mode = frame_sample_mode
        self.max_distractor_gap = max_distractor_gap


    def __len__(self):
        return self.samples_per_epoch


    def _sample_visible_ids(self, visible, num_ids=1, min_id=None, max_id=None):
        if num_ids == 0:
            return []
        if min_id is None or min_id < 0:
            min_id = 0
        if max_id is None or max_id > len(visible):
            max_id = len(visible)

        valid_ids = [i for i in range(min_id, max_id) if visible[i]]

        # No visible ids
        if len(valid_ids) == 0:
            return None

        return random.choices(valid_ids, k=num_ids)

    def _dict_cat(self, *dicts):
        # remove empty lists
        dicts = [d for d in dicts if isinstance(d, dict)]
        dict_cat = {}

        if len(dicts) == 0:
            return {}

        key_list = []
        for d in dicts:
            key_list += list(d.keys())
        key_list = list(set(key_list))

        for key in key_list:
            val_list = [d.get(key, []) for d in dicts]
            dict_cat[key] = list(itertools.chain.from_iterable(val_list))

        return dict_cat

    def _sample_class_distractors(self, dataset, sampled_seq, sequences, num_test_frames, num_train_frames):
        cls_dist_train_frames = []
        cls_dist_train_anno = []
        cls_dist_test_frames = []
        cls_dist_test_anno = []

        i = 0
        while i < num_test_frames:
            dist_seq_id = random.choices(sequences)[0]
            while dist_seq_id == sampled_seq:
                dist_seq_id = random.choices(sequences)[0]

            dist_seq_info_dict = dataset.get_sequence_info(dist_seq_id)
            visible = dist_seq_info_dict['visible']

            dist_train_frame_id = self._sample_visible_ids(visible)
            if dist_train_frame_id is None:
                continue

            dist_test_frame_id = self._sample_visible_ids(visible, min_id=dist_train_frame_id[0] - self.max_distractor_gap,
                                                          max_id=dist_train_frame_id[0] + self.max_distractor_gap)
            if dist_test_frame_id is None:
                continue

            frame, anno_dict, _ = dataset.get_frames(dist_seq_id, dist_test_frame_id, dist_seq_info_dict)

            cls_dist_test_frames += frame
            cls_dist_test_anno = self._dict_cat(anno_dict, cls_dist_test_anno)

            if i < num_train_frames:
                frame, anno_dict, _ = dataset.get_frames(dist_seq_id, dist_train_frame_id, dist_seq_info_dict)
                cls_dist_train_frames += frame
                cls_dist_train_anno = self._dict_cat(anno_dict, cls_dist_train_anno)

            i += 1

        return cls_dist_train_frames, cls_dist_train_anno, cls_dist_test_frames, cls_dist_test_anno

    def _sample_random_distractors(self, num_test_frames, num_train_frames):
        rnd_dist_train_frames = []
        rnd_dist_train_anno = []
        rnd_dist_test_frames = []
        rnd_dist_test_anno = []

        i = 0
        while i < num_test_frames:
            dist_dataset = random.choices(self.datasets, self.p_datasets)[0]
            dist_seq_id = random.randint(0, dist_dataset.get_num_sequences() - 1)

            dist_seq_info_dict = dist_dataset.get_sequence_info(dist_seq_id)
            visible = dist_seq_info_dict['visible']

            dist_train_frame_id = self._sample_visible_ids(visible)
            dist_test_frame_id = self._sample_visible_ids(visible)

            if dist_test_frame_id is None:
                continue
            frame, anno_dict, _ = dist_dataset.get_frames(dist_seq_id, dist_test_frame_id, dist_seq_info_dict)

            rnd_dist_test_frames += frame
            rnd_dist_test_anno = self._dict_cat(rnd_dist_test_anno, anno_dict)

            if i < num_train_frames:
                frame, anno_dict, _ = dist_dataset.get_frames(dist_seq_id, dist_train_frame_id, dist_seq_info_dict)
                rnd_dist_train_frames += frame
                rnd_dist_train_anno = self._dict_cat(rnd_dist_train_anno, anno_dict)

            i += 1

        return rnd_dist_train_frames, rnd_dist_train_anno, rnd_dist_test_frames, rnd_dist_test_anno

    def __getitem__(self, index):
        """
        Args:
            index (int): Index (Ignored since we sample randomly)

        Returns:

        """

        dataset = random.choices(self.datasets, self.p_datasets)[0]
        is_video_dataset = dataset.is_video_sequence()

        enough_visible_frames = False
        # TODO clean this part
        while not enough_visible_frames:
            # Select a class
            if self.sample_mode == 'sequence':
                while not enough_visible_frames:
                    # Sample a sequence
                    seq_id = random.randint(0, dataset.get_num_sequences() - 1)
                    # Sample frames
                    seq_info_dict = dataset.get_sequence_info(seq_id)
                    visible = seq_info_dict['visible']

                    enough_visible_frames = visible.type(torch.int64).sum().item() > 2 * (self.num_seq_test_frames + self.num_seq_train_frames) and \
                        len(visible) >= 20

                    enough_visible_frames = enough_visible_frames or not is_video_dataset
                if self.use_class_info:
                    class_name = dataset.get_class_name(seq_id)
                    class_sequences = dataset.get_sequences_in_class(class_name)
            elif self.sample_mode == 'class':
                class_name = random.choices(dataset.get_class_list())[0]
                class_sequences = dataset.get_sequences_in_class(class_name)

                # Sample test frames from the sequence
                try_ct = 0
                while not enough_visible_frames and try_ct < 5:
                    # Sample a sequence
                    seq_id = random.choices(class_sequences)[0]
                    # Sample frames
                    seq_info_dict = dataset.get_sequence_info(seq_id)
                    visible = seq_info_dict['visible']

                    # TODO probably filter sequences where we don't have enough visible frames in a pre-processing step
                    #  so that we are not stuck in a while loop
                    enough_visible_frames = visible.type(torch.int64).sum().item() > self.num_seq_test_frames + \
                                            self.num_seq_train_frames
                    enough_visible_frames = enough_visible_frames or not is_video_dataset
                    try_ct += 1
            else:
                raise ValueError

        if is_video_dataset:
            train_frame_ids = None
            test_frame_ids = None
            gap_increase = 0
            if self.frame_sample_mode == 'default':
                while test_frame_ids is None:
                    train_frame_ids = self._sample_visible_ids(visible, num_ids=self.num_seq_train_frames)
                    test_frame_ids = self._sample_visible_ids(visible, min_id=train_frame_ids[0] - self.max_gap - gap_increase,
                                                              max_id=train_frame_ids[0] + self.max_gap + gap_increase,
                                                              num_ids=self.num_seq_test_frames)
                    gap_increase += 5   # Increase gap until a frame is found
            elif self.frame_sample_mode == 'causal':
                while test_frame_ids is None:
                    base_frame_id = self._sample_visible_ids(visible, num_ids=1, min_id=self.num_seq_train_frames-1,
                                                             max_id=len(visible)-self.num_seq_test_frames)
                    prev_frame_ids = self._sample_visible_ids(visible, num_ids=self.num_seq_train_frames-1,
                                                              min_id=base_frame_id[0] - self.max_gap - gap_increase,
                                                              max_id=base_frame_id[0])
                    if prev_frame_ids is None:
                        gap_increase += 5
                        continue
                    train_frame_ids = base_frame_id + prev_frame_ids
                    test_frame_ids = self._sample_visible_ids(visible, min_id=train_frame_ids[0]+1,
                                                              max_id=train_frame_ids[0] + self.max_gap + gap_increase,
                                                              num_ids=self.num_seq_test_frames)
                    gap_increase += 5   # Increase gap until a frame is found
            else:
                raise ValueError('Unknown frame_sample_mode.')
        else:
            train_frame_ids = [1]*self.num_seq_train_frames
            test_frame_ids = [1]*self.num_seq_test_frames

        seq_train_frames, seq_train_anno, meta_obj_train = dataset.get_frames(seq_id, train_frame_ids, seq_info_dict)

        seq_test_frames, seq_test_anno, meta_obj_test = dataset.get_frames(seq_id, test_frame_ids, seq_info_dict)

        assert meta_obj_train.get('object_class_name') == meta_obj_test.get('object_class_name'), "Train and test classes don't match!!"

        # Sample from sequences with the same class
        # TODO fix sequences which do not have a single visible frame
        if self.use_class_info and len(class_sequences) > 5:
            cls_dist_train_frames, cls_dist_train_anno, cls_dist_test_frames, cls_dist_test_anno = \
                self._sample_class_distractors(dataset, seq_id, class_sequences, self.num_class_distractor_frames,
                self.num_class_distractor_train_frames)
            num_rnd_distractors = self.num_random_distractor_frames
            num_rnd_train_distractors = self.num_random_distractor_train_frames
        else:
            cls_dist_train_frames = []
            cls_dist_train_anno = []
            cls_dist_test_frames = []
            cls_dist_test_anno = []
            num_rnd_distractors = self.num_random_distractor_frames + self.num_class_distractor_frames
            num_rnd_train_distractors = self.num_random_distractor_train_frames + self.num_class_distractor_train_frames

        # Sample sequences from any class
        rnd_dist_train_frames, rnd_dist_train_anno, rnd_dist_test_frames, rnd_dist_test_anno = \
            self._sample_random_distractors(num_rnd_distractors, num_rnd_train_distractors)

        train_frames = seq_train_frames + cls_dist_train_frames + rnd_dist_train_frames
        test_frames = seq_test_frames + cls_dist_test_frames + rnd_dist_test_frames

        train_anno = self._dict_cat(seq_train_anno, cls_dist_train_anno, rnd_dist_train_anno)
        test_anno = self._dict_cat(seq_test_anno, cls_dist_test_anno, rnd_dist_test_anno)

        is_distractor_train_frame = [False]*self.num_seq_train_frames + \
                                    [True]*(self.num_class_distractor_train_frames + self.num_random_distractor_train_frames)
        is_distractor_test_frame = [False]*self.num_seq_test_frames + [True]*(self.num_class_distractor_frames +
                                                                       self.num_random_distractor_frames)


        # TODO send in class name for each frame
        data = TensorDict({'train_images': train_frames,
                           'train_anno': train_anno['bbox'],
                           'test_images': test_frames,
                           'test_anno': test_anno['bbox'],
                           'dataset': dataset.get_name(),
                           'is_distractor_train_frame': is_distractor_train_frame,
                           'is_distractor_test_frame': is_distractor_test_frame})

        meta_keys = ['motion_class', 'major_class', 'root_class', 'motion_adverb', 'object_class_name', 'object_ids']
        for key in meta_keys:
            if key in meta_obj_train:
                data[key] = meta_obj_train.get(key)

        # Not supported by tracking processing
        # 'train_mask': train_anno.get('mask', [None]*len(train_frames)),
        # 'test_mask': test_anno.get('mask', [None]*len(test_frames))})

        return self.processing(data)


class TrackingSampler(torch.utils.data.Dataset):
    """ Class responsible for sampling frames from training sequences to form batches. Each training sample is a
    tuple consisting of i) a set of train frames, used to learn the DiMP classification model and obtain the
    modulation vector for IoU-Net, and ii) a set of test frames on which target classification loss for the predicted
    DiMP model, and the IoU prediction loss for the IoU-Net is calculated.

    The sampling is done in the following ways. First a dataset is selected at random. Next, a sequence is selected
    from that dataset. A base frame is then sampled randomly from the sequence. Next, a set of 'train frames' and
    'test frames' are sampled from the sequence from the range [base_frame_id - max_gap, base_frame_id]  and
    (base_frame_id, base_frame_id + max_gap] respectively. Only the frames in which the target is visible are sampled.
    If enough visible frames are not found, the 'max_gap' is increased gradually till enough frames are found.

    The sampled frames are then passed through the input 'processing' function for the necessary processing-
    """

    def __init__(self, datasets, p_datasets, samples_per_epoch, max_gap,
                 num_test_frames, num_train_frames=1, processing=no_processing, frame_sample_mode='causal'):
        """
        args:
            datasets - List of datasets to be used for training
            p_datasets - List containing the probabilities by which each dataset will be sampled
            samples_per_epoch - Number of training samples per epoch
            max_gap - Maximum gap, in frame numbers, between the train frames and the test frames.
            num_test_frames - Number of test frames to sample.
            num_train_frames - Number of train frames to sample.
            processing - An instance of Processing class which performs the necessary processing of the data.
            frame_sample_mode - Either 'causal' or 'interval'. If 'causal', then the test frames are sampled in a causally,
                                otherwise randomly within the interval.
        """
        self.datasets = datasets

        # If p not provided, sample uniformly from all videos
        if p_datasets is None:
            p_datasets = [len(d) for d in self.datasets]

        # Normalize
        p_total = sum(p_datasets)
        self.p_datasets = [x/p_total for x in p_datasets]

        self.samples_per_epoch = samples_per_epoch
        self.max_gap = max_gap
        self.num_test_frames = num_test_frames
        self.num_train_frames = num_train_frames
        self.processing = processing
        self.frame_sample_mode = frame_sample_mode

    def __len__(self):
        return self.samples_per_epoch

    def _sample_visible_ids(self, visible, num_ids=1, min_id=None, max_id=None):
        """ Samples num_ids frames between min_id and max_id for which target is visible

        args:
            visible - 1d Tensor indicating whether target is visible for each frame
            num_ids - number of frames to be samples
            min_id - Minimum allowed frame number
            max_id - Maximum allowed frame number

        returns:
            list - List of sampled frame numbers. None if not sufficient visible frames could be found.
        """
        if num_ids == 0:
            return []
        if min_id is None or min_id < 0:
            min_id = 0
        if max_id is None or max_id > len(visible):
            max_id = len(visible)

        valid_ids = [i for i in range(min_id, max_id) if visible[i]]

        # No visible ids
        if len(valid_ids) == 0:
            return None

        return random.choices(valid_ids, k=num_ids)

    def __getitem__(self, index):
        """
        args:
            index (int): Index (Ignored since we sample randomly)

        returns:
            TensorDict - dict containing all the data blocks
        """

        # Select a dataset
        dataset = random.choices(self.datasets, self.p_datasets)[0]
        is_video_dataset = dataset.is_video_sequence()

        # Sample a sequence with enough visible frames
        enough_visible_frames = False
        while not enough_visible_frames:
            # Sample a sequence
            seq_id = random.randint(0, dataset.get_num_sequences() - 1)

            # Sample frames
            seq_info_dict = dataset.get_sequence_info(seq_id)
            visible = seq_info_dict['visible']

            enough_visible_frames = visible.type(torch.int64).sum().item() > 2 * (
                    self.num_test_frames + self.num_train_frames) and len(visible) >= 20

            enough_visible_frames = enough_visible_frames or not is_video_dataset

        if is_video_dataset:
            train_frame_ids = None
            test_frame_ids = None
            gap_increase = 0

            if self.frame_sample_mode == 'interval':
                # Sample frame numbers within interval defined by the first frame
                while test_frame_ids is None:
                    base_frame_id = self._sample_visible_ids(visible, num_ids=1)
                    extra_train_frame_ids = self._sample_visible_ids(visible, num_ids=self.num_train_frames - 1,
                                                              min_id=base_frame_id[0] - self.max_gap - gap_increase,
                                                              max_id=base_frame_id[0] + self.max_gap + gap_increase)
                    if extra_train_frame_ids is None:
                        gap_increase += 5
                        continue
                    train_frame_ids = base_frame_id + extra_train_frame_ids
                    test_frame_ids = self._sample_visible_ids(visible, num_ids=self.num_test_frames,
                                                              min_id=train_frame_ids[0] - self.max_gap - gap_increase,
                                                              max_id=train_frame_ids[0] + self.max_gap + gap_increase)
                    gap_increase += 5   # Increase gap until a frame is found

            elif self.frame_sample_mode == 'causal':
                # Sample test and train frames in a causal manner, i.e. test_frame_ids > train_frame_ids
                while test_frame_ids is None:
                    base_frame_id = self._sample_visible_ids(visible, num_ids=1, min_id=self.num_train_frames - 1,
                                                             max_id=len(visible)-self.num_test_frames)
                    prev_frame_ids = self._sample_visible_ids(visible, num_ids=self.num_train_frames - 1,
                                                              min_id=base_frame_id[0] - self.max_gap - gap_increase,
                                                              max_id=base_frame_id[0])
                    if prev_frame_ids is None:
                        gap_increase += 5
                        continue
                    train_frame_ids = base_frame_id + prev_frame_ids
                    test_frame_ids = self._sample_visible_ids(visible, min_id=train_frame_ids[0]+1,
                                                              max_id=train_frame_ids[0] + self.max_gap + gap_increase,
                                                              num_ids=self.num_test_frames)
                    # Increase gap until a frame is found
                    gap_increase += 5
        else:
            # In case of image dataset, just repeat the image to generate synthetic video
            train_frame_ids = [1]*self.num_train_frames
            test_frame_ids = [1]*self.num_test_frames

        train_frames, train_anno, meta_obj_train = dataset.get_frames(seq_id, train_frame_ids, seq_info_dict)
        test_frames, test_anno, meta_obj_test = dataset.get_frames(seq_id, test_frame_ids, seq_info_dict)

        data = TensorDict({'train_images': train_frames,
                           'train_anno': train_anno['bbox'],
                           'test_images': test_frames,
                           'test_anno': test_anno['bbox'],
                           'dataset': dataset.get_name(),
                           'test_class': meta_obj_test.get('object_class_name')})

        return self.processing(data)



class DiMPSampler(TrackingSampler):
    """ See TrackingSampler."""
    def __init__(self, datasets, p_datasets, samples_per_epoch, max_gap,
                 num_test_frames, num_train_frames=1, processing=no_processing, frame_sample_mode='causal'):
        super().__init__(datasets=datasets, p_datasets=p_datasets, samples_per_epoch=samples_per_epoch, max_gap=max_gap,
                         num_test_frames=num_test_frames, num_train_frames=num_train_frames, processing=processing,
                         frame_sample_mode=frame_sample_mode)

class ATOMSampler(TrackingSampler):
    """ See TrackingSampler."""
    def __init__(self, datasets, p_datasets, samples_per_epoch, max_gap,
                 num_test_frames=1, num_train_frames=1, processing=no_processing, frame_sample_mode='interval'):
        super().__init__(datasets=datasets, p_datasets=p_datasets, samples_per_epoch=samples_per_epoch, max_gap=max_gap,
                         num_test_frames=num_test_frames, num_train_frames=num_train_frames, processing=processing,
                         frame_sample_mode=frame_sample_mode)


# For motion stuff
class SubSequenceSampler(torch.utils.data.Dataset):
    def __init__(self, datasets, p_datasets, samples_per_epoch, sequence_sample_info, processing=no_processing,
                 sample_occluded_sequences=False):
        self.datasets = datasets

        # If p not provided, sample uniformly from all videos
        if p_datasets is None:
            p_datasets = [1 for d in self.datasets]

        # Normalize
        p_total = sum(p_datasets)
        self.p_datasets = [x/p_total for x in p_datasets]

        self.samples_per_epoch = samples_per_epoch
        self.sequence_sample_info = sequence_sample_info
        self.processing = processing

        self.sample_occluded_sequences = sample_occluded_sequences

    def __len__(self):
        return self.samples_per_epoch

    def _sample_ids(self, valid, num_ids=1, min_id=None, max_id=None):
        """ Samples num_ids frames between min_id and max_id for which target is visible

        args:
            visible - 1d Tensor indicating whether target is visible for each frame
            num_ids - number of frames to be samples
            min_id - Minimum allowed frame number
            max_id - Maximum allowed frame number

        returns:
            list - List of sampled frame numbers. None if not sufficient visible frames could be found.
        """
        if min_id is None or min_id < 0:
            min_id = 0
        if max_id is None or max_id > len(valid):
            max_id = len(valid)

        valid_ids = [i for i in range(min_id, max_id) if valid[i]]

        # No visible ids
        if len(valid_ids) == 0:
            return None

        return random.choices(valid_ids, k=num_ids)

    def find_occlusion_end_frame(self, first_occ_frame, target_not_fully_visible):
        for i in range(first_occ_frame, len(target_not_fully_visible)):
            if not target_not_fully_visible[i]:
                return i

        return len(target_not_fully_visible)

    def load_synseq(self, dataset):
        num_train_frames = self.sequence_sample_info['num_train_frames']
        num_test_frames = self.sequence_sample_info['num_test_frames']

        seq_id = random.randint(0, dataset.get_num_sequences() - 1)

        train_frame_ids = random.choices(list(range(0, 6)), k=num_train_frames)
        test_frame_base = random.choice(list(range(0, 64 - num_test_frames + 1)))
        test_frame_ids = [test_frame_base + i_ for i_ in range(0, num_test_frames)]

        # Get frames
        train_frames, train_anno_dict, _ = dataset.get_train_frames(seq_id, train_frame_ids)
        train_anno = train_anno_dict['bbox']

        test_frames, test_anno_dict, _ = dataset.get_test_frames(seq_id, test_frame_ids)
        test_anno = test_anno_dict['bbox']
        test_valid_anno = test_anno_dict['valid']
        test_visible = test_anno_dict['visible']
        test_visible_ratio = test_anno_dict.get('visible_ratio', torch.ones(len(test_visible)))

        test_valid_image = torch.ones(num_test_frames, dtype=torch.int8)

        # Prepare data
        data = TensorDict({'train_images': train_frames,
                           'train_anno': train_anno,
                           'test_images': test_frames,
                           'test_anno': test_anno,
                           'test_valid_anno': test_valid_anno,
                           'test_visible': test_visible,
                           'test_valid_image': test_valid_image,
                           'test_visible_ratio': test_visible_ratio,
                           'dataset': dataset.get_name()})

        # Send for processing
        return self.processing(data)

    def __getitem__(self, index):
        """
        args:
            index (int): Index (Ignored since we sample randomly)

        returns:
            TensorDict - dict containing all the data blocks
        """

        # Select a dataset
        p_datasets = self.p_datasets

        dataset = random.choices(self.datasets, p_datasets)[0]

        if dataset.get_name() == 'synseqv2':
            return self.load_synseq(dataset)

        is_video_dataset = dataset.is_video_sequence()

        num_train_frames = self.sequence_sample_info['num_train_frames']
        num_test_frames = self.sequence_sample_info['num_test_frames']
        max_train_gap = self.sequence_sample_info['max_train_gap']
        allow_missing_target = self.sequence_sample_info['allow_missing_target']
        min_fraction_valid_frames = self.sequence_sample_info.get('min_fraction_valid_frames', 0.0)

        if allow_missing_target:
            min_visible_frames = 0
        else:
            raise NotImplementedError
            # min_visible_frames = 2 * (num_train_frames + num_test_frames)

        valid_sequence = False

        # Sample a sequence with enough visible frames and get anno for the same
        while not valid_sequence:
            seq_id = random.randint(0, dataset.get_num_sequences() - 1)

            seq_info_dict = dataset.get_sequence_info(seq_id)
            visible = seq_info_dict['visible']
            valid_frames = seq_info_dict['valid']
            visible_ratio = seq_info_dict.get('visible_ratio', visible)

            num_visible = visible.type(torch.int64).sum().item()

            enough_visible_frames = not is_video_dataset or (num_visible > min_visible_frames and len(visible) >= 20)

            valid_sequence = enough_visible_frames

        if self.sequence_sample_info['mode'] == 'Sequence':
            if is_video_dataset:
                train_frame_ids = None
                test_frame_ids = None
                gap_increase = 0

                test_valid_image = torch.zeros(num_test_frames, dtype=torch.int8)
                # Sample frame numbers in a causal manner, i.e. test_frame_ids > train_frame_ids
                while test_frame_ids is None:
                    occlusion_sampling = False
                    if dataset.has_occlusion_info() and self.sample_occluded_sequences:
                        target_not_fully_visible = visible_ratio < 0.9
                        if target_not_fully_visible.float().sum() > 0:
                            occlusion_sampling = True

                    if occlusion_sampling:
                        first_occ_frame = target_not_fully_visible.nonzero()[0]

                        occ_end_frame = self.find_occlusion_end_frame(first_occ_frame, target_not_fully_visible)

                        # Make sure target visible in first frame
                        base_frame_id = self._sample_ids(visible, num_ids=1, min_id=max(0, first_occ_frame - 20),
                                                         max_id=first_occ_frame - 5)

                        if base_frame_id is None:
                            base_frame_id = 0
                        else:
                            base_frame_id = base_frame_id[0]

                        prev_frame_ids = self._sample_ids(visible, num_ids=num_train_frames,
                                                          min_id=base_frame_id - max_train_gap - gap_increase - 1,
                                                          max_id=base_frame_id - 1)

                        if prev_frame_ids is None:
                            if base_frame_id - max_train_gap - gap_increase - 1 < 0:
                                prev_frame_ids = [base_frame_id] * num_train_frames
                            else:
                                gap_increase += 5
                                continue

                        train_frame_ids = prev_frame_ids

                        end_frame = min(occ_end_frame + random.randint(5, 20), len(visible) - 1)

                        if (end_frame - base_frame_id) < num_test_frames:
                            rem_frames = num_test_frames - (end_frame - base_frame_id)
                            end_frame = random.randint(end_frame, min(len(visible) - 1, end_frame + rem_frames))
                            base_frame_id = max(0, end_frame - num_test_frames + 1)

                            end_frame = min(end_frame, len(visible) - 1)

                        step_len = float(end_frame - base_frame_id) / float(num_test_frames)

                        test_frame_ids = [base_frame_id + int(x * step_len) for x in range(0, num_test_frames)]
                        test_valid_image[:len(test_frame_ids)] = 1

                        test_frame_ids = test_frame_ids + [0] * (num_test_frames - len(test_frame_ids))
                    else:
                        # Make sure target visible in first frame
                        base_frame_id = self._sample_ids(visible, num_ids=1, min_id=2*num_train_frames,
                                                         max_id=len(visible) - int(num_test_frames * min_fraction_valid_frames))
                        if base_frame_id is None:
                            base_frame_id = 0
                        else:
                            base_frame_id = base_frame_id[0]

                        prev_frame_ids = self._sample_ids(visible, num_ids=num_train_frames,
                                                          min_id=base_frame_id - max_train_gap - gap_increase - 1,
                                                          max_id=base_frame_id - 1)
                        if prev_frame_ids is None:
                            if base_frame_id - max_train_gap - gap_increase - 1 < 0:
                                prev_frame_ids = [base_frame_id] * num_train_frames
                            else:
                                gap_increase += 5
                                continue

                        train_frame_ids = prev_frame_ids

                        test_frame_ids = list(range(base_frame_id, min(len(visible), base_frame_id + num_test_frames)))
                        test_valid_image[:len(test_frame_ids)] = 1

                        test_frame_ids = test_frame_ids + [0]*(num_test_frames - len(test_frame_ids))
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

        # Get frames
        train_frames, train_anno_dict, _ = dataset.get_frames(seq_id, train_frame_ids, seq_info_dict)
        train_anno = train_anno_dict['bbox']

        test_frames, test_anno_dict, _ = dataset.get_frames(seq_id, test_frame_ids, seq_info_dict)
        test_anno = test_anno_dict['bbox']
        test_valid_anno = test_anno_dict['valid']
        test_visible = test_anno_dict['visible']
        test_visible_ratio = test_anno_dict.get('visible_ratio', torch.ones(len(test_visible)))

        # Prepare data
        data = TensorDict({'train_images': train_frames,
                           'train_anno': train_anno,
                           'test_images': test_frames,
                           'test_anno': test_anno,
                           'test_valid_anno': test_valid_anno,
                           'test_visible': test_visible,
                           'test_valid_image': test_valid_image,
                           'test_visible_ratio': test_visible_ratio,
                           'dataset': dataset.get_name()})

        # Send for processing
        return self.processing(data)


class SegmDiMPSampler(torch.utils.data.Dataset):
    """ Class responsible for sampling frames from training sequences to form batches. Each training sample is a
    tuple consisting of i) a set of train frames, used to learn the DiMP classification model and obtain the
    modulation vector for IoU-Net, and ii) a set of test frames on which target classification loss for the predicted
    DiMP model, and the IoU prediction loss for the IoU-Net is calculated.

    The sampling is done in the following ways. First a dataset is selected at random. Next, a sequence is selected
    from that dataset. A base frame is then sampled randomly from the sequence. Next, a set of 'train frames' and
    'test frames' are sampled from the sequence from the range [base_frame_id - max_gap, base_frame_id]  and
    (base_frame_id, base_frame_id + max_gap] respectively. Only the frames in which the target is visible are sampled.
    If enough visible frames are not found, the 'max_gap' is increased gradually till enough frames are found.

    The sampled frames are then passed through the input 'processing' function for the necessary processing-
    """

    def __init__(self, datasets, p_datasets, samples_per_epoch, max_gap,
                 num_test_frames, num_train_frames=1, processing=no_processing, p_reverse=None):
        """
        args:
            datasets - List of datasets to be used for training
            p_datasets - List containing the probabilities by which each dataset will be sampled
            samples_per_epoch - Number of training samples per epoch
            max_gap - Maximum gap, in frame numbers, between the train frames and the test frames.
            num_test_frames - Number of test frames to sample.
            num_train_frames - Number of train frames to sample.
            processing - An instance of Processing class which performs the necessary processing of the data.
        """
        self.datasets = datasets

        # If p not provided, sample uniformly from all videos
        if p_datasets is None:
            p_datasets = [len(d) for d in self.datasets]

        # Normalize
        p_total = sum(p_datasets)
        self.p_datasets = [x/p_total for x in p_datasets]

        self.samples_per_epoch = samples_per_epoch
        self.max_gap = max_gap
        self.num_test_frames = num_test_frames
        self.num_train_frames = num_train_frames
        self.processing = processing

        self.p_reverse = p_reverse

    def __len__(self):
        return self.samples_per_epoch

    def _sample_visible_ids(self, visible, num_ids=1, min_id=None, max_id=None):
        """ Samples num_ids frames between min_id and max_id for which target is visible

        args:
            visible - 1d Tensor indicating whether target is visible for each frame
            num_ids - number of frames to be samples
            min_id - Minimum allowed frame number
            max_id - Maximum allowed frame number

        returns:
            list - List of sampled frame numbers. None if not sufficient visible frames could be found.
        """
        if min_id is None or min_id < 0:
            min_id = 0
        if max_id is None or max_id > len(visible):
            max_id = len(visible)

        valid_ids = [i for i in range(min_id, max_id) if visible[i]]

        # No visible ids
        if len(valid_ids) == 0:
            return None

        return random.choices(valid_ids, k=num_ids)

    def __getitem__(self, index):
        """
        args:
            index (int): Index (dataset index)

        returns:
            TensorDict - dict containing all the data blocks
        """

        # Select a dataset
        # TODO ensure that the dataset can either be used independently, or wrapped with batch sampler
        # dataset = self.datasets[index]
        dataset = random.choices(self.datasets, self.p_datasets)[0]

        is_video_dataset = dataset.is_video_sequence()

        reverse_sequence = False
        if self.p_reverse is not None:
            reverse_sequence = random.random() < self.p_reverse

        # Sample a sequence with enough visible frames
        enough_visible_frames = False
        while not enough_visible_frames:
            # Sample a sequence
            seq_id = random.randint(0, dataset.get_num_sequences() - 1)

            # Sample frames
            seq_info_dict = dataset.get_sequence_info(seq_id)
            visible = seq_info_dict['visible']

            enough_visible_frames = visible.type(torch.int64).sum().item() > 2 * (
                    self.num_test_frames + self.num_train_frames)

            enough_visible_frames = enough_visible_frames or not is_video_dataset

        if is_video_dataset:
            train_frame_ids = None
            test_frame_ids = None
            gap_increase = 0

            # Sample test and train frames in a causal manner, i.e. test_frame_ids > train_frame_ids
            while test_frame_ids is None:
                if gap_increase > 1000:
                    raise Exception('Frame not found')

                if not reverse_sequence:
                    base_frame_id = self._sample_visible_ids(visible, num_ids=1, min_id=self.num_train_frames - 1,
                                                             max_id=len(visible)-self.num_test_frames)
                    prev_frame_ids = self._sample_visible_ids(visible, num_ids=self.num_train_frames - 1,
                                                              min_id=base_frame_id[0] - self.max_gap - gap_increase,
                                                              max_id=base_frame_id[0])
                    if prev_frame_ids is None:
                        gap_increase += 5
                        continue
                    train_frame_ids = base_frame_id + prev_frame_ids
                    test_frame_ids = self._sample_visible_ids(visible, min_id=train_frame_ids[0]+1,
                                                              max_id=train_frame_ids[0] + self.max_gap + gap_increase,
                                                              num_ids=self.num_test_frames)

                    # Increase gap until a frame is found
                    gap_increase += 5
                else:
                    base_frame_id = self._sample_visible_ids(visible, num_ids=1, min_id=self.num_test_frames + 1,
                                                             max_id=len(visible) - self.num_train_frames - 1)
                    prev_frame_ids = self._sample_visible_ids(visible, num_ids=self.num_train_frames - 1,
                                                              min_id=base_frame_id[0],
                                                              max_id=base_frame_id[0] + self.max_gap + gap_increase)
                    if prev_frame_ids is None:
                        gap_increase += 5
                        continue
                    train_frame_ids = base_frame_id + prev_frame_ids
                    test_frame_ids = self._sample_visible_ids(visible, min_id=0,
                                                              max_id=train_frame_ids[0] - 1,
                                                              num_ids=self.num_test_frames)

                    # Increase gap until a frame is found
                    gap_increase += 5
        else:
            # In case of image dataset, just repeat the image to generate synthetic video
            train_frame_ids = [1]*self.num_train_frames
            test_frame_ids = [1]*self.num_test_frames

        test_frame_ids = sorted(test_frame_ids, reverse=reverse_sequence)
        all_frame_ids = train_frame_ids + test_frame_ids

        all_frames, all_anno, meta_obj = dataset.get_frames(seq_id, all_frame_ids, seq_info_dict)

        train_frames = all_frames[:len(train_frame_ids)]
        test_frames = all_frames[len(train_frame_ids):]

        train_anno = {}
        test_anno = {}
        for key, value in all_anno.items():
            train_anno[key] = value[:len(train_frame_ids)]
            test_anno[key] = value[len(train_frame_ids):]

        train_masks = train_anno['mask'] if 'mask' in train_anno else None
        test_masks = test_anno['mask'] if 'mask' in test_anno else None

        data = TensorDict({'train_images': train_frames,
                           'train_masks': train_masks,
                           'train_anno': train_anno['bbox'],
                           'test_images': test_frames,
                           'test_masks': test_masks,
                           'test_anno': test_anno['bbox'],
                           'dataset': dataset.get_name()})

        return self.processing(data)


class LWTLSampler(torch.utils.data.Dataset):
    """ Class responsible for sampling frames from training sequences to form batches. Each training sample is a
    tuple consisting of i) a set of train frames and ii) a set of test frames. The train frames, along with the
    ground-truth masks, are passed to the few-shot learner to obtain the target model parameters \tau. The test frames
    are used to compute the prediction accuracy.

    The sampling is done in the following ways. First a dataset is selected at random. Next, a sequence is randomly
    selected from that dataset. A base frame is then sampled randomly from the sequence. The 'train frames'
    are then sampled from the sequence from the range [base_frame_id - max_gap, base_frame_id], and the 'test frames'
    are sampled from the sequence from the range (base_frame_id, base_frame_id + max_gap] respectively. Only the frames
    in which the target is visible are sampled. If enough visible frames are not found, the 'max_gap' is increased
    gradually until enough frames are found. Both the 'train frames' and the 'test frames' are sorted to preserve the
    temporal order.

    The sampled frames are then passed through the input 'processing' function for the necessary processing-
    """

    def __init__(self, datasets, p_datasets, samples_per_epoch, max_gap,
                 num_test_frames, num_train_frames=1, processing=no_processing, p_reverse=None):
        """
        args:
            datasets - List of datasets to be used for training
            p_datasets - List containing the probabilities by which each dataset will be sampled
            samples_per_epoch - Number of training samples per epoch
            max_gap - Maximum gap, in frame numbers, between the train frames and the test frames.
            num_test_frames - Number of test frames to sample.
            num_train_frames - Number of train frames to sample.
            processing - An instance of Processing class which performs the necessary processing of the data.
            p_reverse - Probability that a sequence is temporally reversed
        """
        self.datasets = datasets

        # If p not provided, sample uniformly from all videos
        if p_datasets is None:
            p_datasets = [len(d) for d in self.datasets]

        # Normalize
        p_total = sum(p_datasets)
        self.p_datasets = [x/p_total for x in p_datasets]

        self.samples_per_epoch = samples_per_epoch
        self.max_gap = max_gap
        self.num_test_frames = num_test_frames
        self.num_train_frames = num_train_frames
        self.processing = processing

        self.p_reverse = p_reverse

    def __len__(self):
        return self.samples_per_epoch

    def _sample_visible_ids(self, visible, num_ids=1, min_id=None, max_id=None):
        """ Samples num_ids frames between min_id and max_id for which target is visible

        args:
            visible - 1d Tensor indicating whether target is visible for each frame
            num_ids - number of frames to be samples
            min_id - Minimum allowed frame number
            max_id - Maximum allowed frame number

        returns:
            list - List of sampled frame numbers. None if not sufficient visible frames could be found.
        """
        if min_id is None or min_id < 0:
            min_id = 0
        if max_id is None or max_id > len(visible):
            max_id = len(visible)

        valid_ids = [i for i in range(min_id, max_id) if visible[i]]

        # No visible ids
        if len(valid_ids) == 0:
            return None

        return random.choices(valid_ids, k=num_ids)

    def __getitem__(self, index):
        """
        args:
            index (int): Index (dataset index)

        returns:
            TensorDict - dict containing all the data blocks
        """

        # Select a dataset
        dataset = random.choices(self.datasets, self.p_datasets)[0]

        is_video_dataset = dataset.is_video_sequence()

        reverse_sequence = False
        if self.p_reverse is not None:
            reverse_sequence = random.random() < self.p_reverse

        # Sample a sequence with enough visible frames
        enough_visible_frames = False
        while not enough_visible_frames:
            # Sample a sequence
            seq_id = random.randint(0, dataset.get_num_sequences() - 1)

            # Sample frames
            seq_info_dict = dataset.get_sequence_info(seq_id)
            visible = seq_info_dict['visible']

            enough_visible_frames = visible.type(torch.int64).sum().item() > 2 * (
                    self.num_test_frames + self.num_train_frames)

            enough_visible_frames = enough_visible_frames or not is_video_dataset

        if is_video_dataset:
            train_frame_ids = None
            test_frame_ids = None
            gap_increase = 0

            # Sample test and train frames in a causal manner, i.e. test_frame_ids > train_frame_ids
            while test_frame_ids is None:
                if gap_increase > 1000:
                    raise Exception('Frame not found')

                if not reverse_sequence:
                    base_frame_id = self._sample_visible_ids(visible, num_ids=1, min_id=self.num_train_frames - 1,
                                                             max_id=len(visible)-self.num_test_frames)
                    prev_frame_ids = self._sample_visible_ids(visible, num_ids=self.num_train_frames - 1,
                                                              min_id=base_frame_id[0] - self.max_gap - gap_increase,
                                                              max_id=base_frame_id[0])
                    if prev_frame_ids is None:
                        gap_increase += 5
                        continue
                    train_frame_ids = base_frame_id + prev_frame_ids
                    test_frame_ids = self._sample_visible_ids(visible, min_id=train_frame_ids[0]+1,
                                                              max_id=train_frame_ids[0] + self.max_gap + gap_increase,
                                                              num_ids=self.num_test_frames)

                    # Increase gap until a frame is found
                    gap_increase += 5
                else:
                    # Sample in reverse order, i.e. train frames come after the test frames
                    base_frame_id = self._sample_visible_ids(visible, num_ids=1, min_id=self.num_test_frames + 1,
                                                             max_id=len(visible) - self.num_train_frames - 1)
                    prev_frame_ids = self._sample_visible_ids(visible, num_ids=self.num_train_frames - 1,
                                                              min_id=base_frame_id[0],
                                                              max_id=base_frame_id[0] + self.max_gap + gap_increase)
                    if prev_frame_ids is None:
                        gap_increase += 5
                        continue
                    train_frame_ids = base_frame_id + prev_frame_ids
                    test_frame_ids = self._sample_visible_ids(visible, min_id=0,
                                                              max_id=train_frame_ids[0] - 1,
                                                              num_ids=self.num_test_frames)

                    # Increase gap until a frame is found
                    gap_increase += 5
        else:
            # In case of image dataset, just repeat the image to generate synthetic video
            train_frame_ids = [1]*self.num_train_frames
            test_frame_ids = [1]*self.num_test_frames

        # Sort frames
        train_frame_ids = sorted(train_frame_ids, reverse=reverse_sequence)
        test_frame_ids = sorted(test_frame_ids, reverse=reverse_sequence)

        all_frame_ids = train_frame_ids + test_frame_ids

        # Load frames
        all_frames, all_anno, meta_obj = dataset.get_frames(seq_id, all_frame_ids, seq_info_dict)

        train_frames = all_frames[:len(train_frame_ids)]
        test_frames = all_frames[len(train_frame_ids):]

        train_anno = {}
        test_anno = {}
        for key, value in all_anno.items():
            train_anno[key] = value[:len(train_frame_ids)]
            test_anno[key] = value[len(train_frame_ids):]

        train_masks = train_anno['mask'] if 'mask' in train_anno else None
        test_masks = test_anno['mask'] if 'mask' in test_anno else None

        data = TensorDict({'train_images': train_frames,
                           'train_masks': train_masks,
                           'train_anno': train_anno['bbox'],
                           'test_images': test_frames,
                           'test_masks': test_masks,
                           'test_anno': test_anno['bbox'],
                           'dataset': dataset.get_name()})

        return self.processing(data)


class BatchGroupSampler(torch.utils.data.sampler.Sampler):
    r"""Yield a mini-batch of indices. Drops the last indices.

    Args:
        batch_size (int): Size of mini-batch.
        p_datasets
        dataset_groups
    """

    def __init__(self, num_samples, batch_size, p_datasets, g_datasets, shuffle=True):
        if not isinstance(batch_size, _int_classes) or isinstance(batch_size, bool) or \
                batch_size <= 0:
            raise ValueError("batch_size should be a positive integer value, "
                             "but got batch_size={}".format(batch_size))
        self.num_samples = num_samples
        self.batch_size = batch_size

        # Normalize
        p_total = sum(p_datasets)
        self.p_datasets = [x / p_total for x in p_datasets]
        self.n_datasets = [math.ceil(num_samples * p) for p in p_datasets]

        self.g_datasets = g_datasets
        self.groups_ids = np.unique(self.g_datasets).tolist()

        self.shuffle = shuffle

    def __iter__(self):
        # Build dataset indices based on p_datasets, and keep them grouped using g_datasets
        ids_grouped = [
            [id for id, (n_id, g_id) in enumerate(zip(self.n_datasets, self.g_datasets)) if g_id == group_id for _ in range(n_id)]
            for group_id in self.groups_ids
        ]

        # Shuffle dataset indices within each group
        if self.shuffle:
            for ids in ids_grouped:
                random.shuffle(ids)

        # Build chunks of size batch_size within each group and drop last if needed
        ids_chunks = [
            ids[i:i+self.batch_size]
            for ids in ids_grouped
            for i in range(0, len(ids) - len(ids) % self.batch_size, self.batch_size)
        ]

        # Shuffle all the chunks
        if self.shuffle:
            random.shuffle(ids_chunks)

        return iter(ids_chunks)

    def __len__(self):
        return self.num_samples // self.batch_size


class SegmBatchSampler(BatchGroupSampler):
    def __init__(self, sampler: SegmDiMPSampler, batch_size, shuffle=True):
        # Builds BatchGroupSampler by grouping datasets that have segmentation info

        g_datasets = [int(d.has_segmentation_info()) for d in sampler.datasets]

        super(SegmBatchSampler, self).__init__(sampler.samples_per_epoch, batch_size, sampler.p_datasets,
                                               g_datasets, shuffle)


class SegmMultiObjSampler(torch.utils.data.Dataset):
    def __init__(self, datasets, p_datasets, samples_per_epoch, max_gap,
                 num_test_frames, num_train_frames=1, processing=no_processing):
        """
        args:
            datasets - List of datasets to be used for training
            p_datasets - List containing the probabilities by which each dataset will be sampled
            samples_per_epoch - Number of training samples per epoch
            max_gap - Maximum gap, in frame numbers, between the train frames and the test frames.
            num_test_frames - Number of test frames to sample.
            num_train_frames - Number of train frames to sample.
            processing - An instance of Processing class which performs the necessary processing of the data.
        """
        self.datasets = datasets

        # If p not provided, sample uniformly from all videos
        if p_datasets is None:
            p_datasets = [len(d) for d in self.datasets]

        # Normalize
        p_total = sum(p_datasets)
        self.p_datasets = [x/p_total for x in p_datasets]

        self.samples_per_epoch = samples_per_epoch
        self.max_gap = max_gap
        self.num_test_frames = num_test_frames
        self.num_train_frames = num_train_frames
        self.processing = processing

    def __len__(self):
        return self.samples_per_epoch

    def _sample_visible_ids(self, visible, num_ids=1, min_id=None, max_id=None):
        """ Samples num_ids frames between min_id and max_id for which target is visible

        args:
            visible - 1d Tensor indicating whether target is visible for each frame
            num_ids - number of frames to be samples
            min_id - Minimum allowed frame number
            max_id - Maximum allowed frame number

        returns:
            list - List of sampled frame numbers. None if not sufficient visible frames could be found.
        """
        if min_id is None or min_id < 0:
            min_id = 0
        if max_id is None or max_id > len(visible):
            max_id = len(visible)

        valid_ids = [i for i in range(min_id, max_id) if visible[i]]

        # No visible ids
        if len(valid_ids) == 0:
            return None

        return random.choices(valid_ids, k=num_ids)

    def __getitem__(self, index):
        """
        args:
            index (int): Index (dataset index)

        returns:
            TensorDict - dict containing all the data blocks
        """

        # Select a dataset
        # TODO ensure that the dataset can either be used independently, or wrapped with batch sampler
        # dataset = self.datasets[index]
        dataset = random.choices(self.datasets, self.p_datasets)[0]

        is_video_dataset = dataset.is_video_sequence()

        # Sample a sequence with enough visible frames
        enough_visible_frames = False
        while not enough_visible_frames:
            # Sample a sequence
            seq_id = random.randint(0, dataset.get_num_sequences() - 1)

            # Sample frames
            seq_info_dict = dataset.get_sequence_info(seq_id)

            seq_length = len(seq_info_dict['frame_names'])
            visible = seq_info_dict['visible']

            visible_all = visible.view(seq_length, -1).all(dim=1)
            visible_some = visible.view(seq_length, -1).any(dim=1)

            enough_visible_frames = (visible_all.type(torch.int64).sum().item() > self.num_train_frames) and \
                                    (visible_some.type(torch.int64).sum().item() > (self.num_train_frames + self.num_test_frames)) and \
                                     seq_length >= (self.num_train_frames + self.num_test_frames)

            enough_visible_frames = enough_visible_frames or not is_video_dataset

        if is_video_dataset:
            train_frame_ids = None
            test_frame_ids = None
            gap_increase = 0

            # Sample test and train frames in a causal manner, i.e. test_frame_ids > train_frame_ids
            try_ct = 0
            while test_frame_ids is None:
                try_ct = try_ct + 1

                if try_ct > 100:
                    raise Exception('Could not sample frames')


                base_frame_id = self._sample_visible_ids(visible_all, num_ids=1, min_id=self.num_train_frames - 1,
                                                         max_id=len(visible_all)-self.num_test_frames)

                if base_frame_id is None:
                    base_frame_id = self._sample_visible_ids(visible_some, num_ids=1, min_id=self.num_train_frames - 1,
                                                             max_id=len(visible_some) - self.num_test_frames)

                if base_frame_id is None:
                    base_frame_id = self._sample_visible_ids(visible_some, num_ids=1, min_id=0,
                                                             max_id=len(visible_some))

                prev_frame_ids = self._sample_visible_ids(visible_some, num_ids=self.num_train_frames - 1,
                                                          min_id=base_frame_id[0] - self.max_gap - gap_increase,
                                                          max_id=base_frame_id[0])
                if prev_frame_ids is None:
                    gap_increase += 5
                    continue
                train_frame_ids = base_frame_id + prev_frame_ids
                test_frame_ids = self._sample_visible_ids(visible_some, min_id=train_frame_ids[0]+1,
                                                          max_id=train_frame_ids[0] + self.max_gap + gap_increase,
                                                          num_ids=self.num_test_frames)

                # Increase gap until a frame is found
                gap_increase += 5
        else:
            # In case of image dataset, just repeat the image to generate synthetic video
            train_frame_ids = [1]*self.num_train_frames
            test_frame_ids = [1]*self.num_test_frames

        test_frame_ids = sorted(test_frame_ids)
        all_frame_ids = train_frame_ids + test_frame_ids

        all_frames, all_anno, meta_obj = dataset.get_frames(seq_id, all_frame_ids, seq_info_dict)

        train_frames = all_frames[:len(train_frame_ids)]
        test_frames = all_frames[len(train_frame_ids):]

        train_anno = {}
        test_anno = {}
        for key, value in all_anno.items():
            train_anno[key] = value[:len(train_frame_ids)]
            test_anno[key] = value[len(train_frame_ids):]

        train_masks = train_anno['mask'] if 'mask' in train_anno else None
        test_masks = test_anno['mask'] if 'mask' in test_anno else None

        data = TensorDict({'train_images': train_frames,
                           'train_masks': train_masks,
                           'train_anno': train_anno['bbox'],
                           'test_images': test_frames,
                           'test_masks': test_masks,
                           'test_anno': test_anno['bbox'],
                           'object_ids': seq_info_dict['object_ids'],
                           'dataset': dataset.get_name()})

        return self.processing(data)


class FewShotSegSampler(torch.utils.data.Dataset):
    def __init__(self, datasets, p_datasets, samples_per_epoch, num_train_frames, num_test_frames, num_classes=1,
                 processing=no_processing):
        self.datasets = datasets

        # If p not provided, sample uniformly from all videos
        if p_datasets is None:
            p_datasets = [len(d) for d in self.datasets]

        # Normalize
        p_total = sum(p_datasets)
        self.p_datasets = [x/p_total for x in p_datasets]

        self.samples_per_epoch = samples_per_epoch
        self.num_test_frames = num_test_frames
        self.num_train_frames = num_train_frames
        self.num_classes = num_classes
        self.processing = processing

    def convert_mask(self, mask, selected_class_ids):
        mask_out = torch.zeros_like(mask)
        mask = mask.long()
        for i, cls_id in enumerate(selected_class_ids):
            mask_out[mask == cls_id] = i + 1.0

        return mask_out

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, index):
        dataset = random.choices(self.datasets, self.p_datasets)[0]

        # Sample a class
        selected_classes = random.sample(dataset.get_class_list(), k=self.num_classes)

        selected_image_ids = []

        # Sample images
        total_num_images_per_class = self.num_train_frames + self.num_test_frames
        for cls in selected_classes:
            num_images_required = total_num_images_per_class

            for try_ct in range(100):
                ids = random.sample(dataset.get_images_in_class(cls), k=num_images_required)

                if len(ids) != len(set(ids)):
                    raise Exception

                # Ensure images are not repeated
                ids = [ii for ii in ids if ii not in selected_image_ids]
                selected_image_ids = selected_image_ids + ids
                num_images_required -= len(ids)

                if num_images_required == 0:
                    break

        all_frames, all_anno, _ = dataset.get_images(selected_image_ids)

        # Generate masks
        selected_class_ids = [dataset.get_class_id(cls_i) for cls_i in selected_classes]

        all_masks = [self.convert_mask(ann['semantic_mask'], selected_class_ids) for ann in all_anno]

        # Select train images for each class
        train_frames = []
        test_frames = []
        train_masks = []
        test_masks = []

        for cls_i in range(self.num_classes):
            train_start_id = cls_i*total_num_images_per_class
            train_end_id = train_start_id + self.num_train_frames

            train_frames += all_frames[train_start_id:train_end_id]
            test_frames += all_frames[train_end_id:train_end_id + self.num_test_frames]

            train_masks += all_masks[train_start_id:train_end_id]
            test_masks += all_masks[train_end_id:train_end_id + self.num_test_frames]

        data = TensorDict({'train_images': train_frames,
                           'train_masks': train_masks,
                           'test_images': test_frames,
                           'test_masks': test_masks,
                           'dataset': dataset.get_name()})

        return self.processing(data)

    # For leaning weights stuff
class ContiuousSubSequenceSampler(torch.utils.data.Dataset):
    def __init__(self, datasets, p_datasets, samples_per_epoch, sequence_sample_info, processing=no_processing,
                 sample_occluded_sequences=False, sample_test_frames_contiuously=True):
        self.datasets = datasets

        # If p not provided, sample uniformly from all videos
        if p_datasets is None:
            p_datasets = [1 for _ in self.datasets]

        # Normalize
        p_total = sum(p_datasets)
        self.p_datasets = [x / p_total for x in p_datasets]

        self.samples_per_epoch = samples_per_epoch
        self.sequence_sample_info = sequence_sample_info
        self.processing = processing

        self.sample_occluded_sequences = sample_occluded_sequences
        self.sample_test_frames_contiuously = sample_test_frames_contiuously

    def __len__(self):
        return self.samples_per_epoch

    def _sample_ids(self, valid, num_ids=1, min_id=None, max_id=None, continuous=False, stop_idx=None):
        """ Samples num_ids frames between min_id and max_id for which target is visible

        args:
            visible - 1d Tensor indicating whether target is visible for each frame
            num_ids - number of frames to be sampled
            min_id - Minimum allowed frame number
            max_id - Maximum allowed frame number

        returns:
            list - List of sampled frame numbers. None if not sufficient visible frames could be found.
        """
        if min_id is None or min_id < 0:
            min_id = 0
        if max_id is None or max_id > len(valid):
            max_id = len(valid)

        valid_ids = [i for i in range(min_id, max_id) if valid[i]]

        # No visible ids
        if len(valid_ids) == 0:
            return None

        if continuous:
            if stop_idx is not None:
                available_ids = valid_ids[:stop_idx+1]

                if len(available_ids) >= num_ids:
                    sample_ids = available_ids[-num_ids:]
                else:
                    sample_ids = random.choices(available_ids, k=num_ids)

            else:
                available_ids = valid_ids

                if len(available_ids) >= num_ids:
                    sample_ids = available_ids[:num_ids]
                elif max_id == len(valid):
                    sample_ids = random.choices(available_ids, k=num_ids)
                else:
                    return None
        else:
            sample_ids = random.choices(valid_ids, k=num_ids)

        return sample_ids

    def find_occlusion_end_frame(self, first_occ_frame, target_visible):
        for i in range(first_occ_frame, len(target_visible)):
            if target_visible[i]:
                return i

        return len(target_visible)

    def load_synseq(self, dataset):
        num_train_frames = self.sequence_sample_info['num_train_frames']
        num_test_frames = self.sequence_sample_info['num_test_frames']

        seq_id = random.randint(0, dataset.get_num_sequences() - 1)

        train_frame_ids = random.choices(list(range(0, 6)), k=num_train_frames)
        test_frame_base = random.choice(list(range(0, 64 - num_test_frames + 1)))
        test_frame_ids = [test_frame_base + i_ for i_ in range(0, num_test_frames)]

        # Get frames
        train_frames, train_anno_dict, _ = dataset.get_train_frames(seq_id, train_frame_ids)
        train_anno = train_anno_dict['bbox']

        test_frames, test_anno_dict, _ = dataset.get_test_frames(seq_id, test_frame_ids)
        test_anno = test_anno_dict['bbox']
        test_valid_anno = test_anno_dict['valid']
        test_visible = test_anno_dict['visible']
        test_visible_ratio = test_anno_dict.get('visible_ratio', torch.ones(len(test_visible)))

        test_valid_image = torch.ones(num_test_frames, dtype=torch.int8)

        # Prepare data
        data = TensorDict({'train_images': train_frames,
                           'train_anno': train_anno,
                           'test_images': test_frames,
                           'test_anno': test_anno,
                           'test_valid_anno': test_valid_anno,
                           'test_visible': test_visible,
                           'test_valid_image': test_valid_image,
                           'test_visible_ratio': test_visible_ratio,
                           'dataset': dataset.get_name()})

        # Send for processing
        return self.processing(data)

    def __getitem__(self, index):
        """
        args:
            index (int): Index (Ignored since we sample randomly)

        returns:
            TensorDict - dict containing all the data blocks
        """

        # Select a dataset
        p_datasets = self.p_datasets

        dataset = random.choices(self.datasets, p_datasets)[0]

        if dataset.get_name() == 'synseqv2':
            return self.load_synseq(dataset)

        is_video_dataset = dataset.is_video_sequence()

        num_train_frames = self.sequence_sample_info['num_train_frames']
        num_train_frames_pool = self.sequence_sample_info['num_train_frames_pool']
        num_test_frames_pool = self.sequence_sample_info['num_test_frames_pool']
        num_test_frames = self.sequence_sample_info['num_test_frames']
        allow_missing_target = self.sequence_sample_info['allow_missing_target']
        min_fraction_valid_frames = self.sequence_sample_info.get('min_fraction_valid_frames', 0.0)

        valid_sequence = False

        # Sample a sequence with enough visible frames and get anno for the same
        counter = 0
        while not valid_sequence:
            seq_id = random.randint(0, dataset.get_num_sequences() - 1)

            seq_info_dict = dataset.get_sequence_info(seq_id)
            visible = seq_info_dict['visible']

            num_visible = visible.type(torch.int64).sum().item()

            enough_visible_frames = (num_visible > (num_train_frames+num_test_frames)/min_fraction_valid_frames)

            valid_sequence = enough_visible_frames and is_video_dataset


        if self.sequence_sample_info['mode'] == 'Sequence':
            if is_video_dataset:

                train_frame_ids = None
                test_frame_ids = None
                gap_increase = 0

                while test_frame_ids is None:

                    base_frame_id = self._sample_ids(visible, num_ids=1, min_id=num_train_frames_pool - 1,
                                                     max_id=len(visible) - num_test_frames_pool)


                    prev_frame_ids = self._sample_ids(visible, num_ids=num_train_frames_pool - 1, min_id=0,
                                                      max_id=base_frame_id[0], stop_idx=base_frame_id[0]-1,
                                                      continuous=True)

                    if prev_frame_ids is None:
                        continue

                    if num_train_frames_pool == num_train_frames:
                        train_frame_ids = prev_frame_ids + base_frame_id
                    elif num_train_frames_pool > num_train_frames:
                        x = np.round(np.linspace(0, len(prev_frame_ids)-1, num_test_frames - 1), decimals=0).astype(int)
                        train_frame_ids = [prev_frame_ids[i] for i in x] + base_frame_id
                    else:
                        raise ValueError()

                    test_frame_ids = self._sample_ids(visible, min_id=base_frame_id[0] + 1,
                                                      max_id=base_frame_id[0] + 1 + num_test_frames_pool + gap_increase,
                                                      num_ids=num_test_frames_pool, continuous=True)

                    if test_frame_ids is None:
                        continue

                    if num_test_frames_pool == num_test_frames:
                        test_frame_ids = test_frame_ids
                    elif num_test_frames_pool > num_test_frames:
                        intermediate_ids = np.sort(np.random.choice(test_frame_ids[1:-1], size=num_test_frames-2, replace=False)).tolist()
                        test_frame_ids = [test_frame_ids[0]] + intermediate_ids + [test_frame_ids[-1]]
                    else:
                        raise ValueError()

                    # Increase gap until a frame is found
                    gap_increase += 5
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

        # Get frames
        train_frames, train_anno_dict, _ = dataset.get_frames(seq_id, train_frame_ids, seq_info_dict)
        train_anno = train_anno_dict['bbox']

        test_frames, test_anno_dict, _ = dataset.get_frames(seq_id, test_frame_ids, seq_info_dict)
        test_anno = test_anno_dict['bbox']
        test_valid_anno = test_anno_dict['valid']
        test_visible = test_anno_dict['visible']
        test_visible_ratio = test_anno_dict.get('visible_ratio', torch.ones(len(test_visible)))

        # Prepare data
        data = TensorDict({'train_images': train_frames,
                           'train_anno': train_anno,
                           'test_images': test_frames,
                           'test_anno': test_anno,
                           'test_valid_anno': test_valid_anno,
                           'test_visible': test_visible,
                           'test_visible_ratio': test_visible_ratio,
                           'dataset': dataset.get_name()})

        # Send for processing
        return self.processing(data)


class DumpedDataSampler(torch.utils.data.Dataset):
    def __init__(self, datasets, p_datasets, samples_per_epoch, max_gap,
                 num_test_frames, num_train_frames=0, processing=no_processing, frame_sample_mode='interval'):
        """
        args:
            datasets - List of datasets to be used for training
            p_datasets - List containing the probabilities by which each dataset will be sampled
            samples_per_epoch - Number of training samples per epoch
            max_gap - Maximum gap, in frame numbers, between the train frames and the test frames.
            num_test_frames - Number of test frames to sample.
            num_train_frames - Number of train frames to sample.
            processing - An instance of Processing class which performs the necessary processing of the data.
            frame_sample_mode - Either 'causal' or 'interval'. If 'causal', then the test frames are sampled in a causally,
                                otherwise randomly within the interval.
        """
        assert (num_train_frames == 0) # We don't need train frames for this
        self.datasets = datasets

        # If p not provided, sample uniformly from all videos
        if p_datasets is None:
            p_datasets = [len(d) for d in self.datasets]

        # Normalize
        p_total = sum(p_datasets)
        self.p_datasets = [x/p_total for x in p_datasets]

        self.samples_per_epoch = samples_per_epoch
        self.max_gap = max_gap
        self.num_test_frames = num_test_frames
        self.num_train_frames = num_train_frames
        self.processing = processing
        self.frame_sample_mode = frame_sample_mode

    def __len__(self):
        return self.samples_per_epoch

    def _sample_valid_ids(self, visible, num_ids=1, min_id=None, max_id=None):
        """ Samples num_ids frames between min_id and max_id for which dumped data is useful

        args:
            visible - 1d Tensor indicating whether target is visible for each frame
            num_ids - number of frames to be sampled
            min_id - Minimum allowed frame number
            max_id - Maximum allowed frame number

        returns:
            list - List of sampled frame numbers. None if not sufficient visible frames could be found.
        """
        if num_ids == 0:
            return []
        if min_id is None or min_id < 2:
            min_id = 2
        if max_id is None or max_id > len(visible):
            max_id = len(visible)

        valid_ids = [i for i in range(min_id, max_id) if visible[i]]

        # No visible ids
        if len(valid_ids) == 0:
            return None

        num_begin = num_ids//2
        num_end = num_ids - num_ids//2
        ids_begin = random.sample(valid_ids[:len(valid_ids)//2], k=num_begin)
        ids_end = random.sample(valid_ids[len(valid_ids)//2:], k=num_end)
        return ids_begin + ids_end

    def __getitem__(self, index):
        """
        args:
            index (int): Index (Ignored since we sample randomly)

        returns:
            TensorDict - dict containing all the data blocks
        """

        # Select a dataset
        dataset = random.choices(self.datasets, self.p_datasets)[0]
        is_video_dataset = dataset.is_video_sequence()

        # Sample a sequence with enough visible frames
        enough_valid_frames = False
        while not enough_valid_frames:
            # Sample a sequence
            seq_id = random.randint(0, dataset.get_num_sequences() - 1)

            # Sample frames
            seq_info_dict = dataset.get_sequence_info(seq_id)
            critical_frames = seq_info_dict['critical_frames'].bool()

            valid = critical_frames

            enough_valid_frames = valid.type(torch.int64).sum().item() > self.num_test_frames
            enough_valid_frames = enough_valid_frames or not is_video_dataset

        if is_video_dataset:
            test_frame_ids = None

            if self.frame_sample_mode == 'interval':
                # Sample frame numbers within interval defined by the first frame
                if valid is None:
                    print(valid)
                    exit()
                test_frame_ids = self._sample_valid_ids(valid, num_ids=self.num_test_frames)

        # print(len(test_frame_ids))
        frames_dict, _ = dataset.get_frames(seq_id, test_frame_ids, seq_info_dict)

        data = TensorDict({'dataset': dataset.get_name()})

        for key, val in frames_dict.items():
            data[key] = val

        # m = []
        # for i, val in enumerate(data['target_label']):
        #     num_dim = len(val.numpy().shape)
        #     if num_dim == 1:
        #         print(seq_id, test_frame_ids[i], val.numpy().mean(), val.numpy().max(), val.numpy().min())


        return self.processing(data)


class FullContinuousSequenceDumpedDataSampler(torch.utils.data.Dataset):
    def __init__(self, datasets, allframes=True, processing=no_processing):
        """
        args:
            datasets - List of datasets to be used for training
            processing - An instance of Processing class which performs the necessary processing of the data.
        """
        self.datasets = datasets
        self.processing = processing
        self.allframes = allframes

        self.samples_per_epoch = sum([len(d) for d in self.datasets])

    def get_dataset_and_seq_id_from_index(self, index):
        lengths = [len(d) for d in self.datasets]

        if len(lengths) == 1:
            out = (0, index)
        else:
            raise NotImplementedError()

        return out

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, index):
        """
        args:
            index (int): Index (Ignored since we sample randomly)

        returns:
            TensorDict - dict containing all the data blocks
        """

        data_index, seq_id = self.get_dataset_and_seq_id_from_index(index)
        dataset = self.datasets[data_index]

        seq_info_dict = dataset.get_sequence_info(seq_id)
        num_frames = seq_info_dict['bbox'].shape[0]

        if self.allframes:
            frame_ids = range(2, num_frames)
        else:
            frame_ids = np.where(seq_info_dict['critical_frames'])[0]
            frame_ids = list(frame_ids[frame_ids >= 2])

            if len(frame_ids) == 0: frame_ids = [2]

        frames_dict, _ = dataset.get_frames(seq_id, frame_ids, seq_info_dict)

        data = TensorDict({'dataset': dataset.get_name(), 'seq_name': dataset.get_sequence_name(seq_id)})

        for key, val in frames_dict.items():
            data[key] = val

        return self.processing(data)


class ContinuousDumpedDataSampler(torch.utils.data.Dataset):
    def __init__(self, dataset, samples_per_epoch, num_test_frames, proc_modes, p_proc_modes=None,
                 num_train_frames=1, processing=no_processing, subseq_modes=None, p_subseq_modes=None,
                 subseq_states=None, frame_states=None, frame_modes=None, p_frame_modes=None):
        """
        args:
            datasets - List of datasets to be used for training
            p_datasets - List containing the probabilities by which each dataset will be sampled
            samples_per_epoch - Number of training samples per epoch
            max_gap - Maximum gap, in frame numbers, between the train frames and the test frames.
            num_test_frames - Number of test frames to sample.
            num_train_frames - Number of train frames to sample.
            processing - An instance of Processing class which performs the necessary processing of the data.
            frame_sample_mode - Either 'causal' or 'interval'. If 'causal', then the test frames are sampled in a causally,
                                otherwise randomly within the interval.
        """
        self.dataset = dataset
        self.samples_per_epoch = samples_per_epoch
        self.num_test_frames = num_test_frames
        self.num_train_frames = num_train_frames
        self.processing = processing
        self.subseq_modes = subseq_modes
        self.frame_modes = frame_modes
        self.proc_modes = proc_modes if proc_modes is not None else ['aug']
        self.p_proc_modes = p_proc_modes

        if p_proc_modes is None:
            self.p_proc_modes = [1. / len(self.proc_modes)] * (len(self.proc_modes))

        if subseq_modes is not None:
            if subseq_states is None:
                self.dataset_subseq_states = self._load_dataset_subsequence_states()
            else:
                self.dataset_subseq_states = subseq_states.copy()

            if p_subseq_modes is None:
                p_subseq_modes = [self.dataset_subseq_states[mode].shape[0] for mode in self.subseq_modes]

            # Normalize
            p_subseq_total = sum(p_subseq_modes)
            self.p_subseq_modes = [x / p_subseq_total for x in p_subseq_modes]

        if frame_modes is not None:
            if frame_states is None:
                self.dataset_frame_states = self._load_dataset_frame_states()
            else:
                self.dataset_frame_states = frame_states.copy()

            if p_frame_modes is None:
                p_frame_modes = [self.dataset_frame_states[mode].shape[0] for mode in self.frame_modes]

            # Normalize
            p_frames_total = sum(p_frame_modes)
            self.p_frame_modes = [x / p_frames_total for x in p_frame_modes]

    def __len__(self):
        return self.samples_per_epoch

    def _load_dataset_subsequence_states(self):
        return self.dataset.build_subsequence_states()

    def _load_dataset_frame_states(self):
        return self.dataset.build_frame_states()

    def _sample_valid_ids(self, visible, num_ids=1, min_id=None, max_id=None):
        """ Samples num_ids frames between min_id and max_id for which dumped data is useful

        args:
            visible - 1d Tensor indicating whether target is visible for each frame
            num_ids - number of frames to be sampled
            min_id - Minimum allowed frame number
            max_id - Maximum allowed frame number

        returns:
            list - List of sampled frame numbers. None if not sufficient visible frames could be found.
        """
        if num_ids == 0:
            return []
        if min_id is None or min_id < 2:
            min_id = 2
        if max_id is None or max_id > len(visible):
            max_id = len(visible)

        valid_ids = [i for i in range(min_id, max_id) if visible[i]]

        # No visible ids
        if len(valid_ids) == 0:
            return None

        num_begin = num_ids//2
        num_end = num_ids - num_ids//2
        ids_begin = random.sample(valid_ids[:len(valid_ids)//2], k=num_begin)
        ids_end = random.sample(valid_ids[len(valid_ids)//2:], k=num_end)
        return ids_begin + ids_end


    def __getitem__(self, index):
        """
        args:
            index (int): Index (Ignored since we sample randomly).

        returns:
            TensorDict - dict containing all the data blocks
        """

        # select a subseq mode
        proc_mode = random.choices(self.proc_modes, self.p_proc_modes, k=1)[0]

        if proc_mode == 'aug':
            mode = random.choices(self.frame_modes, self.p_frame_modes, k=1)[0]

            states = self.dataset_frame_states[mode]
            state = random.choices(states, k=1)[0]
            seq_id = state[0].item()
            baseframe_id = state[1].item()
            test_frame_ids = [baseframe_id]

        else:
            mode = random.choices(self.subseq_modes, self.p_subseq_modes, k=1)[0]

            states = self.dataset_subseq_states[mode]
            state = random.choices(states, k=1)[0]
            seq_id = state[0].item()
            baseframe_id = state[1].item()
            test_frame_ids = [baseframe_id, baseframe_id + 1]


        seq_info_dict = self.dataset.get_sequence_info(seq_id)

        frames_dict, _ = self.dataset.get_frames(seq_id, test_frame_ids, seq_info_dict)

        data = TensorDict({
            'dataset': self.dataset.get_name(),
            'mode': mode,
            'seq_name': self.dataset.get_sequence_name(seq_id),
            'base_frame_id': baseframe_id,
            'proc_mode': proc_mode
        })

        for key, val in frames_dict.items():
            data[key] = val

        return self.processing(data)


class KYSSampler(torch.utils.data.Dataset):
    def __init__(self, datasets, p_datasets, samples_per_epoch, sequence_sample_info, processing=no_processing,
                 sample_occluded_sequences=False):
        """
        args:
            datasets - List of datasets to be used for training
            p_datasets - List containing the probabilities by which each dataset will be sampled
            samples_per_epoch - Number of training samples per epoch
            sequence_sample_info - A dict containing information about how to sample a sequence, e.g. number of frames,
                                    max gap between frames, etc.
            processing - An instance of Processing class which performs the necessary processing of the data.
            sample_occluded_sequences - If true, sub-sequence containing occlusion is sampled whenever possible
        """

        self.datasets = datasets

        # If p not provided, sample uniformly from all videos
        if p_datasets is None:
            p_datasets = [1 for d in self.datasets]

        # Normalize
        p_total = sum(p_datasets)
        self.p_datasets = [x/p_total for x in p_datasets]

        self.samples_per_epoch = samples_per_epoch
        self.sequence_sample_info = sequence_sample_info
        self.processing = processing

        self.sample_occluded_sequences = sample_occluded_sequences

    def __len__(self):
        return self.samples_per_epoch

    def _sample_ids(self, valid, num_ids=1, min_id=None, max_id=None):
        """ Samples num_ids frames between min_id and max_id for which target is visible

        args:
            visible - 1d Tensor indicating whether target is visible for each frame
            num_ids - number of frames to be samples
            min_id - Minimum allowed frame number
            max_id - Maximum allowed frame number

        returns:
            list - List of sampled frame numbers. None if not sufficient visible frames could be found.
        """
        if min_id is None or min_id < 0:
            min_id = 0
        if max_id is None or max_id > len(valid):
            max_id = len(valid)

        valid_ids = [i for i in range(min_id, max_id) if valid[i]]

        # No visible ids
        if len(valid_ids) == 0:
            return None

        return random.choices(valid_ids, k=num_ids)

    def find_occlusion_end_frame(self, first_occ_frame, target_not_fully_visible):
        for i in range(first_occ_frame, len(target_not_fully_visible)):
            if not target_not_fully_visible[i]:
                return i

        return len(target_not_fully_visible)

    def __getitem__(self, index):
        """
        args:
            index (int): Index (Ignored since we sample randomly)

        returns:
            TensorDict - dict containing all the data blocks
        """

        # Select a dataset
        p_datasets = self.p_datasets

        dataset = random.choices(self.datasets, p_datasets)[0]
        is_video_dataset = dataset.is_video_sequence()

        num_train_frames = self.sequence_sample_info['num_train_frames']
        num_test_frames = self.sequence_sample_info['num_test_frames']
        max_train_gap = self.sequence_sample_info['max_train_gap']
        allow_missing_target = self.sequence_sample_info['allow_missing_target']
        min_fraction_valid_frames = self.sequence_sample_info.get('min_fraction_valid_frames', 0.0)

        if allow_missing_target:
            min_visible_frames = 0
        else:
            raise NotImplementedError

        valid_sequence = False

        # Sample a sequence with enough visible frames and get anno for the same
        while not valid_sequence:
            seq_id = random.randint(0, dataset.get_num_sequences() - 1)

            seq_info_dict = dataset.get_sequence_info(seq_id)
            visible = seq_info_dict['visible']
            visible_ratio = seq_info_dict.get('visible_ratio', visible)

            num_visible = visible.type(torch.int64).sum().item()

            enough_visible_frames = not is_video_dataset or (num_visible > min_visible_frames and len(visible) >= 20)

            valid_sequence = enough_visible_frames

        if self.sequence_sample_info['mode'] == 'Sequence':
            if is_video_dataset:
                train_frame_ids = None
                test_frame_ids = None
                gap_increase = 0

                test_valid_image = torch.zeros(num_test_frames, dtype=torch.int8)
                # Sample frame numbers in a causal manner, i.e. test_frame_ids > train_frame_ids
                while test_frame_ids is None:
                    occlusion_sampling = False
                    if dataset.has_occlusion_info() and self.sample_occluded_sequences:
                        target_not_fully_visible = visible_ratio < 0.9
                        if target_not_fully_visible.float().sum() > 0:
                            occlusion_sampling = True

                    if occlusion_sampling:
                        first_occ_frame = target_not_fully_visible.nonzero()[0]

                        occ_end_frame = self.find_occlusion_end_frame(first_occ_frame, target_not_fully_visible)

                        # Make sure target visible in first frame
                        base_frame_id = self._sample_ids(visible, num_ids=1, min_id=max(0, first_occ_frame - 20),
                                                         max_id=first_occ_frame - 5)

                        if base_frame_id is None:
                            base_frame_id = 0
                        else:
                            base_frame_id = base_frame_id[0]

                        prev_frame_ids = self._sample_ids(visible, num_ids=num_train_frames,
                                                          min_id=base_frame_id - max_train_gap - gap_increase - 1,
                                                          max_id=base_frame_id - 1)

                        if prev_frame_ids is None:
                            if base_frame_id - max_train_gap - gap_increase - 1 < 0:
                                prev_frame_ids = [base_frame_id] * num_train_frames
                            else:
                                gap_increase += 5
                                continue

                        train_frame_ids = prev_frame_ids

                        end_frame = min(occ_end_frame + random.randint(5, 20), len(visible) - 1)

                        if (end_frame - base_frame_id) < num_test_frames:
                            rem_frames = num_test_frames - (end_frame - base_frame_id)
                            end_frame = random.randint(end_frame, min(len(visible) - 1, end_frame + rem_frames))
                            base_frame_id = max(0, end_frame - num_test_frames + 1)

                            end_frame = min(end_frame, len(visible) - 1)

                        step_len = float(end_frame - base_frame_id) / float(num_test_frames)

                        test_frame_ids = [base_frame_id + int(x * step_len) for x in range(0, num_test_frames)]
                        test_valid_image[:len(test_frame_ids)] = 1

                        test_frame_ids = test_frame_ids + [0] * (num_test_frames - len(test_frame_ids))
                    else:
                        # Make sure target visible in first frame
                        base_frame_id = self._sample_ids(visible, num_ids=1, min_id=2*num_train_frames,
                                                         max_id=len(visible) - int(num_test_frames * min_fraction_valid_frames))
                        if base_frame_id is None:
                            base_frame_id = 0
                        else:
                            base_frame_id = base_frame_id[0]

                        prev_frame_ids = self._sample_ids(visible, num_ids=num_train_frames,
                                                          min_id=base_frame_id - max_train_gap - gap_increase - 1,
                                                          max_id=base_frame_id - 1)
                        if prev_frame_ids is None:
                            if base_frame_id - max_train_gap - gap_increase - 1 < 0:
                                prev_frame_ids = [base_frame_id] * num_train_frames
                            else:
                                gap_increase += 5
                                continue

                        train_frame_ids = prev_frame_ids

                        test_frame_ids = list(range(base_frame_id, min(len(visible), base_frame_id + num_test_frames)))
                        test_valid_image[:len(test_frame_ids)] = 1

                        test_frame_ids = test_frame_ids + [0]*(num_test_frames - len(test_frame_ids))
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

        # Get frames
        train_frames, train_anno_dict, _ = dataset.get_frames(seq_id, train_frame_ids, seq_info_dict)
        train_anno = train_anno_dict['bbox']

        test_frames, test_anno_dict, _ = dataset.get_frames(seq_id, test_frame_ids, seq_info_dict)
        test_anno = test_anno_dict['bbox']
        test_valid_anno = test_anno_dict['valid']
        test_visible = test_anno_dict['visible']
        test_visible_ratio = test_anno_dict.get('visible_ratio', torch.ones(len(test_visible)))

        # Prepare data
        data = TensorDict({'train_images': train_frames,
                           'train_anno': train_anno,
                           'test_images': test_frames,
                           'test_anno': test_anno,
                           'test_valid_anno': test_valid_anno,
                           'test_visible': test_visible,
                           'test_valid_image': test_valid_image,
                           'test_visible_ratio': test_visible_ratio,
                           'dataset': dataset.get_name()})

        # Send for processing
        return self.processing(data)


# if __name__ == '__main__':
#     import matplotlib.pyplot as plt
#     from ltr.dataset import LasotDumped
#     from pytracking import dcf
#     from ltr.models.memory_learning.peak_prediction import find_local_maxima
#
#
#     dataset = LasotDumped(split='train')
#
#     sampler = ContinuousDumpedDataSampler(dataset=dataset, samples_per_epoch=2000, num_test_frames=1,
#                                           num_train_frames=1, modes=['HK'], p_modes=[1.])
#     for i in range(0, 10):
#         data = sampler[i]
#         scores_old = data['target_scores'][0].view(23,23)
#         scores_cur = data['target_scores'][1].view(23,23)
#         anno_old = data['anno_label'][0].view(23, 23)
#         anno_cur = data['anno_label'][1].view(23, 23)
#
#         print('anno_old', dcf.max2d(anno_old))
#         print('anno_cur', dcf.max2d(anno_cur))
#         print('scores_old', find_local_maxima(scores_old.view(1,1,23,23), 0.05, 5))
#         print('scores_cur', find_local_maxima(scores_cur.view(1,1,23,23), 0.05, 5))
#
#         fig, ax = plt.subplots(2,2,figsize=(8,8))
#         ax[0,0].imshow(scores_old)
#         ax[1,0].imshow(scores_cur)
#         ax[0,1].imshow(anno_old)
#         ax[1,1].imshow(anno_cur)
#         plt.show()
#         print()

