import numpy as np
from PIL import Image

import torch
import torch.utils.data as data

from modules.sample_generator import SampleGenerator
from modules.utils import crop_image2


class RegionDataset(data.Dataset):
    def __init__(self, img_list_vis, img_list_event, gt, opts):
        self.img_list_vis = np.asarray(img_list_vis)
        self.img_list_event = np.asarray(img_list_event)
        self.gt = gt

        self.batch_frames = opts['batch_frames']
        self.batch_pos = opts['batch_pos']
        self.batch_neg = opts['batch_neg']

        self.overlap_pos = opts['overlap_pos']
        self.overlap_neg = opts['overlap_neg']

        self.crop_size = opts['img_size']
        self.padding = opts['padding']

        self.flip = opts.get('flip', False)
        self.rotate = opts.get('rotate', 0)
        self.blur = opts.get('blur', 0)

        self.index = np.random.permutation(len(self.img_list_vis))
        self.pointer = 0

        image_vis = Image.open(self.img_list_vis[0]).convert('RGB')
        image_event = Image.open(self.img_list_event[0]).convert('RGB')

        self.pos_generator = SampleGenerator('uniform', image_vis.size, opts['trans_pos'], opts['scale_pos'])
        self.neg_generator = SampleGenerator('uniform', image_vis.size, opts['trans_neg'], opts['scale_neg'])

    def __iter__(self):
        return self

    def __next__(self):
        next_pointer = min(self.pointer + self.batch_frames, len(self.img_list_vis))
        idx = self.index[self.pointer:next_pointer]
        if len(idx) < self.batch_frames:
            self.index = np.random.permutation(len(self.img_list_vis))
            next_pointer = self.batch_frames - len(idx)
            idx = np.concatenate((idx, self.index[:next_pointer]))
        self.pointer = next_pointer

        pos_regions_vis = np.empty((0, 3, self.crop_size, self.crop_size), dtype='float32')
        neg_regions_vis = np.empty((0, 3, self.crop_size, self.crop_size), dtype='float32')

        pos_regions_event = np.empty((0, 3, self.crop_size, self.crop_size), dtype='float32')
        neg_regions_event = np.empty((0, 3, self.crop_size, self.crop_size), dtype='float32')

        for i, (img_path_vis, img_path_event, bbox) in enumerate(zip(self.img_list_vis[idx], self.img_list_event[idx], self.gt[idx])):
            image_vis = Image.open(img_path_vis).convert('RGB')
            image_vis = np.asarray(image_vis)

            image_event = Image.open(img_path_event).convert('RGB')
            image_event = np.asarray(image_event)

            n_pos = (self.batch_pos - len(pos_regions_vis)) // (self.batch_frames - i)
            n_neg = (self.batch_neg - len(neg_regions_vis)) // (self.batch_frames - i)
            pos_examples = self.pos_generator(bbox, n_pos, overlap_range=self.overlap_pos)
            neg_examples = self.neg_generator(bbox, n_neg, overlap_range=self.overlap_neg)

            pos_regions_vis = np.concatenate((pos_regions_vis, self.extract_regions(image_vis, pos_examples)), axis=0)
            neg_regions_vis = np.concatenate((neg_regions_vis, self.extract_regions(image_vis, neg_examples)), axis=0)

            pos_regions_event = np.concatenate((pos_regions_event, self.extract_regions(image_event, pos_examples)), axis=0)
            neg_regions_event = np.concatenate((neg_regions_event, self.extract_regions(image_event, neg_examples)), axis=0)

        pos_regions_vis = torch.from_numpy(pos_regions_vis)
        neg_regions_vis = torch.from_numpy(neg_regions_vis)
        pos_regions_event = torch.from_numpy(pos_regions_event)
        neg_regions_event = torch.from_numpy(neg_regions_event)

        return pos_regions_vis, pos_regions_event, neg_regions_vis, neg_regions_event

    next = __next__

    def extract_regions(self, image, samples):
        regions = np.zeros((len(samples), self.crop_size, self.crop_size, 3), dtype='uint8')
        for i, sample in enumerate(samples):
            regions[i] = crop_image2(image, sample, self.crop_size, self.padding,
                    self.flip, self.rotate, self.blur)
        regions = regions.transpose(0, 3, 1, 2)
        regions = regions.astype('float32') - 128.
        return regions
