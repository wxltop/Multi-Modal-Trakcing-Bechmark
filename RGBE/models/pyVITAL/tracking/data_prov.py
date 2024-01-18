import sys
import numpy as np
from PIL import Image

import torch
import torch.utils.data as data

from modules.utils import crop_image2


class RegionExtractor():
    def __init__(self, image_vis, image_event, samples, opts):
        self.image_vis = np.asarray(image_vis)
        self.image_event = np.asarray(image_event)
        self.samples = samples

        self.crop_size = opts['img_size']
        self.padding = opts['padding']
        self.batch_size = opts['batch_test']

        self.index = np.arange(len(samples))
        self.pointer = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.pointer == len(self.samples):
            self.pointer = 0
            raise StopIteration
        else:
            next_pointer = min(self.pointer + self.batch_size, len(self.samples))
            index = self.index[self.pointer:next_pointer]
            self.pointer = next_pointer
            regions_vis, regions_event = self.extract_regions(index)
            regions_vis = torch.from_numpy(regions_vis)
            regions_event = torch.from_numpy(regions_event)
            return regions_vis, regions_event

    next = __next__

    def extract_regions(self, index):
        regions_vis = np.zeros((len(index), self.crop_size, self.crop_size, 3), dtype='uint8')
        regions_event = np.zeros((len(index), self.crop_size, self.crop_size, 3), dtype='uint8')
        for i, sample in enumerate(self.samples[index]):
            regions_vis[i] = crop_image2(self.image_vis, sample, self.crop_size, self.padding)
            regions_event[i] = crop_image2(self.image_event, sample, self.crop_size, self.padding)
        regions_vis = regions_vis.transpose(0, 3, 1, 2)
        regions_vis = regions_vis.astype('float32') - 128.
        regions_event = regions_event.transpose(0, 3, 1, 2)
        regions_event = regions_event.astype('float32') - 128.

        return regions_vis, regions_event 





