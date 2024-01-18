import os
from .base_image_dataset import BaseImageDataset
from ltr.data.image_loader import jpeg4py_loader
import torch
import random
from collections import OrderedDict
from ltr.admin.environment import env_settings
import cv2 as cv
import numpy as np
import pickle


DATASET_YEAR_DICT = {
    '2012': {
        'url': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar',
        'filename': 'VOCtrainval_11-May-2012.tar',
        'md5': '6cd6e144f989b92b3379bac3b3de84fd',
        'base_dir': 'VOCdevkit/VOC2012'
    },
    '2011': {
        'url': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2011/VOCtrainval_25-May-2011.tar',
        'filename': 'VOCtrainval_25-May-2011.tar',
        'md5': '6c3384ef61512963050cb5d687e5bf1e',
        'base_dir': 'TrainVal/VOCdevkit/VOC2011'
    },
    '2010': {
        'url': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2010/VOCtrainval_03-May-2010.tar',
        'filename': 'VOCtrainval_03-May-2010.tar',
        'md5': 'da459979d0c395079b5c75ee67908abb',
        'base_dir': 'VOCdevkit/VOC2010'
    },
    '2009': {
        'url': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2009/VOCtrainval_11-May-2009.tar',
        'filename': 'VOCtrainval_11-May-2009.tar',
        'md5': '59065e4b188729180974ef6572f6a212',
        'base_dir': 'VOCdevkit/VOC2009'
    },
    '2008': {
        'url': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2008/VOCtrainval_14-Jul-2008.tar',
        'filename': 'VOCtrainval_11-May-2012.tar',
        'md5': '2629fa636546599198acfcfbfcf1904a',
        'base_dir': 'VOCdevkit/VOC2008'
    },
    '2007': {
        'url': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar',
        'filename': 'VOCtrainval_06-Nov-2007.tar',
        'md5': 'c52e279531787c972589f7e41ab4ae64',
        'base_dir': 'VOCdevkit/VOC2007'
    }
}


voc_classes = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
               'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
               'tvmonitor']


def generate_meta_data_file(root, split, version):
    base_dir = DATASET_YEAR_DICT[version]['base_dir']
    voc_root = os.path.join(root, base_dir)
    img_path = os.path.join(voc_root, 'JPEGImages')

    if split in ["train", "trainval", "val"]:
        mask_dir = os.path.join(voc_root, 'SegmentationClass')
    elif split in ['train_sbd_aug', 'val_sbd_aug']:
        mask_dir = os.path.join(voc_root, 'SegmentationClassSBDAug')
    else:
        raise Exception

    ltr_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
    split_file = '{}/data_specs/voc{}_{}.txt'.format(ltr_path, version, split)
    image_list = [line.rstrip('\n') for line in open(split_file)]
    image_list = sorted(image_list)

    class_ids = list(range(1, 21))
    class_names = {id_: voc_classes[id_] for id_ in class_ids}
    class_name_to_id = {cls_name: id_ for id_, cls_name in class_names.items()}

    image_ids = list(range(len(image_list)))
    image_names = {id_: im_name for id_, im_name in enumerate(image_list)}

    classes_per_image = {}
    im_per_class = {}
    for im_id, im_name in image_names.items():
        mask_path = os.path.join(mask_dir, im_name)
        mask = cv.imread(mask_path + '.png', cv.IMREAD_GRAYSCALE)

        class_ids = np.unique(mask)
        class_ids = [c for c in class_ids if c not in [0, 255]]
        classes_per_image[im_id] = class_ids

        for c_id in class_ids:
            class_name = class_names[c_id]
            if class_name in im_per_class:
                im_per_class[class_name].append(im_id)
            else:
                im_per_class[class_name] = [im_id, ]

    meta_data_dict = {'image_ids': image_ids, 'image_names': image_names,
                      'class_ids': class_ids, 'class_names': class_names, 'class_name_to_id': class_name_to_id,
                      'im_per_class': im_per_class, 'classes_per_image': classes_per_image}

    out_path = '{}/{}_metadata.pkl'.format(voc_root, split)
    with open(out_path, 'wb') as handle:
        pickle.dump(meta_data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


class VOC(BaseImageDataset):
    """ TODO
    """

    def __init__(self, root=None, image_loader=jpeg4py_loader, data_fraction=None, split="train", version="2012",
                 classes=None):
        """
        args:
        """

        root = env_settings().voc_dir if root is None else root
        super().__init__('VOC', root, image_loader)

        base_dir = DATASET_YEAR_DICT[version]['base_dir']
        voc_root = os.path.join(self.root, base_dir)
        self.img_path = os.path.join(voc_root, 'JPEGImages')

        if split in ["train", "trainval", "val"]:
            self.mask_path = os.path.join(voc_root, 'SegmentationClass')
        elif split in ['train_sbd_aug', 'val_sbd_aug']:
            self.mask_path = os.path.join(voc_root, 'SegmentationClassSBDAug')
        else:
            raise Exception

        metadata_path = '{}/{}_metadata.pkl'.format(voc_root, split)
        if not os.path.exists(metadata_path):
            generate_meta_data_file(root, split, version)

        with open(metadata_path, 'rb') as handle:
            metadata_dict = pickle.load(handle)

        self.image_list = metadata_dict['image_ids']            # List of image keys
        self.image_names = metadata_dict['image_names']

        self.class_ids = metadata_dict['class_ids']
        self.class_names = metadata_dict['class_names']
        self.class_name_to_id = metadata_dict['class_name_to_id']

        self.im_per_class = metadata_dict['im_per_class']
        self.classes_per_image = metadata_dict['classes_per_image']

        if classes is not None:
            # Only keep images belonging to the classes
            new_class_ids = [self.class_name_to_id[cls] for cls in classes]

            new_image_list = []
            for cls in classes:
                new_image_list.extend(self.im_per_class[cls])
            new_image_list = sorted(list(set(new_image_list)))

            self.image_list = new_image_list
            self.class_ids = new_class_ids

            # new_image_names = {id_: self.image_names[id_] for id_ in new_image_list}
            # new_class_names = {id_: self.class_names[id_] for id_ in new_class_list}

            # new_class_name_to_id = {cls_name: id_ for id_, cls_name in new_class_names.items()}
            self.im_per_class = {cls: self.im_per_class[cls] for cls in classes}

            # TODO filter classes_per_image
            # TODO when returning masks, set other objects to background

        # Save pos in im_per_class instead of key
        self.key_to_id = {key: id_ for id_, key in enumerate(self.image_list)}
        self.im_per_class = {cls: [self.key_to_id[key] for key in self.im_per_class[cls]] for cls in classes}
        self.class_list = [self.class_names[id] for id in self.class_ids]

    def get_num_classes(self):
        return len(self.class_list)

    def get_name(self):
        return 'voc'

    def has_class_info(self):
        return True

    def has_segmentation_info(self):
        return True

    def get_images_in_class(self, class_name):
        return self.im_per_class[class_name]

    def get_class_id(self, class_name):
        return self.class_name_to_id[class_name]

    def _get_image_name(self, im_id):
        return self.image_names[self.image_list[im_id]]

    def get_image_info(self, im_id):
        mask = cv.imread('{}/{}.png'.format(self.mask_path, self._get_image_name(im_id)),
                         cv.IMREAD_GRAYSCALE)
        mask = torch.from_numpy(mask).float()

        return {'semantic_mask': mask}

    def _get_image(self, im_id):
        img = self.image_loader('{}/{}.jpg'.format(self.img_path, self._get_image_name(im_id)))
        return img

    def get_meta_info(self, im_id):
        object_meta = OrderedDict({'object_class_name': self._get_object_classes(im_id),
                                   'motion_class': None,
                                   'major_class': None,
                                   'root_class': None,
                                   'motion_adverb': None})

        return object_meta

    def _get_object_classes(self, im_id):
        class_ids = self.classes_per_image[self.image_list[im_id]]
        class_names = [self.class_names[cls_id] for cls_id in class_ids]
        return class_names

    def get_image(self, image_id, anno=None):
        frame = self._get_image(image_id)

        if anno is None:
            anno = self.get_image_info(image_id)

        object_meta = self.get_meta_info(image_id)

        return frame, anno, object_meta
