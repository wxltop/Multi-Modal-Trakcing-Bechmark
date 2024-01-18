import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
from pytracking import TensorDict
import ltr.data.processing_utils as prutils
import ltr.data.bounding_box_utils as bbutils
import numpy as np
import math
import random
import cv2 as cv


def stack_tensors(x):
    if isinstance(x, (list, tuple)) and isinstance(x[0], torch.Tensor):
        return torch.stack(x)
    return x


class BaseProcessing:
    """ Base class for Processing. Processing class is used to process the data returned by a dataset, before passing it
     through the network. For example, it can be used to crop a search region around the object, apply various data
     augmentations, etc."""

    def __init__(self, transform=transforms.ToTensor(), train_transform=None, test_transform=None,
                 joint_transform=None):
        """
        args:
            transform       - The set of transformations to be applied on the images. Used only if train_transform or
                                test_transform is None.
            train_transform - The set of transformations to be applied on the train images. If None, the 'transform'
                                argument is used instead.
            test_transform  - The set of transformations to be applied on the test images. If None, the 'transform'
                                argument is used instead.
            joint_transform - The set of transformations to be applied 'jointly' on the train and test images.  For
                                example, it can be used to convert both test and train images to grayscale.
        """
        self.transform = {'train': transform if train_transform is None else train_transform,
                          'test': transform if test_transform is None else test_transform,
                          'joint': joint_transform}

    def __call__(self, data: TensorDict):
        raise NotImplementedError


# !OUTDATED!
class TrackingProcessing(BaseProcessing):
    def __init__(self, search_area_factor, output_sz, center_jitter_factor, scale_jitter_factor, crop_type='replicate',
                 max_scale_change=None, mode='pair',
                 proposal_params=None, label_function_params=None, *args, **kwargs):
        # Mode is either sequence or pair
        super().__init__(*args, **kwargs)
        self.search_area_factor = search_area_factor
        self.output_sz = output_sz
        self.center_jitter_factor = center_jitter_factor
        self.scale_jitter_factor = scale_jitter_factor
        self.crop_type = crop_type
        self.mode = mode
        self.max_scale_change = max_scale_change

        self.proposal_params = proposal_params
        self.label_function_params = label_function_params

    def _get_jittered_box(self, box, mode):
        jittered_size = box[2:4] * torch.exp(torch.randn(2) * self.scale_jitter_factor[mode])

        max_offset = (jittered_size.prod().sqrt() * torch.tensor(self.center_jitter_factor[mode]).float())
        jittered_center = box[0:2] + 0.5 * box[2:4] + max_offset * (torch.rand(2) - 0.5)

        return torch.cat((jittered_center - 0.5 * jittered_size, jittered_size), dim=0)

    def _generate_proposals(self, box):
        """ Generates proposals by adding noise to the input box
        args:
            box - input box

        returns:
            torch.Tensor - Array of shape (num_proposals, 4) containing proposals
            torch.Tensor - Array of shape (num_proposals,) containing IoU overlap of each proposal with the input box. The
                        IoU is mapped to [-1, 1]
        """
        # Generate proposals
        num_proposals = self.proposal_params['boxes_per_frame']
        proposals = torch.zeros((num_proposals, 4))
        gt_iou = torch.zeros(num_proposals)

        for i in range(num_proposals):
            proposals[i, :], gt_iou[i] = prutils.perturb_box(box, min_iou=self.proposal_params['min_iou'],
                                                             sigma_factor=self.proposal_params['sigma_factor']
                                                             )

        # Map to [-1, 1]
        gt_iou = gt_iou * 2 - 1
        return proposals, gt_iou

    def _generate_kl_proposals(self, box):
        """
        """
        # Generate proposals
        proposals, proposal_density, gt_density = prutils.sample_box_gmm(box, self.proposal_params['proposal_sigma'],
                                                                         self.proposal_params['gt_sigma'],
                                                                         self.proposal_params['boxes_per_frame'])

        return proposals, proposal_density, gt_density

    def _generate_label_function(self, target_bb, is_distractor=None):
        gauss_label = prutils.gaussian_label_function(target_bb.view(-1, 4), self.label_function_params['sigma_factor'],
                                                      self.label_function_params['kernel_sz'],
                                                      self.label_function_params['feature_sz'], self.output_sz,
                                                      end_pad_if_even=self.label_function_params.get(
                                                          'end_pad_if_even', True))
        if is_distractor is not None:
            gauss_label *= (1 - is_distractor).view(-1, 1, 1).float()
        return gauss_label

    def _generate_box_segmentation(self, target_bb, im_sz, is_distractor=None):
        bb = target_bb.view(-1, 4)
        seg = torch.zeros(bb.shape[0], im_sz[0], im_sz[1], dtype=torch.uint8)
        for i in range(bb.shape[0]):
            if is_distractor is None or is_distractor[i] == 0:
                seg[i, max(int(bb[i, 1]), 0):min(int(bb[i, 1] + bb[i, 3]), im_sz[0]),
                max(int(bb[i, 0]), 0):min(int(bb[i, 0] + bb[i, 2]), im_sz[1])] = 1

        return seg

    def __call__(self, data: TensorDict):
        if self.transform['joint'] is not None:
            data['train_images'], data['train_anno'] = self.transform['joint'](image=data['train_images'], bbox=data['train_anno'])
            data['test_images'], data['test_anno'] = self.transform['joint'](image=data['test_images'], bbox=data['test_anno'], new_roll=False)

        for s in ['train', 'test']:
            assert self.mode == 'sequence' or len(data[s + '_images']) == 1, \
                "In pair mode, num train/test frames must be 1"

            # Add a uniform noise to the center pos
            jittered_anno = [self._get_jittered_box(a, s) for a in data[s + '_anno']]

            crops, boxes = prutils.target_image_crop(data[s + '_images'], jittered_anno, data[s + '_anno'],
                                                     self.search_area_factor, self.output_sz, mode=self.crop_type,
                                                     max_scale_change=self.max_scale_change)

            data[s + '_images'], data[s + '_anno'] = self.transform[s](image=crops, bbox=boxes, joint=False)

        if self.proposal_params:
            if self.proposal_params.get('mode', 'iou') == 'iou':
                frame2_proposals, gt_iou = zip(*[self._generate_proposals(a) for a in data['test_anno']])

                data['test_proposals'] = list(frame2_proposals)
                data['proposal_iou'] = list(gt_iou)
            else:
                proposals, proposal_density, gt_density = zip(
                    *[self._generate_kl_proposals(a) for a in data['test_anno']])

                data['test_proposals'] = proposals
                data['proposal_density'] = proposal_density
                data['gt_density'] = gt_density

        if 'is_distractor_test_frame' in data:
            data['is_distractor_test_frame'] = torch.tensor(data['is_distractor_test_frame'], dtype=torch.uint8)
        else:
            data['is_distractor_test_frame'] = torch.zeros(len(data['test_images']), dtype=torch.uint8)

        if 'is_distractor_train_frame' in data:
            data['is_distractor_train_frame'] = torch.tensor(data['is_distractor_train_frame'], dtype=torch.uint8)
        else:
            data['is_distractor_train_frame'] = torch.zeros(len(data['train_images']), dtype=torch.uint8)

        if self.mode == 'sequence':
            data = data.apply(stack_tensors)
        else:
            data = data.apply(lambda x: x[0] if isinstance(x, list) else x)

        if self.label_function_params is not None:
            data['train_label'] = self._generate_label_function(data['train_anno'], data['is_distractor_train_frame'])
            data['test_label'] = self._generate_label_function(data['test_anno'], data['is_distractor_test_frame'])

        # data['test_bbseg'] = self._generate_box_segmentation(data['test_anno'], data['test_images'].shape[-2:], data['is_distractor_test_frame'])

        return data


class ATOMProcessing(BaseProcessing):
    """ The processing class used for training ATOM. The images are processed in the following way.
    First, the target bounding box is jittered by adding some noise. Next, a square region (called search region )
    centered at the jittered target center, and of area search_area_factor^2 times the area of the jittered box is
    cropped from the image. The reason for jittering the target box is to avoid learning the bias that the target is
    always at the center of the search region. The search region is then resized to a fixed size given by the
    argument output_sz. A set of proposals are then generated for the test images by jittering the ground truth box.

    """

    def __init__(self, search_area_factor, output_sz, center_jitter_factor, scale_jitter_factor, proposal_params,
                 mode='pair', *args, **kwargs):
        """
        args:
            search_area_factor - The size of the search region  relative to the target size.
            output_sz - An integer, denoting the size to which the search region is resized. The search region is always
                        square.
            center_jitter_factor - A dict containing the amount of jittering to be applied to the target center before
                                    extracting the search region. See _get_jittered_box for how the jittering is done.
            scale_jitter_factor - A dict containing the amount of jittering to be applied to the target size before
                                    extracting the search region. See _get_jittered_box for how the jittering is done.
            proposal_params - Arguments for the proposal generation process. See _generate_proposals for details.
            mode - Either 'pair' or 'sequence'. If mode='sequence', then output has an extra dimension for frames
        """
        super().__init__(*args, **kwargs)
        self.search_area_factor = search_area_factor
        self.output_sz = output_sz
        self.center_jitter_factor = center_jitter_factor
        self.scale_jitter_factor = scale_jitter_factor
        self.proposal_params = proposal_params
        self.mode = mode

    def _get_jittered_box(self, box, mode):
        """ Jitter the input box
        args:
            box - input bounding box
            mode - string 'train' or 'test' indicating train or test data

        returns:
            torch.Tensor - jittered box
        """

        jittered_size = box[2:4] * torch.exp(torch.randn(2) * self.scale_jitter_factor[mode])
        max_offset = (jittered_size.prod().sqrt() * torch.tensor(self.center_jitter_factor[mode]).float())
        jittered_center = box[0:2] + 0.5 * box[2:4] + max_offset * (torch.rand(2) - 0.5)

        return torch.cat((jittered_center - 0.5 * jittered_size, jittered_size), dim=0)

    def _generate_proposals(self, box):
        """ Generates proposals by adding noise to the input box
        args:
            box - input box

        returns:
            torch.Tensor - Array of shape (num_proposals, 4) containing proposals
            torch.Tensor - Array of shape (num_proposals,) containing IoU overlap of each proposal with the input box. The
                        IoU is mapped to [-1, 1]
        """
        # Generate proposals
        num_proposals = self.proposal_params['boxes_per_frame']
        proposal_method = self.proposal_params.get('proposal_method', 'default')

        if proposal_method == 'default':
            proposals = torch.zeros((num_proposals, 4))
            gt_iou = torch.zeros(num_proposals)
            for i in range(num_proposals):
                proposals[i, :], gt_iou[i] = prutils.perturb_box(box, min_iou=self.proposal_params['min_iou'],
                                                                 sigma_factor=self.proposal_params['sigma_factor'])
        elif proposal_method == 'gmm':
            proposals, _, _ = prutils.sample_box_gmm(box, self.proposal_params['proposal_sigma'],
                                                                             num_samples=num_proposals)
            gt_iou = prutils.iou(box.view(1,4), proposals.view(-1,4))

        # Map to [-1, 1]
        gt_iou = gt_iou * 2 - 1
        return proposals, gt_iou

    def __call__(self, data: TensorDict):
        """
        args:
            data - The input data, should contain the following fields:
                'train_images', test_images', 'train_anno', 'test_anno'
        returns:
            TensorDict - output data block with following fields:
                'train_images', 'test_images', 'train_anno', 'test_anno', 'test_proposals', 'proposal_iou'
        """
        # Apply joint transforms
        if self.transform['joint'] is not None:
            data['train_images'], data['train_anno'] = self.transform['joint'](image=data['train_images'], bbox=data['train_anno'])
            data['test_images'], data['test_anno'] = self.transform['joint'](image=data['test_images'], bbox=data['test_anno'], new_roll=False)

        for s in ['train', 'test']:
            assert self.mode == 'sequence' or len(data[s + '_images']) == 1, \
                "In pair mode, num train/test frames must be 1"

            # Add a uniform noise to the center pos
            jittered_anno = [self._get_jittered_box(a, s) for a in data[s + '_anno']]

            # Crop image region centered at jittered_anno box
            crops, boxes = prutils.jittered_center_crop(data[s + '_images'], jittered_anno, data[s + '_anno'],
                                                        self.search_area_factor, self.output_sz)

            # Apply transforms
            data[s + '_images'], data[s + '_anno'] = self.transform[s](image=crops, bbox=boxes, joint=False)

        # Generate proposals
        frame2_proposals, gt_iou = zip(*[self._generate_proposals(a) for a in data['test_anno']])

        data['test_proposals'] = list(frame2_proposals)
        data['proposal_iou'] = list(gt_iou)

        # Prepare output
        if self.mode == 'sequence':
            data = data.apply(stack_tensors)
        else:
            data = data.apply(lambda x: x[0] if isinstance(x, list) else x)

        return data


class KLBBregProcessing(BaseProcessing):
    """ Based on ATOMProcessing. It supports training ATOM using the Maximum Likelihood or KL-divergence based learning
    introduced in [https://arxiv.org/abs/1909.12297] and in PrDiMP [https://arxiv.org/abs/2003.12565].
    """

    def __init__(self, search_area_factor, output_sz, center_jitter_factor, scale_jitter_factor, proposal_params,
                 mode='pair', *args, **kwargs):
        """
        args:
            search_area_factor - The size of the search region  relative to the target size.
            output_sz - An integer, denoting the size to which the search region is resized. The search region is always
                        square.
            center_jitter_factor - A dict containing the amount of jittering to be applied to the target center before
                                    extracting the search region. See _get_jittered_box for how the jittering is done.
            scale_jitter_factor - A dict containing the amount of jittering to be applied to the target size before
                                    extracting the search region. See _get_jittered_box for how the jittering is done.
            proposal_params - Arguments for the proposal generation process. See _generate_proposals for details.
            mode - Either 'pair' or 'sequence'. If mode='sequence', then output has an extra dimension for frames
        """
        super().__init__(*args, **kwargs)
        self.search_area_factor = search_area_factor
        self.output_sz = output_sz
        self.center_jitter_factor = center_jitter_factor
        self.scale_jitter_factor = scale_jitter_factor
        self.proposal_params = proposal_params
        self.mode = mode

    def _get_jittered_box(self, box, mode):
        """ Jitter the input box
        args:
            box - input bounding box
            mode - string 'train' or 'test' indicating train or test data

        returns:
            torch.Tensor - jittered box
        """

        jittered_size = box[2:4] * torch.exp(torch.randn(2) * self.scale_jitter_factor[mode])
        max_offset = (jittered_size.prod().sqrt() * torch.tensor(self.center_jitter_factor[mode]).float())
        jittered_center = box[0:2] + 0.5 * box[2:4] + max_offset * (torch.rand(2) - 0.5)

        return torch.cat((jittered_center - 0.5 * jittered_size, jittered_size), dim=0)

    def _generate_proposals(self, box):
        """
        """
        # Generate proposals
        proposals, proposal_density, gt_density = prutils.sample_box_gmm(box, self.proposal_params['proposal_sigma'],
                                                                         gt_sigma=self.proposal_params['gt_sigma'],
                                                                         num_samples=self.proposal_params[
                                                                             'boxes_per_frame'],
                                                                         add_mean_box=self.proposal_params.get(
                                                                             'add_mean_box', False))

        return proposals, proposal_density, gt_density

    def __call__(self, data: TensorDict):
        """
        args:
            data - The input data, should contain the following fields:
                'train_images', test_images', 'train_anno', 'test_anno'
        returns:
            TensorDict - output data block with following fields:
                'train_images', 'test_images', 'train_anno', 'test_anno', 'test_proposals', 'proposal_density', 'gt_density'
        """
        # Apply joint transforms
        if self.transform['joint'] is not None:
            data['train_images'], data['train_anno'] = self.transform['joint'](image=data['train_images'], bbox=data['train_anno'])
            data['test_images'], data['test_anno'] = self.transform['joint'](image=data['test_images'], bbox=data['test_anno'], new_roll=False)

        for s in ['train', 'test']:
            assert self.mode == 'sequence' or len(data[s + '_images']) == 1, \
                "In pair mode, num train/test frames must be 1"

            # Add a uniform noise to the center pos
            jittered_anno = [self._get_jittered_box(a, s) for a in data[s + '_anno']]

            # Crop image region centered at jittered_anno box
            crops, boxes, _ = prutils.jittered_center_crop(data[s + '_images'], jittered_anno, data[s + '_anno'],
                                                        self.search_area_factor, self.output_sz)

            # Apply transforms
            data[s + '_images'], data[s + '_anno'] = self.transform[s](image=crops, bbox=boxes, joint=False)

        # Generate proposals
        proposals, proposal_density, gt_density = zip(*[self._generate_proposals(a) for a in data['test_anno']])

        data['test_proposals'] = proposals
        data['proposal_density'] = proposal_density
        data['gt_density'] = gt_density

        # Prepare output
        if self.mode == 'sequence':
            data = data.apply(stack_tensors)
        else:
            data = data.apply(lambda x: x[0] if isinstance(x, list) else x)

        return data


class ATOMwKLProcessing(BaseProcessing):
    """Same as ATOMProcessing but using the GMM-based sampling of proposal boxes used in KLBBregProcessing."""
    def __init__(self, search_area_factor, output_sz, center_jitter_factor, scale_jitter_factor, proposal_params,
                 mode='pair', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.search_area_factor = search_area_factor
        self.output_sz = output_sz
        self.center_jitter_factor = center_jitter_factor
        self.scale_jitter_factor = scale_jitter_factor
        self.proposal_params = proposal_params
        self.mode = mode

    def _get_jittered_box(self, box, mode):
        """ Jitter the input box
        args:
            box - input bounding box
            mode - string 'train' or 'test' indicating train or test data

        returns:
            torch.Tensor - jittered box
        """

        jittered_size = box[2:4] * torch.exp(torch.randn(2) * self.scale_jitter_factor[mode])
        max_offset = (jittered_size.prod().sqrt() * torch.tensor(self.center_jitter_factor[mode]).float())
        jittered_center = box[0:2] + 0.5 * box[2:4] + max_offset * (torch.rand(2) - 0.5)

        return torch.cat((jittered_center - 0.5 * jittered_size, jittered_size), dim=0)

    def _generate_proposals(self, box):
        """
        """
        # Generate proposals
        proposals, proposal_density, gt_density = prutils.sample_box_gmm(box, self.proposal_params['proposal_sigma'],
                                                                         self.proposal_params['gt_sigma'],
                                                                         self.proposal_params['boxes_per_frame'])

        iou = prutils.iou_gen(proposals, box.view(1, 4))
        return proposals, proposal_density, gt_density, iou

    def __call__(self, data: TensorDict):
        # Apply joint transforms
        if self.transform['joint'] is not None:
            data['train_images'], data['train_anno'] = self.transform['joint'](image=data['train_images'], bbox=data['train_anno'])
            data['test_images'], data['test_anno'] = self.transform['joint'](image=data['test_images'], bbox=data['test_anno'], new_roll=False)

        for s in ['train', 'test']:
            assert self.mode == 'sequence' or len(data[s + '_images']) == 1, \
                "In pair mode, num train/test frames must be 1"

            # Add a uniform noise to the center pos
            jittered_anno = [self._get_jittered_box(a, s) for a in data[s + '_anno']]

            # Crop image region centered at jittered_anno box
            crops, boxes = prutils.jittered_center_crop(data[s + '_images'], jittered_anno, data[s + '_anno'],
                                                        self.search_area_factor, self.output_sz)

            # Apply transforms
            data[s + '_images'], data[s + '_anno'] = self.transform[s](image=crops, bbox=boxes, joint=False)

        # Generate proposals
        proposals, proposal_density, gt_density, proposal_iou = zip(
            *[self._generate_proposals(a) for a in data['test_anno']])

        data['test_proposals'] = proposals
        data['proposal_density'] = proposal_density
        data['gt_density'] = gt_density
        data['proposal_iou'] = proposal_iou
        # Prepare output
        if self.mode == 'sequence':
            data = data.apply(stack_tensors)
        else:
            data = data.apply(lambda x: x[0] if isinstance(x, list) else x)

        return data


class CorrLearnProcessing(BaseProcessing):
    def __init__(self, search_area_factor, output_sz, center_jitter_factor, scale_jitter_factor, crop_type='replicate',
                 max_scale_change=None, label_function_params=None, *args, **kwargs):
        # Mode is either sequence or pair
        super().__init__(*args, **kwargs)
        self.search_area_factor = search_area_factor
        self.output_sz = output_sz
        self.center_jitter_factor = center_jitter_factor
        self.scale_jitter_factor = scale_jitter_factor
        self.crop_type = crop_type
        self.max_scale_change = max_scale_change
        self.label_function_params = label_function_params

    def _get_jittered_box(self, box, mode):
        jittered_size = box[2:4] * torch.exp(torch.randn(2) * self.scale_jitter_factor[mode])

        max_offset = (jittered_size.prod().sqrt() * torch.tensor(self.center_jitter_factor[mode]).float())
        jittered_center = box[0:2] + 0.5 * box[2:4] + max_offset * (torch.rand(2) - 0.5)

        return torch.cat((jittered_center - 0.5 * jittered_size, jittered_size), dim=0)

    def _generate_label_function(self, bb_ref, bb_test):
        feat_sz = self.label_function_params['feature_sz']
        sc = feat_sz / self.output_sz

        # BB img_coords in feature frame
        c_ref = (bb_ref[:2] - 0.5 * bb_ref[2:]).flip(0) * sc - 0.5
        c_test = (bb_test[:2] - 0.5 * bb_test[2:]).flip(0) * sc - 0.5
        s_ref = bb_ref[2:].flip(0) * sc
        s_test = bb_test[2:].flip(0) * sc

        k0 = torch.arange(feat_sz, dtype=torch.float32).view(-1, 1, 1, 1)
        k1 = torch.arange(feat_sz, dtype=torch.float32).view(1, -1, 1, 1)

        m0 = ((k0 - c_ref[0]) * (s_test[0] / s_ref[0]) + c_test[0])
        m1 = ((k1 - c_ref[1]) * (s_test[1] / s_ref[1]) + c_test[1])

        k0 = k0.view(1, 1, -1, 1)
        k1 = k1.view(1, 1, 1, -1)

        sqr_dist = (k0 - m0) ** 2 + (k1 - m1) ** 2

        sig = self.label_function_params['sigma_factor']

        if sig == 0:
            gauss_label = (sqr_dist == 0).float()
        else:
            gauss_label = torch.exp(-0.5 / sig ** 2 * sqr_dist)
        return gauss_label.view(1, 1, -1, feat_sz, feat_sz)

    def __call__(self, data: TensorDict):
        if self.transform['joint'] is not None:
            num_train_images = len(data['train_images'])
            assert num_train_images == 1

            all_images = data['train_images']
            all_images_trans = self.transform['joint'](image=all_images)

            data['train_images'] = all_images_trans
            data['test_images'] = [all_images_trans[0].copy()]

            data['test_anno'] = [data['train_anno'][0].clone()]

        for s in ['train', 'test']:
            # Add a uniform noise to the center pos
            jittered_anno = [self._get_jittered_box(a, s) for a in data[s + '_anno']]

            crops, boxes = prutils.target_image_crop(data[s + '_images'], jittered_anno, data[s + '_anno'],
                                                     self.search_area_factor, self.output_sz, mode=self.crop_type,
                                                     max_scale_change=self.max_scale_change)

            data[s + '_images'], data[s + '_anno'] = self.transform[s](image=crops, bbox=boxes, joint=False)

        data = data.apply(lambda x: x[0] if isinstance(x, list) else x)

        if self.label_function_params is not None:
            data['test_label'] = self._generate_label_function(data['train_anno'].view(-1), data['test_anno'].view(-1))

        return data


class MotionTrackingSequenceProcessing(BaseProcessing):
    def __init__(self, search_area_factor, output_sz, center_jitter_param, scale_jitter_param,
                 proposal_params=None, label_function_params=None, min_crop_inside_ratio=0,
                 *args, **kwargs):
        # Mode is either sequence or pair
        super().__init__(*args, **kwargs)
        self.search_area_factor = search_area_factor
        self.output_sz = output_sz
        self.center_jitter_param = center_jitter_param
        self.scale_jitter_param = scale_jitter_param

        self.proposal_params = proposal_params
        self.label_function_params = label_function_params
        self.min_crop_inside_ratio = min_crop_inside_ratio

    def _check_if_crop_inside_image(self, box, im_shape):
        x, y, w, h = box.tolist()

        if w <= 0.0 or h <= 0.0:
            return False

        crop_sz = math.ceil(math.sqrt(w * h) * self.search_area_factor)

        x1 = x + 0.5 * w - crop_sz * 0.5
        x2 = x1 + crop_sz

        y1 = y + 0.5 * h - crop_sz * 0.5
        y2 = y1 + crop_sz

        w_inside = max(min(x2, im_shape[1]) - max(x1, 0), 0)
        h_inside = max(min(y2, im_shape[0]) - max(y1, 0), 0)

        crop_area = ((x2 - x1) * (y2 - y1))

        if crop_area > 0:
            inside_ratio = w_inside * h_inside / crop_area
            return inside_ratio > self.min_crop_inside_ratio
        else:
            return False

    def _generate_synthetic_motion(self, boxes, images, mode):
        num_frames = len(boxes)

        out_boxes = []
        # prev_box = boxes[0]

        for i in range(num_frames):
            jittered_box = None
            for _ in range(10):
                orig_box = boxes[i]
                jittered_size = orig_box[2:4] * torch.exp(torch.randn(2) * self.scale_jitter_param[mode + '_factor'])

                if self.center_jitter_param.get(mode + '_mode', 'uniform') == 'uniform':
                    max_offset = (jittered_size.prod().sqrt() * self.center_jitter_param[mode + '_factor']).item()
                    offset_factor = (torch.rand(2) - 0.5)
                    jittered_center = orig_box[0:2] + 0.5 * orig_box[2:4] + max_offset * offset_factor

                    if self.center_jitter_param.get(mode + '_limit_motion', False) and i > 0:
                        prev_out_box_center = out_boxes[-1][:2] + 0.5 * out_boxes[-1][2:]
                        if abs(jittered_center[0] - prev_out_box_center[0]) > out_boxes[-1][2:].prod().sqrt() * 2.5:
                            jittered_center[0] = orig_box[0] + 0.5 * orig_box[2] + max_offset * offset_factor[0] * -1

                        if abs(jittered_center[1] - prev_out_box_center[1]) > out_boxes[-1][2:].prod().sqrt() * 2.5:
                            jittered_center[1] = orig_box[1] + 0.5 * orig_box[3] + max_offset * offset_factor[1] * -1

                jittered_box = torch.cat((jittered_center - 0.5 * jittered_size, jittered_size), dim=0)

                if self._check_if_crop_inside_image(jittered_box, images[i].shape):
                    break
                else:
                    jittered_box = torch.Tensor([1, 1, 10, 10])

            out_boxes.append(jittered_box)
            # prev_box = boxes[i]

        return out_boxes

    def _generate_proposals(self, frame2_gt_crop):
        # Generate proposals
        num_proposals = self.proposal_params['boxes_per_frame']
        frame2_proposals = np.zeros((num_proposals, 4))
        gt_iou = np.zeros(num_proposals)
        sample_p = np.zeros(num_proposals)

        for i in range(num_proposals):
            frame2_proposals[i, :], gt_iou[i], sample_p[i] = bbutils.perturb_box(
                frame2_gt_crop,
                min_iou=self.proposal_params['min_iou'],
                sigma_factor=self.proposal_params['sigma_factor']
            )

        gt_iou = gt_iou * 2 - 1

        return frame2_proposals, gt_iou

    def _generate_label_function(self, target_bb, target_absent=None):
        gauss_label = prutils.gaussian_label_function(target_bb.view(-1, 4), self.label_function_params['sigma_factor'],
                                                      self.label_function_params['kernel_sz'],
                                                      self.label_function_params['feature_sz'], self.output_sz,
                                                      end_pad_if_even=self.label_function_params.get(
                                                          'end_pad_if_even', True))
        if target_absent is not None:
            gauss_label *= (1 - target_absent).view(-1, 1, 1).float()
        return gauss_label

    def __call__(self, data: TensorDict):
        if self.transform['joint'] is not None:
            data['train_images'], data['train_anno'] = self.transform['joint'](image=data['train_images'], bbox=data['train_anno'])
            data['test_images'], data['test_anno'] = self.transform['joint'](image=data['test_images'], bbox=data['test_anno'], new_roll=False)

        for s in ['train', 'test']:
            # Generate synthetic sequence
            jittered_anno = self._generate_synthetic_motion(data[s + '_anno'], data[s + '_images'], s)

            # Crop images
            crops, boxes= prutils.jittered_center_crop(data[s + '_images'], jittered_anno, data[s + '_anno'],
                                                       self.search_area_factor, self.output_sz)

            # Add transforms
            data[s + '_images'], data[s + '_anno'] = self.transform[s](image=crops, bbox=boxes, joint=False)

        if self.proposal_params:
            frame2_proposals, gt_iou = zip(*[self._generate_proposals(a.numpy()) for a in data['test_anno']])

            data['test_proposals'] = [torch.tensor(p, dtype=torch.float32) for p in frame2_proposals]
            data['proposal_iou'] = [torch.tensor(gi, dtype=torch.float32) for gi in gt_iou]

        data = data.apply(stack_tensors)

        if self.label_function_params is not None:
            data['train_label'] = self._generate_label_function(data['train_anno'])
            test_target_absent = 1 - (data['test_visible'] * data['test_valid_anno'])

            data['test_label'] = self._generate_label_function(data['test_anno'], test_target_absent)

        return data


class KYSProcessing(BaseProcessing):
    """ The processing class used for training KYS. The images are processed in the following way.
        First, the target bounding box is jittered by adding some noise. Next, a square region (called search region )
        centered at the jittered target center, and of area search_area_factor^2 times the area of the jittered box is
        cropped from the image. The reason for jittering the target box is to avoid learning the bias that the target is
        always at the center of the search region. The search region is then resized to a fixed size given by the
        argument output_sz. A Gaussian label centered at the target is generated for each image. These label functions are
        used for computing the loss of the predicted classification model on the test images. A set of proposals are
        also generated for the test images by jittering the ground truth box. These proposals can be used to train the
        bounding box estimating branch.
        """
    def __init__(self, search_area_factor, output_sz, center_jitter_param, scale_jitter_param,
                 proposal_params=None, label_function_params=None, min_crop_inside_ratio=0,
                 *args, **kwargs):
        """
        args:
            search_area_factor - The size of the search region  relative to the target size.
            output_sz - An integer, denoting the size to which the search region is resized. The search region is always
                        square.
            center_jitter_factor - A dict containing the amount of jittering to be applied to the target center before
                                    extracting the search region. See _generate_synthetic_motion for how the jittering is done.
            scale_jitter_factor - A dict containing the amount of jittering to be applied to the target size before
                                    extracting the search region. See _generate_synthetic_motion for how the jittering is done.
            proposal_params - Arguments for the proposal generation process. See _generate_proposals for details.
            label_function_params - Arguments for the label generation process. See _generate_label_function for details.
            min_crop_inside_ratio - Minimum amount of cropped search area which should be inside the image.
                                    See _check_if_crop_inside_image for details.
        """
        super().__init__(*args, **kwargs)
        self.search_area_factor = search_area_factor
        self.output_sz = output_sz
        self.center_jitter_param = center_jitter_param
        self.scale_jitter_param = scale_jitter_param

        self.proposal_params = proposal_params
        self.label_function_params = label_function_params
        self.min_crop_inside_ratio = min_crop_inside_ratio

    def _check_if_crop_inside_image(self, box, im_shape):
        x, y, w, h = box.tolist()

        if w <= 0.0 or h <= 0.0:
            return False

        crop_sz = math.ceil(math.sqrt(w * h) * self.search_area_factor)

        x1 = x + 0.5 * w - crop_sz * 0.5
        x2 = x1 + crop_sz

        y1 = y + 0.5 * h - crop_sz * 0.5
        y2 = y1 + crop_sz

        w_inside = max(min(x2, im_shape[1]) - max(x1, 0), 0)
        h_inside = max(min(y2, im_shape[0]) - max(y1, 0), 0)

        crop_area = ((x2 - x1) * (y2 - y1))

        if crop_area > 0:
            inside_ratio = w_inside * h_inside / crop_area
            return inside_ratio > self.min_crop_inside_ratio
        else:
            return False

    def _generate_synthetic_motion(self, boxes, images, mode):
        num_frames = len(boxes)

        out_boxes = []

        for i in range(num_frames):
            jittered_box = None
            for _ in range(10):
                orig_box = boxes[i]
                jittered_size = orig_box[2:4] * torch.exp(torch.randn(2) * self.scale_jitter_param[mode + '_factor'])

                if self.center_jitter_param.get(mode + '_mode', 'uniform') == 'uniform':
                    max_offset = (jittered_size.prod().sqrt() * self.center_jitter_param[mode + '_factor']).item()
                    offset_factor = (torch.rand(2) - 0.5)
                    jittered_center = orig_box[0:2] + 0.5 * orig_box[2:4] + max_offset * offset_factor

                    if self.center_jitter_param.get(mode + '_limit_motion', False) and i > 0:
                        prev_out_box_center = out_boxes[-1][:2] + 0.5 * out_boxes[-1][2:]
                        if abs(jittered_center[0] - prev_out_box_center[0]) > out_boxes[-1][2:].prod().sqrt() * 2.5:
                            jittered_center[0] = orig_box[0] + 0.5 * orig_box[2] + max_offset * offset_factor[0] * -1

                        if abs(jittered_center[1] - prev_out_box_center[1]) > out_boxes[-1][2:].prod().sqrt() * 2.5:
                            jittered_center[1] = orig_box[1] + 0.5 * orig_box[3] + max_offset * offset_factor[1] * -1

                jittered_box = torch.cat((jittered_center - 0.5 * jittered_size, jittered_size), dim=0)

                if self._check_if_crop_inside_image(jittered_box, images[i].shape):
                    break
                else:
                    jittered_box = torch.tensor([1, 1, 10, 10]).float()

            out_boxes.append(jittered_box)

        return out_boxes

    def _generate_proposals(self, frame2_gt_crop):
        # Generate proposals
        num_proposals = self.proposal_params['boxes_per_frame']
        frame2_proposals = np.zeros((num_proposals, 4))
        gt_iou = np.zeros(num_proposals)
        sample_p = np.zeros(num_proposals)

        for i in range(num_proposals):
            frame2_proposals[i, :], gt_iou[i], sample_p[i] = bbutils.perturb_box(
                frame2_gt_crop,
                min_iou=self.proposal_params['min_iou'],
                sigma_factor=self.proposal_params['sigma_factor']
            )

        gt_iou = gt_iou * 2 - 1

        return frame2_proposals, gt_iou

    def _generate_label_function(self, target_bb, target_absent=None):
        gauss_label = prutils.gaussian_label_function(target_bb.view(-1, 4), self.label_function_params['sigma_factor'],
                                                      self.label_function_params['kernel_sz'],
                                                      self.label_function_params['feature_sz'], self.output_sz,
                                                      end_pad_if_even=self.label_function_params.get(
                                                          'end_pad_if_even', True))
        if target_absent is not None:
            gauss_label *= (1 - target_absent).view(-1, 1, 1).float()
        return gauss_label

    def __call__(self, data: TensorDict):
        if self.transform['joint'] is not None:
            data['train_images'], data['train_anno'] = self.transform['joint'](image=data['train_images'],
                                                                               bbox=data['train_anno'])
            data['test_images'], data['test_anno'] = self.transform['joint'](image=data['test_images'], bbox=data['test_anno'], new_roll=False)

        for s in ['train', 'test']:
            # Generate synthetic sequence
            jittered_anno = self._generate_synthetic_motion(data[s + '_anno'], data[s + '_images'], s)

            # Crop images
            crops, boxes, _ = prutils.jittered_center_crop(data[s + '_images'], jittered_anno, data[s + '_anno'],
                                                       self.search_area_factor, self.output_sz)

            # Add transforms
            data[s + '_images'], data[s + '_anno'] = self.transform[s](image=crops, bbox=boxes, joint=False)

        if self.proposal_params:
            frame2_proposals, gt_iou = zip(*[self._generate_proposals(a.numpy()) for a in data['test_anno']])

            data['test_proposals'] = [torch.tensor(p, dtype=torch.float32) for p in frame2_proposals]
            data['proposal_iou'] = [torch.tensor(gi, dtype=torch.float32) for gi in gt_iou]

        data = data.apply(stack_tensors)

        if self.label_function_params is not None:
            data['train_label'] = self._generate_label_function(data['train_anno'])
            test_target_absent = 1 - (data['test_visible'] * data['test_valid_anno'])

            data['test_label'] = self._generate_label_function(data['test_anno'], test_target_absent)

        return data


class DiMPProcessing(BaseProcessing):
    """ The processing class used for training DiMP. The images are processed in the following way.
    First, the target bounding box is jittered by adding some noise. Next, a square region (called search region )
    centered at the jittered target center, and of area search_area_factor^2 times the area of the jittered box is
    cropped from the image. The reason for jittering the target box is to avoid learning the bias that the target is
    always at the center of the search region. The search region is then resized to a fixed size given by the
    argument output_sz. A Gaussian label centered at the target is generated for each image. These label functions are
    used for computing the loss of the predicted classification model on the test images. A set of proposals are
    also generated for the test images by jittering the ground truth box. These proposals are used to train the
    bounding box estimating branch.

    """

    def __init__(self, search_area_factor, output_sz, center_jitter_factor, scale_jitter_factor, crop_type='replicate',
                 max_scale_change=None, mode='pair', proposal_params=None, label_function_params=None, *args, **kwargs):
        """
        args:
            search_area_factor - The size of the search region  relative to the target size.
            output_sz - An integer, denoting the size to which the search region is resized. The search region is always
                        square.
            center_jitter_factor - A dict containing the amount of jittering to be applied to the target center before
                                    extracting the search region. See _get_jittered_box for how the jittering is done.
            scale_jitter_factor - A dict containing the amount of jittering to be applied to the target size before
                                    extracting the search region. See _get_jittered_box for how the jittering is done.
            crop_type - If 'replicate', the boundary pixels are replicated in case the search region crop goes out of image.
                        If 'inside', the search region crop is shifted/shrunk to fit completely inside the image.
                        If 'inside_major', the search region crop is shifted/shrunk to fit completely inside one axis of the image.
            max_scale_change - Maximum allowed scale change when performing the crop (only applicable for 'inside' and 'inside_major')
            mode - Either 'pair' or 'sequence'. If mode='sequence', then output has an extra dimension for frames
            proposal_params - Arguments for the proposal generation process. See _generate_proposals for details.
            label_function_params - Arguments for the label generation process. See _generate_label_function for details.
        """
        super().__init__(*args, **kwargs)
        self.search_area_factor = search_area_factor
        self.output_sz = output_sz
        self.center_jitter_factor = center_jitter_factor
        self.scale_jitter_factor = scale_jitter_factor
        self.crop_type = crop_type
        self.mode = mode
        self.max_scale_change = max_scale_change

        self.proposal_params = proposal_params
        self.label_function_params = label_function_params

    def _get_jittered_box(self, box, mode):
        """ Jitter the input box
        args:
            box - input bounding box
            mode - string 'train' or 'test' indicating train or test data

        returns:
            torch.Tensor - jittered box
        """

        jittered_size = box[2:4] * torch.exp(torch.randn(2) * self.scale_jitter_factor[mode])
        max_offset = (jittered_size.prod().sqrt() * torch.tensor(self.center_jitter_factor[mode]).float())
        jittered_center = box[0:2] + 0.5 * box[2:4] + max_offset * (torch.rand(2) - 0.5)

        return torch.cat((jittered_center - 0.5 * jittered_size, jittered_size), dim=0)

    def _generate_proposals(self, box):
        """ Generates proposals by adding noise to the input box
        args:
            box - input box

        returns:
            torch.Tensor - Array of shape (num_proposals, 4) containing proposals
            torch.Tensor - Array of shape (num_proposals,) containing IoU overlap of each proposal with the input box. The
                        IoU is mapped to [-1, 1]
        """
        # Generate proposals
        num_proposals = self.proposal_params['boxes_per_frame']
        proposal_method = self.proposal_params.get('proposal_method', 'default')

        if proposal_method == 'default':
            proposals = torch.zeros((num_proposals, 4))
            gt_iou = torch.zeros(num_proposals)

            for i in range(num_proposals):
                proposals[i, :], gt_iou[i] = prutils.perturb_box(box, min_iou=self.proposal_params['min_iou'],
                                                                 sigma_factor=self.proposal_params['sigma_factor'])
        elif proposal_method == 'gmm':
            proposals, _, _ = prutils.sample_box_gmm(box, self.proposal_params['proposal_sigma'],
                                                     num_samples=num_proposals)
            gt_iou = prutils.iou(box.view(1, 4), proposals.view(-1, 4))
        else:
            raise ValueError('Unknown proposal method.')

        # Map to [-1, 1]
        gt_iou = gt_iou * 2 - 1
        return proposals, gt_iou

    def _generate_label_function(self, target_bb):
        """ Generates the gaussian label function centered at target_bb
        args:
            target_bb - target bounding box (num_images, 4)

        returns:
            torch.Tensor - Tensor of shape (num_images, label_sz, label_sz) containing the label for each sample
        """

        gauss_label = prutils.gaussian_label_function(target_bb.view(-1, 4), self.label_function_params['sigma_factor'],
                                                      self.label_function_params['kernel_sz'],
                                                      self.label_function_params['feature_sz'], self.output_sz,
                                                      end_pad_if_even=self.label_function_params.get('end_pad_if_even', True))

        return gauss_label

    def __call__(self, data: TensorDict):
        """
        args:
            data - The input data, should contain the following fields:
                'train_images', test_images', 'train_anno', 'test_anno'
        returns:
            TensorDict - output data block with following fields:
                'train_images', 'test_images', 'train_anno', 'test_anno', 'test_proposals', 'proposal_iou',
                'test_label' (optional), 'train_label' (optional), 'test_label_density' (optional), 'train_label_density' (optional)
        """

        if self.transform['joint'] is not None:
            data['train_images'], data['train_anno'] = self.transform['joint'](image=data['train_images'], bbox=data['train_anno'])
            data['test_images'], data['test_anno'] = self.transform['joint'](image=data['test_images'], bbox=data['test_anno'], new_roll=False)

        for s in ['train', 'test']:
            assert self.mode == 'sequence' or len(data[s + '_images']) == 1, \
                "In pair mode, num train/test frames must be 1"

            # Add a uniform noise to the center pos
            jittered_anno = [self._get_jittered_box(a, s) for a in data[s + '_anno']]

            crops, boxes = prutils.target_image_crop(data[s + '_images'], jittered_anno, data[s + '_anno'],
                                                     self.search_area_factor, self.output_sz, mode=self.crop_type,
                                                     max_scale_change=self.max_scale_change)

            data[s + '_images'], data[s + '_anno'] = self.transform[s](image=crops, bbox=boxes, joint=False)

        # Generate proposals
        if self.proposal_params:
            frame2_proposals, gt_iou = zip(*[self._generate_proposals(a) for a in data['test_anno']])

            data['test_proposals'] = list(frame2_proposals)
            data['proposal_iou'] = list(gt_iou)

        # Prepare output
        if self.mode == 'sequence':
            data = data.apply(stack_tensors)
        else:
            data = data.apply(lambda x: x[0] if isinstance(x, list) else x)

        # Generate label functions
        if self.label_function_params is not None:
            data['train_label'] = self._generate_label_function(data['train_anno'])
            data['test_label'] = self._generate_label_function(data['test_anno'])

        return data


class KLDiMPProcessing(BaseProcessing):
    """ The processing class used for training PrDiMP that additionally supports the probabilistic classifier and
    bounding box regressor. See DiMPProcessing for details.
    """

    def __init__(self, search_area_factor, output_sz, center_jitter_factor, scale_jitter_factor, crop_type='replicate',
                 max_scale_change=None, mode='pair', proposal_params=None,
                 label_function_params=None, label_density_params=None, *args, **kwargs):
        """
        args:
            search_area_factor - The size of the search region  relative to the target size.
            output_sz - An integer, denoting the size to which the search region is resized. The search region is always
                        square.
            center_jitter_factor - A dict containing the amount of jittering to be applied to the target center before
                                    extracting the search region. See _get_jittered_box for how the jittering is done.
            scale_jitter_factor - A dict containing the amount of jittering to be applied to the target size before
                                    extracting the search region. See _get_jittered_box for how the jittering is done.
            crop_type - If 'replicate', the boundary pixels are replicated in case the search region crop goes out of image.
                        If 'inside', the search region crop is shifted/shrunk to fit completely inside the image.
                        If 'inside_major', the search region crop is shifted/shrunk to fit completely inside one axis of the image.
            max_scale_change - Maximum allowed scale change when performing the crop (only applicable for 'inside' and 'inside_major')
            mode - Either 'pair' or 'sequence'. If mode='sequence', then output has an extra dimension for frames
            proposal_params - Arguments for the proposal generation process. See _generate_proposals for details.
            label_function_params - Arguments for the label generation process. See _generate_label_function for details.
            label_density_params - Arguments for the label density generation process. See _generate_label_function for details.
        """
        super().__init__(*args, **kwargs)
        self.search_area_factor = search_area_factor
        self.output_sz = output_sz
        self.center_jitter_factor = center_jitter_factor
        self.scale_jitter_factor = scale_jitter_factor
        self.crop_type = crop_type
        self.mode = mode
        self.max_scale_change = max_scale_change

        self.proposal_params = proposal_params
        self.label_function_params = label_function_params
        self.label_density_params = label_density_params

    def _get_jittered_box(self, box, mode):
        """ Jitter the input box
        args:
            box - input bounding box
            mode - string 'train' or 'test' indicating train or test data

        returns:
            torch.Tensor - jittered box
        """

        jittered_size = box[2:4] * torch.exp(torch.randn(2) * self.scale_jitter_factor[mode])
        max_offset = (jittered_size.prod().sqrt() * torch.tensor(self.center_jitter_factor[mode]).float())
        jittered_center = box[0:2] + 0.5 * box[2:4] + max_offset * (torch.rand(2) - 0.5)

        return torch.cat((jittered_center - 0.5 * jittered_size, jittered_size), dim=0)

    def _generate_proposals(self, box):
        """ Generate proposal sample boxes from a GMM proposal distribution and compute their ground-truth density.
        This is used for ML and KL based regression learning of the bounding box regressor.
        args:
            box - input bounding box
        """
        # Generate proposals
        sampling_method = self.proposal_params.get('sampling_method', 'gmm')
        if sampling_method == 'gmm':
            proposals, proposal_density, gt_density = prutils.sample_box_gmm(box, self.proposal_params['proposal_sigma'],
                                                                             gt_sigma=self.proposal_params['gt_sigma'],
                                                                             num_samples=self.proposal_params['boxes_per_frame'],
                                                                             add_mean_box=self.proposal_params.get('add_mean_box', False))

        elif sampling_method == 'ncep_gmm':
            proposals, proposal_density, gt_density = prutils.ncep_sample_box_gmm(box, self.proposal_params['proposal_sigma'],
                                                                                 beta=self.proposal_params['beta'],
                                                                                 gt_sigma=self.proposal_params['gt_sigma'],
                                                                                 num_samples=self.proposal_params['boxes_per_frame'],
                                                                                 add_mean_box=self.proposal_params.get('add_mean_box', False))
        else:
            raise Exception('Wrong sampling method.')

        return proposals, proposal_density, gt_density

    def _generate_label_function(self, target_bb):
        """ Generates the gaussian label function centered at target_bb
        args:
            target_bb - target bounding box (num_images, 4)

        returns:
            torch.Tensor - Tensor of shape (num_images, label_sz, label_sz) containing the label for each sample
        """

        gauss_label = prutils.gaussian_label_function(target_bb.view(-1, 4), self.label_function_params['sigma_factor'],
                                                      self.label_function_params['kernel_sz'],
                                                      self.label_function_params['feature_sz'], self.output_sz,
                                                      end_pad_if_even=self.label_function_params.get('end_pad_if_even', True))

        return gauss_label

    def _generate_label_density(self, target_bb):
        """ Generates the gaussian label density centered at target_bb
        args:
            target_bb - target bounding box (num_images, 4)

        returns:
            torch.Tensor - Tensor of shape (num_images, label_sz, label_sz) containing the label for each sample
        """

        feat_sz = self.label_density_params['feature_sz'] * self.label_density_params.get('interp_factor', 1)
        gauss_label = prutils.gaussian_label_function(target_bb.view(-1, 4), self.label_density_params['sigma_factor'],
                                                      self.label_density_params['kernel_sz'],
                                                      feat_sz, self.output_sz,
                                                      end_pad_if_even=self.label_density_params.get('end_pad_if_even', True),
                                                      density=True,
                                                      uni_bias=self.label_density_params.get('uni_weight', 0.0))

        gauss_label *= (gauss_label > self.label_density_params.get('threshold', 0.0)).float()

        if self.label_density_params.get('normalize', False):
            g_sum = gauss_label.sum(dim=(-2,-1))
            valid = g_sum>0.01
            gauss_label[valid, :, :] /= g_sum[valid].view(-1, 1, 1)
            gauss_label[~valid, :, :] = 1.0 / (gauss_label.shape[-2] * gauss_label.shape[-1])

        gauss_label *= 1.0 - self.label_density_params.get('shrink', 0.0)

        return gauss_label

    def __call__(self, data: TensorDict):
        """
        args:
            data - The input data, should contain the following fields:
                'train_images', test_images', 'train_anno', 'test_anno'
        returns:
            TensorDict - output data block with following fields:
                'train_images', 'test_images', 'train_anno', 'test_anno', 'test_proposals', 'proposal_density', 'gt_density',
                'test_label' (optional), 'train_label' (optional), 'test_label_density' (optional), 'train_label_density' (optional)
        """

        if self.transform['joint'] is not None:
            data['train_images'], data['train_anno'] = self.transform['joint'](image=data['train_images'], bbox=data['train_anno'])
            data['test_images'], data['test_anno'] = self.transform['joint'](image=data['test_images'], bbox=data['test_anno'], new_roll=False)

        for s in ['train', 'test']:
            assert self.mode == 'sequence' or len(data[s + '_images']) == 1, \
                "In pair mode, num train/test frames must be 1"

            # Add a uniform noise to the center pos
            jittered_anno = [self._get_jittered_box(a, s) for a in data[s + '_anno']]

            crops, boxes = prutils.target_image_crop(data[s + '_images'], jittered_anno, data[s + '_anno'],
                                                     self.search_area_factor, self.output_sz, mode=self.crop_type,
                                                     max_scale_change=self.max_scale_change)

            data[s + '_images'], data[s + '_anno'] = self.transform[s](image=crops, bbox=boxes, joint=False)

        # Generate proposals
        proposals, proposal_density, gt_density = zip(*[self._generate_proposals(a) for a in data['test_anno']])

        data['test_proposals'] = proposals
        data['proposal_density'] = proposal_density
        data['gt_density'] = gt_density

        for s in ['train', 'test']:
            is_distractor = data.get('is_distractor_{}_frame'.format(s), None)
            if is_distractor is not None:
                for is_dist, box in zip(is_distractor, data[s+'_anno']):
                    if is_dist:
                        box[0] = 99999999.9
                        box[1] = 99999999.9

        # Prepare output
        if self.mode == 'sequence':
            data = data.apply(stack_tensors)
        else:
            data = data.apply(lambda x: x[0] if isinstance(x, list) else x)

        # Generate label functions
        if self.label_function_params is not None:
            data['train_label'] = self._generate_label_function(data['train_anno'])
            data['test_label'] = self._generate_label_function(data['test_anno'])
        if self.label_density_params is not None:
            data['train_label_density'] = self._generate_label_density(data['train_anno'])
            data['test_label_density'] = self._generate_label_density(data['test_anno'])

        return data


from ltr.models.memory_learning import peak_prediction
from pytracking import dcf
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class PeakMatchingProcessing(BaseProcessing):
    def __init__(self, output_sz, num_peaks=None, peak_th=0.05, ks=5, mode='aug', img_aug_transform=None, score_sz=23,
                 enable_search_area_aug=True, search_area_jitter_value=100, real_peaks_only=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.output_sz = output_sz
        self.num_peaks = num_peaks
        self.peak_th = peak_th
        self.ks = ks
        self.mode = mode
        self.img_aug_transform = img_aug_transform
        self.enable_search_area_aug = enable_search_area_aug
        self.search_area_jitter_value = search_area_jitter_value
        self.real_peaks_only = real_peaks_only
        self.score_sz = score_sz

    def _find_gt_peak_index(self, coords, anno_label):
        val, coord = dcf.max2d(anno_label)
        gt_idx = torch.argmin(torch.sum((coords - coord) ** 2, dim=1))
        return gt_idx

    def _gt_peak_drop_out(self):
        dropout = (torch.rand(1) < 0.25).item()
        # dropout = False
        frameid = torch.randint(0, 2, (1,)).item()
        return frameid, dropout

    def _pad_with_fake_peaks_drop_gt(self, img_coords, dropout, gt_idx, sa_box, img_shape):
        H, W = img_shape[:2]
        num_peaks = min(img_coords.shape[0], self.num_peaks)
        x, y, w, h = sa_box.long().tolist()

        lowx, lowy, highx, highy = max(0, x), max(0, y), min(W, x + w), min(H, y + h)

        img_coords_pad = torch.zeros((self.num_peaks, 2))
        valid = torch.zeros(self.num_peaks)

        img_coords_pad[:num_peaks] = img_coords[:num_peaks]
        valid[:num_peaks] = 1

        gt_coords = img_coords_pad[gt_idx].clone().unsqueeze(0)

        if dropout:
            valid[gt_idx] = 0
            img_coords_pad[gt_idx] = 0

        filled = valid.clone()
        for i in range(0, self.num_peaks):
            if filled[i] == 0:
                cs = torch.cat([
                    torch.rand((20, 1)) * (highy - lowy) + lowy,
                    torch.rand((20, 1)) * (highx - lowx) + lowx
                ], dim=1)

                cs_used = torch.cat([img_coords_pad[filled == 1], gt_coords], dim=0)

                dist = torch.sqrt(torch.sum((cs_used[:, None, :] - cs[None, :, :]) ** 2, dim=2))
                min_dist = torch.min(dist, dim=0).values
                max_min_dist_idx = torch.argmax(min_dist)
                img_coords_pad[i] = cs[max_min_dist_idx]
                filled[i] = 1

        return img_coords_pad, valid

    def _old_and_current_frame(self, data: TensorDict):
        out = TensorDict()
        imgs = data.pop('img')
        img0 = imgs[0]
        img1 = imgs[1]
        sa_box0 = data['search_area_box'][0]
        sa_box1 = data['search_area_box'][1]
        target_score0 = data['target_scores'][0]
        target_score1 = data['target_scores'][1]
        anno_label0 = data['anno_label'][0]
        anno_label1 = data['anno_label'][1]

        out['img_shape0'] = [torch.tensor(img0.shape[:2])]
        out['img_shape1'] = [torch.tensor(img1.shape[:2])]

        frame_crop0, _ = prutils.sample_target_from_crop_region(img0, sa_box0, self.output_sz)
        frame_crop1, _ = prutils.sample_target_from_crop_region(img1, sa_box1, self.output_sz)

        frame_crop0 = self.transform['train'](image=frame_crop0)
        frame_crop1 = self.transform['train'](image=frame_crop1)

        out['img_cropped0'] = [frame_crop0]
        out['img_cropped1'] = [frame_crop1]

        target_score0 = target_score0.squeeze()
        target_score1 = target_score1.squeeze()

        ts_coords0, scores0 = peak_prediction.find_local_maxima(target_score0, th=self.peak_th, ks=self.ks)
        ts_coords1, scores1 = peak_prediction.find_local_maxima(target_score1, th=self.peak_th, ks=self.ks)

        gt_idx0 = self._find_gt_peak_index(ts_coords0, anno_label0.squeeze())
        gt_idx1 = self._find_gt_peak_index(ts_coords1, anno_label1.squeeze())

        x0, y0, w0, h0 = sa_box0.tolist()
        x1, y1, w1, h1 = sa_box1.tolist()

        img_coords0 = torch.stack([
            h0 * (ts_coords0[:, 0].float() / (target_score0.shape[0] - 1)) + y0,
            w0 * (ts_coords0[:, 1].float() / (target_score0.shape[1] - 1)) + x0
        ]).permute(1, 0)

        img_coords1 = torch.stack([
            h1 * (ts_coords1[:, 0].float() / (target_score1.shape[0] - 1)) + y1,
            w1 * (ts_coords1[:, 1].float() / (target_score1.shape[1] - 1)) + x1
        ]).permute(1, 0)

        frameid, dropout = self._gt_peak_drop_out()

        drop0 = dropout & (frameid == 0)
        drop1 = dropout & (frameid == 1)

        img_coords_pad0, valid0 = self._pad_with_fake_peaks_drop_gt(img_coords0, drop0, gt_idx0, sa_box0, img0.shape)
        img_coords_pad1, valid1 = self._pad_with_fake_peaks_drop_gt(img_coords1, drop1, gt_idx1, sa_box1, img1.shape)

        scores_pad0 = self._add_fake_peak_scores(scores0, valid0)
        scores_pad1 = self._add_fake_peak_scores(scores1, valid1)

        x0, y0, w0, h0 = sa_box0.long().tolist()
        x1, y1, w1, h1 = sa_box1.long().tolist()


        ts_coords_pad0 = torch.stack([
            torch.round((img_coords_pad0[:, 0] - y0) / h0 * (target_score0.shape[0] - 1)).long(),
            torch.round((img_coords_pad0[:, 1] - x0) / w0 * (target_score0.shape[1] - 1)).long()
        ]).permute(1, 0)

        ts_coords_pad1 = torch.stack([
            torch.round((img_coords_pad1[:, 0] - y1) / h1 * (target_score1.shape[0] - 1)).long(),
            torch.round((img_coords_pad1[:, 1] - x1) / w1 * (target_score1.shape[1] - 1)).long()
        ]).permute(1, 0)

        assert torch.all(ts_coords_pad0 >= 0) and torch.all(ts_coords_pad0 < target_score0.shape[0])
        assert torch.all(ts_coords_pad1 >= 0) and torch.all(ts_coords_pad1 < target_score1.shape[0])

        out['peak_img_coords0'] = [img_coords_pad0]
        out['peak_img_coords1'] = [img_coords_pad1]
        out['peak_tsm_coords0'] = [ts_coords_pad0]
        out['peak_tsm_coords1'] = [ts_coords_pad1]
        out['peak_scores0'] = [scores_pad0]
        out['peak_scores1'] = [scores_pad1]
        out['peak_valid0'] = [valid0]
        out['peak_valid1'] = [valid1]

        # Prepare gt labels
        gt_assignment = torch.zeros((self.num_peaks, self.num_peaks))
        gt_assignment[gt_idx0, gt_idx1] = valid0[gt_idx0]*valid1[gt_idx1]

        gt_matches0 = torch.zeros(self.num_peaks) - 2
        gt_matches1 = torch.zeros(self.num_peaks) - 2

        if drop0:
            gt_matches0[gt_idx0] = -2
            gt_matches1[gt_idx1] = -1
        elif drop1:
            gt_matches0[gt_idx0] = -1
            gt_matches0[gt_idx1] = -2
        else:
            gt_matches0[gt_idx0] = gt_idx1
            gt_matches1[gt_idx1] = gt_idx0

        out['gt_matches0'] = [gt_matches0]
        out['gt_matches1'] = [gt_matches1]
        out['gt_assignment'] = [gt_assignment]

        return out

    def _peak_drop_out(self, coords0, coords1):
        num_peaks = min(coords1.shape[0], self.num_peaks)
        # TODO: HOW MANY PEAKS SHOULD BE DROPPED? 0.5?
        num_peaks_to_drop = torch.round(0.25*num_peaks*torch.rand(1)).long()
        idx = torch.randperm(num_peaks)[:num_peaks_to_drop]

        coords_pad0 = torch.zeros((self.num_peaks, 2))
        valid0 = torch.zeros(self.num_peaks)
        coords_pad1 = torch.zeros((self.num_peaks, 2))
        valid1 = torch.zeros(self.num_peaks)

        coords_pad0[:num_peaks] = coords0[:num_peaks]
        coords_pad1[:num_peaks] = coords1[:num_peaks]

        valid0[:num_peaks] = 1
        valid1[:num_peaks] = 1

        if torch.rand(1) < 0.5:
            coords_pad0[idx] = 0
            valid0[idx] = 0
        else:
            coords_pad1[idx] = 0
            valid1[idx] = 0

        return coords_pad0, coords_pad1, valid0, valid1

    def _pad_with_fake_peaks(self, img_coords_pad0, img_coords_pad1, valid0, valid1, sa_box0, sa_box1, img_shape):
        H, W = img_shape[:2]

        x0, y0, w0, h0 = sa_box0.long().tolist()
        x1, y1, w1, h1 = sa_box1.long().tolist()

        lowx = [max(0, x0), max(0, x1)]
        lowy = [max(0, y0), max(0, y1)]
        highx = [min(W, x0 + w0), min(W, x1 + w1)]
        highy = [min(H, y0 + h0), min(H, y1 + h1)]

        filled = [valid0.clone(), valid1.clone()]
        img_coords_pad = [img_coords_pad0.clone(), img_coords_pad1.clone()]

        for i in range(0, self.num_peaks):
            for k in range(0, 2):
                if filled[k][i] == 0:
                    cs = torch.cat([
                        torch.rand((20, 1)) * (highy[k] - lowy[k]) + lowy[k],
                        torch.rand((20, 1)) * (highx[k] - lowx[k]) + lowx[k]
                    ], dim=1)

                    cs_used = torch.cat([img_coords_pad[0][filled[0]==1], img_coords_pad[1][filled[1]==1]], dim=0)

                    dist = torch.sqrt(torch.sum((cs_used[:, None, :] - cs[None, :, :]) ** 2, dim=2))
                    min_dist = torch.min(dist, dim=0).values
                    max_min_dist_idx = torch.argmax(min_dist)
                    img_coords_pad[k][i] = cs[max_min_dist_idx]
                    filled[k][i] = 1

        return img_coords_pad[0], img_coords_pad[1]

    def _add_fake_peak_scores(self, scores, valid):
        scores_pad = torch.zeros(valid.shape[0])
        scores_pad[valid == 1] = scores[:self.num_peaks][valid[:scores.shape[0]] == 1]
        scores_pad[valid == 0] = (torch.abs(torch.randn((valid==0).sum()))/50).clamp_max(0.025) + 0.05
        return scores_pad

    def _augment_scores(self, scores, valid, drop):
        num_valid = (valid==1).sum()

        noise = 0.1 * torch.randn(num_valid)

        if num_valid > 2 and not drop:
            if scores[1] > 0.5*scores[0] and torch.all(scores[:2] > 0.2):
                # two valid peaks with a high score that are relatively close.
                mode = torch.randint(0, 3, size=(1,))
                if mode == 0:
                    # augment randomly.
                    scores_aug = torch.sort(noise + scores[valid==1], descending=True)[0]
                elif mode == 1:
                    # move peaks closer
                    scores_aug = torch.sort(noise + scores[valid == 1], descending=True)[0]
                    scores_aug[0] = scores[valid==1][0] - torch.abs(noise[0])
                    scores_aug[1] = scores[valid==1][1] + torch.abs(noise[1])
                    scores_aug[:2] = torch.sort(scores_aug[:2], descending=True)[0]
                else:
                    # move peaks closer and switch
                    scores_aug = torch.sort(noise + scores[valid == 1], descending=True)[0]
                    scores_aug[0] = scores[valid==1][0] - torch.abs(noise[0])
                    scores_aug[1] = scores[valid==1][1] + torch.abs(noise[1])
                    scores_aug[:2] = torch.sort(scores_aug[:2], descending=True)[0]

                    idx = torch.arange(num_valid)
                    idx[:2] = torch.tensor([1, 0])
                    scores_aug = scores_aug[idx]
            else:
                scores_aug = torch.sort(scores[valid==1] + noise, descending=True)[0]

        else:
            scores_aug = torch.sort(scores[valid == 1] + noise, descending=True)[0]

        scores_aug = scores_aug.clamp_min(0.075)

        scores[valid==1] = scores_aug.clone()

        return scores

    def _proj_coords_inside_image(self, coords0, coords1, img_shape):
        H, W = img_shape[:2]

        xmin = torch.min(coords0[:, 1].min(), coords1[:, 1].min())
        ymin = torch.min(coords0[:, 0].min(), coords1[:, 0].min())

        if xmin < 0:
            tx = torch.rand(1)*torch.abs(xmin) + torch.abs(xmin)
            coords0[:, 1] += tx
            coords1[:, 1] += tx

        if ymin < 0:
            ty = torch.rand(1)*torch.abs(ymin) + torch.abs(ymin)
            coords0[:, 0] += ty
            coords1[:, 0] += ty

        xmax = torch.max(coords0[:, 1].max(), coords1[:, 1].max())
        ymax = torch.max(coords0[:, 0].max(), coords1[:, 0].max())

        if xmax > W:
            sx = torch.rand(1)*0.2*xmax + xmax
            coords0[:, 1] /= sx
            coords1[:, 1] /= sx

        if ymax > H:
            sy = torch.rand(1)*0.2*ymax + ymax
            coords0[:, 0] /= sy
            coords1[:, 0] /= sy

        return coords0, coords1

    def _augment_coords(self, coords, img_shape, search_area_box):
        H, W = img_shape[:2]

        # # add x,y offset to all img_coords
        # ymax, ymin = H - coords[:, 0].max(), 0 - coords[:, 0].min()
        # xmax, xmin = W - coords[:, 1].max(), 0 - coords[:, 1].min()
        #
        # _, _, w, h = search_area_box.float()
        # # make sure sampling range is valid and not too big only
        # ymin = torch.max(ymin, -1*h/self.score_sz)
        # ymax = torch.min(ymax, h/self.score_sz)
        # xmin = torch.max(xmin, -w/self.score_sz)
        # xmax = torch.min(ymax, w/self.score_sz)
        #
        # ty = (torch.rand(1)*(ymax - ymin) + ymin)
        # tx = (torch.rand(1)*(xmax - xmin) + xmin)
        #
        # coords[:, 0] += ty
        # coords[:, 1] += tx
        #
        # # scale all img_coords along x,y dim
        # # h = coords[:, 0].max() - coords[:, 0].min()
        # # w = coords[:, 1].max() - coords[:, 1].min()
        #
        # sxmax = torch.min(1.1*torch.ones(1), W/coords[:, 1].max())
        # symax = torch.min(1.1*torch.ones(1), H/coords[:, 0].max())
        #
        # sxmin = 0.9*torch.ones(1)
        # symin = 0.9*torch.ones(1)
        #
        # sy = torch.rand(1)*(symax - symin) + symin
        # sx = torch.rand(1)*(sxmax - sxmin) + sxmin

        # # subtract mean before scaling
        # center = coords.mean(0)
        # coords -= center
        #
        # # coords[:, 0] *= sy
        # # coords[:, 1] *= sx
        #
        # # add independent small scaling for each coord
        # syi = torch.rand(coords.shape[0])*(0.05) + 0.95
        # sxi = torch.rand(coords.shape[0])*(0.05) + 0.95
        #
        # coords[:, 0] *= syi
        # coords[:, 1] *= sxi
        #
        # # add mean back
        # coords += center

        _, _, w, h = search_area_box.float()

        # add independent offset to each coord
        d = torch.sqrt(torch.sum((coords[None, :] - coords[:, None])**2, dim=2))

        if torch.all(d == 0):
            xmin = 0.5*w/self.score_sz
            ymin = 0.5*h/self.score_sz
        else:
            dmin = torch.min(d[d>0])
            xmin = (math.sqrt(2)*dmin/4).clamp_max(w/self.score_sz)
            ymin = (math.sqrt(2)*dmin/4).clamp_max(h/self.score_sz)



        # dx = torch.abs(coords[:, 1][None, :] - coords[:, 1][:, None])
        # dy = torch.abs(coords[:, 0][None, :] - coords[:, 0][:, None])
        # if torch.all(dx == 0):
        #     xmin = 0.5*w/self.score_sz
        # else:
        #     xmin = torch.min(dx[dx>0].min()*0.1, 0.5*w/self.score_sz)
        #
        # if torch.all(dy == 0):
        #     ymin = 0.5*h/self.score_sz
        # else:
        #     ymin = torch.min(dy[dy > 0].min() * 0.1, 0.5*h/self.score_sz)

        txi = torch.rand(coords.shape[0])*2*xmin - xmin
        tyi = torch.rand(coords.shape[0])*2*ymin - ymin

        coords[:, 0] += tyi
        coords[:, 1] += txi

        coords[:, 0] = coords[:, 0].clamp(0, H)
        coords[:, 1] = coords[:, 1].clamp(0, W)

        return coords

    def _original_and_augmented_frame(self, data: TensorDict):
        out = TensorDict()
        img = data.pop('img')[0]
        sa_box = data['search_area_box'][0]
        sa_box0 = sa_box.clone()
        sa_box1 = sa_box.clone()
        target_score = data['target_scores'][0]

        out['img_shape0'] = [torch.tensor(img.shape[:2])]
        out['img_shape1'] = [torch.tensor(img.shape[:2])]

        # prepared cropped image
        frame_crop0, _ = prutils.sample_target_from_crop_region(img, sa_box, self.output_sz)

        x, y, w, h = sa_box.long().tolist()

        if self.enable_search_area_aug:
            l = self.search_area_jitter_value
            sa_box1 = torch.tensor([x + torch.randint(-w//l, w//l+1, (1,)),
                                    y + torch.randint(-h//l, h//l+1, (1,)),
                                    w + torch.randint(-w//l, w//l+1, (1,)),
                                    h + torch.randint(-h//l, h//l+1, (1,))])

        frame_crop1, _ = prutils.sample_target_from_crop_region(img, sa_box1, self.output_sz)

        frame_crop0 = self.transform['train'](image=frame_crop0)
        frame_crop1 = self.img_aug_transform(image=frame_crop1)
        # frame_crop1 = self.transform['train'](image=frame_crop1)

        out['img_cropped0'] = [frame_crop0]
        out['img_cropped1'] = [frame_crop1]

        # prepare peaks
        target_score = target_score.squeeze()
        ts_coords, scores = peak_prediction.find_local_maxima(target_score, th=self.peak_th, ks=self.ks)

        x, y, w, h = sa_box0.tolist()
        img_coords = torch.stack([
            h * (ts_coords[:, 0].float() / (target_score.shape[0] - 1)) + y,
            w * (ts_coords[:, 1].float() / (target_score.shape[1] - 1)) + x
        ]).permute(1, 0)

        img_coords_pad0, img_coords_pad1, valid0, valid1 = self._peak_drop_out(img_coords, img_coords.clone())

        img_coords_pad0, img_coords_pad1 = self._pad_with_fake_peaks(img_coords_pad0, img_coords_pad1, valid0, valid1,
                                                                     sa_box0, sa_box1, img.shape)

        scores_pad0 = self._add_fake_peak_scores(scores, valid0)
        scores_pad1 = self._add_fake_peak_scores(scores, valid1)

        x0, y0, w0, h0 = sa_box0.long().tolist()


        ts_coords_pad0 = torch.stack([
            torch.round((img_coords_pad0[:, 0] - y0) / h0 * (target_score.shape[0] - 1)).long(),
            torch.round((img_coords_pad0[:, 1] - x0) / w0 * (target_score.shape[1] - 1)).long()
        ]).permute(1, 0)

        # make sure that the augmented search_are_box is only used for the fake img_coords the other need the original.
        x1, y1, w1, h1 = sa_box1.long().tolist()
        y = torch.where(valid1 == 1, torch.tensor(y0), torch.tensor(y1))
        x = torch.where(valid1 == 1, torch.tensor(x0), torch.tensor(x1))
        h = torch.where(valid1 == 1, torch.tensor(h0), torch.tensor(h1))
        w = torch.where(valid1 == 1, torch.tensor(w0), torch.tensor(w1))

        ts_coords_pad1 = torch.stack([
            torch.round((img_coords_pad1[:, 0] - y) / h * (target_score.shape[0] - 1)).long(),
            torch.round((img_coords_pad1[:, 1] - x) / w * (target_score.shape[1] - 1)).long()
        ]).permute(1, 0)

        assert torch.all(ts_coords_pad0 >= 0) and torch.all(ts_coords_pad0 < target_score.shape[0])
        assert torch.all(ts_coords_pad1 >= 0) and torch.all(ts_coords_pad1 < target_score.shape[0])

        # img_coords_pad0, img_coords_pad1 = self._proj_coords_inside_image(img_coords_pad0, img_coords_pad1, img.shape)


        # fig, axes = plt.subplots(3,1, figsize=(10, 15))
        # img0 = torch.zeros(img.shape)
        # for i, c in enumerate(img_coords_pad0):
        #     val = torch.tensor([255, 0, 0]) if valid0[i] == 0 else torch.tensor([0, 255, 0])
        #     img0[max(0, c[0].long()-10):min(c[0].long()+9, img.shape[0]), max(0, c[1].long()-10):min(c[1].long()+9, img.shape[1])] = val
        # axes[0].imshow(img0)

        # img1 = torch.zeros(img.shape)
        # for i, c in enumerate(img_coords_pad1):
        #     val = torch.tensor([255, 0, 0]) if valid1[i] == 0 else torch.tensor([0, 255, 0])
        #     img1[max(0, c[0].long()-10):min(c[0].long()+9, img.shape[0]), max(0, c[1].long()-10):min(c[1].long()+9, img.shape[1])] = val
        # axes[1].imshow(img1)

        img_coords_pad1 = self._augment_coords(img_coords_pad1, img.shape, sa_box1)
        scores_pad1 = self._augment_scores(scores_pad1, valid1, ~torch.all(valid0 == valid1))

        # img1 = torch.zeros(img.shape)
        # for i, c in enumerate(img_coords_pad1):
        #     val = torch.tensor([255,0,0]) if valid1[i] == 0 else torch.tensor([0,255,0])
        #     img1[max(0, c[0].long()-10):min(c[0].long()+9, img.shape[0]), max(0, c[1].long()-10):min(c[1].long()+9, img.shape[1])] = val
        # axes[2].imshow(img1)
        # plt.show()

        # if (torch.any(img_coords_pad0[:, 0] > img.shape[0]) or torch.any(img_coords_pad0[:, 1] > img.shape[1]) or
        #         torch.any(img_coords_pad1[:, 0] > img.shape[0]) or torch.any(img_coords_pad1[:, 1] > img.shape[1]) or
        #         torch.any(img_coords_pad0 < 0) or torch.any(img_coords_pad1 < 0)):
        #     print()

        out['peak_img_coords0'] = [img_coords_pad0]
        out['peak_img_coords1'] = [img_coords_pad1]
        out['peak_tsm_coords0'] = [ts_coords_pad0]
        out['peak_tsm_coords1'] = [ts_coords_pad1]
        out['peak_scores0'] = [scores_pad0]
        out['peak_scores1'] = [scores_pad1]
        out['peak_valid0'] = [valid0]
        out['peak_valid1'] = [valid1]

        # Prepare gt labels

        gt_assignment = torch.zeros((self.num_peaks, self.num_peaks))
        gt_assignment[torch.arange(self.num_peaks), torch.arange(self.num_peaks)] = valid0*valid1

        gt_matches0 = torch.arange(0, self.num_peaks).float()
        gt_matches1 = torch.arange(0, self.num_peaks).float()

        gt_matches0[(valid0==0) | (valid1==0)] = -1
        gt_matches1[(valid0==0) | (valid1==0)] = -1

        out['gt_matches0'] = [gt_matches0]
        out['gt_matches1'] = [gt_matches1]
        out['gt_assignment'] = [gt_assignment]

        return out

    def _old_and_current_frame_detected_peaks_only(self, data: TensorDict):
        out = TensorDict()
        imgs = data.pop('img')
        img0 = imgs[0]
        img1 = imgs[1]
        sa_box0 = data['search_area_box'][0]
        sa_box1 = data['search_area_box'][1]
        target_score0 = data['target_scores'][0]
        target_score1 = data['target_scores'][1]
        anno_label0 = data['anno_label'][0]
        anno_label1 = data['anno_label'][1]

        out['img_shape0'] = [torch.tensor(imgs[1].shape[:2])]
        out['img_shape1'] = [torch.tensor(imgs[0].shape[:2])]

        frame_crop0, _ = prutils.sample_target_from_crop_region(img0, sa_box0, self.output_sz)
        frame_crop1, _ = prutils.sample_target_from_crop_region(img1, sa_box1, self.output_sz)

        frame_crop0 = self.transform['train'](image=frame_crop0)
        frame_crop1 = self.transform['train'](image=frame_crop1)

        out['img_cropped0'] = [frame_crop0]
        out['img_cropped1'] = [frame_crop1]

        target_score0 = target_score0.squeeze()
        target_score1 = target_score1.squeeze()

        ts_coords0, scores0 = peak_prediction.find_local_maxima(target_score0, th=self.peak_th, ks=self.ks)
        ts_coords1, scores1 = peak_prediction.find_local_maxima(target_score1, th=self.peak_th, ks=self.ks)

        gt_idx0 = self._find_gt_peak_index(ts_coords0, anno_label0.squeeze())
        gt_idx1 = self._find_gt_peak_index(ts_coords1, anno_label1.squeeze())

        x0, y0, w0, h0 = sa_box0.tolist()
        x1, y1, w1, h1 = sa_box1.tolist()

        img_coords0 = torch.stack([
            h0 * (ts_coords0[:, 0].float() / (target_score0.shape[0] - 1)) + y0,
            w0 * (ts_coords0[:, 1].float() / (target_score0.shape[1] - 1)) + x0
        ]).permute(1, 0)

        img_coords1 = torch.stack([
            h1 * (ts_coords1[:, 0].float() / (target_score1.shape[0] - 1)) + y1,
            w1 * (ts_coords1[:, 1].float() / (target_score1.shape[1] - 1)) + x1
        ]).permute(1, 0)

        # frameid, dropout = self._gt_peak_drop_out()

        # drop0 = dropout & (frameid == 0)
        # drop1 = dropout & (frameid == 1)
        #
        # img_coords_pad0, valid0 = self._pad_with_fake_peaks_drop_gt(img_coords0, drop0, gt_idx0, sa_box0, img0.shape)
        # img_coords_pad1, valid1 = self._pad_with_fake_peaks_drop_gt(img_coords1, drop1, gt_idx1, sa_box1, img1.shape)

        # scores_pad0 = self._add_fake_peak_scores(scores0, valid0, small_peaks=True)
        # scores_pad1 = self._add_fake_peak_scores(scores1, valid1, small_peaks=True)

        # x0, y0, w0, h0 = sa_box0.long().tolist()
        # x1, y1, w1, h1 = sa_box1.long().tolist()


        # ts_coords_pad0 = torch.stack([
        #     torch.round((img_coords_pad0[:, 0] - y0) / h0 * (target_score0.shape[0] - 1)).long(),
        #     torch.round((img_coords_pad0[:, 1] - x0) / w0 * (target_score0.shape[1] - 1)).long()
        # ]).permute(1, 0)
        #
        # ts_coords_pad1 = torch.stack([
        #     torch.round((img_coords_pad1[:, 0] - y1) / h1 * (target_score1.shape[0] - 1)).long(),
        #     torch.round((img_coords_pad1[:, 1] - x1) / w1 * (target_score1.shape[1] - 1)).long()
        # ]).permute(1, 0)

        # assert torch.all(ts_coords_pad0 >= 0) and torch.all(ts_coords_pad0 < target_score0.shape[0])
        # assert torch.all(ts_coords_pad1 >= 0) and torch.all(ts_coords_pad1 < target_score1.shape[0])

        out['peak_img_coords0'] = [img_coords0]
        out['peak_img_coords1'] = [img_coords1]
        out['peak_tsm_coords0'] = [ts_coords0]
        out['peak_tsm_coords1'] = [ts_coords1]
        out['peak_scores0'] = [scores0]
        out['peak_scores1'] = [scores1]
        out['peak_valid0'] = [torch.ones_like(scores0)]
        out['peak_valid1'] = [torch.ones_like(scores1)]

        # Prepare gt labels
        gt_assignment = torch.zeros((scores0.shape[0], scores1.shape[0]))
        gt_assignment[gt_idx0, gt_idx1] = 1

        gt_matches0 = torch.zeros(scores0.shape[0]) - 2
        gt_matches1 = torch.zeros(scores1.shape[0]) - 2

        gt_matches0[gt_idx0] = gt_idx1
        gt_matches1[gt_idx1] = gt_idx0

        out['gt_matches0'] = [gt_matches0]
        out['gt_matches1'] = [gt_matches1]
        out['gt_assignment'] = [gt_assignment]

        return out


    def __call__(self, data: TensorDict):
        if data['proc_mode'] == 'aug':
            data = self._original_and_augmented_frame(data)
        elif data['proc_mode'] == 'temporal' and self.real_peaks_only == False:
            data = self._old_and_current_frame(data)
        elif data['proc_mode'] == 'temporal' and self.real_peaks_only == True:
            data = self._old_and_current_frame_detected_peaks_only(data)
        else:
            raise NotImplementedError()

        data = data.apply(stack_tensors)

        return data


class KLDiMPCascadeProcessing(BaseProcessing):
    """ The processing class used for training DiMP. The images are processed in the following way.
    First, the target bounding box is jittered by adding some noise. Next, a square region (called search region )
    centered at the jittered target center, and of area search_area_factor^2 times the area of the jittered box is
    cropped from the image. The reason for jittering the target box is to avoid learning the bias that the target is
    always at the center of the search region. The search region is then resized to a fixed size given by the
    argument output_sz. A Gaussian label centered at the target is generated for each image. These label functions are
    used for computing the loss of the predicted classification model on the test images. A set of proposals are
    also generated for the test images by jittering the ground truth box. These proposals are used to train the
    bounding box estimating branch.

    """

    def __init__(self, search_area_factor, output_sz, center_jitter_factor, scale_jitter_factor, crop_type='replicate',
                 max_scale_change=None, mode='pair',
                 proposal_params=None, label_function_params=None, label_density_params=None, *args, **kwargs):
        """
        args:
            search_area_factor - The size of the search region  relative to the target size.
            output_sz - An integer, denoting the size to which the search region is resized. The search region is always
                        square.
            center_jitter_factor - A dict containing the amount of jittering to be applied to the target center before
                                    extracting the search region. See _get_jittered_box for how the jittering is done.
            scale_jitter_factor - A dict containing the amount of jittering to be applied to the target size before
                                    extracting the search region. See _get_jittered_box for how the jittering is done.
            crop_type - If 'replicate', the boundary pixels are replicated in case the search region crop goes out of image.
                        If 'nopad', the search region crop is shifted/shrunk to fit completely inside the image.
            mode - Either 'pair' or 'sequence'. If mode='sequence', then output has an extra dimension for frames
            proposal_params - Arguments for the proposal generation process. See _generate_proposals for details.
            label_function_params - Arguments for the label generation process. See _generate_label_function for details.
        """
        super().__init__(*args, **kwargs)
        self.search_area_factor = search_area_factor
        self.output_sz = output_sz
        self.center_jitter_factor = center_jitter_factor
        self.scale_jitter_factor = scale_jitter_factor
        self.crop_type = crop_type
        self.mode = mode
        self.max_scale_change = max_scale_change

        self.proposal_params = proposal_params
        self.label_function_params = label_function_params
        self.label_density_params = label_density_params
        self.num_cascades = len(self.proposal_params['proposal_sigma'])

    def _get_jittered_box(self, box, mode):
        """ Jitter the input box
        args:
            box - input bounding box
            mode - string 'train' or 'test' indicating train or test data

        returns:
            torch.Tensor - jittered box
        """

        jittered_size = box[2:4] * torch.exp(torch.randn(2) * self.scale_jitter_factor[mode])
        max_offset = (jittered_size.prod().sqrt() * self.center_jitter_factor[mode]).item()
        jittered_center = box[0:2] + 0.5 * box[2:4] + max_offset * (torch.rand(2) - 0.5)

        return torch.cat((jittered_center - 0.5 * jittered_size, jittered_size), dim=0)

    def _generate_proposals(self, box, proposal_sigma, gt_sigma, num_samples):
        """
        """
        # Generate proposals

        proposals, proposal_density, gt_density = prutils.sample_box_gmm(box, proposal_sigma,
                                                                         gt_sigma=gt_sigma,
                                                                         num_samples=num_samples,
                                                                         add_mean_box=self.proposal_params.get('add_mean_box', False))

        return proposals, proposal_density, gt_density

    def _generate_label_function(self, target_bb):
        """ Generates the gaussian label function centered at target_bb
        args:
            target_bb - target bounding box (num_images, 4)

        returns:
            torch.Tensor - Tensor of shape (num_images, label_sz, label_sz) containing the label for each sample
        """

        gauss_label = prutils.gaussian_label_function(target_bb.view(-1, 4), self.label_function_params['sigma_factor'],
                                                      self.label_function_params['kernel_sz'],
                                                      self.label_function_params['feature_sz'], self.output_sz,
                                                      end_pad_if_even=self.label_function_params.get('end_pad_if_even', True))

        return gauss_label

    def _generate_label_density(self, target_bb):
        """ Generates the gaussian label density centered at target_bb
        args:
            target_bb - target bounding box (num_images, 4)

        returns:
            torch.Tensor - Tensor of shape (num_images, label_sz, label_sz) containing the label for each sample
        """

        feat_sz = self.label_density_params['feature_sz'] * self.label_density_params.get('interp_factor', 1)
        gauss_label = prutils.gaussian_label_function(target_bb.view(-1, 4), self.label_density_params['sigma_factor'],
                                                      self.label_density_params['kernel_sz'],
                                                      feat_sz, self.output_sz,
                                                      end_pad_if_even=self.label_density_params.get('end_pad_if_even', True),
                                                      density=True,
                                                      uni_bias=self.label_density_params.get('uni_weight', 0.0))

        gauss_label *= (gauss_label > self.label_density_params.get('threshold', 0.0)).float()

        if self.label_density_params.get('normalize', False):
            g_sum = gauss_label.sum(dim=(-2,-1))
            valid = g_sum>0.01
            gauss_label[valid, :, :] /= g_sum[valid].view(-1, 1, 1)
            gauss_label[~valid, :, :] = 1.0 / (gauss_label.shape[-2] * gauss_label.shape[-1])

        gauss_label *= 1.0 - self.label_density_params.get('shrink', 0.0)

        return gauss_label

    def __call__(self, data: TensorDict):
        """
        args:
            data - The input data, should contain the following fields:
                'train_images'  -
                'test_images'   -
                'train_anno'    -
                'test_anno'     -

        returns:
            TensorDict - output data block with following fields:
                'train_images'  -
                'test_images'   -
                'train_anno'    -
                'test_anno'     -
                'test_proposals' (optional) -
                'proposal_iou'  (optional)  -
                'test_label' (optional)     -
                'train_label' (optional)    -
        """

        if self.transform['joint'] is not None:
            data['train_images'], data['train_anno'] = self.transform['joint'](image=data['train_images'], bbox=data['train_anno'])
            data['test_images'], data['test_anno'] = self.transform['joint'](image=data['test_images'], bbox=data['test_anno'], new_roll=False)

        for s in ['train', 'test']:
            assert self.mode == 'sequence' or len(data[s + '_images']) == 1, \
                "In pair mode, num train/test frames must be 1"

            # Add a uniform noise to the center pos
            jittered_anno = [self._get_jittered_box(a, s) for a in data[s + '_anno']]

            crops, boxes = prutils.target_image_crop(data[s + '_images'], jittered_anno, data[s + '_anno'],
                                                     self.search_area_factor, self.output_sz, mode=self.crop_type,
                                                     max_scale_change=self.max_scale_change)

            data[s + '_images'], data[s + '_anno'] = self.transform[s](image=crops, bbox=boxes, joint=False)

        # Generate proposals
        for casc_level, ps in enumerate(self.proposal_params['proposal_sigma']):
            gt_sigma = self.proposal_params['gt_sigma']
            num_proposals = self.proposal_params['boxes_per_frame_per_cascade']
            proposals, proposal_density, gt_density = zip(*[self._generate_proposals(a, ps, gt_sigma, num_proposals) for a in data['test_anno']])

            data['test_proposals_{}'.format(casc_level)] = proposals
            data['proposal_density_{}'.format(casc_level)] = proposal_density
            data['gt_density_{}'.format(casc_level)] = gt_density

        for s in ['train', 'test']:
            is_distractor = data.get('is_distractor_{}_frame'.format(s), None)
            if is_distractor is not None:
                for is_dist, box in zip(is_distractor, data[s+'_anno']):
                    if is_dist:
                        box[0] = 99999999.9
                        box[1] = 99999999.9

        # Prepare output
        if self.mode == 'sequence':
            data = data.apply(stack_tensors)
        else:
            data = data.apply(lambda x: x[0] if isinstance(x, list) else x)

        # Generate label functions
        if self.label_function_params is not None:
            data['train_label'] = self._generate_label_function(data['train_anno'])
            data['test_label'] = self._generate_label_function(data['test_anno'])
        if self.label_density_params is not None:
            data['train_label_density'] = self._generate_label_density(data['train_anno'])
            data['test_label_density'] = self._generate_label_density(data['test_anno'])

        return data


class SegmDiMPProcessing(BaseProcessing):
    """ The processing class used for training DiMP. The images are processed in the following way.
    First, the target bounding box is jittered by adding some noise. Next, a square region (called search region )
    centered at the jittered target center, and of area search_area_factor^2 times the area of the jittered box is
    cropped from the image. The reason for jittering the target box is to avoid learning the bias that the target is
    always at the center of the search region. The search region is then resized to a fixed size given by the
    argument output_sz. A Gaussian label centered at the target is generated for each image. These label functions are
    used for computing the loss of the predicted classification model on the test images. A set of proposals are
    also generated for the test images by jittering the ground truth box. These proposals are used to train the
    bounding box estimating branch.

    """

    def __init__(self, search_area_factor, output_sz, center_jitter_factor, scale_jitter_factor, crop_type='replicate',
                 max_scale_change=None, mode='pair',
                 proposal_params=None, label_function_params=None, label_density_params=None,
                 tc_label_function_params=None, new_roll=False, *args, **kwargs):
        """
        args:
            search_area_factor - The size of the search region  relative to the target size.
            output_sz - An integer, denoting the size to which the search region is resized. The search region is always
                        square.
            center_jitter_factor - A dict containing the amount of jittering to be applied to the target center before
                                    extracting the search region. See _get_jittered_box for how the jittering is done.
            scale_jitter_factor - A dict containing the amount of jittering to be applied to the target size before
                                    extracting the search region. See _get_jittered_box for how the jittering is done.
            crop_type - If 'replicate', the boundary pixels are replicated in case the search region crop goes out of image.
                        If 'nopad', the search region crop is shifted/shrunk to fit completely inside the image.
            mode - Either 'pair' or 'sequence'. If mode='sequence', then output has an extra dimension for frames
            proposal_params - Arguments for the proposal generation process. See _generate_proposals for details.
            label_function_params - Arguments for the label generation process. See _generate_label_function for details.
        """
        super().__init__(*args, **kwargs)
        self.search_area_factor = search_area_factor
        self.output_sz = output_sz
        self.center_jitter_factor = center_jitter_factor
        self.scale_jitter_factor = scale_jitter_factor
        self.crop_type = crop_type
        self.mode = mode
        self.max_scale_change = max_scale_change

        self.proposal_params = proposal_params
        self.label_function_params = label_function_params
        self.tc_label_function_params = tc_label_function_params
        self.label_density_params = label_density_params

        self.new_roll = new_roll

    def _get_jittered_box(self, box, mode):
        """ Jitter the input box
        args:
            box - input bounding box
            mode - string 'train' or 'test' indicating train or test data

        returns:
            torch.Tensor - jittered box
        """

        if self.scale_jitter_factor.get('mode', 'gauss') == 'gauss':
            jittered_size = box[2:4] * torch.exp(torch.randn(2) * self.scale_jitter_factor[mode])
        elif self.scale_jitter_factor.get('mode', 'gauss') == 'uniform':
            jittered_size = box[2:4] * torch.exp(torch.FloatTensor(2).uniform_(-self.scale_jitter_factor[mode],
                                                                               self.scale_jitter_factor[mode]))
        else:
            raise Exception

        max_offset = (jittered_size.prod().sqrt() * torch.tensor(self.center_jitter_factor[mode])).float()
        jittered_center = box[0:2] + 0.5 * box[2:4] + max_offset * (torch.rand(2) - 0.5)

        return torch.cat((jittered_center - 0.5 * jittered_size, jittered_size), dim=0)

    def _generate_proposals(self, box):
        """
        """
        # Generate proposals
        proposals, proposal_density, gt_density = prutils.sample_box_gmm(box, self.proposal_params['proposal_sigma'],
                                                                         gt_sigma=self.proposal_params['gt_sigma'],
                                                                         num_samples=self.proposal_params['boxes_per_frame'],
                                                                         add_mean_box=self.proposal_params.get('add_mean_box', False))

        return proposals, proposal_density, gt_density

    def _generate_label_function(self, target_bb):
        """ Generates the gaussian label function centered at target_bb
        args:
            target_bb - target bounding box (num_images, 4)

        returns:
            torch.Tensor - Tensor of shape (num_images, label_sz, label_sz) containing the label for each sample
        """
        gauss_label = prutils.gaussian_label_function(target_bb.view(-1, 4), self.label_function_params['sigma_factor'],
                                                      self.label_function_params['kernel_sz'],
                                                      self.label_function_params['feature_sz'], self.output_sz,
                                                      end_pad_if_even=self.label_function_params.get('end_pad_if_even', True))

        return gauss_label

    def _generate_label_function_ada(self, target_bb, crop_bb):
        """ Generates the gaussian label function centered at target_bb
        args:
            target_bb - target bounding box (num_images, 4)

        returns:
            torch.Tensor - Tensor of shape (num_images, label_sz, label_sz) containing the label for each sample
        """
        crop_sz = crop_bb[:, 2:].tolist()
        gauss_label = prutils.gaussian_label_function(target_bb.view(-1, 4), self.label_function_params['sigma_factor'],
                                                      self.label_function_params['kernel_sz'],
                                                      self.label_function_params['feature_sz'], crop_sz,
                                                      end_pad_if_even=self.label_function_params.get('end_pad_if_even',
                                                                                                     True))
        return gauss_label


    def _generate_tc_label_function(self, target_bb):
        """ Generates the gaussian label function centered at target_bb
        args:
            target_bb - target bounding box (num_images, 4)

        returns:
            torch.Tensor - Tensor of shape (num_images, label_sz, label_sz) containing the label for each sample
        """
        if self.tc_label_function_params.get('jitter_center', False):
            target_bb_jit = target_bb.view(-1, 4).clone()
            target_sz = target_bb[:, 2:].prod(dim=1).sqrt()
            jitter_factor = torch.FloatTensor(target_bb_jit.shape[0], 2).uniform_(
                -self.tc_label_function_params['max_jitter_factor'],
                self.tc_label_function_params['max_jitter_factor'])

            target_bb_jit[:, :2] += jitter_factor*target_sz.view(-1, 1)
        else:
            target_bb_jit = target_bb.view(-1, 4)
        gauss_label = prutils.gaussian_label_function(target_bb_jit, self.tc_label_function_params['sigma_factor'],
                                                      self.tc_label_function_params['kernel_sz'],
                                                      self.tc_label_function_params['feature_sz'], self.output_sz,
                                                      end_pad_if_even=self.tc_label_function_params.get('end_pad_if_even', True))

        return gauss_label

    def _generate_label_density(self, target_bb):
        """ Generates the gaussian label density centered at target_bb
        args:
            target_bb - target bounding box (num_images, 4)

        returns:
            torch.Tensor - Tensor of shape (num_images, label_sz, label_sz) containing the label for each sample
        """

        feat_sz = self.label_density_params['feature_sz'] * self.label_density_params.get('interp_factor', 1)
        gauss_label = prutils.gaussian_label_function(target_bb.view(-1, 4), self.label_density_params['sigma_factor'],
                                                      self.label_density_params['kernel_sz'],
                                                      feat_sz, self.output_sz,
                                                      end_pad_if_even=self.label_density_params.get('end_pad_if_even', True),
                                                      density=True,
                                                      uni_bias=self.label_density_params.get('uni_weight', 0.0))

        gauss_label *= (gauss_label > self.label_density_params.get('threshold', 0.0)).float()

        if self.label_density_params.get('normalize', False):
            g_sum = gauss_label.sum(dim=(-2,-1))
            valid = g_sum>0.01
            gauss_label[valid, :, :] /= g_sum[valid].view(-1, 1, 1)
            gauss_label[~valid, :, :] = 1.0 / (gauss_label.shape[-2] * gauss_label.shape[-1])

        gauss_label *= 1.0 - self.label_density_params.get('shrink', 0.0)

        return gauss_label

    def _generate_search_bb(self, boxes_crop, crops, boxes_orig, boxes_jittered):
        search_bb = []
        anno_search_bb = []
        for b_crop, im, b_orig, b_jit in zip(boxes_crop, crops, boxes_orig, boxes_jittered):
            output_sz = self.output_sz
            if isinstance(output_sz, (float, int)):
                output_sz = (output_sz, output_sz)

            output_sz = torch.Tensor(output_sz)

            resize_factor = b_crop[-1] / b_orig[-1]

            b_jit_crop_sz = b_jit[2:] * resize_factor

            search_bb_sz = (
                        output_sz * (b_jit_crop_sz.prod() / output_sz.prod()).sqrt() * self.search_area_factor).ceil()
            search_bb.append(torch.cat((torch.zeros(2), search_bb_sz)))

            b_sh = b_crop.clone()

            anno_search_bb.append(b_sh)
        return search_bb, anno_search_bb

    def __call__(self, data: TensorDict):
        """
        args:
            data - The input data, should contain the following fields:
                'train_images'  -
                'test_images'   -
                'train_anno'    -
                'test_anno'     -

        returns:
            TensorDict - output data block with following fields:
                'train_images'  -
                'test_images'   -
                'train_anno'    -
                'test_anno'     -
                'test_proposals' (optional) -
                'proposal_iou'  (optional)  -
                'test_label' (optional)     -
                'train_label' (optional)    -
        """

        if self.transform['joint'] is not None:
            data['train_images'], data['train_anno'], data['train_masks'] = self.transform['joint'](image=data['train_images'], bbox=data['train_anno'], mask=data['train_masks'])
            data['test_images'], data['test_anno'], data['test_masks'] = self.transform['joint'](image=data['test_images'], bbox=data['test_anno'], mask=data['test_masks'], new_roll=self.new_roll)

        for s in ['train', 'test']:
            assert self.mode == 'sequence' or len(data[s + '_images']) == 1, \
                "In pair mode, num train/test frames must be 1"

            # Add a uniform noise to the center pos
            jittered_anno = [self._get_jittered_box(a, s) for a in data[s + '_anno']]
            orig_anno = data[s + '_anno']

            crops, boxes, mask_crops = prutils.target_image_crop(data[s + '_images'], jittered_anno,
                                                                 data[s + '_anno'], self.search_area_factor,
                                                                 self.output_sz, mode=self.crop_type,
                                                                 max_scale_change=self.max_scale_change,
                                                                 masks=data[s + '_masks'])

            data[s + '_images'], data[s + '_anno'], data[s + '_masks'] = self.transform[s](image=crops, bbox=boxes, mask=mask_crops, joint=False)

            # Generate search_bb
            sa_bb, anno_in_sa = self._generate_search_bb(boxes, crops, orig_anno, jittered_anno)

            data[s + '_sa_bb'] = sa_bb
            data[s + '_anno_in_sa'] = anno_in_sa

        # Generate proposals
        if self.proposal_params is not None:
            proposals, proposal_density, gt_density = zip(*[self._generate_proposals(a) for a in data['test_anno']])

            data['test_proposals'] = proposals
            data['proposal_density'] = proposal_density
            data['gt_density'] = gt_density

        for s in ['train', 'test']:
            is_distractor = data.get('is_distractor_{}_frame'.format(s), None)
            if is_distractor is not None:
                for is_dist, box in zip(is_distractor, data[s+'_anno']):
                    if is_dist:
                        box[0] = 99999999.9
                        box[1] = 99999999.9

        # Prepare output
        if self.mode == 'sequence':
            data = data.apply(stack_tensors)
        else:
            data = data.apply(lambda x: x[0] if isinstance(x, list) else x)

        # Generate label functions
        if self.label_function_params is not None:
            if self.label_function_params.get('adaptive_resampling', False):
                data['train_label'] = self._generate_label_function_ada(data['train_anno_in_sa'], data['train_sa_bb'])
                data['test_label'] = self._generate_label_function_ada(data['test_anno_in_sa'], data['test_sa_bb'])
            else:
                data['train_label'] = self._generate_label_function(data['train_anno'])
                data['test_label'] = self._generate_label_function(data['test_anno'])

        if self.tc_label_function_params is not None:
            data['test_tc_label'] = self._generate_tc_label_function(data['test_anno'])

        return data


class DiMPFPNProcessing(BaseProcessing):
    """ The processing class used for training DiMP. The images are processed in the following way.
    First, the target bounding box is jittered by adding some noise. Next, a square region (called search region )
    centered at the jittered target center, and of area search_area_factor^2 times the area of the jittered box is
    cropped from the image. The reason for jittering the target box is to avoid learning the bias that the target is
    always at the center of the search region. The search region is then resized to a fixed size given by the
    argument output_sz. A Gaussian label centered at the target is generated for each image. These label functions are
    used for computing the loss of the predicted classification model on the test images. A set of proposals are
    also generated for the test images by jittering the ground truth box. These proposals are used to train the
    bounding box estimating branch.

    """

    def __init__(self, base_sz_per_level, output_sz, scale_jitter_factor, mode='pair',
                 proposal_params=None, mix_scales=True, *args, **kwargs):
        """
        args:
            search_area_factor - The size of the search region  relative to the target size.
            output_sz - An integer, denoting the size to which the search region is resized. The search region is always
                        square.
            center_jitter_factor - A dict containing the amount of jittering to be applied to the target center before
                                    extracting the search region. See _get_jittered_box for how the jittering is done.
            scale_jitter_factor - A dict containing the amount of jittering to be applied to the target size before
                                    extracting the search region. See _get_jittered_box for how the jittering is done.
            crop_type - If 'replicate', the boundary pixels are replicated in case the search region crop goes out of image.
                        If 'nopad', the search region crop is shifted/shrunk to fit completely inside the image.
            mode - Either 'pair' or 'sequence'. If mode='sequence', then output has an extra dimension for frames
            proposal_params - Arguments for the proposal generation process. See _generate_proposals for details.
            label_function_params - Arguments for the label generation process. See _generate_label_function for details.
        """
        super().__init__(*args, **kwargs)
        self.base_sz_per_level = base_sz_per_level
        self.num_feature_levels = len(base_sz_per_level)

        max_sz_per_level = []
        for i in range(1, self.num_feature_levels):
            max_sz_per_level.append((base_sz_per_level[i-1] + base_sz_per_level[i] / 2.0))
        max_sz_per_level.append(float('inf'))
        self.max_sz_per_level = max_sz_per_level

        self.output_sz = output_sz
        self.scale_jitter_factor = scale_jitter_factor
        self.mode = mode

        self.mix_scales = mix_scales
        self.proposal_params = proposal_params

    def _generate_proposals(self, box):
        """
        """
        # Generate proposals
        proposals, proposal_density, gt_density = prutils.sample_box_gmm(box, self.proposal_params['proposal_sigma'],
                                                                         gt_sigma=self.proposal_params['gt_sigma'],
                                                                         num_samples=self.proposal_params['boxes_per_frame'],
                                                                         add_mean_box=self.proposal_params.get('add_mean_box', False))

        return proposals, proposal_density, gt_density

    def _get_base_target_sz(self, anno):
        base_sz = np.sqrt(anno[2] * anno[3])

        if base_sz <= 1.0:
            return 10.0
        else:
            return base_sz

    def _get_base_level(self, sz):
        min_sz = 0

        for i, max_sz in enumerate(self.max_sz_per_level):
            if min_sz <= sz < max_sz:
                return i
            min_sz = max_sz

        raise Exception('Size is {}'.format(sz))

    def _get_new_level(self, base_level):
        return random.randint(0, max(base_level, self.num_feature_levels - 1))

    def _get_new_target_sz(self, new_level, mode):
        base_sz = self.base_sz_per_level[new_level]
        jittered_size = base_sz * np.exp(random.uniform(-self.scale_jitter_factor[mode], self.scale_jitter_factor[mode]))

        return jittered_size

    def _get_resized_target_sz(self, anno, mode, input_feat_level=None):
        if self.sample_params[mode + '_mix_scale']:
            resize_factors = []
            new_level_all = []
            for a in anno:
                base_sz = self._get_base_target_sz(a)
                base_level = self._get_base_level(base_sz)
                new_level = self._get_new_level(base_level)

                new_sz = self._get_new_target_sz(new_level, mode)
                resize_factors.append(new_sz / base_sz)
                new_level_all.append(torch.tensor([new_level]))
        else:
            base_sz_all = [self._get_base_target_sz(a) for a in anno]

            if input_feat_level is not None and self.sample_params[mode + '_use_input_level']:
                new_level = input_feat_level
            else:
                base_level = self._get_base_level(base_sz_all[0])
                new_level = self._get_new_level(base_level)

            new_sz_all = [self._get_new_target_sz(new_level, mode) for _ in anno]
            resize_factors = [ns / bs for ns, bs in zip(new_sz_all, base_sz_all)]

            new_level_all = [torch.tensor([new_level]) for _ in anno]
        return resize_factors, new_level_all

    def __call__(self, data: TensorDict):
        """
        args:
            data - The input data, should contain the following fields:
                'train_images'  -
                'test_images'   -
                'train_anno'    -
                'test_anno'     -

        returns:
            TensorDict - output data block with following fields:
                'train_images'  -
                'test_images'   -
                'train_anno'    -
                'test_anno'     -
                'test_proposals' (optional) -
                'proposal_iou'  (optional)  -
                'test_label' (optional)     -
                'train_label' (optional)    -
        """

        if self.transform['joint'] is not None:
            data['train_images'], data['train_anno'] = self.transform['joint'](image=data['train_images'], bbox=data['train_anno'])
            data['test_images'], data['test_anno'] = self.transform['joint'](image=data['test_images'], bbox=data['test_anno'], new_roll=False)

        # Hack
        input_feat_level = None
        for s in ['train', 'test']:
            assert self.mode == 'sequence' or len(data[s + '_images']) == 1, \
                "In pair mode, num train/test frames must be 1"

            resized_factors, feat_level = self._get_resized_target_sz(data[s + '_anno'], s, input_feat_level)

            if s == 'train':
                input_feat_level = feat_level

            crops, boxes = prutils.target_image_crop_adaptive(data[s + '_images'], data[s + '_anno'],
                                                              resized_factors, self.output_sz)

            data[s + '_images'], data[s + '_anno'] = self.transform[s](image=crops, bbox=boxes, joint=False)
            data[s + '_feature_level'] = feat_level

        # Generate proposals
        proposals, proposal_density, gt_density = zip(*[self._generate_proposals(a) for a in data['test_anno']])

        data['test_proposals'] = proposals
        data['proposal_density'] = proposal_density
        data['gt_density'] = gt_density

        for s in ['train', 'test']:
            is_distractor = data.get('is_distractor_{}_frame'.format(s), None)
            if is_distractor is not None:
                for is_dist, box in zip(is_distractor, data[s+'_anno']):
                    if is_dist:
                        box[0] = 99999999.9
                        box[1] = 99999999.9

        # Prepare output
        if self.mode == 'sequence':
            data = data.apply(stack_tensors)
        else:
            data = data.apply(lambda x: x[0] if isinstance(x, list) else x)

        # Generate label functions
        # if self.label_function_params is not None:
        #     data['train_label'] = self._generate_label_function(data['train_anno'])
        #     data['test_label'] = self._generate_label_function(data['test_anno'])

        return data


class DiMPAdaProcessing(BaseProcessing):
    def __init__(self, search_area_scale, crop_sample_params, output_sz, scale_jitter_factor, mode='pair',
                 proposal_params=None, label_function_params=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.search_area_scale = search_area_scale
        self.crop_sample_params = crop_sample_params

        # self.target_size_range = target_size_range

        self.output_sz = output_sz
        self.scale_jitter_factor = scale_jitter_factor
        self.mode = mode

        self.label_function_params = label_function_params
        self.proposal_params = proposal_params

    def _generate_proposals(self, box):
        """
        """
        # Generate proposals
        proposals, proposal_density, gt_density = prutils.sample_box_gmm(box, self.proposal_params['proposal_sigma'],
                                                                         gt_sigma=self.proposal_params['gt_sigma'],
                                                                         num_samples=self.proposal_params['boxes_per_frame'],
                                                                         add_mean_box=self.proposal_params.get('add_mean_box', False))

        return proposals, proposal_density, gt_density

    def _get_base_target_sz(self, anno):
        base_sz = np.sqrt(anno[2] * anno[3])

        if base_sz <= 1.0:
            return 10.0
        else:
            return base_sz

    def _get_resize_factors(self, anno, images):
        resize_factors = []
        for a, im in zip(anno, images):
            if self.crop_sample_params.get('jitter_mode', 'random') == 'random':
                base_sz = self._get_base_target_sz(a)

                target_size_range = self.crop_sample_params['target_size_range']
                jittered_size = random.uniform(target_size_range[0],
                                               max(min(target_size_range[1], base_sz), target_size_range[0]))

                resize_factors.append(jittered_size / base_sz)
            elif self.crop_sample_params['jitter_mode'] == 'inside':
                # TODO
                im_h = im.shape[0]
                im_w = im.shape[1]

                base_sz = self._get_base_target_sz(a)

                assert self.output_sz[0] == self.output_sz[1], "Not tested if rows/cols are interchanged"
                if base_sz * self.search_area_scale <= min(im_w, im_h):
                    resize_factors.append(self.output_sz[0] / (base_sz * self.search_area_scale))
                else:
                    resize_factors.append(self.output_sz[0] / max(im_w, im_h))

        return resize_factors

    def _get_crop_box(self, target_bb, im_size, output_sz):
        bbx, bby, bbw, bbh = target_bb.tolist()

        x1_lo = max(bbx + bbw - output_sz[0], min(0, im_size[0] - output_sz[0]))
        x1_hi = min(bbx, max(im_size[0] - output_sz[0], 0))

        y1_lo = max(bby + bbh - output_sz[1], min(0, im_size[1] - output_sz[1]))
        y1_hi = min(bby, max(im_size[1] - output_sz[1], 0))

        crop_x1 = random.randint(int(x1_lo), int(max(x1_lo, x1_hi)))
        crop_y1 = random.randint(int(y1_lo), int(max(y1_lo, y1_hi)))

        crop_bb = [crop_x1, crop_y1, output_sz[0], output_sz[1]]
        return torch.tensor(crop_bb)

    def _generate_search_bb(self, boxes, crops, mode):
        search_bb = []
        anno_search_bb = []
        for b, im in zip(boxes, crops):
            sz = b[2:].prod().sqrt()

            max_jitter = self.scale_jitter_factor[mode]
            resize_factor = math.exp(random.uniform(-max_jitter, max_jitter))
            search_bb_sz = int(sz * self.search_area_scale) * resize_factor

            assert im.shape[0] == im.shape[1], "Not tested if rows/cols are interchanged"
            crop_bb = self._get_crop_box(b, (im.shape[0], im.shape[1]), (search_bb_sz, search_bb_sz))

            search_bb.append(crop_bb)
            b_sh = b.clone()
            b_sh[:2] -= crop_bb[:2]

            anno_search_bb.append(b_sh)
        return search_bb, anno_search_bb

    def _generate_label_function(self, target_bb, crop_bb):
        """ Generates the gaussian label function centered at target_bb
        args:
            target_bb - target bounding box (num_images, 4)

        returns:
            torch.Tensor - Tensor of shape (num_images, label_sz, label_sz) containing the label for each sample
        """
        crop_sz = crop_bb[:, 2:].tolist()
        gauss_label = prutils.gaussian_label_function(target_bb.view(-1, 4), self.label_function_params['sigma_factor'],
                                                      self.label_function_params['kernel_sz'],
                                                      self.label_function_params['feature_sz'], crop_sz,
                                                      end_pad_if_even=self.label_function_params.get('end_pad_if_even', True))

        return gauss_label

    def __call__(self, data: TensorDict):
        if self.transform['joint'] is not None:
            data['train_images'], data['train_anno'] = self.transform['joint'](image=data['train_images'], bbox=data['train_anno'])
            data['test_images'], data['test_anno'] = self.transform['joint'](image=data['test_images'], bbox=data['test_anno'], new_roll=False)

        for s in ['train', 'test']:
            assert self.mode == 'sequence' or len(data[s + '_images']) == 1, \
                "In pair mode, num train/test frames must be 1"

            resize_factors = self._get_resize_factors(data[s + '_anno'], data[s + '_images'])

            crops, boxes = prutils.target_image_crop_adaptive(data[s + '_images'], data[s + '_anno'],
                                                              resize_factors, self.output_sz)

            data[s + '_images'], data[s + '_anno'] = self.transform[s](image=crops, bbox=boxes, joint=False)

            # Generate search_bb
            search_bb, anno_search_bb = self._generate_search_bb(boxes, crops, s)

            data[s + '_search_bb'] = search_bb
            data[s + '_anno_search_bb'] = anno_search_bb

        # Generate proposals
        proposals, proposal_density, gt_density = zip(*[self._generate_proposals(a) for a in data['test_anno']])

        data['test_proposals'] = proposals
        data['proposal_density'] = proposal_density
        data['gt_density'] = gt_density

        for s in ['train', 'test']:
            is_distractor = data.get('is_distractor_{}_frame'.format(s), None)
            if is_distractor is not None:
                for is_dist, box in zip(is_distractor, data[s+'_anno']):
                    if is_dist:
                        box[0] = 99999999.9
                        box[1] = 99999999.9

        # Prepare output
        if self.mode == 'sequence':
            data = data.apply(stack_tensors)
        else:
            data = data.apply(lambda x: x[0] if isinstance(x, list) else x)

        # Generate label functions
        if self.label_function_params is not None:
            data['train_label'] = self._generate_label_function(data['train_anno_search_bb'], data['train_search_bb'])
            data['test_label'] = self._generate_label_function(data['test_anno_search_bb'], data['test_search_bb'])

        return data


class KLDiMPAdaProcessing(BaseProcessing):
    def __init__(self, search_area_factor, output_sz, center_jitter_factor, scale_jitter_factor, crop_type='replicate',
                 max_scale_change=None, mode='pair',
                 proposal_params=None, label_function_params=None, label_density_params=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.search_area_factor = search_area_factor
        self.output_sz = output_sz
        self.center_jitter_factor = center_jitter_factor
        self.scale_jitter_factor = scale_jitter_factor
        self.crop_type = crop_type
        self.mode = mode
        self.max_scale_change = max_scale_change

        self.proposal_params = proposal_params
        self.label_function_params = label_function_params
        self.label_density_params = label_density_params

    def _get_jittered_box(self, box, mode):
        """ Jitter the input box
        args:
            box - input bounding box
            mode - string 'train' or 'test' indicating train or test data

        returns:
            torch.Tensor - jittered box
        """

        jittered_size = box[2:4] * torch.exp(torch.randn(2) * self.scale_jitter_factor[mode])
        max_offset = (jittered_size.prod().sqrt() * torch.tensor(self.center_jitter_factor[mode]).float())
        jittered_center = box[0:2] + 0.5 * box[2:4] + max_offset * (torch.rand(2) - 0.5)

        return torch.cat((jittered_center - 0.5 * jittered_size, jittered_size), dim=0)

    def _generate_proposals(self, box):
        """
        """
        # Generate proposals
        proposals, proposal_density, gt_density = prutils.sample_box_gmm(box, self.proposal_params['proposal_sigma'],
                                                                         gt_sigma=self.proposal_params['gt_sigma'],
                                                                         num_samples=self.proposal_params['boxes_per_frame'],
                                                                         add_mean_box=self.proposal_params.get('add_mean_box', False))

        return proposals, proposal_density, gt_density

    def _generate_search_bb(self, boxes_crop, crops, boxes_orig, boxes_jittered):
        search_bb = []
        anno_search_bb = []
        for b_crop, im, b_orig, b_jit in zip(boxes_crop, crops, boxes_orig, boxes_jittered):
            output_sz = self.output_sz
            if isinstance(output_sz, (float, int)):
                output_sz = (output_sz, output_sz)

            output_sz = torch.Tensor(output_sz)

            resize_factor = b_crop[-1] / b_orig[-1]

            b_jit_crop_sz = b_jit[2:] * resize_factor

            search_bb_sz = (output_sz * (b_jit_crop_sz.prod() / output_sz.prod()).sqrt() * self.search_area_factor).ceil()
            search_bb.append(torch.cat((torch.zeros(2), search_bb_sz)))

            b_sh = b_crop.clone()

            anno_search_bb.append(b_sh)
        return search_bb, anno_search_bb

    def _generate_label_function(self, target_bb, crop_bb):
        """ Generates the gaussian label function centered at target_bb
        args:
            target_bb - target bounding box (num_images, 4)

        returns:
            torch.Tensor - Tensor of shape (num_images, label_sz, label_sz) containing the label for each sample
        """

        crop_sz = crop_bb[:, 2:].tolist()
        gauss_label = prutils.gaussian_label_function(target_bb.view(-1, 4), self.label_function_params['sigma_factor'],
                                                      self.label_function_params['kernel_sz'],
                                                      self.label_function_params['feature_sz'], crop_sz,
                                                      end_pad_if_even=self.label_function_params.get('end_pad_if_even', True))

        return gauss_label

    def __call__(self, data: TensorDict):
        """
        args:
            data - The input data, should contain the following fields:
                'train_images'  -
                'test_images'   -
                'train_anno'    -
                'test_anno'     -

        returns:
            TensorDict - output data block with following fields:
                'train_images'  -
                'test_images'   -
                'train_anno'    -
                'test_anno'     -
                'test_proposals' (optional) -
                'proposal_iou'  (optional)  -
                'test_label' (optional)     -
                'train_label' (optional)    -
        """

        if self.transform['joint'] is not None:
            data['train_images'], data['train_anno'] = self.transform['joint'](image=data['train_images'], bbox=data['train_anno'])
            data['test_images'], data['test_anno'] = self.transform['joint'](image=data['test_images'], bbox=data['test_anno'], new_roll=False)

        for s in ['train', 'test']:
            assert self.mode == 'sequence' or len(data[s + '_images']) == 1, \
                "In pair mode, num train/test frames must be 1"

            # Add a uniform noise to the center pos
            jittered_anno = [self._get_jittered_box(a, s) for a in data[s + '_anno']]
            orig_anno = data[s + '_anno']

            crops, boxes = prutils.target_image_crop(data[s + '_images'], jittered_anno, data[s + '_anno'],
                                                     self.search_area_factor, self.output_sz, mode=self.crop_type,
                                                     max_scale_change=self.max_scale_change)

            data[s + '_images'], data[s + '_anno'] = self.transform[s](image=crops, bbox=boxes, joint=False)

            # Generate search_bb
            search_bb, anno_search_bb = self._generate_search_bb(boxes, crops, orig_anno, jittered_anno)

            data[s + '_search_bb'] = search_bb
            data[s + '_anno_search_bb'] = anno_search_bb

        # Generate proposals
        proposals, proposal_density, gt_density = zip(*[self._generate_proposals(a) for a in data['test_anno']])

        data['test_proposals'] = proposals
        data['proposal_density'] = proposal_density
        data['gt_density'] = gt_density

        for s in ['train', 'test']:
            is_distractor = data.get('is_distractor_{}_frame'.format(s), None)
            if is_distractor is not None:
                for is_dist, box in zip(is_distractor, data[s+'_anno']):
                    if is_dist:
                        box[0] = 99999999.9
                        box[1] = 99999999.9

        # Prepare output
        if self.mode == 'sequence':
            data = data.apply(stack_tensors)
        else:
            data = data.apply(lambda x: x[0] if isinstance(x, list) else x)

        # Generate label functions
        if self.label_function_params is not None:
            data['train_label'] = self._generate_label_function(data['train_anno_search_bb'], data['train_search_bb'])
            data['test_label'] = self._generate_label_function(data['test_anno_search_bb'], data['test_search_bb'])

        if self.label_density_params is not None:
            raise NotImplementedError

        return data


class SegmFullImageProcessing(BaseProcessing):
    def __init__(self, output_sz, max_occlusion_factor, scale_jitter_factor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.output_sz = output_sz
        self.max_occlusion_factor = max_occlusion_factor
        self.scale_jitter_factor = scale_jitter_factor

    def _get_scale_jitter_factor(self, mode):
        if self.scale_jitter_factor.get('mode', 'gauss') == 'gauss':
            return math.exp(random.random() * self.scale_jitter_factor[mode])
        elif self.scale_jitter_factor.get('mode', 'gauss') == 'uniform':
            return math.exp(random.uniform(-self.scale_jitter_factor[mode], self.scale_jitter_factor[mode]))
        else:
            raise Exception

    def __call__(self, data: TensorDict):
        if self.transform['joint'] is not None:
            data['train_images'], data['train_anno'], data['train_masks'] = self.transform['joint'](image=data['train_images'], bbox=data['train_anno'], mask=data['train_masks'])
            data['test_images'], data['test_anno'], data['test_masks'] = self.transform['joint'](image=data['test_images'], bbox=data['test_anno'], mask=data['test_masks'], new_roll=False)

        im_h, im_w = data['train_images'][0].shape[:2]

        # Set a base size between output size and image size
        max_resize_factor = max([self.output_sz[0] / im_w, self.output_sz[1] / im_h])

        if max_resize_factor >= 1.0:
            base_resize_factor = max_resize_factor
        else:
            base_resize_factor = random.uniform(max_resize_factor, 1.0)

        for s in ['train', 'test']:
            # Determine random scale jittering
            im_crop_all = []
            ann_crop_all = []
            mask_crop_all = []
            for im, mask, ann in zip(data[s + '_images'], data[s + '_masks'], data[s + '_anno']):
                scale_jitter_factor = self._get_scale_jitter_factor(s)

                resize_factor = base_resize_factor * scale_jitter_factor

                crop_sz = [int(o / resize_factor) for o in self.output_sz]

                # Determine crop box
                occlusion_factor = self.max_occlusion_factor[s]

                if occlusion_factor is not None:
                    x1_max = int(min(ann[0] + occlusion_factor * ann[2], im_w - crop_sz[0]))
                    x1_min = int(max(0, ann[0] + (1.0 - occlusion_factor) * ann[2] - crop_sz[0]))

                    y1_max = int(min(ann[1] + occlusion_factor * ann[3], im_h - crop_sz[1]))
                    y1_min = int(max(0, ann[1] + (1.0 - occlusion_factor) * ann[3] - crop_sz[1]))
                else:
                    x1_max = int(im_w - crop_sz[0])
                    x1_min = 0

                    y1_max = int(im_h - crop_sz[1])
                    y1_min = 0

                # TODO make sure there is no error
                if x1_max > x1_min:
                    crop_x = random.randint(x1_min, x1_max - 1)
                else:
                    # Image smaller than crop size, needs padding
                    crop_x = random.randint(x1_max, x1_min)

                if y1_max > y1_min:
                    crop_y = random.randint(y1_min, y1_max - 1)
                else:
                    # Image smaller than crop size, needs padding
                    crop_y = random.randint(y1_max, y1_min)

                crop_box = [crop_x, crop_y, *crop_sz]

                # Crop and resize
                im_crop, ann_crop, mask_crop = prutils.crop_and_resize(im, ann, crop_box, self.output_sz,
                                                                       mask=mask)
                im_crop_all.append(im_crop)
                ann_crop_all.append(ann_crop)
                mask_crop_all.append(mask_crop)

            data[s + '_images'], data[s + '_anno'], data[s + '_masks'] = self.transform[s](image=im_crop_all,
                                                                                           bbox=ann_crop_all,
                                                                                           mask=mask_crop_all, joint=False)

        data = data.apply(stack_tensors)

        return data


class SegmMultiObjFullImageProcessing(BaseProcessing):
    def __init__(self, output_sz, max_occlusion_factor, scale_jitter_factor, max_num_objects, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.output_sz = output_sz
        self.max_occlusion_factor = max_occlusion_factor
        self.scale_jitter_factor = scale_jitter_factor
        self.max_num_objects = max_num_objects

    def _get_scale_jitter_factor(self, mode):
        if self.scale_jitter_factor.get('mode', 'gauss') == 'gauss':
            return math.exp(random.random() * self.scale_jitter_factor[mode])
        elif self.scale_jitter_factor.get('mode', 'gauss') == 'uniform':
            return math.exp(random.uniform(-self.scale_jitter_factor[mode], self.scale_jitter_factor[mode]))
        else:
            raise Exception

    def __call__(self, data: TensorDict):
        # Remove bbox field, since we always use mask to determine the box
        data.pop('train_anno', None)
        data.pop('test_anno', None)

        # DO not transform the box. Get it again from the mask instead
        if self.transform['joint'] is not None:
            data['train_images'], data['train_masks'] = self.transform['joint'](image=data['train_images'], mask=data['train_masks'])
            data['test_images'], data['test_masks'] = self.transform['joint'](image=data['test_images'], mask=data['test_masks'], new_roll=False)

        im_h, im_w = data['train_images'][0].shape[:2]

        # Set a base size between output size and image size
        max_resize_factor = max([self.output_sz[0] / im_w, self.output_sz[1] / im_h])

        if max_resize_factor >= 1.0:
            base_resize_factor = max_resize_factor
        else:
            base_resize_factor = random.uniform(max_resize_factor, 1.0)

        object_ids_orig = data['object_ids']
        object_ids_orig =[int(o) for o in object_ids_orig]
        num_objects = len(object_ids_orig)

        if num_objects > self.max_num_objects:
            object_ids_selected = random.sample(object_ids_orig, self.max_num_objects)
        else:
            object_ids_selected = object_ids_orig

        object_map = torch.zeros(max(object_ids_orig)+1, dtype=data['train_masks'][0].dtype)
        object_ids_mapped = []
        for i, oid in enumerate(object_ids_selected):
            object_map[oid] = i + 1
            object_ids_mapped.append(i + 1)

        # Convert mask
        for s in ['train', 'test']:
            for i in range(len(data[s + '_masks'])):
                data[s + '_masks'][i] = object_map[data[s + '_masks'][i].long()].float()

        for s in ['train', 'test']:
            # Determine random scale jittering
            im_crop_all = []
            mask_crop_all = []
            for im, mask in zip(data[s + '_images'], data[s + '_masks']):
                scale_jitter_factor = self._get_scale_jitter_factor(s)

                resize_factor = base_resize_factor * scale_jitter_factor

                crop_sz = [int(o / resize_factor) for o in self.output_sz]

                # Determine crop box
                occlusion_factor = self.max_occlusion_factor[s]

                ann = bbutils.masks_to_bboxes_multi(mask, object_ids_mapped, fmt='t')

                ann_t = torch.stack(ann, dim=0)
                if occlusion_factor is not None:
                    left_most_edge = (ann_t[:, 0] + occlusion_factor * ann_t[:, 2]).min().item()
                    x1_max = int(min(left_most_edge, im_w - crop_sz[0]))

                    right_most_edge = (ann_t[:, 0] + (1.0 - occlusion_factor) * ann_t[:, 2]).max().item()
                    x1_min = int(max(0, right_most_edge - crop_sz[0]))

                    top_most_edge = (ann_t[:, 1] + occlusion_factor * ann_t[:, 3]).min().item()
                    y1_max = int(min(top_most_edge, im_h - crop_sz[1]))

                    bottom_most_edge = (ann_t[:, 1] + (1.0 - occlusion_factor) * ann_t[:, 3]).max().item()
                    y1_min = int(max(0, bottom_most_edge - crop_sz[1]))
                else:
                    x1_max = int(im_w - crop_sz[0])
                    x1_min = 0

                    y1_max = int(im_h - crop_sz[1])
                    y1_min = 0

                # TODO make sure there is no error
                if x1_max > x1_min:
                    crop_x = random.randint(x1_min, x1_max - 1)
                else:
                    # Image smaller than crop size, or objects are far apart
                    crop_x = random.randint(x1_max, x1_min)

                if y1_max > y1_min:
                    crop_y = random.randint(y1_min, y1_max - 1)
                else:
                    # Image smaller than crop size, or objects are far apart
                    crop_y = random.randint(y1_max, y1_min)

                crop_box = [crop_x, crop_y, *crop_sz]

                # Crop and resize
                im_crop, _, mask_crop = prutils.crop_and_resize(im, None, crop_box, self.output_sz,
                                                                mask=mask)
                assert mask_crop.max() <= 3
                im_crop_all.append(im_crop)
                mask_crop_all.append(mask_crop)

            data[s + '_images'], data[s + '_masks'] = self.transform[s](image=im_crop_all, mask=mask_crop_all,
                                                                        joint=False)

        data = data.apply(stack_tensors)

        valid_object = torch.zeros((1, self.max_num_objects), dtype=torch.bool)
        for oid in object_ids_mapped:
            valid_object[0, oid-1] = True

        data['valid_object'] = valid_object
        return data


class SegmFullImageProcessingNoCrop(BaseProcessing):
    def __init__(self, output_sz, border_mode='inside_major', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.output_sz = output_sz
        self.border_mode = border_mode

    def __call__(self, data: TensorDict):
        if self.transform['joint'] is not None:
            data['train_images'], data['train_anno'], data['train_masks'] = self.transform['joint'](image=data['train_images'], bbox=data['train_anno'], mask=data['train_masks'])
            data['test_images'], data['test_anno'], data['test_masks'] = self.transform['joint'](image=data['test_images'], bbox=data['test_anno'], mask=data['test_masks'], new_roll=False)

        mode = self.border_mode
        im_h, im_w = data['train_images'][0].shape[:2]

        if mode == 'inside' or mode == 'inside_major':
            # Calculate rescaling factor if outside the image
            rescale_factor = [self.output_sz[0] / im_w, self.output_sz[1] / im_h]
            if mode == 'inside':
                rescale_factor = max(rescale_factor)
            elif mode == 'inside_major':
                rescale_factor = min(rescale_factor)
            else:
                raise Exception

            crop_sz_x = math.floor(self.output_sz[0] / rescale_factor)
            crop_sz_y = math.floor(self.output_sz[1] / rescale_factor)

        for s in ['train', 'test']:
            # Determine random scale jittering
            im_crop_all = []
            ann_crop_all = []
            mask_crop_all = []
            for im, mask, ann in zip(data[s + '_images'], data[s + '_masks'], data[s + '_anno']):
                # Crop and resize
                crop_x1 = random.randint(im_w - crop_sz_x, 0)
                crop_y1 = random.randint(im_h - crop_sz_y, 0)

                crop_box = [crop_x1, crop_y1, crop_sz_x, crop_sz_y]

                im_crop, ann_crop, mask_crop = prutils.crop_and_resize(im, ann, crop_box, self.output_sz,
                                                                       mask=mask)
                im_crop_all.append(im_crop)
                ann_crop_all.append(ann_crop)
                mask_crop_all.append(mask_crop)

            data[s + '_images'], data[s + '_anno'], data[s + '_masks'] = self.transform[s](image=im_crop_all,
                                                                                           bbox=ann_crop_all,
                                                                                           mask=mask_crop_all, joint=False)

        data = data.apply(stack_tensors)

        return data


class SegmFullImageMultiObjProcessingNoCrop(BaseProcessing):
    def __init__(self, output_sz, max_num_objects, border_mode='inside_major', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.output_sz = output_sz
        self.max_num_objects = max_num_objects
        self.border_mode = border_mode

    def __call__(self, data: TensorDict):
        data.pop('train_anno', None)
        data.pop('test_anno', None)

        if self.transform['joint'] is not None:
            data['train_images'], data['train_masks'] = self.transform['joint'](image=data['train_images'], mask=data['train_masks'])
            data['test_images'], data['test_masks'] = self.transform['joint'](image=data['test_images'], mask=data['test_masks'], new_roll=False)

        object_ids_orig = data['object_ids']
        object_ids_orig = [int(o) for o in object_ids_orig]
        num_objects = len(object_ids_orig)

        if num_objects > self.max_num_objects:
            object_ids_selected = random.sample(object_ids_orig, self.max_num_objects)
        else:
            object_ids_selected = object_ids_orig

        object_map = torch.zeros(max(object_ids_orig) + 1, dtype=data['train_masks'][0].dtype)
        object_ids_mapped = []
        for i, oid in enumerate(object_ids_selected):
            object_map[oid] = i + 1
            object_ids_mapped.append(i + 1)

        # Convert mask
        for s in ['train', 'test']:
            for i in range(len(data[s + '_masks'])):
                data[s + '_masks'][i] = object_map[data[s + '_masks'][i].long()].float()

        mode = self.border_mode
        im_h, im_w = data['train_images'][0].shape[:2]

        if mode == 'inside' or mode == 'inside_major':
            # Calculate rescaling factor if outside the image
            rescale_factor = [self.output_sz[0] / im_w, self.output_sz[1] / im_h]
            if mode == 'inside':
                rescale_factor = max(rescale_factor)
            elif mode == 'inside_major':
                rescale_factor = min(rescale_factor)
            else:
                raise Exception

            crop_sz_x = math.floor(self.output_sz[0] / rescale_factor)
            crop_sz_y = math.floor(self.output_sz[1] / rescale_factor)

        for s in ['train', 'test']:
            # Determine random scale jittering
            im_crop_all = []
            mask_crop_all = []
            for im, mask in zip(data[s + '_images'], data[s + '_masks']):
                # Crop and resize
                crop_x1 = random.randint(im_w - crop_sz_x, 0)
                crop_y1 = random.randint(im_h - crop_sz_y, 0)

                crop_box = [crop_x1, crop_y1, crop_sz_x, crop_sz_y]

                im_crop, ann_crop, mask_crop = prutils.crop_and_resize(im, None, crop_box, self.output_sz,
                                                                       mask=mask)
                im_crop_all.append(im_crop)
                mask_crop_all.append(mask_crop)

            data[s + '_images'], masks_t = self.transform[s](image=im_crop_all, mask=mask_crop_all,
                                                             joint=False)

            data[s + '_masks_label'] = masks_t

            # Convert mask to one-hot
            masks_oh_all = []
            for m in masks_t:
                one_hot_mask = torch.zeros((self.max_num_objects, *m.shape[-2:]), dtype=m.dtype)

                for i in range(self.max_num_objects):
                    one_hot_mask[i] = m == (i + 1)

                masks_oh_all.append(one_hot_mask)

            data[s + '_masks'] = torch.stack(masks_oh_all, dim=0)

        data = data.apply(stack_tensors)

        valid_object = torch.zeros((1, self.max_num_objects), dtype=torch.bool)
        for oid in object_ids_mapped:
            valid_object[0, oid - 1] = True

        data['valid_object'] = valid_object

        return data


class SegmDiMPMultiObjProcessing(BaseProcessing):
    def __init__(self, search_area_factor, output_sz, max_num_objects, center_jitter_factor,
                 scale_jitter_factor, border_mode='inside_major', new_roll=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.search_area_factor = search_area_factor
        self.output_sz = output_sz
        self.max_num_objects = max_num_objects

        self.center_jitter_factor = center_jitter_factor
        self.scale_jitter_factor = scale_jitter_factor
        self.border_mode = border_mode
        self.new_roll = new_roll

    def _get_jittered_box(self, box, mode):
        jittered_size = box[2:4] * torch.exp(torch.randn(2) * self.scale_jitter_factor[mode])
        max_offset = (jittered_size.prod().sqrt() * torch.tensor(self.center_jitter_factor[mode]).float())
        jittered_center = box[0:2] + 0.5 * box[2:4] + max_offset * (torch.rand(2) - 0.5)

        return torch.cat((jittered_center - 0.5 * jittered_size, jittered_size), dim=0)

    def _get_crop_box(self, box, im):
        output_sz = self.output_sz
        if isinstance(output_sz, (float, int)):
            output_sz = (output_sz, output_sz)
        output_sz = torch.Tensor(output_sz)

        im_h = im.shape[0]
        im_w = im.shape[1]

        bbx, bby, bbw, bbh = box.tolist()

        # Crop image
        crop_sz_x, crop_sz_y = (output_sz * (
                    box[2:].prod() / output_sz.prod()).sqrt() * self.search_area_factor).ceil().long().tolist()

        mode = self.border_mode

        # Get new sample size if forced inside the image
        if mode == 'inside' or mode == 'inside_major':
            # Calculate rescaling factor if outside the image
            rescale_factor = [crop_sz_x / im_w, crop_sz_y / im_h]
            if mode == 'inside':
                rescale_factor = max(rescale_factor)
            elif mode == 'inside_major':
                rescale_factor = min(rescale_factor)
            rescale_factor = max(1, rescale_factor)

            crop_sz_x = math.floor(crop_sz_x / rescale_factor)
            crop_sz_y = math.floor(crop_sz_y / rescale_factor)

        if crop_sz_x < 1 or crop_sz_y < 1:
            raise Exception('Too small bounding box.')

        x1 = round(bbx + 0.5 * bbw - crop_sz_x * 0.5)
        x2 = x1 + crop_sz_x

        y1 = round(bby + 0.5 * bbh - crop_sz_y * 0.5)
        y2 = y1 + crop_sz_y

        # Move box inside image
        shift_x = max(0, -x1) + min(0, im_w - x2)
        x1 += shift_x
        x2 += shift_x

        shift_y = max(0, -y1) + min(0, im_h - y2)
        y1 += shift_y
        y2 += shift_y

        out_x = (max(0, -x1) + max(0, x2 - im_w)) // 2
        out_y = (max(0, -y1) + max(0, y2 - im_h)) // 2
        shift_x = (-x1 - out_x) * (out_x > 0)
        shift_y = (-y1 - out_y) * (out_y > 0)

        x1 += shift_x
        x2 += shift_x
        y1 += shift_y
        y2 += shift_y

        crop_box_x1y1x2y2 = [x1, y1, x2, y2]

        x1_pad = max(0, -x1)
        x2_pad = max(x2 - im.shape[1] + 1, 0)

        y1_pad = max(0, -y1)
        y2_pad = max(y2 - im.shape[0] + 1, 0)

        crop_box_inside_x1y1x2y1 = [x1 + x1_pad, y1 + y1_pad, x2 - x2_pad, y2 - y2_pad]
        return crop_box_x1y1x2y2, crop_box_inside_x1y1x2y1

    def _crop_with_padding(self, crop_box_x1y1x2y2, im, mask):
        output_sz = self.output_sz
        if isinstance(output_sz, (float, int)):
            output_sz = (output_sz, output_sz)
        output_sz = torch.Tensor(output_sz)

        x1, y1, x2, y2 = crop_box_x1y1x2y2

        x1_pad = max(0, -x1)
        x2_pad = max(x2 - im.shape[1] + 1, 0)

        y1_pad = max(0, -y1)
        y2_pad = max(y2 - im.shape[0] + 1, 0)

        # Crop target
        im_crop = im[y1 + y1_pad:y2 - y2_pad, x1 + x1_pad:x2 - x2_pad, :]

        if mask is not None:
            mask_crop = mask[y1 + y1_pad:y2 - y2_pad, x1 + x1_pad:x2 - x2_pad]

        # Pad
        im_crop_padded = cv.copyMakeBorder(im_crop, y1_pad, y2_pad, x1_pad, x2_pad, cv.BORDER_REPLICATE)

        mask_crop_padded = F.pad(mask_crop, pad=(x1_pad, x2_pad, y1_pad, y2_pad), mode='constant', value=0)

        # Resize image
        im_out = cv.resize(im_crop_padded, tuple(output_sz.long().tolist()))

        mask_out = F.interpolate(mask_crop_padded[None, None], tuple(output_sz.flip(0).long().tolist()),
                                 mode='nearest')[0, 0]

        return im_out, mask_out

    def __call__(self, data: TensorDict):
        assert len(data['train_images']) == 1

        # Calculate box from the segmentation mask
        data.pop('train_anno', None)
        data.pop('test_anno', None)

        if self.transform['joint'] is not None:
            data['train_images'], data['train_masks'] = self.transform['joint'](image=data['train_images'], mask=data['train_masks'])
            data['test_images'], data['test_masks'] = self.transform['joint'](image=data['test_images'], mask=data['test_masks'], new_roll=self.new_roll)

        object_ids_orig = data['object_ids']
        object_ids_orig = [int(o) for o in object_ids_orig]

        num_objects = len(object_ids_orig)
        max_object_id = max(object_ids_orig)

        # Select an object at random
        base_object_id = random.sample(object_ids_orig, 1)[0]

        # Remove base object from ids
        object_ids_orig.remove(base_object_id)

        # Select crop box
        base_object_gt = bbutils.masks_to_bboxes_multi(data['train_masks'][0], [base_object_id, ], fmt='t')[0]

        crop_box_x1y1x2y2, crop_box_inside_x1y1x2y1 = self._get_crop_box(self._get_jittered_box(base_object_gt, 'train'),
                                                                         data['train_images'][0])

        if num_objects > self.max_num_objects - 1:
            train_mask = data['train_masks'][0]
            train_mask_in_crop = data['train_masks'][0][crop_box_inside_x1y1x2y1[1]:crop_box_inside_x1y1x2y1[3],
                                                        crop_box_inside_x1y1x2y1[0]:crop_box_inside_x1y1x2y1[2]]
            object_in_crop_ratio = [(train_mask_in_crop == id).sum() / ((train_mask == id).sum() + 1.0)for id in object_ids_orig]

            selected_object_indices = np.argsort(object_in_crop_ratio)[-(self.max_num_objects - 1):]

            object_ids_selected = [base_object_id,] + [object_ids_orig[i] for i in selected_object_indices]
        else:
            object_ids_selected = [base_object_id,] + object_ids_orig

        object_map = torch.zeros(max_object_id + 1, dtype=data['train_masks'][0].dtype)
        object_ids_mapped = []
        for i, oid in enumerate(object_ids_selected):
            object_map[oid] = i + 1
            object_ids_mapped.append(i + 1)

        # Convert mask
        for s in ['train', 'test']:
            for i in range(len(data[s + '_masks'])):
                data[s + '_masks'][i] = object_map[data[s + '_masks'][i].long()].float()

        # Train frame
        im_crop_train, mask_crop_train = self._crop_with_padding(crop_box_x1y1x2y2, data['train_images'][0], data['train_masks'][0])

        data['train_images'], data['train_masks_label'] = self.transform['train'](image=[im_crop_train, ],
                                                                                  mask=[mask_crop_train, ], joint=False)

        # Determine random scale jittering
        im_test_crop_all = []
        mask_test_crop_all = []
        for im, mask in zip(data['test_images'], data['test_masks']):
            # we have mapped the labels. So base objects always gets label 1
            for id in range(self.max_num_objects):
                object_sz = (mask == (id + 1)).sum()

                if object_sz > 100:
                    base_object_gt = bbutils.masks_to_bboxes_multi(mask, [id+1, ], fmt='t')[0]
                    crop_box_x1y1x2y2, crop_box_inside_x1y1x2y1 = self._get_crop_box(
                        self._get_jittered_box(base_object_gt, 'test'),
                        im)

                    break

            im_crop, mask_crop = self._crop_with_padding(crop_box_x1y1x2y2, im, mask)

            im_test_crop_all.append(im_crop)
            mask_test_crop_all.append(mask_crop)

        data['test_images'], data['test_masks_label'] = self.transform['test'](image=im_test_crop_all,
                                                                               mask=mask_test_crop_all, joint=False)

        for s in ['train', 'test']:
            # Convert mask to one-hot
            masks_oh_all = []
            for m in data[s + '_masks_label']:
                one_hot_mask = torch.zeros((self.max_num_objects, *m.shape[-2:]), dtype=m.dtype)

                for i in range(self.max_num_objects):
                    one_hot_mask[i] = m == (i + 1)

                masks_oh_all.append(one_hot_mask)

            data[s + '_masks'] = torch.stack(masks_oh_all, dim=0)

        data = data.apply(stack_tensors)

        valid_object = torch.zeros((1, self.max_num_objects), dtype=torch.bool)
        for oid in object_ids_mapped:
            valid_object[0, oid - 1] = True

        data['valid_object'] = valid_object

        return data


class SegmBoxProcessing(BaseProcessing):
    def __init__(self, search_area_factor, output_sz, center_jitter_factor, scale_jitter_factor, crop_type='replicate',
                 max_scale_change=None, mode='pair',
                 label_function_params=None, new_roll=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.search_area_factor = search_area_factor
        self.output_sz = output_sz
        self.center_jitter_factor = center_jitter_factor
        self.scale_jitter_factor = scale_jitter_factor
        self.crop_type = crop_type
        self.mode = mode
        self.max_scale_change = max_scale_change

        self.label_function_params = label_function_params

        self.new_roll = new_roll

    def _get_jittered_box(self, box, mode):
        """ Jitter the input box
        args:
            box - input bounding box
            mode - string 'train' or 'test' indicating train or test data

        returns:
            torch.Tensor - jittered box
        """

        if self.scale_jitter_factor.get('mode', 'gauss') == 'gauss':
            jittered_size = box[2:4] * torch.exp(torch.randn(2) * self.scale_jitter_factor[mode])
        elif self.scale_jitter_factor.get('mode', 'gauss') == 'uniform':
            jittered_size = box[2:4] * torch.exp(torch.FloatTensor(2).uniform_(-self.scale_jitter_factor[mode],
                                                                               self.scale_jitter_factor[mode]))
        else:
            raise Exception

        max_offset = (jittered_size.prod().sqrt() * torch.tensor(self.center_jitter_factor[mode])).float()
        jittered_center = box[0:2] + 0.5 * box[2:4] + max_offset * (torch.rand(2) - 0.5)

        return torch.cat((jittered_center - 0.5 * jittered_size, jittered_size), dim=0)

    def _generate_label_function(self, target_bb):
        """ Generates the gaussian label function centered at target_bb
        args:
            target_bb - target bounding box (num_images, 4)

        returns:
            torch.Tensor - Tensor of shape (num_images, label_sz, label_sz) containing the label for each sample
        """
        gauss_label = prutils.gaussian_label_function(target_bb.view(-1, 4), self.label_function_params['sigma_factor'],
                                                      self.label_function_params['kernel_sz'],
                                                      self.label_function_params['feature_sz'], self.output_sz,
                                                      end_pad_if_even=self.label_function_params.get('end_pad_if_even', True))

        return gauss_label

    def __call__(self, data: TensorDict):
        if self.transform['joint'] is not None:
            data['train_images'], data['train_anno'], data['train_masks'] = self.transform['joint'](
                image=data['train_images'], bbox=data['train_anno'], mask=data['train_masks'])
            data['test_images'], data['test_anno'], data['test_masks'] = self.transform['joint'](
                image=data['test_images'], bbox=data['test_anno'], mask=data['test_masks'], new_roll=self.new_roll)

        # TODO make sure boxes are correct
        for s in ['train', 'test']:
            assert self.mode == 'sequence' or len(data[s + '_images']) == 1, \
                "In pair mode, num train/test frames must be 1"

            # Add a uniform noise to the center pos
            jittered_anno = [self._get_jittered_box(a, s) for a in data[s + '_anno']]
            orig_anno = data[s + '_anno']

            # TODO fix optional mask input
            out = prutils.target_image_crop(data[s + '_images'], jittered_anno,
                                                                 data[s + '_anno'], self.search_area_factor,
                                                                 self.output_sz, mode=self.crop_type,
                                                                 max_scale_change=self.max_scale_change,
                                                                 masks=data[s + '_masks'])

            if len(out) == 2:
                crops, boxes = out
                data[s + '_images'], data[s + '_anno'] = self.transform[s](image=crops, bbox=boxes, joint=False)
                data[s + '_masks'] = [None for _ in data[s + '_images']]
            else:
                crops, boxes, mask_crops = out
                data[s + '_images'], data[s + '_anno'], data[s + '_masks'] = self.transform[s](image=crops, bbox=boxes,
                                                                                               mask=mask_crops,
                                                                                               joint=False)

            data[s + '_has_mask'] = []

            for i in range(len(data[s + '_images'])):
                im = data[s + '_images'][i]
                m = data[s + '_masks'][i]
                if m is None:
                    data[s + '_masks'][i] = torch.zeros((im.shape[-2], im.shape[-1])).float()
                    data[s + '_has_mask'].append(torch.tensor([0.0]))
                else:
                    data[s + '_has_mask'].append(torch.tensor([1.0]))


        # Prepare output
        if self.mode == 'sequence':
            data = data.apply(stack_tensors)
        else:
            data = data.apply(lambda x: x[0] if isinstance(x, list) else x)

        # Generate label functions
        if self.label_function_params is not None:
            data['train_label'] = self._generate_label_function(data['train_anno'])
            data['test_label'] = self._generate_label_function(data['test_anno'])

        return data


class CascadeDiMPProcessing(BaseProcessing):
    """ The processing class used for training PrDiMP that additionally supports the probabilistic classifier and
    bounding box regressor. See DiMPProcessing for details.
    """

    def __init__(self, search_area_factor, output_sz, center_jitter_factor, scale_jitter_factor, crop_type='replicate',
                 max_scale_change=None, mode='pair', bbr_proposal_params=None, iou_proposal_params=None,
                 label_function_params=None, label_density_params=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.search_area_factor = search_area_factor
        self.output_sz = output_sz
        self.center_jitter_factor = center_jitter_factor
        self.scale_jitter_factor = scale_jitter_factor
        self.crop_type = crop_type
        self.mode = mode
        self.max_scale_change = max_scale_change

        self.bbr_proposal_params = bbr_proposal_params
        self.iou_proposal_params = iou_proposal_params
        self.label_function_params = label_function_params
        self.label_density_params = label_density_params

    def _get_jittered_box(self, box, mode):
        """ Jitter the input box
        args:
            box - input bounding box
            mode - string 'train' or 'test' indicating train or test data

        returns:
            torch.Tensor - jittered box
        """

        jittered_size = box[2:4] * torch.exp(torch.randn(2) * self.scale_jitter_factor[mode])
        max_offset = (jittered_size.prod().sqrt() * torch.tensor(self.center_jitter_factor[mode]).float())
        jittered_center = box[0:2] + 0.5 * box[2:4] + max_offset * (torch.rand(2) - 0.5)

        return torch.cat((jittered_center - 0.5 * jittered_size, jittered_size), dim=0)

    def _generate_proposals(self, box, type='iou'):
        """ Generate proposal sample boxes from a GMM proposal distribution and compute their ground-truth density.
        This is used for ML and KL based regression learning of the bounding box regressor.
        args:
            box - input bounding box
        """
        # Generate proposals
        if type == 'iou':
            proposal_params = self.iou_proposal_params
        elif type == 'bbr':
            proposal_params = self.bbr_proposal_params
        else:
            raise Exception

        sampling_method = proposal_params.get('sampling_method', 'gmm')
        if sampling_method == 'gmm':
            proposals, proposal_density, gt_density = prutils.sample_box_gmm(box, proposal_params['proposal_sigma'],
                                                                             gt_sigma=proposal_params['gt_sigma'],
                                                                             num_samples=proposal_params['boxes_per_frame'],
                                                                             add_mean_box=proposal_params.get('add_mean_box', False))

        elif sampling_method == 'ncep_gmm':
            proposals, proposal_density, gt_density = prutils.ncep_sample_box_gmm(box, proposal_params['proposal_sigma'],
                                                                                 beta=proposal_params['beta'],
                                                                                 gt_sigma=proposal_params['gt_sigma'],
                                                                                 num_samples=proposal_params['boxes_per_frame'],
                                                                                 add_mean_box=proposal_params.get('add_mean_box', False))
        else:
            raise Exception('Wrong sampling method.')

        return proposals, proposal_density, gt_density

    def _generate_label_function(self, target_bb):
        """ Generates the gaussian label function centered at target_bb
        args:
            target_bb - target bounding box (num_images, 4)

        returns:
            torch.Tensor - Tensor of shape (num_images, label_sz, label_sz) containing the label for each sample
        """

        gauss_label = prutils.gaussian_label_function(target_bb.view(-1, 4), self.label_function_params['sigma_factor'],
                                                      self.label_function_params['kernel_sz'],
                                                      self.label_function_params['feature_sz'], self.output_sz,
                                                      end_pad_if_even=self.label_function_params.get('end_pad_if_even', True))

        return gauss_label

    def _generate_label_density(self, target_bb):
        """ Generates the gaussian label density centered at target_bb
        args:
            target_bb - target bounding box (num_images, 4)

        returns:
            torch.Tensor - Tensor of shape (num_images, label_sz, label_sz) containing the label for each sample
        """

        feat_sz = self.label_density_params['feature_sz'] * self.label_density_params.get('interp_factor', 1)
        gauss_label = prutils.gaussian_label_function(target_bb.view(-1, 4), self.label_density_params['sigma_factor'],
                                                      self.label_density_params['kernel_sz'],
                                                      feat_sz, self.output_sz,
                                                      end_pad_if_even=self.label_density_params.get('end_pad_if_even', True),
                                                      density=True,
                                                      uni_bias=self.label_density_params.get('uni_weight', 0.0))

        gauss_label *= (gauss_label > self.label_density_params.get('threshold', 0.0)).float()

        if self.label_density_params.get('normalize', False):
            g_sum = gauss_label.sum(dim=(-2,-1))
            valid = g_sum>0.01
            gauss_label[valid, :, :] /= g_sum[valid].view(-1, 1, 1)
            gauss_label[~valid, :, :] = 1.0 / (gauss_label.shape[-2] * gauss_label.shape[-1])

        gauss_label *= 1.0 - self.label_density_params.get('shrink', 0.0)

        return gauss_label

    def __call__(self, data: TensorDict):
        """
        args:
            data - The input data, should contain the following fields:
                'train_images', test_images', 'train_anno', 'test_anno'
        returns:
            TensorDict - output data block with following fields:
                'train_images', 'test_images', 'train_anno', 'test_anno', 'test_proposals', 'proposal_density', 'gt_density',
                'test_label' (optional), 'train_label' (optional), 'test_label_density' (optional), 'train_label_density' (optional)
        """

        if self.transform['joint'] is not None:
            data['train_images'], data['train_anno'] = self.transform['joint'](image=data['train_images'], bbox=data['train_anno'])
            data['test_images'], data['test_anno'] = self.transform['joint'](image=data['test_images'], bbox=data['test_anno'], new_roll=False)

        for s in ['train', 'test']:
            assert self.mode == 'sequence' or len(data[s + '_images']) == 1, \
                "In pair mode, num train/test frames must be 1"

            # Add a uniform noise to the center pos
            jittered_anno = [self._get_jittered_box(a, s) for a in data[s + '_anno']]

            crops, boxes = prutils.target_image_crop(data[s + '_images'], jittered_anno, data[s + '_anno'],
                                                     self.search_area_factor, self.output_sz, mode=self.crop_type,
                                                     max_scale_change=self.max_scale_change)

            data[s + '_images'], data[s + '_anno'] = self.transform[s](image=crops, bbox=boxes, joint=False)

        # Generate proposals
        bbr_proposals, _, _ = zip(*[self._generate_proposals(a, 'bbr') for a in data['test_anno']])
        iou_proposals, proposal_density, gt_density = zip(*[self._generate_proposals(a, 'iou') for a in data['test_anno']])

        data['test_bbr_proposals'] = bbr_proposals
        data['test_iou_proposals'] = iou_proposals
        data['proposal_density'] = proposal_density
        data['gt_density'] = gt_density

        for s in ['train', 'test']:
            is_distractor = data.get('is_distractor_{}_frame'.format(s), None)
            if is_distractor is not None:
                for is_dist, box in zip(is_distractor, data[s+'_anno']):
                    if is_dist:
                        box[0] = 99999999.9
                        box[1] = 99999999.9

        # Prepare output
        if self.mode == 'sequence':
            data = data.apply(stack_tensors)
        else:
            data = data.apply(lambda x: x[0] if isinstance(x, list) else x)

        # Generate label functions
        if self.label_function_params is not None:
            data['train_label'] = self._generate_label_function(data['train_anno'])
            data['test_label'] = self._generate_label_function(data['test_anno'])
        if self.label_density_params is not None:
            data['train_label_density'] = self._generate_label_density(data['train_anno'])
            data['test_label_density'] = self._generate_label_density(data['test_anno'])

        return data


class LWTLProcessing(BaseProcessing):
    """ The processing class used for training LWTL. The images are processed in the following way.
    First, the target bounding box (computed using the segmentation mask)is jittered by adding some noise.
    Next, a rectangular region (called search region ) centered at the jittered target center, and of area
    search_area_factor^2 times the area of the jittered box is cropped from the image.
    The reason for jittering the target box is to avoid learning the bias that the target is
    always at the center of the search region. The search region is then resized to a fixed size given by the
    argument output_sz. The argument 'crop_type' determines how out-of-frame regions are handled when cropping the
    search region. For instance, if crop_type == 'replicate', the boundary pixels are replicated in case the search
    region crop goes out of frame. If crop_type == 'inside_major', the search region crop is shifted/shrunk to fit
    completely inside one axis of the image.
    """

    def __init__(self, search_area_factor, output_sz, center_jitter_factor, scale_jitter_factor, crop_type='replicate',
                 max_scale_change=None, mode='pair', new_roll=False, *args, **kwargs):
        """
        args:
            search_area_factor - The size of the search region  relative to the target size.
            output_sz - The size (width, height) to which the search region is resized. The aspect ratio is always
                        preserved when resizing the search region
            center_jitter_factor - A dict containing the amount of jittering to be applied to the target center before
                                    extracting the search region. See _get_jittered_box for how the jittering is done.
            scale_jitter_factor - A dict containing the amount of jittering to be applied to the target size before
                                    extracting the search region. See _get_jittered_box for how the jittering is done.
            crop_type - Determines how out-of-frame regions are handled when cropping the search region.
                        If 'replicate', the boundary pixels are replicated in case the search region crop goes out of
                                        image.
                        If 'inside', the search region crop is shifted/shrunk to fit completely inside the image.
                        If 'inside_major', the search region crop is shifted/shrunk to fit completely inside one axis
                        of the image.
            max_scale_change - Maximum allowed scale change when shrinking the search region to fit the image
                               (only applicable to 'inside' and 'inside_major' cropping modes). In case the desired
                               shrink factor exceeds the max_scale_change, the search region is only shrunk to the
                               factor max_scale_change. Out-of-frame regions are then handled by replicating the
                               boundary pixels. If max_scale_change is set to None, unbounded shrinking is allowed.

            mode - Either 'pair' or 'sequence'. If mode='sequence', then output has an extra dimension for frames
            new_roll - Whether to use the same random roll values for train and test frames when applying the joint
                       transformation. If True, a new random roll is performed for the test frame transformations. Thus,
                       if performing random flips, the set of train frames and the set of test frames will be flipped
                       independently.
        """
        super().__init__(*args, **kwargs)
        self.search_area_factor = search_area_factor
        self.output_sz = output_sz
        self.center_jitter_factor = center_jitter_factor
        self.scale_jitter_factor = scale_jitter_factor
        self.crop_type = crop_type
        self.mode = mode
        self.max_scale_change = max_scale_change

        self.new_roll = new_roll

    def _get_jittered_box(self, box, mode):
        """ Jitter the input box
        args:
            box - input bounding box
            mode - string 'train' or 'test' indicating train or test data

        returns:
            torch.Tensor - jittered box
        """

        if self.scale_jitter_factor.get('mode', 'gauss') == 'gauss':
            jittered_size = box[2:4] * torch.exp(torch.randn(2) * self.scale_jitter_factor[mode])
        elif self.scale_jitter_factor.get('mode', 'gauss') == 'uniform':
            jittered_size = box[2:4] * torch.exp(torch.FloatTensor(2).uniform_(-self.scale_jitter_factor[mode],
                                                                               self.scale_jitter_factor[mode]))
        else:
            raise Exception

        max_offset = (jittered_size.prod().sqrt() * torch.tensor(self.center_jitter_factor[mode])).float()
        jittered_center = box[0:2] + 0.5 * box[2:4] + max_offset * (torch.rand(2) - 0.5)

        return torch.cat((jittered_center - 0.5 * jittered_size, jittered_size), dim=0)

    def __call__(self, data: TensorDict):
        # Apply joint transformations. i.e. All train/test frames in a sequence are applied the transformation with the
        # same parameters
        if self.transform['joint'] is not None:
            data['train_images'], data['train_anno'], data['train_masks'] = self.transform['joint'](
                image=data['train_images'], bbox=data['train_anno'], mask=data['train_masks'])
            data['test_images'], data['test_anno'], data['test_masks'] = self.transform['joint'](
                image=data['test_images'], bbox=data['test_anno'], mask=data['test_masks'], new_roll=self.new_roll)

        for s in ['train', 'test']:
            assert self.mode == 'sequence' or len(data[s + '_images']) == 1, \
                "In pair mode, num train/test frames must be 1"

            # Add a uniform noise to the center pos
            jittered_anno = [self._get_jittered_box(a, s) for a in data[s + '_anno']]
            orig_anno = data[s + '_anno']

            # Extract a crop containing the target
            crops, boxes, mask_crops = prutils.target_image_crop(data[s + '_images'], jittered_anno,
                                                                 data[s + '_anno'], self.search_area_factor,
                                                                 self.output_sz, mode=self.crop_type,
                                                                 max_scale_change=self.max_scale_change,
                                                                 masks=data[s + '_masks'])

            # Apply independent transformations to each image
            data[s + '_images'], data[s + '_anno'], data[s + '_masks'] = self.transform[s](image=crops, bbox=boxes, mask=mask_crops, joint=False)

        # Prepare output
        if self.mode == 'sequence':
            data = data.apply(stack_tensors)
        else:
            data = data.apply(lambda x: x[0] if isinstance(x, list) else x)

        return data


class FewShotSegProcessing(BaseProcessing):
    def __init__(self, output_sz, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.output_sz = output_sz

    def __call__(self, data: TensorDict):
        if self.transform['joint'] is not None:
            data['train_images'], data['train_masks'] = self.transform['joint'](image=data['train_images'], mask=data['train_masks'])
            data['test_images'], data['test_masks'] = self.transform['joint'](image=data['test_images'], mask=data['test_masks'], new_roll=False)

        for s in ['train', 'test']:
            # Determine random scale jittering
            im_crop_all = []
            mask_crop_all = []
            for im, mask in zip(data[s + '_images'], data[s + '_masks']):
                resize_factors = (self.output_sz[0] / im.shape[1], self.output_sz[1] / im.shape[0])
                min_resize_factor = min(resize_factors)
                crop_sz = [int(o / min_resize_factor) for o in self.output_sz]

                x1_max = crop_sz[0] - im.shape[1]
                y1_max = crop_sz[1] - im.shape[0]

                crop_x1 = random.randint(min(-x1_max, 0), 0)
                crop_y1 = random.randint(min(-y1_max, 0), 0)

                crop_box = [crop_x1, crop_y1, crop_sz[0], crop_sz[1]]

                # Crop and resize
                im_crop, _, mask_crop = prutils.crop_and_resize(im, None, crop_box, self.output_sz, mask=mask,
                                                                border_mode='zeros')
                im_crop_all.append(im_crop)
                mask_crop_all.append(mask_crop)

            data[s + '_images'], data[s + '_masks'] = self.transform[s](image=im_crop_all, mask=mask_crop_all,
                                                                        joint=False)

        data = data.apply(stack_tensors)

        return data
