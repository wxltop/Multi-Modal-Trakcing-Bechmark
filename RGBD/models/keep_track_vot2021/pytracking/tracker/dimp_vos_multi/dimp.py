from pytracking.tracker.base import BaseTracker
import torch
import torch.nn.functional as F
import numpy as np
import math
import time
from pytracking import dcf, TensorList
from pytracking.features.preprocessing import numpy_to_torch
from collections import OrderedDict
from pytracking.utils.plotting import show_tensor, plot_graph
from pytracking.features.preprocessing import sample_patch_multiscale, sample_patch_transformed, sample_patch, crop_and_resize
from pytracking.features import augmentation
import ltr.data.bounding_box_utils as bbutils
import ltr.data.processing_utils as prutils
from ltr.models.target_classifier.initializer import FilterInitializerZero
import cv2
import matplotlib.cm as cm
from ltr.models.layers import activation


class DiMPVosMulti(BaseTracker):

    multiobj_mode = 'default'

    def initialize_features(self):
        if not getattr(self, 'features_initialized', False):
            self.params.net.initialize()
        self.features_initialized = True

        if self.params.get('run_in_train_mode', False):
            self.params.net.train(True)

    def initialize(self, image, info: dict) -> dict:
        # Initialize some stuff
        self.frame_num = 1
        if not self.params.has('device'):
            self.params.device = 'cuda' if self.params.use_gpu else 'cpu'

        # Initialize network
        self.initialize_features()

        # The DiMP network
        self.net = self.params.net

        # Time initialization
        tic = time.time()

        init_mask = info.get('init_mask', None)
        self.object_ids_key = info['sequence_object_ids']
        self.object_ids = [int(i) for i in info['sequence_object_ids']]
        self.total_num_objects = len(self.object_ids)

        self.current_object_ids = [int(i) for i in info['init_object_ids']]

        self.label_map = torch.zeros(self.total_num_objects + 1).float()

        for i, id in enumerate(self.current_object_ids):
            self.label_map[i+1] = id

        self.id_to_pos_map = {id: i for i, id in enumerate(self.current_object_ids)}

        if init_mask is None:
            raise Exception

        init_mask = torch.tensor(init_mask).unsqueeze(0).unsqueeze(0).float()

        # Set sizes
        sz = self.params.image_sample_size
        self.img_sample_sz = torch.Tensor([sz, sz] if isinstance(sz, int) else sz)
        self.img_support_sz = self.img_sample_sz

        # Convert image
        im = numpy_to_torch(image)

        # Setup scale bounds
        self.image_sz = torch.Tensor([im.shape[2], im.shape[3]])

        # Extract and transform sample
        init_backbone_feat, init_masks = self.generate_init_samples(im, init_mask)

        init_masks_oh = self.convert_to_oh(init_masks, self.current_object_ids)

        # Initialize classifier
        self.init_classifier(init_backbone_feat, init_masks_oh)

        self.prev_test_x = None

        out = {'time': time.time() - tic}
        return out

    def convert_to_oh(self, masks, ids):
        out = []

        for i in ids:
            out.append(masks == i)

        return torch.cat(out, dim=1).float()

    def track(self, image, info: dict = None) -> dict:
        self.debug_info = {}

        self.frame_num += 1

        self.debug_info['frame_num'] = self.frame_num

        # Convert image
        im = numpy_to_torch(image)

        # ------- LOCALIZATION ------- #
        # Extract backbone features
        backbone_feat, im_patches = self.extract_backbone_features(im, self.img_sample_sz)

        # Extract classification features
        test_x = self.get_classification_features(backbone_feat)

        seg_mask = self.segment_target(test_x, backbone_feat)
        seg_mask_prob = self.merge_segmentation_results(seg_mask)

        if 'init_object_ids' in info.keys():
            init_mask = torch.tensor(info['init_mask']).unsqueeze(0).unsqueeze(0).float()

            _, init_mask_crop = crop_and_resize(im, self.crop_box, self.img_sample_sz, mask=init_mask)
            new_object_ids = [int(id) for id in info['init_object_ids']]
            init_masks_oh = self.convert_to_oh(init_mask_crop, new_object_ids).squeeze(1).to(self.params.device)

            init_masks_oh = init_masks_oh.view(len(new_object_ids), *init_masks_oh.shape[-2:])
            seg_mask_prob[:, init_masks_oh.sum(dim=0) > 0] = 0

            seg_mask_prob = torch.cat((seg_mask_prob, init_masks_oh), dim=0)

            for new_id in new_object_ids:
                self.id_to_pos_map[new_id] = len(self.current_object_ids)
                self.label_map[len(self.current_object_ids) + 1] = new_id
                self.current_object_ids.append(new_id)

            new_samples_pos = [self.id_to_pos_map[new_id] for new_id in new_object_ids]
            self.init_new_classifier(test_x, seg_mask_prob[1:, :, :], new_samples_pos)
        else:
            new_samples_pos = None

        # Update
        if self.params.get('update_classifier', False):
            if self.params.get('thresh_before_update', False):
                raise NotImplementedError
            else:
                seg_mask_prob_train = seg_mask_prob[1:, :, :].clone().contiguous()

            self.update_classifier(test_x, seg_mask_prob_train, new_samples_pos=new_samples_pos)

        seg_mask_prob_im = self.convert_mask_crop_to_im(seg_mask_prob, im)

        seg_label_im = seg_mask_prob_im.argmax(dim=0)
        seg_label_im = self.label_map[seg_label_im]
        seg_label_im_np = seg_label_im.cpu().numpy().astype(np.uint8)

        if self.visdom is not None:
            #self.visdom.register(seg_label, 'heatmap', 2, 'Seg Scores' + self.id_str)
            self.visdom.register(self.debug_info, 'info_dict', 1, 'Status')

        pred_boxes = OrderedDict()
        for id in self.object_ids_key:
            pred_boxes[id] = [0, 0, 1, 1]

        out = {'segmentation': seg_label_im_np, 'target_bbox': pred_boxes}
        return out

    def convert_mask_crop_to_im(self, seg_mask, im):
        seg_mask_re = F.interpolate(seg_mask.unsqueeze(0), (self.crop_box[-1], self.crop_box[-2]), mode='bilinear')
        seg_mask_re = seg_mask_re.view(-1, *seg_mask_re.shape[-2:])

        # Regions outside search area get very low score
        seg_mask_im = seg_mask_re[:, :im.shape[-2], :im.shape[-1]]

        return seg_mask_im

    def merge_segmentation_results(self, segmentation_maps):
        # Soft aggregation from RGMP
        eps = 1e-7

        segmentation_maps_t_prob = torch.sigmoid(segmentation_maps)
        bg_p = torch.prod(1 - segmentation_maps_t_prob, dim=0).clamp(eps, 1.0 - eps)  # bg prob
        bg_score = (bg_p / (1.0 - bg_p)).log()
        scores_all = torch.cat((bg_score.unsqueeze(0), segmentation_maps), dim=0)

        out = []
        for s in scores_all:
            s_out = 1.0 / (scores_all - s.unsqueeze(0)).exp().sum(dim=0)
            out.append(s_out)

        segmentation_maps_t_agg = torch.stack(out, dim=0)
        return segmentation_maps_t_agg

    def segment_target(self, sample_clf_feat, sample_x):
        """Classify target by applying the DiMP filter."""
        with torch.no_grad():
            mask = self.net.segment_target(self.target_filter, sample_clf_feat, sample_x, len(self.current_object_ids),
                                           serial=True)
        return mask

    def extract_backbone_features(self, im: torch.Tensor, sz: torch.Tensor):
        im_patches = crop_and_resize(im, self.crop_box, self.img_sample_sz)

        with torch.no_grad():
            backbone_feat = self.net.extract_backbone(im_patches)
        return backbone_feat, im_patches

    def get_classification_features(self, backbone_feat):
        with torch.no_grad():
            return self.net.extract_classification_feat(backbone_feat)

    def generate_init_samples(self, im: torch.Tensor, init_mask) -> TensorList:
        """Perform data augmentation to generate initial training samples."""

        # Get new sample size if forced inside the image
        im_sz = torch.Tensor([im.shape[2], im.shape[3]])

        mode = self.params.get('border_mode', 'inside_major')
        if mode == 'inside' or mode == 'inside_major':
            # Calculate rescaling factor if outside the image
            rescale_factor = self.img_sample_sz / im_sz
            if mode == 'inside':
                rescale_factor = rescale_factor.max().item()
            elif mode == 'inside_major':
                rescale_factor = rescale_factor.min().item()
            else:
                raise Exception

            crop_sz_x = math.floor(self.img_sample_sz[1] / rescale_factor)
            crop_sz_y = math.floor(self.img_sample_sz[0] / rescale_factor)
        else:
            raise Exception

        crop_x1 = 0
        crop_y1 = 0

        crop_box = [crop_x1, crop_y1, crop_sz_x, crop_sz_y]

        self.crop_box = crop_box
        im_patches, init_masks = crop_and_resize(im, crop_box, self.img_sample_sz, mask=init_mask)

        init_masks = init_masks.to(self.params.device)

        # Extract initial backbone features
        with torch.no_grad():
            init_backbone_feat = self.net.extract_backbone(im_patches)

        return init_backbone_feat, init_masks

    def init_memory(self, train_x: TensorList, masks):
        assert masks.dim() == 4

        # Initialize first-frame spatial training samples
        self.num_init_samples = train_x.size(0)
        init_sample_weights = TensorList([x.new_ones(1) / x.shape[0] for x in train_x])

        # Sample counters and weights for spatial
        self.num_stored_samples = self.num_init_samples.copy()
        self.previous_replace_ind = [None] * len(self.num_stored_samples)
        self.sample_weights = TensorList([x.new_zeros(self.params.sample_memory_size, self.total_num_objects)
                                          for x in train_x])
        for sw, init_sw, num in zip(self.sample_weights, init_sample_weights, self.num_init_samples):
            sw[:num, :len(self.current_object_ids)] = init_sw

        # Initialize memory
        self.training_samples = TensorList(
            [x.new_zeros(self.params.sample_memory_size, x.shape[1], x.shape[2], x.shape[3]) for x in train_x])

        self.target_masks = masks.new_zeros(self.params.sample_memory_size, self.total_num_objects, masks.shape[-2],
                                            masks.shape[-1])
        self.target_masks[:masks.shape[0], :masks.shape[1], :, :] = masks
        self.init_positions = [0 for _ in self.current_object_ids]

        for ts, x in zip(self.training_samples, train_x):
            ts[:x.shape[0],...] = x


    def update_memory(self, sample_x: TensorList, mask, learning_rate = None, new_samples_pos=None):
        # Update weights and get replace ind
        replace_ind = self.update_sample_weights(self.sample_weights, self.previous_replace_ind, self.num_stored_samples, self.num_init_samples, learning_rate, new_samples_pos)
        self.previous_replace_ind = replace_ind

        # Update sample and label memory
        for train_samp, x, ind in zip(self.training_samples, sample_x, replace_ind):
            train_samp[ind:ind+1,...] = x

        # Update bb memory
        # self.target_boxes[replace_ind[0],:] = target_box
        self.target_masks[replace_ind[0], :len(self.current_object_ids), :, :] = mask

        if self.num_stored_samples[0] < self.params.sample_memory_size:
            self.num_stored_samples += 1


    def update_sample_weights(self, sample_weights, previous_replace_ind, num_stored_samples, num_init_samples, learning_rate = None, new_samples_pos=None):
        # Update weights and get index to replace
        replace_ind = []
        for sw, prev_ind, num_samp, num_init in zip(sample_weights, previous_replace_ind, num_stored_samples, num_init_samples):
            assert num_init == 1
            lr = learning_rate
            if lr is None:
                lr = self.params.learning_rate

            init_samp_weight = self.params.get('init_samples_minimum_weight', None)
            if init_samp_weight == 0:
                init_samp_weight = None

            if num_samp == 0 or lr == 1:
                sw[:, :] = 0
                sw[0, :] = 1
                r_ind = 0
            else:
                # Get index to replace
                if num_samp < sw.shape[0]:
                    r_ind = num_samp
                else:
                    sw_cp = sw.clone()
                    sw_cp[list(set(self.init_positions)), :] = 100.0
                    _, r_ind = torch.min(sw_cp.max(dim=1)[0], dim=0)
                    r_ind = r_ind.item()

                if new_samples_pos is not None:
                    sw[r_ind, new_samples_pos] = 1.0
                    for _ in new_samples_pos:
                        self.init_positions.append(r_ind)

                    prev_object_ids = len(self.current_object_ids) - len(new_samples_pos)
                else:
                    prev_object_ids = len(self.current_object_ids)
                # Update weights
                if prev_ind is None:
                    sw /= 1 - lr
                    sw[r_ind, :prev_object_ids] = lr
                else:
                    sw[r_ind, :prev_object_ids] = sw[prev_ind, :prev_object_ids] / (1 - lr)

            sw /= sw.sum(dim=0, keepdim=True) + 1e-7
            if init_samp_weight is not None:
                for i, ip in enumerate(self.init_positions):
                    if sw[ip, i] < init_samp_weight:
                        sw[:, i] /= init_samp_weight + sw[:, i].sum() - sw[ip, i]
                        sw[ip, i] = init_samp_weight / num_init

            replace_ind.append(r_ind)

        return replace_ind

    def init_classifier(self, init_backbone_feat, init_masks):
        # Get classification features
        x = self.get_classification_features(init_backbone_feat)

        # Add the dropout augmentation here, since it requires extraction of the classification features
        if 'dropout' in self.params.augmentation and self.params.get('use_augmentation', True):
            raise NotImplementedError

        # Set number of iterations
        plot_loss = self.params.debug > 0
        num_iter = self.params.get('net_opt_iter', None)

        num_train_images = init_masks.shape[0]
        mask_enc = self.net.label_encoder(init_masks.unsqueeze(1), feature=x.unsqueeze(1), object_ids=None)

        if isinstance(mask_enc, (tuple, list)):
            train_mask_enc = mask_enc[0]
            train_mask_sw = mask_enc[1]

            train_mask_enc = train_mask_enc.view(num_train_images, 1, -1, *train_mask_enc.shape[-2:])
            train_mask_sw = train_mask_sw.view(num_train_images, 1, -1, *train_mask_sw.shape[-2:])
        else:
            train_mask_enc = mask_enc
            train_mask_enc = train_mask_enc.view(num_train_images, 1, -1, *train_mask_enc.shape[-2:])

            train_mask_sw = None

        # Get target filter by running the discriminative model prediction module
        with torch.no_grad():
            self.target_filter, _, losses = self.net.classifier.get_filter(x.unsqueeze(1), (
                                                                           train_mask_enc, train_mask_sw),
                                                                           num_iter=num_iter,
                                                                           num_objects=len(self.current_object_ids))

        # Init memory
        if self.params.get('update_classifier', True):
            if self.params.get('store_enc', False):
                raise Exception
            else:
                self.init_memory(TensorList([x]), masks=init_masks.view(-1, *init_masks.shape[-3:]))

    def update_classifier(self, train_x, mask, learning_rate=None, new_samples_pos=None):
        # Set flags and learning rate
        if learning_rate is None:
            learning_rate = self.params.learning_rate

        # Update the tracker memory
        if self.frame_num % self.params.get('train_sample_interval', 1) == 0:
            self.update_memory(TensorList([train_x]), mask, learning_rate, new_samples_pos)

        # Decide the number of iterations to run
        num_iter = 0
        if (self.frame_num - 1) % self.params.train_skipping == 0:
            num_iter = self.params.get('net_opt_update_iter', None)

        if num_iter > 0:
            # Get inputs for the DiMP filter optimizer module
            samples = self.training_samples[0][:self.num_stored_samples[0],...]
            masks = self.target_masks[:self.num_stored_samples[0], :len(self.current_object_ids), ...]

            mask_enc_info = self.net.label_encoder(masks.unsqueeze(1), samples.unsqueeze(1), object_ids=None)

            # target_boxes = self.target_boxes[:self.num_stored_samples[0],:].clone()
            sample_weights_im = self.sample_weights[0][:self.num_stored_samples[0], :len(self.current_object_ids)]

            if isinstance(mask_enc_info, (tuple, list)):
                mask_enc = mask_enc_info[0]
                mask_enc = mask_enc.view(self.num_stored_samples[0], 1, -1, *mask_enc.shape[-2:])

                sample_weights_spatial = mask_enc_info[1]
                sample_weights = sample_weights_spatial * sample_weights_im.view(self.num_stored_samples[0], 1,
                                                                                 len(self.current_object_ids), 1, 1, 1)
                sample_weights = sample_weights.view(self.num_stored_samples[0], 1, -1, *sample_weights.shape[-2:])
            else:
                mask_enc = mask_enc_info
                mask_enc = mask_enc.view(self.num_stored_samples[0], 1, -1, *mask_enc.shape[-2:])
                sample_weights = sample_weights_im

            # Run the filter optimizer module
            with torch.no_grad():
                target_filter, _, losses = self.net.classifier.filter_optimizer(TensorList([self.target_filter]),
                                                                                num_iter=num_iter, feat=samples.unsqueeze(1),
                                                                                mask=mask_enc,
                                                                                sample_weight=sample_weights)

            self.target_filter = target_filter[0]


    def init_new_classifier(self, train_feat, train_masks, new_ids):
        train_masks = train_masks.view(1, 1, *train_masks.shape[-3:])
        num_iter = self.params.get('net_opt_iter', None)
        mask_enc = self.net.label_encoder(train_masks, feature=train_feat.unsqueeze(1), object_ids=new_ids)

        if isinstance(mask_enc, (tuple, list)):
            train_mask_enc = mask_enc[0]
            train_mask_sw = mask_enc[1]

            train_mask_enc = train_mask_enc.view(1, 1, -1, *train_mask_enc.shape[-2:])
            train_mask_sw = train_mask_sw.view(1, 1, -1, *train_mask_sw.shape[-2:])
        else:
            train_mask_enc = mask_enc
            train_mask_enc = train_mask_enc.view(1, 1, -1, *train_mask_enc.shape[-2:])

            train_mask_sw = None

        # Get target filter by running the discriminative model prediction module
        with torch.no_grad():
            new_target_filter, _, losses = self.net.classifier.get_filter(train_feat.unsqueeze(1), (
                                                                           train_mask_enc, train_mask_sw),
                                                                           num_iter=num_iter,
                                                                           num_objects=len(new_ids))

        self.target_filter = torch.cat((self.target_filter, new_target_filter), dim=1)

    def visdom_draw_tracking(self, image, box, segmentation=None):
        if isinstance(box, OrderedDict):
            box = [v for k, v in box.items()]
        else:
            box = (box,)

        seg_all = []
        for i in self.object_ids:
            seg_all.append(segmentation == i)

        self.visdom.register((image, *box, *seg_all), 'Tracking', 1, 'Tracking')