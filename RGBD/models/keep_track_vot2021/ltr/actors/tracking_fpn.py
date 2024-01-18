from . import BaseActor
import torch
import torch.nn.functional as F
from ltr.data.processing_utils import iou_gen
import random
import math


class DiMPFPNActor(BaseActor):
    """Actor for training the DiMP network."""
    def __init__(self, net, objective, base_feat_sz, loss_weight=None):
        super().__init__(net, objective)
        if loss_weight is None:
            loss_weight = {'bb_ce': 1.0}
        self.loss_weight = loss_weight
        self.base_feat_sz = base_feat_sz

    def _crop_feature_map(self, feat, bb, output_sz, feat_stride):
        # Feat shape c x h x w (assume square)
        if feat.shape[-1] < output_sz:
            # Pad
            if random.random() < 0.5:
                pad_x1 = output_sz - feat.shape[-1]
                pad_x2 = 0
            else:
                pad_x1 = 0
                pad_x2 = output_sz - feat.shape[-1]

            if random.random() < 0.5:
                pad_y1 = output_sz - feat.shape[-2]
                pad_y2 = 0
            else:
                pad_y1 = 0
                pad_y2 = output_sz - feat.shape[-2]

            return F.pad(feat, (pad_x1, pad_x2, pad_y1, pad_y2)), (-pad_x1*feat_stride, -pad_y1*feat_stride)
        elif feat.shape[-1] == output_sz:
            return feat, (0, 0)
        else:
            bb = (bb / feat_stride).tolist()

            x_ll = max(0, math.ceil(bb[0] + bb[2] - output_sz))
            x_up = min(feat.shape[-2] - output_sz - 1, math.floor(bb[0]))

            if x_ll <= x_up:
                x1 = random.randint(x_ll, x_up)
            else:
                x1 = min(max((x_ll + x_up) // 2, 0), feat.shape[-2] - output_sz - 1)

            y_ll = max(0, math.ceil(bb[1] + bb[3] - output_sz))
            y_up = min(feat.shape[-1] - output_sz - 1, math.floor(bb[1]))

            if y_ll <= y_up:
                y1 = random.randint(y_ll, y_up)
            else:
                y1 = min(max((y_ll + y_up) // 2, 0), feat.shape[-1] - output_sz - 1)

            return feat[:, y1:y1+output_sz, x1:x1+output_sz], (x1*feat_stride, y1*feat_stride)

    def _transform_box(self, box, t_info):
        box[:, :2] = box[:, :2] - torch.tensor(t_info, dtype=box.dtype).view(1, 2).to(box.device)

        return box

    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'train_images', 'test_images', 'train_anno',
                    'test_proposals', 'proposal_iou' and 'test_label'.

        returns:
            loss    - the training loss
            stats  -  dict containing detailed losses
        """
        # Extract features
        train_imgs = data['train_images']
        test_imgs = data['test_images']

        train_bb = data['train_anno']
        test_bb = data['test_anno']
        test_proposals = data['test_proposals']

        train_feature_level = data['train_feature_level'].view(-1)
        test_feature_level = data['test_feature_level'].view(-1)

        assert train_imgs.dim() == 5 and test_imgs.dim() == 5, 'Expect 5 dimensional inputs'

        # Extract backbone features
        train_feat = self.net.extract_backbone_features(train_imgs.view(-1, *train_imgs.shape[-3:]))
        test_feat = self.net.extract_backbone_features(test_imgs.view(-1, *test_imgs.shape[-3:]))

        # Generate a feature map in the base scale
        train_feat_base = []
        train_feat_strides = []
        test_feat_base = []
        test_feat_strides = []

        num_sequences = train_bb.shape[1]
        num_frames_train = train_bb.shape[0]
        num_frames_test = test_bb.shape[0]

        train_bb = train_bb.view(-1, 4)
        test_bb = test_bb.view(-1, 4)
        test_proposals = test_proposals.view(num_sequences*num_frames_test, -1, 4)

        for i in range(train_feature_level.numel()):
            train_feat_i = self.net.get_fpn_feature(train_feat, train_feature_level[i])[i, ...]
            train_feat_i_stride = self.net.feature_layers[train_feature_level[i]]['stride']
            train_feat_cropped, crop_info = self._crop_feature_map(train_feat_i, train_bb[i, :],
                                                                   self.base_feat_sz, train_feat_i_stride)
            train_feat_strides.append(train_feat_i_stride)
            train_bb[i:i+1, :] = self._transform_box(train_bb[i:i+1, :], crop_info)
            train_feat_base.append(train_feat_cropped)

        for i in range(test_feature_level.numel()):
            test_feat_i = self.net.get_fpn_feature(test_feat, test_feature_level[i])[i, ...]
            test_feat_i_stride = self.net.feature_layers[test_feature_level[i]]['stride']

            test_feat_cropped, crop_info = self._crop_feature_map(test_feat_i, test_bb[i, :], self.base_feat_sz,
                                                                  test_feat_i_stride)
            test_feat_strides.append(test_feat_i_stride)

            test_bb[i:i+1, :] = self._transform_box(test_bb[i:i+1, :], crop_info)
            test_proposals[i, :, :] = self._transform_box(test_proposals[i, :, :], crop_info)
            test_feat_base.append(test_feat_cropped)

        # Reshape bb reg variables
        test_bb = test_bb.view(num_frames_test, num_sequences, 4)
        is_valid = test_bb[:, :, 0] < 99999.0
        proposal_density = data['proposal_density'][is_valid, :]
        gt_density = data['gt_density'][is_valid, :]

        train_feat_base = torch.stack(train_feat_base)
        test_feat_base = torch.stack(test_feat_base)

        train_feat_strides = torch.tensor(train_feat_strides, dtype=train_bb.dtype).view(-1, 1).to(train_bb.device)
        test_feat_strides = torch.tensor(test_feat_strides, dtype=train_bb.dtype).view(-1, 1).to(train_bb.device)

        # Convert anno to feat scale to avoid stride
        train_bb_feat = train_bb / train_feat_strides
        train_bb_feat = train_bb_feat.view(num_frames_train, num_sequences, 4)

        test_bb_feat = test_bb.view(-1, 4) / test_feat_strides
        test_bb_feat = test_bb_feat.view(num_frames_test, num_sequences, 4)
        # Run classifier module
        target_scores = self.net.classifier(train_feat_base, test_feat_base, train_bb_feat)

        # Run the IoUNet module
        test_proposals_feat = test_proposals / test_feat_strides.view(-1, 1, 1)
        test_proposals_feat = test_proposals_feat.view(num_frames_test, num_sequences, -1, 4)
        bb_scores = self.net.bb_regressor(train_feat_base, test_feat_base, train_bb_feat, test_proposals_feat)

        bb_scores = bb_scores[is_valid, :]

        # Compute loss
        bb_ce = self.objective['bb_ce'](bb_scores, sample_density=proposal_density, gt_density=gt_density, mc_dim=1)
        loss_bb_ce = self.loss_weight['bb_ce'] * bb_ce

        loss_target_classifier = 0
        loss_test_init_clf = 0
        loss_test_iter_clf = 0
        if 'test_clf' in self.loss_weight.keys():
            # Classification losses for the different optimization iterations
            clf_losses_test = [self.objective['test_clf'](s, target_bb=test_bb_feat) for s in target_scores]

            # Loss of the final filter
            clf_loss_test = clf_losses_test[-1]
            loss_target_classifier = self.loss_weight['test_clf'] * clf_loss_test

            # Loss for the initial filter iteration
            if 'test_init_clf' in self.loss_weight.keys():
                loss_test_init_clf = self.loss_weight['test_init_clf'] * clf_losses_test[0]

            # Loss for the intermediate filter iterations
            if 'test_iter_clf' in self.loss_weight.keys():
                test_iter_weights = self.loss_weight['test_iter_clf']
                if isinstance(test_iter_weights, list):
                    loss_test_iter_clf = sum([a * b for a, b in zip(test_iter_weights, clf_losses_test[1:-1])])
                else:
                    loss_test_iter_clf = (test_iter_weights / (len(clf_losses_test) - 2)) * sum(clf_losses_test[1:-1])

        # Total loss
        loss = loss_bb_ce + loss_target_classifier + loss_test_init_clf + loss_test_iter_clf

        if torch.isinf(loss) or torch.isnan(loss):
            raise Exception('ERROR: Loss was nan or inf!!!')

        # Log stats
        stats = {'Loss/total': loss.item(),
                 'Loss/bb_ce': bb_ce.item(),
                 'Loss/loss_bb_ce': loss_bb_ce.item()}
        if 'test_clf' in self.loss_weight.keys():
            stats['Loss/target_clf'] = loss_target_classifier.item()
        if 'test_init_clf' in self.loss_weight.keys():
            stats['Loss/test_init_clf'] = loss_test_init_clf.item()
        if 'test_iter_clf' in self.loss_weight.keys():
            stats['Loss/test_iter_clf'] = loss_test_iter_clf.item()

        if 'test_clf' in self.loss_weight.keys():
            stats['ClfTrain/test_loss'] = clf_loss_test.item()
            if len(clf_losses_test) > 0:
                stats['ClfTrain/test_init_loss'] = clf_losses_test[0].item()
                if len(clf_losses_test) > 2:
                    stats['ClfTrain/test_iter_loss'] = sum(clf_losses_test[1:-1]).item() / (len(clf_losses_test) - 2)

        for i in range(len(self.net.feature_layers)):
            stats['Log/train_l{}'.format(i)] = (train_feature_level == i).sum().item()
            stats['Log/test_l{}'.format(i)] = (test_feature_level == i).sum().item()
        return loss, stats


class DiMPAdaActor(BaseActor):
    """Actor for training the DiMP network."""
    def __init__(self, net, objective, loss_weight=None):
        super().__init__(net, objective)
        if loss_weight is None:
            loss_weight = {'bb_ce': 1.0}
        self.loss_weight = loss_weight

    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'train_images', 'test_images', 'train_anno',
                    'test_proposals', 'proposal_iou' and 'test_label'.

        returns:
            loss    - the training loss
            stats  -  dict containing detailed losses
        """
        # Run network
        # TODO in case train_anno and train_anno_search_bb are different, than we have an issue since iounet uses train_anno
        target_scores, bb_scores = self.net(train_imgs=data['train_images'],
                                            test_imgs=data['test_images'],
                                            train_bb_crop=data['train_anno'],
                                            train_bb_search_area=data.get('train_anno_search_bb', None),
                                            train_search_area_bb=data.get('train_search_bb', None),
                                            test_search_area_bb=data.get('test_search_bb', None),
                                            test_proposals=data['test_proposals'])

        # Reshape bb reg variables
        is_valid = data['test_anno'][:, :, 0] < 99999.0
        bb_scores = bb_scores[is_valid, :]
        proposal_density = data['proposal_density'][is_valid, :]
        gt_density = data['gt_density'][is_valid, :]

        # Compute loss
        bb_ce = self.objective['bb_ce'](bb_scores, sample_density=proposal_density, gt_density=gt_density, mc_dim=1)
        loss_bb_ce = self.loss_weight['bb_ce'] * bb_ce

        loss_target_classifier = 0
        loss_test_init_clf = 0
        loss_test_iter_clf = 0
        if 'test_clf' in self.loss_weight.keys():
            # Classification losses for the different optimization iterations
            clf_losses_test = [self.objective['test_clf'](s, data['test_label']) for s in target_scores]

            # Loss of the final filter
            clf_loss_test = clf_losses_test[-1]
            loss_target_classifier = self.loss_weight['test_clf'] * clf_loss_test

            # Loss for the initial filter iteration
            if 'test_init_clf' in self.loss_weight.keys():
                loss_test_init_clf = self.loss_weight['test_init_clf'] * clf_losses_test[0]

            # Loss for the intermediate filter iterations
            if 'test_iter_clf' in self.loss_weight.keys():
                test_iter_weights = self.loss_weight['test_iter_clf']
                if isinstance(test_iter_weights, list):
                    loss_test_iter_clf = sum([a * b for a, b in zip(test_iter_weights, clf_losses_test[1:-1])])
                else:
                    loss_test_iter_clf = (test_iter_weights / (len(clf_losses_test) - 2)) * sum(clf_losses_test[1:-1])

        loss_clf_ce = 0
        loss_clf_ce_init = 0
        loss_clf_ce_iter = 0

        # Total loss
        loss = loss_bb_ce + loss_clf_ce + loss_clf_ce_init + loss_clf_ce_iter + \
                            loss_target_classifier + loss_test_init_clf + loss_test_iter_clf

        if torch.isinf(loss) or torch.isnan(loss):
            raise Exception('ERROR: Loss was nan or inf!!!')

        # Log stats
        stats = {'Loss/total': loss.item(),
                 'Loss/bb_ce': bb_ce.item(),
                 'Loss/loss_bb_ce': loss_bb_ce.item()}
        if 'test_clf' in self.loss_weight.keys():
            stats['Loss/target_clf'] = loss_target_classifier.item()
        if 'test_init_clf' in self.loss_weight.keys():
            stats['Loss/test_init_clf'] = loss_test_init_clf.item()
        if 'test_iter_clf' in self.loss_weight.keys():
            stats['Loss/test_iter_clf'] = loss_test_iter_clf.item()

        if 'test_clf' in self.loss_weight.keys():
            stats['ClfTrain/test_loss'] = clf_loss_test.item()
            if len(clf_losses_test) > 0:
                stats['ClfTrain/test_init_loss'] = clf_losses_test[0].item()
                if len(clf_losses_test) > 2:
                    stats['ClfTrain/test_iter_loss'] = sum(clf_losses_test[1:-1]).item() / (len(clf_losses_test) - 2)

        return loss, stats
