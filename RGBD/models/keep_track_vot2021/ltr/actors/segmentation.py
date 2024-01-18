from . import BaseActor
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from pytracking import TensorList
from pytracking.analysis.vos_utils import davis_jaccard_measure, davis_jaccard_measure_torch

# Based on KLDiMPActor
class SegmActor(BaseActor):
    """Actor for training the DiMP network."""
    def __init__(self, net, objective, loss_weight=None,
                 use_focal_loss=False, use_lovasz_loss=False,
                 disable_all_bn=False):
        super().__init__(net, objective)
        if loss_weight is None:
            loss_weight = {'segm': 1.0}
        self.loss_weight = loss_weight

        self.use_focal_loss = use_focal_loss
        self.use_lovasz_loss = use_lovasz_loss
        self.disable_all_bn = disable_all_bn

    def train(self, mode=True):
        """ Set whether the network is in train mode.
        args:
            mode (True) - Bool specifying whether in training mode.
        """
        self.net.train(mode)

        if self.disable_all_bn:
            for m in self.net.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()

    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'train_images', 'test_images', 'train_anno',
                    'test_proposals', 'proposal_iou', 'test_label', 'train_masks' and 'test_masks'
        returns:
            loss    - the training loss
            stats  -  dict containing detailed losses
        """
        # Run network
        segm_pred, target_scores, mask_enc_test, aux_mask_pred, bb_pred = self.net(train_imgs=data['train_images'],
                                                                                   test_imgs=data['test_images'],
                                                                                   train_masks=data['train_masks'],
                                                                                   test_masks=data['test_masks'])

        loss_target_classifier = 0
        loss_test_init_clf = 0
        loss_test_iter_clf = 0
        clf_loss_test = 0
        if 'test_clf' in self.loss_weight.keys() and target_scores is not None:
            # Classification losses for the different optimization iterations
            clf_losses_test = [self.objective['test_clf'](input=s.view(-1, 1, *s.shape[-2:]),
                                                          target=data['test_masks'].view(-1, 1,
                                                                                          *data['test_masks'].shape[
                                                                                           -2:])) for s in
                               target_scores]

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

        gt_segm = data['test_masks'].view(-1, 1, *data['test_masks'].shape[-2:])

        loss_segm = self.loss_weight['segm'] * self.objective['segm'](segm_pred.view(gt_segm.shape), gt_segm)

        loss_segm_aux = 0
        loss_segm_aux_dict = {}
        if 'aux_segm' in self.loss_weight.keys():
            loss_segm_aux_dict = {k: self.objective['segm'](aux_mask_pred[k].view(gt_segm.shape), gt_segm)
                                  for k in self.loss_weight['aux_segm'].keys()}
            for k in self.loss_weight['aux_segm'].keys():
                loss_segm_aux += loss_segm_aux_dict[k] * self.loss_weight['aux_segm'][k]

        aux_enc_loss = 0
        if 'enc_aux_loss' in self.loss_weight.keys() and mask_enc_test is not None:
            aux_enc_loss = self.objective['enc_aux_loss'](mask_enc_test, target_scores[-1]) * self.loss_weight['enc_aux_loss']

        acc_l = [davis_jaccard_measure(torch.sigmoid(rm.detach()).cpu().numpy() > 0.5, lb.cpu().numpy()) for
                 rm, lb in zip(segm_pred.view(-1, *segm_pred.shape[-2:]), gt_segm.view(-1, *segm_pred.shape[-2:]))]
        acc = sum(acc_l)
        cnt = len(acc_l)

        # Total loss
        loss = loss_segm + loss_target_classifier + loss_test_init_clf + loss_test_iter_clf + aux_enc_loss + loss_segm_aux

        if torch.isinf(loss) or torch.isnan(loss):
            raise Exception('ERROR: Loss was nan or inf!!!')

        # Log stats
        stats = {'Loss/total': loss.item()}
        stats['Loss/segm'] = loss_segm.item()
        stats['acc'] = acc / cnt

        if target_scores is not None:
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
                        stats['ClfTrain/test_iter_loss'] = sum(clf_losses_test[1:-1]).item() / (
                                    len(clf_losses_test) - 2)

        if 'enc_aux_loss' in self.loss_weight.keys() and mask_enc_test is not None:
            stats['Loss/aux/mask_enc'] = aux_enc_loss.item()

        if 'aux_segm' in self.loss_weight.keys():
            stats['Loss/aux_segm'] = loss_segm_aux.item()
            for k, v in loss_segm_aux_dict.items():
                stats['Loss/aux/{}'.format(k)] = v.item()
        return loss, stats


class SegmDimpActor(BaseActor):
    """Actor for training the DiMP network."""
    def __init__(self, net, objective, loss_weight=None,
                 use_focal_loss=False, use_lovasz_loss=False,
                 disable_all_bn=False):
        super().__init__(net, objective)
        if loss_weight is None:
            loss_weight = {'segm': 1.0}
        self.loss_weight = loss_weight

        self.use_focal_loss = use_focal_loss
        self.use_lovasz_loss = use_lovasz_loss
        self.disable_all_bn = disable_all_bn

    def train(self, mode=True):
        """ Set whether the network is in train mode.
        args:
            mode (True) - Bool specifying whether in training mode.
        """
        self.net.train(mode)

        if self.disable_all_bn:
            for m in self.net.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()

    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'train_images', 'test_images', 'train_anno',
                    'test_proposals', 'proposal_iou', 'test_label', 'train_masks' and 'test_masks'
        returns:
            loss    - the training loss
            stats  -  dict containing detailed losses
        """
        # Run network
        segm_pred, target_scores_seg, aux_mask_pred, bb_pred, target_scores_dimp = self.net(
            train_imgs=data['train_images'],
            test_imgs=data['test_images'],
            train_masks=data['train_masks'],
            test_masks=data['test_masks'],
            train_bb=data['train_anno'],
            test_center_label=data['test_tc_label'],
            train_bb_in_sa=data.get('train_anno_in_sa', None),
            train_sa_bb=data.get('train_sa_bb', None),
            test_sa_bb=data.get('test_sa_bb', None))

        loss_dimp_classifier = 0
        loss_dimp_test_init_clf = 0
        loss_dimp_test_iter_clf = 0
        dimp_clf_loss_test = 0

        if 'dimp_test_clf' in self.loss_weight.keys() and target_scores_dimp is not None:
            # Classification losses for the different optimization iterations
            dimp_clf_losses_test = [self.objective['dimp_test_clf'](s, data['test_label'], data['test_anno']) for s in
                                    target_scores_dimp]

            dimp_clf_loss_test = dimp_clf_losses_test[-1]
            loss_dimp_classifier = self.loss_weight['dimp_test_clf'] * dimp_clf_loss_test

            # Loss for the initial filter iteration
            if 'dimp_test_init_clf' in self.loss_weight.keys():
                loss_dimp_test_init_clf = self.loss_weight['dimp_test_init_clf'] * dimp_clf_losses_test[0]

            # Loss for the intermediate filter iterations
            if 'dimp_test_iter_clf' in self.loss_weight.keys():
                dimp_test_iter_weights = self.loss_weight['dimp_test_iter_clf']
                if isinstance(dimp_test_iter_weights, list):
                    loss_dimp_test_iter_clf = sum([a * b for a, b in zip(dimp_test_iter_weights, dimp_clf_losses_test[1:-1])])
                else:
                    loss_dimp_test_iter_clf = (dimp_test_iter_weights / (len(dimp_clf_losses_test) - 2)) * sum(dimp_clf_losses_test[1:-1])

        loss_bbreg_w = 0
        if 'bbreg' in self.loss_weight.keys():
            gt_box = data['test_anno'].view(-1, 4).clone()
            gt_box_r1r2c1c2 = torch.stack((gt_box[:, 1], gt_box[:, 1] + gt_box[:, 3],
                                           gt_box[:, 0], gt_box[:, 0] + gt_box[:, 2]), dim=1)

            im_size = data['test_images'].shape[-2:]
            gt_box_r1r2c1c2[:, :2] /= im_size[0]
            gt_box_r1r2c1c2[:, 2:] /= im_size[1]
            gt_box_r1r2c1c2 = gt_box_r1r2c1c2.clamp(0.0, 1.0)
            loss_bbreg = self.objective['bbreg'](bb_pred.view(-1, 4), gt_box_r1r2c1c2)
            loss_bbreg_w = loss_bbreg * self.loss_weight['bbreg']

        gt_segm = data['test_masks'].view(-1, 1, *data['test_masks'].shape[-2:])

        loss_segm = self.loss_weight['segm'] * self.objective['segm'](segm_pred.view(gt_segm.shape), gt_segm)

        acc_l = [davis_jaccard_measure(torch.sigmoid(rm.detach()).cpu().numpy() > 0.5, lb.cpu().numpy()) for
                 rm, lb in zip(segm_pred.view(-1, *segm_pred.shape[-2:]), gt_segm.view(-1, *segm_pred.shape[-2:]))]
        acc = sum(acc_l)
        cnt = len(acc_l)

        # Total loss
        loss = loss_segm + loss_dimp_classifier + loss_dimp_test_init_clf + loss_dimp_test_iter_clf + loss_bbreg_w

        if torch.isinf(loss) or torch.isnan(loss):
            raise Exception('ERROR: Loss was nan or inf!!!')

        # Log stats
        stats = {'Loss/total': loss.item()}
        stats['Loss/segm'] = loss_segm.item()
        stats['Stats/acc'] = acc / cnt

        if target_scores_dimp is not None:
            if 'dimp_test_clf' in self.loss_weight.keys():
                stats['Loss/dimp_clf'] = loss_dimp_classifier.item()
            if 'dimp_test_init_clf' in self.loss_weight.keys():
                stats['Loss/dimp_test_init_clf'] = loss_dimp_test_init_clf.item()
            if 'dimp_test_iter_clf' in self.loss_weight.keys():
                stats['Loss/dimp_test_iter_clf'] = loss_dimp_test_iter_clf.item()

            if 'dimp_test_clf' in self.loss_weight.keys():
                stats['ClfTrain/dimp_test_loss'] = dimp_clf_loss_test.item()
                if len(dimp_clf_losses_test) > 0:
                    stats['ClfTrain/dimp_test_init_loss'] = dimp_clf_losses_test[0].item()
                    if len(dimp_clf_losses_test) > 2:
                        stats['ClfTrain/dimp_test_iter_loss'] = sum(dimp_clf_losses_test[1:-1]).item() / (
                                    len(dimp_clf_losses_test) - 2)

        if 'bbreg' in self.loss_weight.keys():
            stats['Loss/bbreg'] = loss_bbreg_w.item()
            stats['Loss/aux/bbreg'] = loss_bbreg.item()

        return loss, stats


class SegmSeqActor(BaseActor):
    """Actor for training the DiMP network."""
    def __init__(self, net, objective, loss_weight=None,
                 use_focal_loss=False, use_lovasz_loss=False,
                 detach_pred=True,
                 num_refinement_iter=3,
                 disable_backbone_bn=False,
                 disable_all_bn=False):
        super().__init__(net, objective)
        if loss_weight is None:
            loss_weight = {'segm': 1.0}
        self.loss_weight = loss_weight

        self.use_focal_loss = use_focal_loss
        self.use_lovasz_loss = use_lovasz_loss
        self.detach_pred = detach_pred
        self.num_refinement_iter = num_refinement_iter
        self.disable_backbone_bn = disable_backbone_bn
        self.disable_all_bn = disable_all_bn

    def train(self, mode=True):
        """ Set whether the network is in train mode.
        args:
            mode (True) - Bool specifying whether in training mode.
        """
        self.net.train(mode)

        if self.disable_all_bn:
            self.net.eval()
        elif self.disable_backbone_bn:
            for m in self.net.feature_extractor.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()

    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'train_images', 'test_images', 'train_anno',
                    'test_proposals', 'proposal_iou', 'test_label', 'train_masks' and 'test_masks'
        returns:
            loss    - the training loss
            stats  -  dict containing detailed losses
        """
        train_imgs = data['train_images']
        test_imgs = data['test_images']
        train_masks = data['train_masks']
        test_masks = data['test_masks']

        num_sequences = train_imgs.shape[1]
        num_train_frames = train_imgs.shape[0]
        num_test_frames = test_imgs.shape[0]

        im_size = test_imgs.shape[-2:]

        # Extract backbone features
        train_feat = self.net.extract_backbone_features(
            train_imgs.view(-1, train_imgs.shape[-3], train_imgs.shape[-2], train_imgs.shape[-1]))
        test_feat = self.net.extract_backbone_features(
            test_imgs.view(-1, test_imgs.shape[-3], test_imgs.shape[-2], test_imgs.shape[-1]))

        # Extract classification features
        #train_feat_bbon_clf = self.net.get_backbone_clf_feat(train_feat)
        train_feat_clf = self.net.extract_classification_feat(train_feat)  # seq*frames, channels, height, width
        train_feat_clf = train_feat_clf.view(num_train_frames, num_sequences, *train_feat_clf.shape[-3:])

        train_feat_clf_all = [train_feat_clf, ]

        train_mask_enc_info = self.net.label_encoder(train_masks, train_feat_clf)

        if isinstance(train_mask_enc_info, (tuple, list)):
            train_mask_enc = train_mask_enc_info[0]
            train_mask_sw = train_mask_enc_info[1]
        else:
            train_mask_enc = train_mask_enc_info
            train_mask_sw = None

        train_mask_enc_all = [train_mask_enc, ]
        train_mask_sw_all = None if train_mask_sw is None else [train_mask_sw, ]

        test_feat_clf = self.net.extract_classification_feat(test_feat)  # seq*frames, channels, height, width

        loss_target_classifier_all = 0
        loss_test_init_clf_all = 0
        loss_test_iter_clf_all = 0
        loss_segm_all = 0
        loss_bbreg_all = 0

        if 'aux_segm' in self.loss_weight.keys():
            loss_segm_aux_all = {k: 0 for k in self.loss_weight['aux_segm'].keys()}
        else:
            loss_segm_aux_all = 0
        acc = 0
        cnt = 0

        filter, filter_iter, _ = self.net.classifier.get_filter(train_feat_clf, (train_mask_enc, train_mask_sw))

        for i in range(num_test_frames):
            test_feat_clf_it = test_feat_clf.view(num_test_frames, num_sequences, *test_feat_clf.shape[-3:])[i:i+1, ...]
            target_scores = [self.net.classifier.classify(f, test_feat_clf_it) for f in filter_iter]

            test_feat_it = {k: v.view(num_test_frames, num_sequences, *v.shape[-3:])[i, ...] for k, v in test_feat.items()}
            target_scores_last_iter = target_scores[-1]
            mask_pred, decoder_feat = self.net.decoder(target_scores_last_iter, test_feat_it, test_imgs.shape[-2:],
                                                       ('layer4_dec', 'layer3_dec', 'layer2_dec', 'layer1_dec'))
            mask_pred = mask_pred.view(1, num_sequences, *mask_pred.shape[-2:])

            decoder_feat['mask_enc'] = target_scores_last_iter.view(-1, *target_scores_last_iter.shape[-3:])
            aux_mask_pred = {}
            for L, predictor in self.net.aux_layers.items():
                aux_mask_pred[L] = predictor(decoder_feat[L], test_imgs.shape[-2:])

            if self.net.bb_regressor is not None:
                bb_pred = self.net.bb_regressor(decoder_feat[self.net.bbreg_decoder_layer])

            if self.detach_pred:
                mask_pred_clone = mask_pred.clone().detach()
            else:
                mask_pred_clone = mask_pred.clone()

            mask_pred_clone = torch.sigmoid(mask_pred_clone)

            mask_pred_enc_info = self.net.label_encoder(mask_pred_clone, test_feat_clf_it)
            if isinstance(mask_pred_enc_info, (tuple, list)):
                mask_pred_enc = mask_pred_enc_info[0]
                mask_pred_enc_sw = mask_pred_enc_info[1]
            else:
                mask_pred_enc = mask_pred_enc_info
                mask_pred_enc_sw = None

            train_mask_enc_all.append(mask_pred_enc)
            if train_mask_sw_all is not None:
                train_mask_sw_all.append(mask_pred_enc_sw)
            train_feat_clf_all.append(test_feat_clf_it)

            if 'test_clf' in self.loss_weight.keys() and target_scores is not None:
                # Classification losses for the different optimization iterations
                clf_losses_test = [self.objective['test_clf'](input=s.view(-1, 1, *s.shape[-2:]),
                                                              target=test_masks[i:i+1, ...].view(-1, 1,
                                                                                                 *test_masks.shape[
                                                                                                  -2:])) for s in
                                   target_scores]

                # Loss of the final filter
                clf_loss_test = clf_losses_test[-1]
                loss_target_classifier = self.loss_weight['test_clf'] * clf_loss_test
                loss_target_classifier_all += loss_target_classifier

                # Loss for the initial filter iteration
                if 'test_init_clf' in self.loss_weight.keys():
                    loss_test_init_clf = self.loss_weight['test_init_clf'] * clf_losses_test[0]
                    loss_test_init_clf_all += loss_test_init_clf

                # Loss for the intermediate filter iterations
                if 'test_iter_clf' in self.loss_weight.keys():
                    test_iter_weights = self.loss_weight['test_iter_clf']
                    if isinstance(test_iter_weights, list):
                        loss_test_iter_clf = sum([a * b for a, b in zip(test_iter_weights, clf_losses_test[1:-1])])
                    else:
                        loss_test_iter_clf = (test_iter_weights / (len(clf_losses_test) - 2)) * sum(
                            clf_losses_test[1:-1])
                    loss_test_iter_clf_all += loss_test_iter_clf

            gt_segm = test_masks[i:i+1, ...].view(-1, 1, *test_masks.shape[-2:])

            loss_segm = self.loss_weight['segm'] * self.objective['segm'](mask_pred.view(gt_segm.shape), gt_segm)
            loss_segm_all += loss_segm

            if 'aux_segm' in self.loss_weight.keys():
                loss_segm_aux = {k: self.objective['segm'](aux_mask_pred[k].view(gt_segm.shape), gt_segm)
                                 for k in self.loss_weight['aux_segm'].keys()}
                for k in self.loss_weight['aux_segm'].keys():
                    loss_segm_aux_all[k] += loss_segm_aux[k]

            if 'bbreg' in self.loss_weight.keys():
                gt_box = data['test_anno'][i, :, :].clone()
                gt_box_r1r2c1c2 = torch.stack((gt_box[:, 1], gt_box[:, 1] + gt_box[:, 3],
                                             gt_box[:, 0], gt_box[:, 0] + gt_box[:, 2]), dim=1)

                gt_box_r1r2c1c2[:, :2] /= im_size[0]
                gt_box_r1r2c1c2[:, 2:] /= im_size[1]
                loss_bbreg = self.objective['bbreg'](bb_pred, gt_box_r1r2c1c2)
                loss_bbreg_all += loss_bbreg

            acc_l = [davis_jaccard_measure(torch.sigmoid(rm.detach()).cpu().numpy() > 0.5, lb.cpu().numpy()) for
                     rm, lb in zip(mask_pred.view(-1, *mask_pred.shape[-2:]), gt_segm.view(-1, *mask_pred.shape[-2:]))]
            acc += sum(acc_l)
            cnt += len(acc_l)

            ## Update
            if i < (num_test_frames - 1) and (self.num_refinement_iter > 0):
                train_feat_clf_it = torch.cat(train_feat_clf_all, dim=0)
                train_mask_enc_it = torch.cat(train_mask_enc_all, dim=0)

                if train_mask_sw_all is not None:
                    train_mask_sw_it = torch.cat(train_mask_sw_all, dim=0)
                else:
                    train_mask_sw_it = None

                filter_tl, _, _ = self.net.classifier.filter_optimizer(TensorList([filter]),
                                                                       feat=train_feat_clf_it,
                                                                       mask=train_mask_enc_it,
                                                                       sample_weight=train_mask_sw_it,
                                                                       num_iter=self.num_refinement_iter)

                filter = filter_tl[0]

        # Total loss
        loss_segm_all /= num_test_frames
        loss_target_classifier_all /= num_test_frames
        loss_test_init_clf_all /= num_test_frames
        loss_test_iter_clf_all /= num_test_frames
        loss_bbreg_all /= num_test_frames
        loss_bbreg_all_w = loss_bbreg_all * self.loss_weight.get('bbreg', 0.0)

        if 'aux_segm' in self.loss_weight.keys():
            loss_segm_aux_all_sum = 0
            for k in self.loss_weight['aux_segm'].keys():
                loss_segm_aux_all_sum += loss_segm_aux_all[k] * self.loss_weight['aux_segm'][k]
            loss_segm_aux_all_sum /= num_test_frames
        else:
            loss_segm_aux_all_sum = 0

        loss = loss_segm_all + loss_target_classifier_all + loss_test_init_clf_all + loss_test_iter_clf_all + \
               loss_segm_aux_all_sum + loss_bbreg_all_w

        if torch.isinf(loss) or torch.isnan(loss):
            raise Exception('ERROR: Loss was nan or inf!!!')

        # Log stats
        stats = {'Loss/total': loss.item()}
        stats['Loss/segm'] = loss_segm_all.item()
        if 'aux_segm' in self.loss_weight.keys():
            stats['Loss/segm_aux'] = loss_segm_aux_all_sum.item()

            for k in self.loss_weight['aux_segm'].keys():
                stats['Loss/aux/{}'.format(k)] = loss_segm_aux_all[k]

        if 'bbreg' in self.loss_weight.keys():
            stats['Loss/bbreg'] = loss_bbreg_all_w.item()
            stats['Loss/aux/bbreg'] = loss_bbreg_all.item()

        if 'test_clf' in self.loss_weight.keys():
            stats['Loss/target_clf'] = loss_target_classifier_all.item()
        if 'test_init_clf' in self.loss_weight.keys():
            stats['Loss/test_init_clf'] = loss_test_init_clf_all.item()
        if 'test_iter_clf' in self.loss_weight.keys():
            stats['Loss/test_iter_clf'] = loss_test_iter_clf_all.item()

        stats['Stats/acc'] = acc / cnt
        return loss, stats


class SegmSeqActorMGPU(BaseActor):
    """Actor for training the DiMP network."""
    def __init__(self, net, objective, loss_weight=None,
                 use_focal_loss=False, use_lovasz_loss=False,
                 detach_pred=True,
                 num_refinement_iter=3,
                 disable_backbone_bn=False,
                 disable_all_bn=False):
        super().__init__(net, objective)
        if loss_weight is None:
            loss_weight = {'segm': 1.0}
        self.loss_weight = loss_weight

        self.use_focal_loss = use_focal_loss
        self.use_lovasz_loss = use_lovasz_loss
        self.detach_pred = detach_pred
        self.num_refinement_iter = num_refinement_iter
        self.disable_backbone_bn = disable_backbone_bn
        self.disable_all_bn = disable_all_bn

    def train(self, mode=True):
        """ Set whether the network is in train mode.
        args:
            mode (True) - Bool specifying whether in training mode.
        """
        self.net.train(mode)

        if self.disable_all_bn:
            self.net.eval()
        elif self.disable_backbone_bn:
            for m in self.net.feature_extractor.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()

    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'train_images', 'test_images', 'train_anno',
                    'test_proposals', 'proposal_iou', 'test_label', 'train_masks' and 'test_masks'
        returns:
            loss    - the training loss
            stats  -  dict containing detailed losses
        """
        segm_pred = self.net(train_imgs=data['train_images'],
                             test_imgs=data['test_images'],
                             train_masks=data['train_masks'],
                             test_masks=data['test_masks'],
                             num_refinement_iter=self.num_refinement_iter)

        acc = 0
        cnt = 0

        segm_pred = segm_pred.view(-1, 1, *segm_pred.shape[-2:])
        gt_segm = data['test_masks']
        gt_segm = gt_segm.view(-1, 1, *gt_segm.shape[-2:])

        loss_segm = self.loss_weight['segm'] * self.objective['segm'](segm_pred, gt_segm)

        acc_l = [davis_jaccard_measure(torch.sigmoid(rm.detach()).cpu().numpy() > 0.5, lb.cpu().numpy()) for
                 rm, lb in zip(segm_pred.view(-1, *segm_pred.shape[-2:]), gt_segm.view(-1, *segm_pred.shape[-2:]))]
        acc += sum(acc_l)
        cnt += len(acc_l)

        loss = loss_segm

        if torch.isinf(loss) or torch.isnan(loss):
            raise Exception('ERROR: Loss was nan or inf!!!')

        # Log stats
        stats = {'Loss/total': loss.item()}
        stats['Loss/segm'] = loss_segm.item()

        stats['Stats/acc'] = acc / cnt
        return loss, stats


class SegmSeqActorv2(BaseActor):
    """Actor for training the DiMP network."""
    def __init__(self, net, objective, loss_weight=None,
                 use_focal_loss=False, use_lovasz_loss=False,
                 detach_pred=True,
                 num_refinement_iter=3,
                 disable_backbone_bn=False,
                 feat_dropout=None,
                 enc_dropout=None):
        super().__init__(net, objective)
        if loss_weight is None:
            loss_weight = {'segm': 1.0}
        self.loss_weight = loss_weight

        self.use_focal_loss = use_focal_loss
        self.use_lovasz_loss = use_lovasz_loss
        self.detach_pred = detach_pred
        self.num_refinement_iter = num_refinement_iter
        self.disable_backbone_bn = disable_backbone_bn

        self.feat_dropout = feat_dropout
        self.enc_dropout = enc_dropout

    def train(self, mode=True):
        """ Set whether the network is in train mode.
        args:
            mode (True) - Bool specifying whether in training mode.
        """
        self.net.train(mode)

        if self.disable_backbone_bn:
            for m in self.net.feature_extractor.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()

    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'train_images', 'test_images', 'train_anno',
                    'test_proposals', 'proposal_iou', 'test_label', 'train_masks' and 'test_masks'
        returns:
            loss    - the training loss
            stats  -  dict containing detailed losses
        """
        train_imgs = data['train_images']
        test_imgs = data['test_images']
        train_masks = data['train_masks']
        test_masks = data['test_masks']

        num_sequences = train_imgs.shape[1]
        num_train_frames = train_imgs.shape[0]
        num_test_frames = test_imgs.shape[0]

        im_size = test_imgs.shape[-2:]

        # Extract backbone features
        train_feat = self.net.extract_backbone_features(
            train_imgs.view(-1, train_imgs.shape[-3], train_imgs.shape[-2], train_imgs.shape[-1]))
        test_feat = self.net.extract_backbone_features(
            test_imgs.view(-1, test_imgs.shape[-3], test_imgs.shape[-2], test_imgs.shape[-1]))

        # Extract classification features
        train_feat_backbone = self.net.get_backbone_clf_feat(train_feat)
        if self.feat_dropout is not None:
            train_feat_backbone = F.dropout2d(train_feat_backbone, p=self.feat_dropout, inplace=True)

        train_feat_clf = self.net.classifier.extract_classification_feat(train_feat_backbone)
        # train_feat_clf = self.net.extract_classification_feat(train_feat)  # seq*frames, channels, height, width

        train_feat_clf = train_feat_clf.view(num_train_frames, num_sequences, *train_feat_clf.shape[-3:])

        train_feat_clf_all = [train_feat_clf, ]

        train_mask_enc_info = self.net.label_encoder(train_masks, train_feat_clf)

        if isinstance(train_mask_enc_info, (tuple, list)):
            train_mask_enc = train_mask_enc_info[0]
            train_mask_sw = train_mask_enc_info[1]
        else:
            train_mask_enc = train_mask_enc_info
            train_mask_sw = None

        train_mask_enc_all = [train_mask_enc, ]
        train_mask_sw_all = None if train_mask_sw is None else [train_mask_sw, ]

        # test_feat_clf = self.net.extract_classification_feat(test_feat)  # seq*frames, channels, height, width

        # Extract classification features
        test_feat_backbone = self.net.get_backbone_clf_feat(test_feat)
        if self.feat_dropout is not None:
            test_feat_backbone = F.dropout2d(test_feat_backbone, p=self.feat_dropout, inplace=True)

        test_feat_clf = self.net.classifier.extract_classification_feat(test_feat_backbone)

        loss_target_classifier_all = 0
        loss_test_init_clf_all = 0
        loss_test_iter_clf_all = 0
        loss_segm_all = 0
        loss_bbreg_all = 0

        if 'aux_segm' in self.loss_weight.keys():
            loss_segm_aux_all = {k: 0 for k in self.loss_weight['aux_segm'].keys()}
        else:
            loss_segm_aux_all = 0
        acc = 0
        cnt = 0

        filter, filter_iter, _ = self.net.classifier.get_filter(train_feat_clf, (train_mask_enc, train_mask_sw))

        for i in range(num_test_frames):
            test_feat_clf_it = test_feat_clf.view(num_test_frames, num_sequences, *test_feat_clf.shape[-3:])[i:i+1, ...]
            target_scores = [self.net.classifier.classify(f, test_feat_clf_it) for f in filter_iter]

            test_feat_it = {k: v.view(num_test_frames, num_sequences, *v.shape[-3:])[i, ...] for k, v in test_feat.items()}
            target_scores_last_iter = target_scores[-1]

            if self.enc_dropout is not None:
                ts_shape = target_scores_last_iter.shape
                target_scores_last_iter = target_scores_last_iter.view(-1, *target_scores_last_iter.shape[-3:])
                target_scores_last_iter = F.dropout2d(target_scores_last_iter, p=self.enc_dropout, inplace=True)
                target_scores_last_iter = target_scores_last_iter.view(ts_shape)

            mask_pred, decoder_feat = self.net.decoder(target_scores_last_iter, test_feat_it, test_imgs.shape[-2:],
                                                       ('layer4_dec', 'layer3_dec', 'layer2_dec', 'layer1_dec'))
            mask_pred = mask_pred.view(1, num_sequences, *mask_pred.shape[-2:])

            decoder_feat['mask_enc'] = target_scores_last_iter.view(-1, *target_scores_last_iter.shape[-3:])
            aux_mask_pred = {}
            for L, predictor in self.net.aux_layers.items():
                aux_mask_pred[L] = predictor(decoder_feat[L], test_imgs.shape[-2:])

            if self.net.bb_regressor is not None:
                bb_pred = self.net.bb_regressor(decoder_feat[self.net.bbreg_decoder_layer])

            if self.detach_pred:
                mask_pred_clone = mask_pred.clone().detach()
            else:
                mask_pred_clone = mask_pred.clone()

            mask_pred_clone = torch.sigmoid(mask_pred_clone)

            mask_pred_enc_info = self.net.label_encoder(mask_pred_clone, test_feat_clf_it)
            if isinstance(mask_pred_enc_info, (tuple, list)):
                mask_pred_enc = mask_pred_enc_info[0]
                mask_pred_enc_sw = mask_pred_enc_info[1]
            else:
                mask_pred_enc = mask_pred_enc_info
                mask_pred_enc_sw = None

            train_mask_enc_all.append(mask_pred_enc)
            if train_mask_sw_all is not None:
                train_mask_sw_all.append(mask_pred_enc_sw)
            train_feat_clf_all.append(test_feat_clf_it)

            if 'test_clf' in self.loss_weight.keys() and target_scores is not None:
                # Classification losses for the different optimization iterations
                clf_losses_test = [self.objective['test_clf'](input=s.view(-1, 1, *s.shape[-2:]),
                                                              target=test_masks[i:i+1, ...].view(-1, 1,
                                                                                                 *test_masks.shape[
                                                                                                  -2:])) for s in
                                   target_scores]

                # Loss of the final filter
                clf_loss_test = clf_losses_test[-1]
                loss_target_classifier = self.loss_weight['test_clf'] * clf_loss_test
                loss_target_classifier_all += loss_target_classifier

                # Loss for the initial filter iteration
                if 'test_init_clf' in self.loss_weight.keys():
                    loss_test_init_clf = self.loss_weight['test_init_clf'] * clf_losses_test[0]
                    loss_test_init_clf_all += loss_test_init_clf

                # Loss for the intermediate filter iterations
                if 'test_iter_clf' in self.loss_weight.keys():
                    test_iter_weights = self.loss_weight['test_iter_clf']
                    if isinstance(test_iter_weights, list):
                        loss_test_iter_clf = sum([a * b for a, b in zip(test_iter_weights, clf_losses_test[1:-1])])
                    else:
                        loss_test_iter_clf = (test_iter_weights / (len(clf_losses_test) - 2)) * sum(
                            clf_losses_test[1:-1])
                    loss_test_iter_clf_all += loss_test_iter_clf

            gt_segm = test_masks[i:i+1, ...].view(-1, 1, *test_masks.shape[-2:])

            loss_segm = self.loss_weight['segm'] * self.objective['segm'](mask_pred.view(gt_segm.shape), gt_segm)
            loss_segm_all += loss_segm

            if 'aux_segm' in self.loss_weight.keys():
                loss_segm_aux = {k: self.objective['segm'](aux_mask_pred[k].view(gt_segm.shape), gt_segm)
                                 for k in self.loss_weight['aux_segm'].keys()}
                for k in self.loss_weight['aux_segm'].keys():
                    loss_segm_aux_all[k] += loss_segm_aux[k]

            if 'bbreg' in self.loss_weight.keys():
                gt_box = data['test_anno'][i, :, :].clone()
                gt_box_r1r2c1c2 = torch.stack((gt_box[:, 1], gt_box[:, 1] + gt_box[:, 3],
                                             gt_box[:, 0], gt_box[:, 0] + gt_box[:, 2]), dim=1)

                gt_box_r1r2c1c2[:, :2] /= im_size[0]
                gt_box_r1r2c1c2[:, 2:] /= im_size[1]
                loss_bbreg = self.objective['bbreg'](bb_pred, gt_box_r1r2c1c2)
                loss_bbreg_all += loss_bbreg

            acc_l = [davis_jaccard_measure(torch.sigmoid(rm.detach()).cpu().numpy() > 0.5, lb.cpu().numpy()) for
                     rm, lb in zip(mask_pred.view(-1, *mask_pred.shape[-2:]), gt_segm.view(-1, *mask_pred.shape[-2:]))]
            acc += sum(acc_l)
            cnt += len(acc_l)

            ## Update
            if i < (num_test_frames - 1):
                train_feat_clf_it = torch.cat(train_feat_clf_all, dim=0)
                train_mask_enc_it = torch.cat(train_mask_enc_all, dim=0)

                if train_mask_sw_all is not None:
                    train_mask_sw_it = torch.cat(train_mask_sw_all, dim=0)
                else:
                    train_mask_sw_it = None

                filter_tl, _, _ = self.net.classifier.filter_optimizer(TensorList([filter]),
                                                                       feat=train_feat_clf_it,
                                                                       mask=train_mask_enc_it,
                                                                       sample_weight=train_mask_sw_it,
                                                                       num_iter=self.num_refinement_iter)

                filter = filter_tl[0]

        # Total loss
        loss_segm_all /= num_test_frames
        loss_target_classifier_all /= num_test_frames
        loss_test_init_clf_all /= num_test_frames
        loss_test_iter_clf_all /= num_test_frames
        loss_bbreg_all /= num_test_frames
        loss_bbreg_all_w = loss_bbreg_all * self.loss_weight.get('bbreg', 0.0)

        if 'aux_segm' in self.loss_weight.keys():
            loss_segm_aux_all_sum = 0
            for k in self.loss_weight['aux_segm'].keys():
                loss_segm_aux_all_sum += loss_segm_aux_all[k] * self.loss_weight['aux_segm'][k]
            loss_segm_aux_all_sum /= num_test_frames
        else:
            loss_segm_aux_all_sum = 0

        loss = loss_segm_all + loss_target_classifier_all + loss_test_init_clf_all + loss_test_iter_clf_all + \
               loss_segm_aux_all_sum + loss_bbreg_all_w

        if torch.isinf(loss) or torch.isnan(loss):
            raise Exception('ERROR: Loss was nan or inf!!!')

        # Log stats
        stats = {'Loss/total': loss.item()}
        stats['Loss/segm'] = loss_segm_all.item()
        if 'aux_segm' in self.loss_weight.keys():
            stats['Loss/segm_aux'] = loss_segm_aux_all_sum.item()

            for k in self.loss_weight['aux_segm'].keys():
                stats['Loss/aux/{}'.format(k)] = loss_segm_aux_all[k]

        if 'bbreg' in self.loss_weight.keys():
            stats['Loss/bbreg'] = loss_bbreg_all_w.item()
            stats['Loss/aux/bbreg'] = loss_bbreg_all.item()

        if 'test_clf' in self.loss_weight.keys():
            stats['Loss/target_clf'] = loss_target_classifier_all.item()
        if 'test_init_clf' in self.loss_weight.keys():
            stats['Loss/test_init_clf'] = loss_test_init_clf_all.item()
        if 'test_iter_clf' in self.loss_weight.keys():
            stats['Loss/test_iter_clf'] = loss_test_iter_clf_all.item()

        stats['Stats/acc'] = acc / cnt
        return loss, stats

class SegmMultiObjActor(BaseActor):
    """Actor for training the DiMP network."""
    def __init__(self, net, objective, loss_weight=None,
                 use_focal_loss=False, use_lovasz_loss=False,
                 disable_all_bn=False,
                 max_num_objects=3):
        super().__init__(net, objective)
        if loss_weight is None:
            loss_weight = {'segm': 1.0}
        self.loss_weight = loss_weight

        self.use_focal_loss = use_focal_loss
        self.use_lovasz_loss = use_lovasz_loss
        self.disable_all_bn = disable_all_bn
        self.max_num_objects = max_num_objects

    def train(self, mode=True):
        """ Set whether the network is in train mode.
        args:
            mode (True) - Bool specifying whether in training mode.
        """
        self.net.train(mode)

        if self.disable_all_bn:
            for m in self.net.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()

    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'train_images', 'test_images', 'train_anno',
                    'test_proposals', 'proposal_iou', 'test_label', 'train_masks' and 'test_masks'
        returns:
            loss    - the training loss
            stats  -  dict containing detailed losses
        """
        # Run network
        segm_pred, target_scores, mask_enc_test, aux_mask_pred, bb_pred = self.net(train_imgs=data['train_images'],
                                                                                   test_imgs=data['test_images'],
                                                                                   train_masks=data['train_masks'],
                                                                                   test_masks=data['test_masks'],
                                                                                   valid_object=data['valid_object'])

        loss_target_classifier = 0
        loss_test_init_clf = 0
        loss_test_iter_clf = 0
        clf_loss_test = 0
        if 'test_clf' in self.loss_weight.keys() and target_scores is not None:
            raise NotImplementedError

        gt_segm = data['test_masks'].view(-1, *data['test_masks'].shape[-2:])
        segm_pred = segm_pred.view(-1, *segm_pred.shape[-3:])
        loss_segm = self.loss_weight['segm'] * self.objective['segm'](segm_pred, gt_segm)

        loss_segm_aux = 0
        loss_segm_aux_dict = {}
        if 'aux_segm' in self.loss_weight.keys():
            raise NotImplementedError

        aux_enc_loss = 0
        if 'enc_aux_loss' in self.loss_weight.keys() and mask_enc_test is not None:
            raise NotImplementedError

        segm_pred_labels = segm_pred.argmax(dim=1)
        acc_l = [davis_jaccard_measure(rm.detach().cpu().numpy(), lb.cpu().numpy()) for
                 rm, lb in zip(segm_pred_labels.view(-1, *segm_pred.shape[-2:]), gt_segm.view(-1, *segm_pred.shape[-2:]))]
        acc = sum(acc_l)
        cnt = len(acc_l)

        # Total loss
        loss = loss_segm + loss_target_classifier + loss_test_init_clf + loss_test_iter_clf + aux_enc_loss + loss_segm_aux

        if torch.isinf(loss) or torch.isnan(loss):
            raise Exception('ERROR: Loss was nan or inf!!!')

        # Log stats
        stats = {'Loss/total': loss.item()}
        stats['Loss/segm'] = loss_segm.item()
        stats['acc'] = acc / cnt

        # if target_scores is not None:
        #     raise NotImplementedError

        if 'enc_aux_loss' in self.loss_weight.keys() and mask_enc_test is not None:
            raise NotImplementedError

        if 'aux_segm' in self.loss_weight.keys():
            raise NotImplementedError

        return loss, stats


class SegmSeqMultiObjActor(BaseActor):
    """Actor for training the DiMP network."""
    def __init__(self, net, objective, loss_weight=None,
                 use_focal_loss=False, use_lovasz_loss=False,
                 detach_pred=True,
                 num_refinement_iter=3):
        super().__init__(net, objective)
        if loss_weight is None:
            loss_weight = {'segm': 1.0}
        self.loss_weight = loss_weight

        self.use_focal_loss = use_focal_loss
        self.use_lovasz_loss = use_lovasz_loss
        self.detach_pred = detach_pred
        self.num_refinement_iter = num_refinement_iter

    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'train_images', 'test_images', 'train_anno',
                    'test_proposals', 'proposal_iou', 'test_label', 'train_masks' and 'test_masks'
        returns:
            loss    - the training loss
            stats  -  dict containing detailed losses
        """
        train_imgs = data['train_images']
        test_imgs = data['test_images']
        train_masks = data['train_masks']
        test_masks = data['test_masks']
        valid_object = data['valid_object']

        num_sequences = train_imgs.shape[1]
        num_train_frames = train_imgs.shape[0]
        num_test_frames = test_imgs.shape[0]

        im_size = test_imgs.shape[-2:]

        # Extract backbone features
        train_feat = self.net.extract_backbone_features(
            train_imgs.view(-1, train_imgs.shape[-3], train_imgs.shape[-2], train_imgs.shape[-1]))
        test_feat = self.net.extract_backbone_features(
            test_imgs.view(-1, test_imgs.shape[-3], test_imgs.shape[-2], test_imgs.shape[-1]))

        # Extract classification features
        train_feat_clf = self.net.extract_classification_feat(train_feat)  # seq*frames, channels, height, width
        train_feat_clf = train_feat_clf.view(num_train_frames, num_sequences, *train_feat_clf.shape[-3:])

        train_feat_clf_all = [train_feat_clf, ]

        object_ids = list(range(1, valid_object.shape[2] + 1))
        train_mask_enc_info = self.net.label_encoder(train_masks, feature=train_feat_clf, object_ids=object_ids)
        valid_object = valid_object.view(-1, len(object_ids), 1, 1).float()

        if isinstance(train_mask_enc_info, (tuple, list)):
            train_mask_enc = train_mask_enc_info[0]
            train_mask_sw = train_mask_enc_info[1]

            train_mask_enc = train_mask_enc.view(num_train_frames, num_sequences, -1, *train_mask_enc.shape[-2:])
            train_mask_sw = train_mask_sw.view(num_train_frames, num_sequences, -1, *train_mask_sw.shape[-2:])
        else:
            train_mask_enc = train_mask_enc_info
            train_mask_enc = train_mask_enc.view(num_train_frames, num_sequences, -1, *train_mask_enc.shape[-2:])

            train_mask_sw = None

        train_mask_enc_all = [train_mask_enc, ]
        train_mask_sw_all = None if train_mask_sw is None else [train_mask_sw, ]

        test_feat_clf = self.net.extract_classification_feat(test_feat)  # seq*frames, channels, height, width

        loss_target_classifier_all = 0
        loss_test_init_clf_all = 0
        loss_test_iter_clf_all = 0
        loss_segm_all = 0
        loss_bbreg_all = 0

        if 'aux_segm' in self.loss_weight.keys():
            loss_segm_aux_all = {k: 0 for k in self.loss_weight['aux_segm'].keys()}
        else:
            loss_segm_aux_all = 0
        acc = 0
        cnt = 0

        filter, filter_iter, _ = self.net.classifier.get_filter(train_feat_clf, (train_mask_enc, train_mask_sw),
                                                                num_objects=len(object_ids))

        for i in range(num_test_frames):
            test_feat_clf_it = test_feat_clf.view(num_test_frames, num_sequences, *test_feat_clf.shape[-3:])[i:i+1, ...]
            target_scores = [self.net.classifier.classify(f, test_feat_clf_it) for f in filter_iter]

            test_feat_it = {k: v.view(num_test_frames, num_sequences, *v.shape[-3:])[i, ...] for k, v in test_feat.items()}
            target_scores_last_iter = target_scores[-1]

            target_scores_last_iter = target_scores_last_iter.view(1, num_sequences, len(object_ids),
                                                                   -1, *target_scores_last_iter.shape[-2:])

            mask_pred, decoder_feat = self.net.decoder(target_scores_last_iter, test_feat_it, test_imgs.shape[-2:],
                                                       ('layer4_dec', 'layer3_dec', 'layer2_dec', 'layer1_dec'),
                                                       num_objects=len(object_ids))

            mask_pred = mask_pred.view(num_sequences, len(object_ids), *test_imgs.shape[-2:])

            mask_pred_valid = mask_pred * valid_object + -100.0 * (1.0 - valid_object)
            mask_pred_merged = self.net.merge_segmentation_results(mask_pred_valid)

            if self.detach_pred:
                mask_pred_clone = mask_pred_merged.clone().detach()
            else:
                mask_pred_clone = mask_pred_merged.clone()

            mask_pred_label = mask_pred_clone.argmax(dim=1).view(1, num_sequences, *mask_pred.shape[-2:])

            mask_pred_enc_info = self.net.label_encoder(mask_pred_label, feature=test_feat_clf_it,
                                                        object_ids=object_ids)
            if isinstance(mask_pred_enc_info, (tuple, list)):
                mask_pred_enc = mask_pred_enc_info[0]
                mask_pred_enc_sw = mask_pred_enc_info[1]

                mask_pred_enc = mask_pred_enc.view(1, num_sequences, -1, *mask_pred_enc.shape[-2:])
                mask_pred_enc_sw = mask_pred_enc_sw.view(1, num_sequences, -1, *mask_pred_enc_sw.shape[-2:])
            else:
                mask_pred_enc = mask_pred_enc_info
                mask_pred_enc_sw = None

                mask_pred_enc = mask_pred_enc.view(1, num_sequences, -1, *mask_pred_enc.shape[-2:])

            train_mask_enc_all.append(mask_pred_enc)
            if train_mask_sw_all is not None:
                train_mask_sw_all.append(mask_pred_enc_sw)
            train_feat_clf_all.append(test_feat_clf_it)

            if 'test_clf' in self.loss_weight.keys() and target_scores is not None:
               raise NotImplementedError

            gt_segm = test_masks[i:i+1, ...].view(-1, *test_masks.shape[-2:])
            mask_pred_merged = mask_pred_merged.view(-1, *mask_pred_merged.shape[-3:])

            loss_segm = self.loss_weight['segm'] * self.objective['segm'](mask_pred_merged, gt_segm)
            loss_segm_all += loss_segm

            if 'aux_segm' in self.loss_weight.keys():
                raise NotImplementedError

            if 'bbreg' in self.loss_weight.keys():
                raise NotImplementedError

            acc_l = [davis_jaccard_measure(rm.detach().cpu().numpy(), lb.cpu().numpy()) for
                     rm, lb in zip(mask_pred_label.view(-1, *mask_pred.shape[-2:]), gt_segm.view(-1, *mask_pred.shape[-2:]))]
            acc += sum(acc_l)
            cnt += len(acc_l)

            ## Update
            if i < (num_test_frames - 1):
                train_feat_clf_it = torch.cat(train_feat_clf_all, dim=0)
                train_mask_enc_it = torch.cat(train_mask_enc_all, dim=0)

                if train_mask_sw_all is not None:
                    train_mask_sw_it = torch.cat(train_mask_sw_all, dim=0)
                else:
                    train_mask_sw_it = None

                filter_tl, _, _ = self.net.classifier.filter_optimizer(TensorList([filter]),
                                                                       feat=train_feat_clf_it,
                                                                       mask=train_mask_enc_it,
                                                                       sample_weight=train_mask_sw_it,
                                                                       num_iter=self.num_refinement_iter)

                filter = filter_tl[0]

        # Total loss
        loss_segm_all /= num_test_frames
        loss_target_classifier_all /= num_test_frames
        loss_test_init_clf_all /= num_test_frames
        loss_test_iter_clf_all /= num_test_frames
        loss_bbreg_all /= num_test_frames
        loss_bbreg_all_w = loss_bbreg_all * self.loss_weight.get('bbreg', 0.0)

        if 'aux_segm' in self.loss_weight.keys():
            raise NotImplementedError
        else:
            loss_segm_aux_all_sum = 0

        loss = loss_segm_all + loss_target_classifier_all + loss_test_init_clf_all + loss_test_iter_clf_all + \
               loss_segm_aux_all_sum + loss_bbreg_all_w

        if torch.isinf(loss) or torch.isnan(loss):
            raise Exception('ERROR: Loss was nan or inf!!!')

        # Log stats
        stats = {'Loss/total': loss.item()}
        stats['Loss/segm'] = loss_segm_all.item()

        stats['Stats/acc'] = acc / cnt
        return loss, stats


class SegmSeqMultiObjActorv2(BaseActor):
    """Actor for training the DiMP network."""
    def __init__(self, net, objective, loss_weight=None,
                 use_focal_loss=False, use_lovasz_loss=False,
                 detach_pred=True,
                 num_refinement_iter=3,
                 single_obj_mode=False,
                 merge_results=True,
                 disable_all_bn=False):
        super().__init__(net, objective)
        if loss_weight is None:
            loss_weight = {'segm': 1.0}
        self.loss_weight = loss_weight

        self.use_focal_loss = use_focal_loss
        self.use_lovasz_loss = use_lovasz_loss
        self.detach_pred = detach_pred
        self.num_refinement_iter = num_refinement_iter
        self.single_obj_mode = single_obj_mode
        self.merge_results = merge_results
        self.disable_all_bn = disable_all_bn


    def train(self, mode=True):
        """ Set whether the network is in train mode.
        args:
            mode (True) - Bool specifying whether in training mode.
        """
        self.net.train(mode)

        if self.disable_all_bn:
            self.net.eval()

    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'train_images', 'test_images', 'train_anno',
                    'test_proposals', 'proposal_iou', 'test_label', 'train_masks' and 'test_masks'
        returns:
            loss    - the training loss
            stats  -  dict containing detailed losses
        """
        if self.single_obj_mode:
            assert not self.merge_results
            object_ids = [0,]
        else:
            assert self.merge_results
            object_ids = None
        pred_scores_all, pred_labels_all = self.net(train_imgs=data['train_images'],
                                                    test_imgs=data['test_images'],
                                                    train_masks=data['train_masks'],
                                                    test_masks=data['test_masks'],
                                                    valid_object=data['valid_object'],
                                                    num_refinement_iter=self.num_refinement_iter,
                                                    object_ids=object_ids,
                                                    merge_results=self.merge_results)



        if self.single_obj_mode:
            assert pred_scores_all.shape[-3] == 1
            gt_segm = data['test_masks']
            gt_segm = gt_segm[:, :, object_ids[0], :, :].contiguous()

            gt_segm = gt_segm.view(-1, 1, *gt_segm.shape[-2:])
            pred_scores_all = pred_scores_all.view(-1, 1, *pred_scores_all.shape[-2:])

            loss_segm = self.loss_weight['segm'] * self.objective['segm'](pred_scores_all, gt_segm)

            # TODO handle invalid objects in multi mode
            acc_l = [davis_jaccard_measure(rm.detach().cpu().numpy(), lb.cpu().numpy()) for
                     rm, lb in
                     zip(pred_labels_all.view(-1, *pred_labels_all.shape[-2:]), gt_segm.view(-1, *gt_segm.shape[-2:]))]
            acc = sum(acc_l)
            cnt = len(acc_l)

            loss = loss_segm
        else:
            gt_segm = data['test_masks_label']
            gt_segm_oh = data['test_masks']

            num_frames, num_seq = gt_segm.shape[0], gt_segm.shape[1]

            gt_segm = gt_segm.view(-1, *gt_segm.shape[-2:])
            pred_scores_all = pred_scores_all.view(-1, *pred_scores_all.shape[-3:])

            loss_segm = self.loss_weight['segm'] * self.objective['segm'](pred_scores_all, gt_segm)

            pred_label = pred_scores_all.argmax(dim=1).view(num_frames, num_seq, *pred_scores_all.shape[-2:])
            gt_segm_oh = gt_segm_oh.view(num_frames, num_seq, *gt_segm_oh.shape[-3:])

            valid_object = data['valid_object'].view(num_seq, -1)

            # TODO speed this up?
            acc = 0
            cnt = 0
            for i in range(num_frames):
                for j in range(num_seq):
                    for k in range(valid_object.shape[1]):
                        if valid_object[j, k]:
                            acc_l = davis_jaccard_measure_torch(pred_label[i, j, :, :] == (k + 1), gt_segm_oh[i, j, k, :, :].bool())
                            acc += acc_l
                            cnt += 1

            loss = loss_segm

        if torch.isinf(loss) or torch.isnan(loss):
            raise Exception('ERROR: Loss was nan or inf!!!')

        # Log stats
        stats = {'Loss/total': loss.item()}
        stats['Loss/segm'] = loss_segm.item()

        stats['Stats/acc'] = acc / (cnt + 1e-7)
        return loss, stats


class LWTLActor(BaseActor):
    """Actor for training the LWTL network."""
    def __init__(self, net, objective, loss_weight=None,
                 num_refinement_iter=3,
                 disable_backbone_bn=False,
                 disable_all_bn=False):
        """
        args:
            net - The network model to train
            objective - Loss functions
            loss_weight - Weights for each training loss
            num_refinement_iter - Number of update iterations N^{train}_{update} used to update the target model in
                                  each frame
            disable_backbone_bn - If True, all batch norm layers in the backbone feature extractor are disabled, i.e.
                                  set to eval mode.
            disable_all_bn - If True, all the batch norm layers in network are disabled, i.e. set to eval mode.
        """
        super().__init__(net, objective)
        if loss_weight is None:
            loss_weight = {'segm': 1.0}
        self.loss_weight = loss_weight

        self.num_refinement_iter = num_refinement_iter
        self.disable_backbone_bn = disable_backbone_bn
        self.disable_all_bn = disable_all_bn

    def train(self, mode=True):
        """ Set whether the network is in train mode.
        args:
            mode (True) - Bool specifying whether in training mode.
        """
        self.net.train(mode)

        if self.disable_all_bn:
            self.net.eval()
        elif self.disable_backbone_bn:
            for m in self.net.feature_extractor.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()

    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'train_images', 'test_images', 'train_masks',
                    'test_masks'
        returns:
            loss    - the training loss
            stats  -  dict containing detailed losses
        """
        segm_pred = self.net(train_imgs=data['train_images'],
                             test_imgs=data['test_images'],
                             train_masks=data['train_masks'],
                             test_masks=data['test_masks'],
                             num_refinement_iter=self.num_refinement_iter)

        acc = 0
        cnt = 0

        segm_pred = segm_pred.view(-1, 1, *segm_pred.shape[-2:])
        gt_segm = data['test_masks']
        gt_segm = gt_segm.view(-1, 1, *gt_segm.shape[-2:])

        loss_segm = self.loss_weight['segm'] * self.objective['segm'](segm_pred, gt_segm)

        acc_l = [davis_jaccard_measure(torch.sigmoid(rm.detach()).cpu().numpy() > 0.5, lb.cpu().numpy()) for
                 rm, lb in zip(segm_pred.view(-1, *segm_pred.shape[-2:]), gt_segm.view(-1, *segm_pred.shape[-2:]))]
        acc += sum(acc_l)
        cnt += len(acc_l)

        loss = loss_segm

        if torch.isinf(loss) or torch.isnan(loss):
            raise Exception('ERROR: Loss was nan or inf!!!')

        # Log stats
        stats = {'Loss/total': loss.item(),
                 'Loss/segm': loss_segm.item(),
                 'Stats/acc': acc / cnt}

        return loss, stats

