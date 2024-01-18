from . import BaseActor
import torch
import math
from pytracking.libs.dcf import max2d
from ltr.data.processing_utils import gauss_2d
from pytracking import TensorList
import torch.nn.functional as F
from ltr.data.processing_utils import iou_gen
import torch.nn as nn



class DiMPActor(BaseActor):
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
        target_scores, bb_scores = self.net(train_imgs=data['train_images'],
                                            test_imgs=data['test_images'],
                                            train_bb=data['train_anno'],
                                            test_proposals=data['test_proposals'],
                                            train_label=data['train_label'])

        # Reshape bb reg variables
        is_valid = data['test_anno'][:, :, 0] < 99999.0
        bb_scores = bb_scores[is_valid, :]
        proposal_density = data['proposal_density'][is_valid, :]
        gt_density = data['gt_density'][is_valid, :]

        # Compute loss
        bb_ce = self.objective['bb_ce'](bb_scores, sample_density=proposal_density, gt_density=gt_density, mc_dim=1)
        loss_bb_ce = self.loss_weight['bb_ce'] * bb_ce

        loss_test_init_clf = 0
        loss_test_iter_clf = 0

        # Classification losses for the different optimization iterations
        clf_losses_test = [self.objective['test_clf'](s, data['test_label'], data['test_anno']) for s in target_scores]

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

        return loss, stats


class ContinuousDiMPActor(BaseActor):
    """Actor for training the DiMP network in a continuous fashio over multiple frames in a sequence."""

    def __init__(self, net, objective, loss_weight=None, params=None):
        super().__init__(net, objective)
        if loss_weight is None:
            loss_weight = {'bb_ce': 1.0}
        self.loss_weight = loss_weight
        self.params = params

    def freeze_bn_modules(self, bn_modules_to_be_frozen):
        for module in bn_modules_to_be_frozen:
            if isinstance(module, nn.BatchNorm2d):
                module.eval()

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
        # Write forward path here explicitly without calling forward on the corresponding modules.
        target_scores, bb_scores = self.run_manual_forward(train_imgs=data['train_images'],
                                                           test_imgs=data['test_images'],
                                                           train_bb=data['train_anno'],
                                                           test_proposals=data['test_proposals'],
                                                           train_label=data['train_label'])


        # Reshape bb reg variables
        is_valid = data['test_anno'][:, :, 0] < 99999.0
        bb_scores = bb_scores[is_valid, :]
        proposal_density = data['proposal_density'][is_valid, :]
        gt_density = data['gt_density'][is_valid, :]

        # Compute loss
        bb_ce = self.objective['bb_ce'](bb_scores, sample_density=proposal_density, gt_density=gt_density, mc_dim=1)
        loss_bb_ce = self.loss_weight['bb_ce'] * bb_ce


        # Loss of the final filter
        clf_loss_test = self.objective['test_clf'](target_scores, data['test_label'], data['test_anno'])
        loss_target_classifier = self.loss_weight['test_clf'] * clf_loss_test

        # Tracking accuracy for logging

        test_label = data['test_label']
        test_clf_acc, test_pred_correct = self.objective['clf_acc'](target_scores, test_label, valid_samples=is_valid)

        test_clf_accs = []
        for i in range(0, target_scores.shape[0]):
            test_clf_accs.append(self.objective['clf_acc'](target_scores[i], test_label[i],
                                                           valid_samples=is_valid[i])[0])

        # Total loss
        loss = loss_bb_ce + loss_target_classifier

        if torch.isinf(loss) or torch.isnan(loss):
            raise Exception('ERROR: Loss was nan or inf!!!')

        # Log stats
        stats = {'Loss/total': loss.item(),
                 'Loss/bb_ce': bb_ce.item(),
                 'Loss/loss_bb_ce': loss_bb_ce.item(),
                 'Acc_all': test_clf_acc.item()}

        for i in range(0, len(test_clf_accs)):
            stats['Acc_{}'.format(i)] = test_clf_accs[i]

        if 'test_clf' in self.loss_weight.keys():
            stats['Loss/target_clf'] = loss_target_classifier.item()

        if 'test_clf' in self.loss_weight.keys():
            stats['ClfTrain/test_loss'] = clf_loss_test.item()

        return loss, stats

    def run_manual_forward(self, train_imgs, test_imgs, train_bb, test_proposals, *args, **kwargs):
        assert train_imgs.dim() == 5 and test_imgs.dim() == 5, 'Expect 5 dimensional inputs'

        # Extract backbone features
        train_feat = self.net.extract_backbone_features(train_imgs.view(-1, *train_imgs.shape[-3:]))
        test_feat = self.net.extract_backbone_features(test_imgs.view(-1, *test_imgs.shape[-3:]))

        # Classification features
        train_feat_clf = self.net.get_backbone_clf_feat(train_feat)
        test_feat_clf = self.net.get_backbone_clf_feat(test_feat)

        # Get bb_regressor features
        train_feat_iou = self.net.get_backbone_bbreg_feat(train_feat)
        test_feat_iou = self.net.get_backbone_bbreg_feat(test_feat)

        # Run the IoUNet module
        iou_pred = self.net.bb_regressor(train_feat_iou, test_feat_iou, train_bb, test_proposals)

        target_scores = self.run_manual_classifier(train_feat=train_feat_clf,
                                                   test_feat=test_feat_clf,
                                                   train_bb=train_bb,
                                                   iou_pred=iou_pred,
                                                   test_proposals=test_proposals, *args, **kwargs)

        return target_scores, iou_pred

    def run_manual_classifier(self, train_feat, test_feat, train_bb, train_label, iou_pred, test_proposals, **kwargs):
        # Setup training of the classifier

        # iou_pred (ntest, nbatch, nproposals)
        # test_proposals (ntest, nbatch, nproposals, 4) last dim is (x,y,w,h) in image img_coords

        num_sequences = train_bb.shape[1]

        if train_feat.dim() == 5:
            train_feat = train_feat.view(-1, *train_feat.shape[-3:])
        if test_feat.dim() == 5:
            test_feat = test_feat.view(-1, *test_feat.shape[-3:])

        # Extract features (independent of memory)
        train_feat_clf = self.net.classifier.extract_classification_feat(train_feat, num_sequences)
        test_feat_clf = self.net.classifier.extract_classification_feat(test_feat, num_sequences)

        # Optimize the filter for the memory (train frames)
        filter, _, _ = self.net.classifier.get_filter(train_feat_clf, train_bb,
                                                      train_label=train_label, **kwargs)

        test_scores_iter = []

        mem_feats = train_feat_clf
        mem_labels = train_label

        num_test_frames = test_feat_clf.shape[0]
        for i in range(0, num_test_frames):
            test_feat_next_frame = torch.unsqueeze(test_feat_clf[i], dim=0)

            # prelimilary score for test frame
            test_score = self.net.classifier(filter, test_feat_next_frame, mem_feats, mem_labels)

            # # optimize filter based on predicted weights for the current test frame
            filter, _, _ = self.net.classifier.filter_optimizer(filter, num_iter=1, feat=mem_feats,
                                                                bb=None, train_label=mem_labels)

            # Update memories with test features and predicted scores instead of gth.
            mem_feats, mem_labels = self.update_memory(mem_feats, mem_labels, test_feat_next_frame, test_score)

            # build output
            test_scores_iter.append(test_score)

        return torch.cat(test_scores_iter, dim=0)

    def update_memory(self, mem_feats, mem_labels, test_feat_next_frame, test_score):
        # label memory
        _, centers = max2d(test_score)
        sz = torch.Tensor([test_score.shape[-2], test_score.shape[-1]])
        centers[:, :, 0] = centers[:, :, 0] - (sz[0].item() - sz[0].item() % 2) / 2
        centers[:, :, 1] = centers[:, :, 1] - (sz[1].item() - sz[1].item() % 2) / 2
        pred_labels = gauss_2d(sz, sigma=self.params['sigma'], center=centers.view(-1, 2).to(sz.device)).unsqueeze(0)
        pred_labels = pred_labels.permute(0,1,3,2).to(centers.device)

        mem_labels = torch.cat([mem_labels, pred_labels], dim=0)
        mem_feats = torch.cat([mem_feats, test_feat_next_frame], dim=0)

        return mem_feats, mem_labels

    # def predict_bbox(self, test_score, iou_pred, test_proposals):
    #     test_bb = test_proposals[torch.arange(0, test_proposals.shape[0]), torch.argmax(iou_pred, dim=1)]
    #     return test_bb


# class ContinuousDiMPActor(BaseActor):
#     """Actor for training the DiMP network in a continuous fashio over multiple frames in a sequence."""
#
#     def __init__(self, net, objective, loss_weight=None, params=None):
#         super().__init__(net, objective)
#         if loss_weight is None:
#             loss_weight = {'bb_ce': 1.0}
#         self.loss_weight = loss_weight
#         self.params = params
#
#     def __call__(self, data):
#         """
#         args:
#             data - The input data, should contain the fields 'train_images', 'test_images', 'train_anno',
#                     'test_proposals', 'proposal_iou' and 'test_label'.
#
#         returns:
#             loss    - the training loss
#             stats  -  dict containing detailed losses
#         """
#         # Run network
#         # Write forward path here explicitly without calling forward on the corresponding modules.
#         target_scores, bb_scores = self.run_manual_forward(train_imgs=data['train_images'],
#                                                            test_imgs=data['test_images'],
#                                                            train_bb=data['train_anno'],
#                                                            test_proposals=data['test_proposals'],
#                                                            train_label=data['train_label'])
#
#
#         # Reshape bb reg variables
#         is_valid = data['test_anno'][:, :, 0] < 99999.0
#         bb_scores = bb_scores[is_valid, :]
#         proposal_density = data['proposal_density'][is_valid, :]
#         gt_density = data['gt_density'][is_valid, :]
#
#         # Compute loss
#         bb_ce = self.objective['bb_ce'](bb_scores, sample_density=proposal_density, gt_density=gt_density, mc_dim=1)
#         loss_bb_ce = self.loss_weight['bb_ce'] * bb_ce
#
#
#         # Loss of the final filter
#         clf_loss_test = self.objective['test_clf'](target_scores, data['test_label'], data['test_anno'])
#         loss_target_classifier = self.loss_weight['test_clf'] * clf_loss_test
#
#         # Tracking accuracy for logging
#
#         test_label = data['test_label']
#         test_clf_acc, test_pred_correct = self.objective['clf_acc'](target_scores, test_label, valid_samples=is_valid)
#
#         test_clf_accs = []
#         for i in range(0, target_scores.shape[0]):
#             test_clf_accs.append(self.objective['clf_acc'](target_scores[i], test_label[i],
#                                                            valid_samples=is_valid[i])[0])
#
#         # Total loss
#         loss = loss_bb_ce + loss_target_classifier
#
#         if torch.isinf(loss) or torch.isnan(loss):
#             raise Exception('ERROR: Loss was nan or inf!!!')
#
#         # Log stats
#         stats = {'Loss/total': loss.item(),
#                  'Loss/bb_ce': bb_ce.item(),
#                  'Loss/loss_bb_ce': loss_bb_ce.item(),
#                  'Acc_all': test_clf_acc.item()}
#
#         for i in range(0, len(test_clf_accs)):
#             stats['Acc_{}'.format(i)] = test_clf_accs[i]
#
#         if 'test_clf' in self.loss_weight.keys():
#             stats['Loss/target_clf'] = loss_target_classifier.item()
#
#         if 'test_clf' in self.loss_weight.keys():
#             stats['ClfTrain/test_loss'] = clf_loss_test.item()
#
#         return loss, stats
#
#     def run_manual_forward(self, train_imgs, test_imgs, train_bb, test_proposals, *args, **kwargs):
#         assert train_imgs.dim() == 5 and test_imgs.dim() == 5, 'Expect 5 dimensional inputs'
#
#         # Extract backbone features
#         train_feat = self.net.extract_backbone_features(train_imgs.view(-1, *train_imgs.shape[-3:]))
#         test_feat = self.net.extract_backbone_features(test_imgs.view(-1, *test_imgs.shape[-3:]))
#
#         # Classification features
#         train_feat_clf = self.net.get_backbone_clf_feat(train_feat)
#         test_feat_clf = self.net.get_backbone_clf_feat(test_feat)
#
#         # Get bb_regressor features
#         train_feat_iou = self.net.get_backbone_bbreg_feat(train_feat)
#         test_feat_iou = self.net.get_backbone_bbreg_feat(test_feat)
#
#         # Run the IoUNet module
#         iou_pred = self.net.bb_regressor(train_feat_iou, test_feat_iou, train_bb, test_proposals)
#
#         target_scores = self.run_manual_classifier(train_feat=train_feat_clf,
#                                                    test_feat=test_feat_clf,
#                                                    train_bb=train_bb,
#                                                    iou_pred=iou_pred,
#                                                    test_proposals=test_proposals, *args, **kwargs)
#
#         return target_scores, iou_pred
#
#     def run_manual_classifier(self, train_feat, test_feat, train_bb, train_label, iou_pred, test_proposals, **kwargs):
#         # Setup training of the classifier
#
#         # iou_pred (ntest, nbatch, nproposals)
#         # test_proposals (ntest, nbatch, nproposals, 4) last dim is (x,y,w,h) in image img_coords
#
#         num_sequences = train_bb.shape[1]
#
#         if train_feat.dim() == 5:
#             train_feat = train_feat.view(-1, *train_feat.shape[-3:])
#         if test_feat.dim() == 5:
#             test_feat = test_feat.view(-1, *test_feat.shape[-3:])
#
#         # Extract features (independent of memory)
#         train_feat_clf = self.net.classifier.extract_classification_feat(train_feat, num_sequences)
#         test_feat_clf = self.net.classifier.extract_classification_feat(test_feat, num_sequences)
#
#         # Optimize the filter for the memory (train frames)
#         filter, _, _ = self.net.classifier.get_filter(train_feat_clf, train_bb,
#                                                       train_label=train_label, **kwargs)
#
#         test_scores_iter = []
#
#         mem_feats = train_feat_clf
#         mem_labels = train_label
#         mem_bboxes = train_bb
#         mem_certainties = 10.*torch.ones((train_bb.shape[0], train_bb.shape[1], 1, 1), device=mem_bboxes.device)
#
#         num_test_frames = test_feat_clf.shape[0]
#         for i in range(0, num_test_frames):
#             test_feat_next_frame = torch.unsqueeze(test_feat_clf[i], dim=0)
#
#             # prelimilary score for test frame
#             test_score_pre = self.net.classifier.classify(filter, test_feat_next_frame)
#
#             # predict weights based on preliminary score and test frame
#             predicted_weights = self.net.classifier.weight_predictor(test_feat_next_frame, mem_feats,
#                                                                      test_score_pre, mem_labels, mem_bboxes,
#                                                                      mem_certainties, **kwargs)
#
#             # optimize filter based on predicted weights for the current test frame
#             filter, _, _ = self.net.classifier.filter_optimizer(filter, num_iter=1, feat=mem_feats,
#                                                                 bb=None, train_label=mem_labels,
#                                                                 sample_weight=predicted_weights)
#
#             # obtain final enhanced test score
#             test_score = self.net.classifier.classify(filter, test_feat_next_frame)
#
#             # predict final bbox
#             test_bb = self.predict_bbox(test_score, iou_pred[i], test_proposals[i])
#
#             # Update memories with test features and predicted scores instead of gth.
#             mem_bboxes, mem_feats, mem_labels, mem_certainties = self.update_memory(mem_bboxes, mem_feats, mem_labels,
#                                                                                     mem_certainties, test_bb,
#                                                                                     test_feat_next_frame, test_score)
#
#             # self.objective['clf_acc'](test_score, test_label_cur, valid_samples=is_valid)
#
#             # build output
#             test_scores_iter.append(test_score)
#
#         return torch.cat(test_scores_iter, dim=0)
#
#     def update_memory(self, mem_bboxes, mem_feats, mem_labels, mem_certainties, test_bb, test_feat_next_frame, test_score):
#         # label memory
#         _, centers = max2d(test_score)
#         sz = torch.Tensor([test_score.shape[-2], test_score.shape[-1]])
#         centers[:, :, 0] = centers[:, :, 0] - (sz[0].item() - sz[0].item() % 2) / 2
#         centers[:, :, 1] = centers[:, :, 1] - (sz[1].item() - sz[1].item() % 2) / 2
#         pred_labels = gauss_2d(sz, sigma=self.params['sigma'], center=centers.view(-1, 2).to(sz.device)).unsqueeze(0)
#         pred_labels = pred_labels.permute(0,1,3,2).to(centers.device)
#
#         certainties = torch.max(test_score.view(test_score.shape[0], test_score.shape[1], -1), dim=2)[0]
#         certainties = certainties.reshape(test_score.shape[0], test_score.shape[1], 1, 1)
#
#         mem_certainties = torch.cat([mem_certainties, certainties], dim=0)
#         mem_labels = torch.cat([mem_labels, pred_labels], dim=0)
#         mem_bboxes = torch.cat([mem_bboxes, test_bb.unsqueeze(0)], dim=0)
#         mem_feats = torch.cat([mem_feats, test_feat_next_frame], dim=0)
#
#         return mem_bboxes, mem_feats, mem_labels, mem_certainties
#
#     def predict_bbox(self, test_score, iou_pred, test_proposals):
#         test_bb = test_proposals[torch.arange(0, test_proposals.shape[0]), torch.argmax(iou_pred, dim=1)]
#         return test_bb

# def print_tensor(z):
#     z = z.detach().cpu().numpy()
#
#     assert len(z.shape) == 2
#
#     for iy in range(0, z.shape[0]):
#         msg = []
#         for ix in range(0, z.shape[1]):
#             if z[iy,ix] < 0.05:
#                 msg.append('_._')
#             else:
#                 msg.append('{:0.1f}'.format(z[iy, ix]))
#         print(' '.join(msg))
#
#     print('')