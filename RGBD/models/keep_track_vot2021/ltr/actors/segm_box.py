from . import BaseActor
import torch
from pytracking import TensorList
from pytracking.analysis.vos_utils import davis_jaccard_measure


class SegmActorBox(BaseActor):
    """Actor for training the DiMP network."""
    def __init__(self, net, objective, loss_weight=None):
        super().__init__(net, objective)
        if loss_weight is None:
            loss_weight = {'segm': 1.0}
        self.loss_weight = loss_weight

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
        bb_train = data['train_anno_in_sa']

        num_sequences = train_imgs.shape[1]
        num_train_frames = train_imgs.shape[0]

        # Extract backbone features
        train_feat = self.net.extract_backbone_features(
            train_imgs.view(-1, train_imgs.shape[-3], train_imgs.shape[-2], train_imgs.shape[-1]))

        # Extract classification features
        train_feat_clf = self.net.extract_target_model_features(train_feat)  # seq*frames, channels, height, width
        bb_train = bb_train.view(-1, *bb_train.shape[-1:])
        train_box_enc = self.net.box_label_encoder(bb_train, train_feat_clf, train_imgs.shape)
        train_box_enc = train_box_enc.view(num_train_frames, num_sequences, *train_box_enc.shape[-3:])

        mask_pred_box_train, decoder_feat_train = self.net.decoder(train_box_enc, train_feat, train_imgs.shape[-2:])

        loss_segm_box = self.loss_weight['segm_box'] * self.objective['segm'](mask_pred_box_train, data['train_masks'].view(mask_pred_box_train.shape))
        loss_segm_box = loss_segm_box / num_train_frames
        stats = {}

        loss = loss_segm_box

        acc_box = 0
        cnt_box = 0
        acc_lbox = [davis_jaccard_measure(torch.sigmoid(rm.detach()).cpu().numpy() > 0.5, lb.cpu().numpy()) for
                 rm, lb in zip(mask_pred_box_train.view(-1, *mask_pred_box_train.shape[-2:]), data['train_masks'].view(-1, *mask_pred_box_train.shape[-2:]))]
        acc_box += sum(acc_lbox)
        cnt_box += len(acc_lbox)

        stats['Loss/total'] = loss.item()
        stats['Stats/acc_box_train'] = acc_box/cnt_box

        return loss, stats


class LWTLBoxActor(BaseActor):
    """Actor for training bounding box encoder """
    def __init__(self, net, objective, loss_weight=None):
        super().__init__(net, objective)
        if loss_weight is None:
            loss_weight = {'segm': 1.0}
        self.loss_weight = loss_weight

    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'train_images', 'train_anno', and 'train_masks'
        returns:
            loss    - the training loss
            stats  -  dict containing detailed losses
        """
        train_imgs = data['train_images']
        bb_train = data['train_anno']

        num_sequences = train_imgs.shape[1]
        num_train_frames = train_imgs.shape[0]

        # Extract backbone features
        train_feat = self.net.extract_backbone_features(
            train_imgs.view(-1, train_imgs.shape[-3], train_imgs.shape[-2], train_imgs.shape[-1]))

        # Extract classification features
        train_feat_clf = self.net.extract_target_model_features(train_feat)  # seq*frames, channels, height, width
        bb_train = bb_train.view(-1, *bb_train.shape[-1:])
        train_box_enc = self.net.box_label_encoder(bb_train, train_feat_clf, train_imgs.shape)
        train_box_enc = train_box_enc.view(num_train_frames, num_sequences, *train_box_enc.shape[-3:])

        mask_pred_box_train, decoder_feat_train = self.net.decoder(train_box_enc, train_feat, train_imgs.shape[-2:])

        loss_segm_box = self.loss_weight['segm_box'] * self.objective['segm'](mask_pred_box_train, data['train_masks'].view(mask_pred_box_train.shape))
        loss_segm_box = loss_segm_box / num_train_frames
        stats = {}

        loss = loss_segm_box

        acc_box = 0
        cnt_box = 0
        acc_lbox = [davis_jaccard_measure(torch.sigmoid(rm.detach()).cpu().numpy() > 0.5, lb.cpu().numpy()) for
                 rm, lb in zip(mask_pred_box_train.view(-1, *mask_pred_box_train.shape[-2:]), data['train_masks'].view(-1, *mask_pred_box_train.shape[-2:]))]
        acc_box += sum(acc_lbox)
        cnt_box += len(acc_lbox)

        stats['Loss/total'] = loss.item()
        stats['Stats/acc_box_train'] = acc_box/cnt_box

        return loss, stats


class SegmSeqActorBox(BaseActor):
    """Actor for training the DiMP network."""
    def __init__(self, net, objective, loss_weight=None,
                 use_focal_loss=False, use_lovasz_loss=False,
                 detach_pred=True,
                 num_refinement_iter=3,
                 train_box_enc=True,
                 visdom=None,
                 update=True):
        super().__init__(net, objective)
        if loss_weight is None:
            loss_weight = {'segm': 1.0}
        self.loss_weight = loss_weight

        self.use_focal_loss = use_focal_loss
        self.use_lovasz_loss = use_lovasz_loss
        self.detach_pred = detach_pred
        self.num_refinement_iter = num_refinement_iter
        self.train_only_box_enc = train_box_enc
        self.visdom = visdom
        self.update = update

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
        bb_train = data['train_anno_in_sa']

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
        bb_train = bb_train.view(-1, *bb_train.shape[-1:])
        train_box_enc = self.net.box_label_encoder(bb_train, train_feat_clf, train_imgs.shape)
        train_box_enc = train_box_enc.view(num_train_frames, num_sequences, *train_box_enc.shape[-3:])
        train_feat_clf = train_feat_clf.view(num_train_frames, num_sequences, *train_feat_clf.shape[-3:])

        train_feat_clf_all = [train_feat_clf, ]
        mask_pred_box_train, decoder_feat_train = self.net.decoder(train_box_enc, train_feat, train_imgs.shape[-2:],
                                                   ('layer4_dec', 'layer3_dec', 'layer2_dec', 'layer1_dec'))

        loss_segm_box = self.loss_weight['segm_box'] * self.objective['segm'](mask_pred_box_train, data['train_masks'].view(mask_pred_box_train.shape))
        loss_segm_box = loss_segm_box / num_train_frames
        stats = {}
        if not self.train_only_box_enc:
            train_mask_enc_info = self.net.label_encoder(train_masks, train_feat_clf)

            if isinstance(train_mask_enc_info, (tuple, list)):
                train_mask_enc = train_mask_enc_info[0]
                train_mask_sw = train_mask_enc_info[1]
            else:
                train_mask_enc = train_mask_enc_info
                train_mask_sw = None

            mask_pred_train, decoder_feat_train = self.net.decoder(train_mask_enc, train_feat, test_imgs.shape[-2:],
                                                                       ('layer4_dec', 'layer3_dec', 'layer2_dec',
                                                                        'layer1_dec'))

            loss_segm_train = self.loss_weight['segm_train'] * self.objective['segm'](mask_pred_train,
                                                                                  data['train_masks'].view(
                                                                                      mask_pred_train.shape))
            loss_segm_train = loss_segm_train / num_train_frames
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
                   raise NotImplementedError

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
                if i < (num_test_frames - 1) and self.update:
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
                   loss_segm_aux_all_sum + loss_bbreg_all_w + loss_segm_box + loss_segm_train

            if torch.isinf(loss) or torch.isnan(loss):
                raise Exception('ERROR: Loss was nan or inf!!!')

            # Log stats
            stats['Loss/segm'] = loss_segm_all.item()
            if 'aux_segm' in self.loss_weight.keys():
                stats['Loss/segm_aux'] = loss_segm_aux_all_sum.item()

                for k in self.loss_weight['aux_segm'].keys():
                    stats['Loss/aux/{}'.format(k)] = loss_segm_aux_all[k]

            if 'bbreg' in self.loss_weight.keys():
                stats['Loss/bbreg'] = loss_bbreg_all_w.item()
                stats['Loss/aux/bbreg'] = loss_bbreg_all.item()
            stats['Stats/acc'] = acc / cnt

        else:
            loss = loss_segm_box

        acc_box = 0
        cnt_box = 0
        acc_lbox = [davis_jaccard_measure(torch.sigmoid(rm.detach()).cpu().numpy() > 0.5, lb.cpu().numpy()) for
                 rm, lb in zip(mask_pred_box_train.view(-1, *mask_pred_box_train.shape[-2:]), data['train_masks'].view(-1, *mask_pred_box_train.shape[-2:]))]
        acc_box += sum(acc_lbox)
        cnt_box += len(acc_lbox)


        stats['Loss/total'] = loss.item()
        stats['Stats/acc_box_train'] = acc_box/cnt_box

        batch_ind = 0
        if not self.visdom is None: #(self, data, mode, debug_level=0, title='Data', **kwargs)
            self.visdom.register(data=torch.sigmoid(mask_pred_box_train[batch_ind, :, :, :].detach()), mode='image', debug_level=0,
                                 title='boxsegtrain')
            #            self.visdom.register(sample_weights[:,0,0].detach(), 'lineplot', 2, 'sample_weights')
            self.visdom.register(data=data['train_masks'][0][batch_ind], mode='image', debug_level=0, title='trainmask')
            self.visdom.register(data=(data['train_images'][0][batch_ind] - data['train_images'][0][batch_ind].min()) / (
                    data['train_images'][0][batch_ind].max() - data['train_images'][0][batch_ind].min()), mode='image',
                                 debug_level=0, title='trainimage')
                #self.visdom.register(data=label_encoding_box[batch_ind].detach(), mode='image', debug_level=0, title='labelencodingbox')

        return loss, stats
