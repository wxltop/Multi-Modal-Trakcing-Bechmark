import math
import torch
import torch.nn as nn
from collections import OrderedDict
import ltr.models.segmentation.linear_filter as target_clf
import ltr.models.target_classifier.features as clf_features
import ltr.models.segmentation.initializer as seg_initializer
import ltr.models.segmentation.label_encoder as seg_label_encoder
import ltr.models.segmentation.loss_residual_modules as loss_residual_modules
import ltr.models.segmentation.stm_decoder as stm_decoder
import ltr.models.segmentation.dolf_decoder as dolf_decoder
import ltr.models.segmentation.features as seg_features
import ltr.models.backbone as backbones
import ltr.models.meta.steepestdescent as steepestdescent
import ltr.models.segmentation.aux_layers as seg_aux_layers
from ltr import model_constructor
from pytracking import TensorList
import ltr.models.backbone.resnet_mrcnn as mrcnn_backbones


class SegMultiDolfTracker(nn.Module):
    def __init__(self, feature_extractor, classifier, decoder, classification_layer, refinement_layers,
                 label_encoder=None, aux_layers=None, bb_regressor=None, bbreg_decoder_layer=None):
        super().__init__()

        self.feature_extractor = feature_extractor
        self.classifier = classifier
        self.decoder = decoder

        # self.output_layers = ['layer1', 'layer2', 'layer3']

        self.classification_layer = (classification_layer,) if isinstance(classification_layer,
                                                                         str) else classification_layer
        self.refinement_layers = refinement_layers
        self.output_layers = sorted(list(set(self.classification_layer + self.refinement_layers)))
        # self.classification_layer = ['layer3']
        self.label_encoder = label_encoder

        if aux_layers is None:
            self.aux_layers = nn.ModuleDict()
        else:
            self.aux_layers = aux_layers

        self.bb_regressor = bb_regressor
        self.bbreg_decoder_layer = bbreg_decoder_layer

    def forward(self, train_imgs, test_imgs, train_masks, test_masks, valid_object=None, object_ids=None,
                num_refinement_iter=None, merge_results=True):
        assert train_imgs.dim() == 5 and test_imgs.dim() == 5, 'Expect 5 dimensional inputs'
        assert train_masks.dim() == 5, 'Expect 5 dimensional masks'

        num_sequences = train_imgs.shape[1]
        num_train_frames = train_imgs.shape[0]
        num_test_frames = test_imgs.shape[0]

        # Extract backbone features
        train_feat = self.extract_backbone_features(
            train_imgs.view(-1, train_imgs.shape[-3], train_imgs.shape[-2], train_imgs.shape[-1]))
        test_feat = self.extract_backbone_features(
            test_imgs.view(-1, test_imgs.shape[-3], test_imgs.shape[-2], test_imgs.shape[-1]))

        # Extract classification features
        train_feat_clf = self.extract_classification_feat(train_feat)  # seq*frames, channels, height, width
        train_feat_clf = train_feat_clf.view(num_train_frames, num_sequences, *train_feat_clf.shape[-3:])

        train_feat_clf_all = [train_feat_clf, ]

        # if object_ids is None:
        #     object_ids = list(range(1, valid_object.shape[2] + 1))
        train_mask_enc_info = self.label_encoder(train_masks, feature=train_feat_clf, object_ids=object_ids)
        valid_object = valid_object.view(num_sequences, -1, 1, 1).float()

        if object_ids is not None:
            num_objects = len(object_ids)
            valid_object = valid_object[:, object_ids, :, :]
        else:
            num_objects = valid_object.shape[1]

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

        test_feat_clf = self.extract_classification_feat(test_feat)  # seq*frames, channels, height, width

        filter, filter_iter, _ = self.classifier.get_filter(train_feat_clf, (train_mask_enc, train_mask_sw),
                                                            num_objects=num_objects)

        pred_scores_all = []
        pred_labels_all = []
        for i in range(num_test_frames):
            test_feat_clf_it = test_feat_clf.view(num_test_frames, num_sequences, *test_feat_clf.shape[-3:])[i:i + 1,
                               ...]
            target_scores = [self.classifier.classify(f, test_feat_clf_it) for f in filter_iter]

            test_feat_it = {k: v.view(num_test_frames, num_sequences, *v.shape[-3:])[i, ...] for k, v in
                            test_feat.items()}
            target_scores_last_iter = target_scores[-1]

            target_scores_last_iter = target_scores_last_iter.view(1, num_sequences, num_objects,
                                                                   -1, *target_scores_last_iter.shape[-2:])

            mask_pred, decoder_feat = self.decoder(target_scores_last_iter, test_feat_it, test_imgs.shape[-2:],
                                                   ('layer4_dec', 'layer3_dec', 'layer2_dec', 'layer1_dec'),
                                                   num_objects=num_objects)

            mask_pred = mask_pred.view(num_sequences, num_objects, *test_imgs.shape[-2:])

            mask_pred_valid = mask_pred * valid_object + -100.0 * (1.0 - valid_object)

            if merge_results:
                mask_pred_merged = self.merge_segmentation_results(mask_pred_valid)
                mask_pred_clone = mask_pred_merged.clone().detach()

                mask_pred_label = mask_pred_clone.softmax(dim=1)[:, 1:, :, :].view(1, num_sequences, num_objects,
                                                                                   *mask_pred.shape[-2:]).contiguous()

            else:
                mask_pred_merged = mask_pred_valid
                mask_pred_clone = mask_pred_merged.clone().detach()

                mask_pred_label = mask_pred_clone.sigmoid().view(1, num_sequences, num_objects, *mask_pred.shape[-2:])

            # We only have predictions for the target. So no need to send object ids
            mask_pred_enc_info = self.label_encoder(mask_pred_label, feature=test_feat_clf_it,
                                                    object_ids=None)
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

            mask_pred_merged = mask_pred_merged.view(-1, *mask_pred_merged.shape[-3:])

            ## Update
            if i < (num_test_frames - 1):
                train_feat_clf_it = torch.cat(train_feat_clf_all, dim=0)
                train_mask_enc_it = torch.cat(train_mask_enc_all, dim=0)

                if train_mask_sw_all is not None:
                    train_mask_sw_it = torch.cat(train_mask_sw_all, dim=0)
                else:
                    train_mask_sw_it = None

                filter_tl, _, _ = self.classifier.filter_optimizer(TensorList([filter]),
                                                                   feat=train_feat_clf_it,
                                                                   mask=train_mask_enc_it,
                                                                   sample_weight=train_mask_sw_it,
                                                                   num_iter=num_refinement_iter)

                filter = filter_tl[0]

            pred_scores_all.append(mask_pred_merged)
            pred_labels_all.append(mask_pred_label.view(num_sequences, *mask_pred_label.shape[-3:]))

        pred_scores_all = torch.stack(pred_scores_all, dim=0)
        pred_labels_all = torch.stack(pred_labels_all, dim=0)
        return pred_scores_all, pred_labels_all


    def merge_segmentation_results(self, scores):
        assert scores.dim() == 4 # (masks, obj, h, w)

        # Soft aggregation from RGMP
        eps = 1e-7

        prob = torch.sigmoid(scores)

        bg_p = torch.prod(1 - prob, dim=1).clamp(eps, 1.0 - eps)  # bg prob
        bg_score = (bg_p / (1.0 - bg_p)).log()

        scores_all = torch.cat((bg_score.unsqueeze(1), scores), dim=1)

        return scores_all

    def segment_target(self, target_filter, test_feat_clf, test_feat, num_objects, serial=False):
        # Classification features
        assert target_filter.dim() == 5     # seq, filters, ch, h, w
        test_feat_clf = test_feat_clf.view(1, 1, *test_feat_clf.shape[-3:])

        target_scores = self.classifier.classify(target_filter, test_feat_clf)

        target_scores = target_scores.view(1, 1, num_objects, -1, *target_scores.shape[-2:])

        if serial:
            mask_out = []
            for i in range(num_objects):
                mask_pred_i, _ = self.decoder(target_scores[:, :, i:i+1, :, :], test_feat,
                                              (test_feat_clf.shape[-2] * 16, test_feat_clf.shape[-1] * 16),
                                              (),
                                              num_objects=1)
                mask_out.append(mask_pred_i.view(*mask_pred_i.shape[-2:]))
            mask_pred = torch.stack(mask_out, dim=0)
        else:
            mask_pred, decoder_feat = self.decoder(target_scores, test_feat,
                                                   (test_feat_clf.shape[-2]*16, test_feat_clf.shape[-1]*16),
                                                   (),
                                                   num_objects=num_objects)

        mask_pred = mask_pred.view(num_objects, *mask_pred.shape[-2:])

        # Output is 1, 1, h, w
        return mask_pred

    def get_backbone_clf_feat(self, backbone_feat):
        feat = OrderedDict({l: backbone_feat[l] for l in self.classification_layer})
        if len(self.classification_layer) == 1:
            return feat[self.classification_layer[0]]
        return feat

    def get_backbone_bbreg_feat(self, backbone_feat):
        return [backbone_feat[l] for l in self.bb_regressor_layer]

    def extract_classification_feat(self, backbone_feat):
        return self.classifier.extract_classification_feat(self.get_backbone_clf_feat(backbone_feat))

    def extract_backbone_features(self, im, layers=None):
        if layers is None:
            layers = self.output_layers
        return self.feature_extractor(im, layers)

    def extract_features(self, im, layers=None):
        if layers is None:
            layers = self.output_layers + ['classification']
        if 'classification' not in layers:
            return self.feature_extractor(im, layers)
        backbone_layers = sorted(list(set([l for l in layers + self.classification_layer if l != 'classification'])))
        all_feat = self.feature_extractor(im, backbone_layers)
        all_feat['classification'] = self.extract_classification_feat(all_feat)
        return OrderedDict({l: all_feat[l] for l in layers})


@model_constructor
def steepest_descent_resnet50(filter_size=1, num_filters=1, optim_iter=3, optim_init_reg=0.01,
                              backbone_pretrained=False, clf_feat_blocks=1,
                              clf_feat_norm=True, final_conv=False,
                              out_feature_dim=512,
                              classification_layer='layer3',
                              refinement_layers = ("layer4", "layer3", "layer2", "layer1",),
                              detach_length=float('Inf'),
                              label_encoder_dims=(1, 1),
                              frozen_backbone_layers=(),
                              label_encoder_type='identity',
                              decoder_mdim=64, filter_groups=1,
                              upsample_residuals=True,
                              ppm_use_res_block=False,
                              aux_mask_loss_layers=(),
                              use_bn_in_label_enc=True,
                              cls_feat_extractor='ppm',
                              decoder_type='rofl',
                              att_inter_dim=-1,
                              dilation_factors=None,
                              use_aux_pred_for_enc=False,
                              backbone_type='imagenet'):
    # backbone
    if backbone_type == 'imagenet':
        backbone_net = backbones.resnet50(pretrained=backbone_pretrained, frozen_layers=frozen_backbone_layers)
    elif backbone_type == 'mrcnn':
        backbone_net = mrcnn_backbones.resnet50(pretrained=False, frozen_layers=frozen_backbone_layers)
    else:
        raise Exception

    norm_scale = math.sqrt(1.0 / (out_feature_dim * filter_size * filter_size))

    # classifier
    layer_channels = backbone_net.out_feature_channels()

    if cls_feat_extractor=='res_block':
        clf_feature_extractor = clf_features.residual_basic_block(feature_dim=layer_channels[classification_layer],
                                                                  num_blocks=clf_feat_blocks, l2norm=clf_feat_norm,
                                                                  final_conv=final_conv, norm_scale=norm_scale,
                                                                  out_dim=out_feature_dim)
    elif cls_feat_extractor == 'ppm':
        clf_feature_extractor = seg_features.SegBlockPPM_keepsize(feature_dim=layer_channels[classification_layer],
                                                                  l2norm=clf_feat_norm,
                                                                  norm_scale=norm_scale,
                                                                  out_dim=out_feature_dim,
                                                                  use_res_block=ppm_use_res_block)
    else:
        raise Exception

    initializer = seg_initializer.FilterInitializerZero(filter_size=filter_size, num_filters=num_filters,
                                                        feature_dim=out_feature_dim, filter_groups=filter_groups)

    if label_encoder_type == 'res_ds16_sw':
        label_encoder = seg_label_encoder.ResidualDS16SWMulti(layer_dims=label_encoder_dims + (num_filters, ), use_bn=use_bn_in_label_enc)
    else:
        raise Exception

    residual_module = loss_residual_modules.LinearFilterSeg(init_filter_reg=optim_init_reg, upsample_residuals=upsample_residuals,
                                                  filter_dilation_factors=dilation_factors)

    optimizer = steepestdescent.GNSteepestDescent(residual_module=residual_module, num_iter=optim_iter, detach_length=detach_length,
                                                  residual_batch_dim=1, compute_losses=True)

    if dilation_factors is not None:
        assert num_filters == sum(dilation_factors.values())

    classifier = target_clf.LinearFilter(filter_size=filter_size, filter_initializer=initializer,
                                         filter_optimizer=optimizer, feature_extractor=clf_feature_extractor,
                                         filter_dilation_factors=dilation_factors)

    refinement_layers_channels = {L: layer_channels[L] for L in refinement_layers}

    if decoder_type == 'rofl':
        decoder = dolf_decoder.RefelixNetwork2(num_filters, decoder_mdim, refinement_layers_channels,
                                               new_upsampler=True, use_bn=True)
    elif decoder_type == 'rofl_att':
        decoder = dolf_decoder.RefelixNetwork2Att(num_filters, decoder_mdim, refinement_layers_channels,
                                                  new_upsampler=True, use_bn=True,
                                                  att_dim=att_inter_dim)
    elif decoder_type == 'stm_maskall':
        decoder = stm_decoder.DecoderResnet50Mask(filter_out_dim=num_filters, mdim=decoder_mdim)
    else:
        raise Exception



    if len(aux_mask_loss_layers) > 0:
        aux_layers = nn.ModuleDict()
        decoder_channels = decoder.out_feature_channels()
        decoder_channels['mask_enc'] = num_filters
        for l in aux_mask_loss_layers:
            aux_layers[l] = seg_aux_layers.ConvPredictor(decoder_channels[l])
    else:
        aux_layers = None

    if use_aux_pred_for_enc:
        aux_layers = nn.ModuleDict()
        aux_layers['mask_enc_iter'] = seg_aux_layers.DeConvPredictor4x(num_filters)

    net = SegMultiDolfTracker(feature_extractor=backbone_net, classifier=classifier, decoder=decoder,
                              label_encoder=label_encoder,
                              classification_layer=classification_layer, refinement_layers=refinement_layers,
                              aux_layers=aux_layers)
    return net
