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
import ltr.models.backbone as backbones
import ltr.models.meta.steepestdescent as steepestdescent
from ltr import model_constructor
import ltr.models.segmentation.features as seg_features


class SegTracker(nn.Module):
    def __init__(self, feature_extractor, classifier, decoder, label_encoder=None,
                 aux_layers=None, bb_regressor=None, bbreg_decoder_layer=None):
        super().__init__()

        self.feature_extractor = feature_extractor
        self.classifier = classifier
        self.decoder = decoder

        self.output_layers = ['layer1', 'layer2', 'layer3']
        self.classification_layer = ['layer3']
        self.label_encoder = label_encoder

        if aux_layers is None:
            self.aux_layers = nn.ModuleDict()
        else:
            self.aux_layers = aux_layers

        self.bb_regressor = bb_regressor
        self.bbreg_decoder_layer = bbreg_decoder_layer

    def forward(self, train_imgs, test_imgs, train_masks, test_masks=None):
        assert train_imgs.dim() == 5 and test_imgs.dim() == 5, 'Expect 5 dimensional inputs'
        assert train_masks.dim() == 4, 'Expect 4 dimensional masks'

        num_sequences = train_imgs.shape[1]
        num_train_frames = train_imgs.shape[0]
        num_test_frames = test_imgs.shape[0]

        # Extract backbone features
        train_feat = self.extract_backbone_features(train_imgs.view(-1, train_imgs.shape[-3], train_imgs.shape[-2], train_imgs.shape[-1]))
        test_feat = self.extract_backbone_features(test_imgs.view(-1, test_imgs.shape[-3], test_imgs.shape[-2], test_imgs.shape[-1]))

        # Extract classification features
        train_feat_clf = self.extract_classification_feat(train_feat)   # seq*frames, channels, height, width
        test_feat_clf = self.extract_classification_feat(test_feat)     # seq*frames, channels, height, width

        if self.label_encoder is not None:
            mask_enc = self.label_encoder(train_masks, train_feat_clf)
        else:
            mask_enc = train_masks

        train_feat_clf = train_feat_clf.view(num_train_frames, num_sequences, *train_feat_clf.shape[-3:])
        filter, filter_iter, _ = self.classifier.get_filter(train_feat_clf, mask_enc)

        test_feat_clf = test_feat_clf.view(num_test_frames, num_sequences, *test_feat_clf.shape[-3:])
        target_scores = [self.classifier.classify(f, test_feat_clf) for f in filter_iter]

        # Shape of target scores is num_frames, num_seq, ch, h, w
        # Test feat has 4 dimensions (num_frames*num_seq, ch, h, w)
        mask_pred, _ = self.decoder(target_scores[-1], test_feat)
        return mask_pred, target_scores, None, None, None

    def segment_target(self, target_filter, test_feat_clf, test_feat):
        # Classification features
        assert target_filter.dim() == 5  # seq, filters, ch, h, w
        test_feat_clf = test_feat_clf.view(1, 1, *test_feat_clf.shape[-3:])

        target_scores = self.classifier.classify(target_filter, test_feat_clf)

        mask_pred, _ = self.decoder(target_scores, test_feat)
        return mask_pred, None

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
def steepest_descent_resnet18(filter_size=1, num_filters=1, optim_iter=3, optim_init_reg=0.01,
                              backbone_pretrained=False, clf_feat_blocks=1,
                              clf_feat_norm=True, final_conv=False,
                              out_feature_dim=256,
                              detach_length=float('Inf'),
                              label_encoder_dims=(1, 1),
                              frozen_backbone_layers=(),
                              label_encoder_type='res_ds16',
                              upsample_residuals=False,
                              decoder_mdim=64, filter_groups=1,
                              cls_feat_extractor='res_block'):
    # backbone
    backbone_net = backbones.resnet18(pretrained=backbone_pretrained, frozen_layers=frozen_backbone_layers)

    norm_scale = math.sqrt(1.0 / (out_feature_dim * filter_size * filter_size))

    # classifier
    if cls_feat_extractor == 'res_block':
        clf_feature_extractor = clf_features.residual_basic_block(num_blocks=clf_feat_blocks, l2norm=clf_feat_norm,
                                                                  final_conv=final_conv, norm_scale=norm_scale,
                                                                  out_dim=out_feature_dim)
    elif cls_feat_extractor == 'ppm':
        clf_feature_extractor = seg_features.SegBlockPPM_keepsize(feature_dim=256,
                                                                  num_blocks=clf_feat_blocks, l2norm=clf_feat_norm,
                                                                  final_conv=final_conv, norm_scale=norm_scale,
                                                                  out_dim=out_feature_dim)
    else:
        raise Exception

    initializer = seg_initializer.FilterInitializerZero(filter_size=filter_size, num_filters=num_filters,
                                                        feature_dim=out_feature_dim, filter_groups=filter_groups)

    if label_encoder_type == 'res_ds16':
        label_encoder = seg_label_encoder.ResidualDS16(layer_dims=label_encoder_dims + (num_filters, ))
    elif label_encoder_type == 'res_ds16_feat':
        label_encoder = seg_label_encoder.ResidualDS16Feat(layer_dims=label_encoder_dims + (num_filters, ), feat_dim=out_feature_dim)
    elif label_encoder_type == 'pool_ds16':
        label_encoder = seg_label_encoder.PoolDS16()
    elif label_encoder_type == 'identity':
        label_encoder = seg_label_encoder.Identity()
    else:
        raise Exception

    residual_module = loss_residual_modules.LinearFilterSeg(init_filter_reg=optim_init_reg, label_encoder=label_encoder,
                                                            upsample_residuals=upsample_residuals)

    optimizer = steepestdescent.GNSteepestDescent(residual_module=residual_module, num_iter=optim_iter, detach_length=detach_length,
                                                  residual_batch_dim=1, compute_losses=True)

    classifier = target_clf.LinearFilter(filter_size=filter_size, filter_initializer=initializer,
                                         filter_optimizer=optimizer, feature_extractor=clf_feature_extractor)

    decoder = stm_decoder.DecoderResnet18(filter_out_dim=num_filters, mdim=decoder_mdim)
    net = SegTracker(feature_extractor=backbone_net, classifier=classifier, decoder=decoder)
    return net


@model_constructor
def steepest_descent_resnet18_v2(filter_size=1, num_filters=1, optim_iter=3, optim_init_reg=0.01,
                                 backbone_pretrained=False, clf_feat_blocks=1,
                                 clf_feat_norm=True, final_conv=False,
                                 out_feature_dim=256,
                                 detach_length=float('Inf'),
                                 label_encoder_dims=(1, 1),
                                 frozen_backbone_layers=(),
                                 label_encoder_type='res_ds16',
                                 upsample_residuals=False,
                                 decoder_mdim=64, filter_groups=1,
                                 cls_feat_extractor='res_block',
                                 decoder_type='stm',
                                 use_bn_in_label_enc=True):
    # backbone
    backbone_net = backbones.resnet18(pretrained=backbone_pretrained, frozen_layers=frozen_backbone_layers)

    norm_scale = math.sqrt(1.0 / (out_feature_dim * filter_size * filter_size))

    # classifier
    if cls_feat_extractor == 'res_block':
        clf_feature_extractor = clf_features.residual_basic_block(num_blocks=clf_feat_blocks, l2norm=clf_feat_norm,
                                                                  final_conv=final_conv, norm_scale=norm_scale,
                                                                  out_dim=out_feature_dim)
    elif cls_feat_extractor == 'ppm':
        clf_feature_extractor = seg_features.SegBlockPPM_keepsize(feature_dim=256,
                                                                  num_blocks=clf_feat_blocks, l2norm=clf_feat_norm,
                                                                  final_conv=final_conv, norm_scale=norm_scale,
                                                                  out_dim=out_feature_dim)
    else:
        raise Exception

    initializer = seg_initializer.FilterInitializerZero(filter_size=filter_size, num_filters=num_filters,
                                                        feature_dim=out_feature_dim, filter_groups=filter_groups)

    if label_encoder_type == 'res_ds16':
        label_encoder = seg_label_encoder.ResidualDS16(layer_dims=label_encoder_dims + (num_filters, ))
    elif label_encoder_type == 'res_ds16_sw':
        label_encoder = seg_label_encoder.ResidualDS16SW(layer_dims=label_encoder_dims + (num_filters, ), use_bn=use_bn_in_label_enc)
    elif label_encoder_type == 'res_ds16_feat':
        label_encoder = seg_label_encoder.ResidualDS16Feat(layer_dims=label_encoder_dims + (num_filters, ), feat_dim=out_feature_dim)
    elif label_encoder_type == 'pool_ds16':
        label_encoder = seg_label_encoder.PoolDS16()
    elif label_encoder_type == 'identity':
        label_encoder = seg_label_encoder.Identity()
    else:
        raise Exception

    residual_module = loss_residual_modules.LinearFilterSeg(init_filter_reg=optim_init_reg,
                                                            upsample_residuals=upsample_residuals)

    optimizer = steepestdescent.GNSteepestDescent(residual_module=residual_module, num_iter=optim_iter, detach_length=detach_length,
                                                  residual_batch_dim=1, compute_losses=True)

    classifier = target_clf.LinearFilter(filter_size=filter_size, filter_initializer=initializer,
                                         filter_optimizer=optimizer, feature_extractor=clf_feature_extractor)

    if decoder_type == 'stm':
        decoder = stm_decoder.DecoderResnet18(filter_out_dim=num_filters, mdim=decoder_mdim)
    elif decoder_type == 'stm_maskall':
        decoder = stm_decoder.DecoderResnet18Mask(filter_out_dim=num_filters, mdim=decoder_mdim)
    else:
        raise Exception

    net = SegTracker(feature_extractor=backbone_net, classifier=classifier, decoder=decoder,
                     label_encoder=label_encoder)
    return net


@model_constructor
def steepest_descent_resnet50(filter_size=1, num_filters=1, optim_iter=3, optim_init_reg=0.01,
                              backbone_pretrained=False, clf_feat_blocks=1,
                              clf_feat_norm=True, final_conv=False,
                              out_feature_dim=512,
                              detach_length=float('Inf'),
                              label_encoder_dims=(1, 1),
                              frozen_backbone_layers=(),
                              label_encoder_type='res_ds16',
                              upsample_residuals=False,
                              decoder_mdim=256, filter_groups=1,
                              cls_feat_extractor='res_block',
                              backbone_net_weights_path=None,
                              decoder_type='stm'):
    # backbone
    backbone_net = backbones.resnet50(pretrained=backbone_pretrained, frozen_layers=frozen_backbone_layers)

    norm_scale = math.sqrt(1.0 / (out_feature_dim * filter_size * filter_size))

    # classifier
    if cls_feat_extractor == 'res_block':
        clf_feature_extractor = clf_features.residual_basic_block(num_blocks=clf_feat_blocks, l2norm=clf_feat_norm,
                                                                  final_conv=final_conv, norm_scale=norm_scale,
                                                                  out_dim=out_feature_dim)
    elif cls_feat_extractor == 'ppm':
        clf_feature_extractor = seg_features.SegBlockPPM_keepsize(feature_dim=1024,
                                                                  l2norm=clf_feat_norm,
                                                                  norm_scale=norm_scale,
                                                                  out_dim=out_feature_dim)
    else:
        raise Exception

    initializer = seg_initializer.FilterInitializerZero(filter_size=filter_size, num_filters=num_filters,
                                                        feature_dim=out_feature_dim, filter_groups=filter_groups)

    if label_encoder_type == 'res_ds16':
        label_encoder = seg_label_encoder.ResidualDS16(layer_dims=label_encoder_dims + (num_filters, ))
    elif label_encoder_type == 'res_ds16_feat':
        label_encoder = seg_label_encoder.ResidualDS16Feat(layer_dims=label_encoder_dims + (num_filters, ), feat_dim=out_feature_dim)
    elif label_encoder_type == 'pool_ds16':
        label_encoder = seg_label_encoder.PoolDS16()
    elif label_encoder_type == 'identity':
        label_encoder = seg_label_encoder.Identity()
    else:
        raise Exception

    residual_module = loss_residual_modules.LinearFilterSeg(init_filter_reg=optim_init_reg,
                                                            upsample_residuals=upsample_residuals)

    optimizer = steepestdescent.GNSteepestDescent(residual_module=residual_module, num_iter=optim_iter, detach_length=detach_length,
                                                  residual_batch_dim=1, compute_losses=True)

    classifier = target_clf.LinearFilter(filter_size=filter_size, filter_initializer=initializer,
                                         filter_optimizer=optimizer, feature_extractor=clf_feature_extractor)

    if decoder_type == 'stm':
        decoder = stm_decoder.DecoderResnet50(filter_out_dim=num_filters, mdim=decoder_mdim)
    elif decoder_type == 'stm_maskall':
        decoder = stm_decoder.DecoderResnet50Mask(filter_out_dim=num_filters, mdim=decoder_mdim)
    else:
        raise Exception

    net = SegTracker(feature_extractor=backbone_net, classifier=classifier, decoder=decoder,
                     label_encoder=label_encoder)
    return net