import math
import torch
import torch.nn as nn
import torch.nn.functional as F
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
import ltr.models.target_classifier.residual_modules as trk_loss_residual_modules
import ltr.models.target_classifier.initializer as trk_initializer
import ltr.models.target_classifier.linear_filter as trk_target_clf


class SegDolfDimpTracker(nn.Module):
    def __init__(self, feature_extractor, seg_classifier, decoder, classification_layer, refinement_layers,
                 label_encoder=None, aux_layers=None, bb_regressor=None, bbreg_decoder_layer=None,
                 dimp_classifier=None, ada_dimp=False):
        super().__init__()

        self.feature_extractor = feature_extractor
        self.seg_classifier = seg_classifier
        self.decoder = decoder
        self.dimp_classifier = dimp_classifier

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
        self.ada_dimp = ada_dimp

    def forward(self, train_imgs, test_imgs, train_masks, test_masks, train_bb, test_center_label,
                train_bb_in_sa=None, train_sa_bb=None, test_sa_bb=None):
        assert train_imgs.dim() == 5 and test_imgs.dim() == 5, 'Expect 5 dimensional inputs'
        assert train_masks.dim() == 4, 'Expect 4 dimensional masks'

        num_sequences = train_imgs.shape[1]
        num_train_frames = train_imgs.shape[0]
        num_test_frames = test_imgs.shape[0]

        # Extract backbone features
        train_feat = self.extract_backbone_features(train_imgs.contiguous().view(-1, train_imgs.shape[-3], train_imgs.shape[-2], train_imgs.shape[-1]))
        test_feat = self.extract_backbone_features(test_imgs.contiguous().view(-1, test_imgs.shape[-3], test_imgs.shape[-2], test_imgs.shape[-1]))

        # DiMP
        # Classification features
        train_feat_backbone_clf = self.get_backbone_clf_feat(train_feat)
        test_feat_backbone_clf = self.get_backbone_clf_feat(test_feat)

        if self.ada_dimp:
            target_scores_dimp, _ = self.dimp_classifier(train_feat_backbone_clf, test_feat_backbone_clf,
                                                         train_bb_in_sa,
                                                         train_sa_bb,
                                                         test_sa_bb)
        else:
            target_scores_dimp = self.dimp_classifier(train_feat_backbone_clf, test_feat_backbone_clf, train_bb)

        # target_scores_dimp = [F.interpolate(s, test_feat_backbone_clf.shape[-2:], mode='bicubic', align_corners=False)
        #                       for s in target_scores_dimp]

        # Extract classification features
        train_feat_clf_seg = self.extract_seg_classification_feat(train_feat)     # seq*frames, channels, height, width
        test_feat_clf_seg = self.extract_seg_classification_feat(test_feat)       # seq*frames, channels, height, width

        if self.label_encoder is not None:
            mask_enc = self.label_encoder(train_masks.contiguous(), train_feat_clf_seg)
        else:
            mask_enc = train_masks.contiguous()

        train_feat_clf_seg = train_feat_clf_seg.view(num_train_frames, num_sequences, *train_feat_clf_seg.shape[-3:])
        filter_seg, filter_iter_seg, _ = self.seg_classifier.get_filter(train_feat_clf_seg, mask_enc)

        test_feat_clf_seg = test_feat_clf_seg.view(num_test_frames, num_sequences, *test_feat_clf_seg.shape[-3:])
        target_scores_seg = [self.seg_classifier.classify(f, test_feat_clf_seg) for f in filter_iter_seg]
        # target_scores = [s.unsqueeze(dim=2) for s in target_scores]

        target_scores_seg_last_iter = target_scores_seg[-1]

        decoder_input = torch.cat((target_scores_seg_last_iter, test_center_label.unsqueeze(2)), dim=2)
        mask_pred, decoder_feat = self.decoder(decoder_input, test_feat, test_imgs.shape[-2:],
                                               ('layer4_dec', 'layer3_dec', 'layer2_dec', 'layer1_dec'))

        decoder_feat['mask_enc'] = target_scores_seg_last_iter.view(-1, *target_scores_seg_last_iter.shape[-3:])
        aux_mask_pred = {}
        for L, predictor in self.aux_layers.items():
            aux_mask_pred[L] = predictor(decoder_feat[L], test_imgs.shape[-2:])

        bb_pred = None
        if self.bb_regressor is not None:
            bb_pred = self.bb_regressor(decoder_feat[self.bbreg_decoder_layer])

        return mask_pred, target_scores_seg, aux_mask_pred, bb_pred, target_scores_dimp

    def segment_target(self, target_filter, test_feat_clf, test_feat, center_label):
        # Classification features
        assert target_filter.dim() == 5     # seq, filters, ch, h, w
        test_feat_clf = test_feat_clf.view(1, 1, *test_feat_clf.shape[-3:])

        target_scores = self.seg_classifier.classify(target_filter, test_feat_clf)
        center_label = center_label.view(*target_scores.shape[:2], 1, *target_scores.shape[-2:])
        decoder_input = torch.cat((target_scores, center_label), dim=2)
        mask_pred, decoder_feat = self.decoder(decoder_input, test_feat,
                                               (test_feat_clf.shape[-2]*16, test_feat_clf.shape[-1]*16),
                                               (self.bbreg_decoder_layer, ))

        bb_pred = None
        if self.bb_regressor is not None:
            bb_pred = self.bb_regressor(decoder_feat[self.bbreg_decoder_layer])
            bb_pred[:, :2] *= test_feat_clf.shape[-2] * 16
            bb_pred[:, 2:] *= test_feat_clf.shape[-1] * 16
            bb_pred = torch.stack((bb_pred[:, 2], bb_pred[:, 0],
                                   bb_pred[:, 3] - bb_pred[:, 2],
                                   bb_pred[:, 1] - bb_pred[:, 0]), dim=1)
        # Output is 1, 1, h, w
        return mask_pred, bb_pred

    def get_backbone_clf_feat(self, backbone_feat):
        feat = OrderedDict({l: backbone_feat[l] for l in self.classification_layer})
        if len(self.classification_layer) == 1:
            return feat[self.classification_layer[0]]
        return feat

    def get_backbone_bbreg_feat(self, backbone_feat):
        return [backbone_feat[l] for l in self.bb_regressor_layer]

    def extract_seg_classification_feat(self, backbone_feat):
        return self.seg_classifier.extract_classification_feat(self.get_backbone_clf_feat(backbone_feat))

    def extract_dimp_classification_feat(self, backbone_feat):
        return self.dimp_classifier.extract_classification_feat(self.get_backbone_clf_feat(backbone_feat))

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
                              backbone_pretrained=False,
                              clf_feat_norm=True,
                              seg_out_feature_dim=256,
                              classification_layer='layer3',
                              refinement_layers=("layer4", "layer3", "layer2", "layer1",),
                              detach_length=float('Inf'),
                              label_encoder_dims=(1, 1),
                              frozen_backbone_layers=(),
                              label_encoder_type='identity',
                              decoder_mdim=64, filter_groups=1,
                              upsample_residuals=True,
                              ppm_use_res_block=False,
                              use_bn_in_label_enc=True,
                              residual_activation_fn=None,
                              aux_mask_loss_layers=(),
                              bb_regressor_type=None,
                              dimp_out_feature_dim=256,
                              dimp_cls_feat_norm=True,
                              dimp_cls_feat_blocks=1,
                              dimp_cls_final_conv=True,
                              dimp_cls_init_filter_norm=False,
                              dimp_init_gauss_sigma=None,
                              dimp_feat_stride=32,
                              dimp_cls_feat_pool=True):
    # backbone
    backbone_net = backbones.resnet18(pretrained=backbone_pretrained, frozen_layers=frozen_backbone_layers)

    norm_scale = math.sqrt(1.0 / (seg_out_feature_dim * filter_size * filter_size))

    # ******************************** Segmentation *************************************************
    # classifier
    layer_channels = backbone_net.out_feature_channels()
    clf_feature_extractor = seg_features.SegBlockPPM_keepsize(feature_dim=layer_channels[classification_layer],
                                                              l2norm=clf_feat_norm,
                                                              norm_scale=norm_scale,
                                                              out_dim=seg_out_feature_dim,
                                                              use_res_block=ppm_use_res_block)
    initializer = seg_initializer.FilterInitializerZero(filter_size=filter_size, num_filters=num_filters,
                                                        feature_dim=seg_out_feature_dim, filter_groups=filter_groups)

    if label_encoder_type == 'res_ds16':
        label_encoder = seg_label_encoder.ResidualDS16(layer_dims=label_encoder_dims + (num_filters, ))
    elif label_encoder_type == 'res_ds16_sw':
        label_encoder = seg_label_encoder.ResidualDS16SW(layer_dims=label_encoder_dims + (num_filters, ), use_bn=use_bn_in_label_enc)
    elif label_encoder_type == 'res_ds16_feat':
        label_encoder = seg_label_encoder.ResidualDS16Feat(layer_dims=label_encoder_dims + (num_filters, ), feat_dim=seg_out_feature_dim)
    elif label_encoder_type == 'res_ds16_feat_sw':
        label_encoder = seg_label_encoder.ResidualDS16FeatSW(layer_dims=label_encoder_dims + (num_filters, ), feat_dim=seg_out_feature_dim)
    elif label_encoder_type == 'pool_ds16':
        label_encoder = seg_label_encoder.PoolDS16()
    elif label_encoder_type == 'identity':
        label_encoder = seg_label_encoder.Identity()
    else:
        raise Exception

    residual_module = loss_residual_modules.LinearFilterSeg(init_filter_reg=optim_init_reg,
                                                            upsample_residuals=upsample_residuals,
                                                            score_act=residual_activation_fn)

    optimizer = steepestdescent.GNSteepestDescent(residual_module=residual_module, num_iter=optim_iter, detach_length=detach_length,
                                                  residual_batch_dim=1, compute_losses=True)

    seg_classifier = target_clf.LinearFilter(filter_size=filter_size, filter_initializer=initializer,
                                             filter_optimizer=optimizer, feature_extractor=clf_feature_extractor)

    refinement_layers_channels = {L: layer_channels[L] for L in refinement_layers}

    decoder = dolf_decoder.RefelixNetwork2(num_filters + 1, decoder_mdim, refinement_layers_channels,
                                           new_upsampler=True, use_bn=True)

    decoder_channels = decoder.out_feature_channels()
    if len(aux_mask_loss_layers) > 0:
        aux_layers = nn.ModuleDict()
        decoder_channels['mask_enc'] = num_filters
        for l in aux_mask_loss_layers:
            aux_layers[l] = seg_aux_layers.ConvPredictor(decoder_channels[l])
    else:
        aux_layers = None

    bbreg_decoder_layer = None
    if bb_regressor_type is None:
        bb_regressor = None
    elif bb_regressor_type == 'pool_fc_l1':
        bb_regressor = seg_aux_layers.BBRegressor(decoder_channels['layer1_dec'], (30, 52), 32, 512)
        bbreg_decoder_layer = 'layer1_dec'
    else:
        raise Exception

    # ************************************** DiMP ******************************************************
    dimp_cls_norm_scale = math.sqrt(
        1.0 / (dimp_out_feature_dim * filter_size * filter_size))
    target_cls_feature_extractor = clf_features.residual_basic_block(num_blocks=dimp_cls_feat_blocks,
                                                                     l2norm=dimp_cls_feat_norm,
                                                                     final_conv=dimp_cls_final_conv,
                                                                     norm_scale=dimp_cls_norm_scale,
                                                                     out_dim=dimp_out_feature_dim,
                                                                     init_pool=dimp_cls_feat_pool)

    target_cls_initializer = trk_initializer.FilterInitializerLinear(filter_size=filter_size,
                                                                     filter_norm=dimp_cls_init_filter_norm,
                                                                     feature_dim=dimp_out_feature_dim)

    num_dist_bins = 100
    bin_displacement = 0.1
    mask_init_factor = 3.0
    score_act = 'relu'
    act_param = None
    target_mask_act = 'sigmoid'
    target_cls_residual_module = trk_loss_residual_modules.LinearFilterLearnGen(feat_stride=dimp_feat_stride,
                                                                                init_filter_reg=optim_init_reg,
                                                                                init_gauss_sigma=dimp_init_gauss_sigma,
                                                                                num_dist_bins=num_dist_bins,
                                                                                bin_displacement=bin_displacement,
                                                                                mask_init_factor=mask_init_factor,
                                                                                score_act=score_act,
                                                                                act_param=act_param,
                                                                                mask_act=target_mask_act)

    target_cls_optimizer = steepestdescent.GNSteepestDescent(residual_module=target_cls_residual_module,
                                                             num_iter=optim_iter,
                                                             detach_length=detach_length,
                                                             residual_batch_dim=1, compute_losses=True)

    dimp_classifier = trk_target_clf.LinearFilterMeta(filter_size=filter_size, filter_initializer=target_cls_initializer,
                                                      filter_optimizer=target_cls_optimizer,
                                                      feature_extractor=target_cls_feature_extractor)

    # ****************************************************************************************************
    net = SegDolfDimpTracker(feature_extractor=backbone_net, seg_classifier=seg_classifier, decoder=decoder,
                             label_encoder=label_encoder,
                             classification_layer=classification_layer, refinement_layers=refinement_layers,
                             aux_layers=aux_layers,
                             bb_regressor=bb_regressor,
                             bbreg_decoder_layer=bbreg_decoder_layer,
                             dimp_classifier=dimp_classifier)
    return net


@model_constructor
def steepest_descent_resnet18_ada(filter_size=1, num_filters=1, optim_iter=3, optim_init_reg=0.01,
                                  backbone_pretrained=False,
                                  clf_feat_norm=True,
                                  seg_out_feature_dim=256,
                                  classification_layer='layer3',
                                  refinement_layers=("layer4", "layer3", "layer2", "layer1",),
                                  detach_length=float('Inf'),
                                  label_encoder_dims=(1, 1),
                                  frozen_backbone_layers=(),
                                  label_encoder_type='identity',
                                  decoder_mdim=64, filter_groups=1,
                                  upsample_residuals=True,
                                  ppm_use_res_block=False,
                                  use_bn_in_label_enc=True,
                                  residual_activation_fn=None,
                                  aux_mask_loss_layers=(),
                                  bb_regressor_type=None,
                                  dimp_out_feature_dim=256,
                                  dimp_cls_feat_norm=True,
                                  dimp_cls_feat_blocks=1,
                                  dimp_cls_final_conv=True,
                                  dimp_cls_init_filter_norm=False,
                                  dimp_init_gauss_sigma=None,
                                  dimp_feat_stride=16,
                                  dimp_feat_pool_size=None,
                                  ):
    # backbone
    backbone_net = backbones.resnet18(pretrained=backbone_pretrained, frozen_layers=frozen_backbone_layers)

    norm_scale = math.sqrt(1.0 / (seg_out_feature_dim * filter_size * filter_size))

    # ******************************** Segmentation *************************************************
    # classifier
    layer_channels = backbone_net.out_feature_channels()
    clf_feature_extractor = seg_features.SegBlockPPM_keepsize(feature_dim=layer_channels[classification_layer],
                                                              l2norm=clf_feat_norm,
                                                              norm_scale=norm_scale,
                                                              out_dim=seg_out_feature_dim,
                                                              use_res_block=ppm_use_res_block)
    initializer = seg_initializer.FilterInitializerZero(filter_size=filter_size, num_filters=num_filters,
                                                        feature_dim=seg_out_feature_dim, filter_groups=filter_groups)

    if label_encoder_type == 'res_ds16':
        label_encoder = seg_label_encoder.ResidualDS16(layer_dims=label_encoder_dims + (num_filters, ))
    elif label_encoder_type == 'res_ds16_sw':
        label_encoder = seg_label_encoder.ResidualDS16SW(layer_dims=label_encoder_dims + (num_filters, ), use_bn=use_bn_in_label_enc)
    elif label_encoder_type == 'res_ds16_feat':
        label_encoder = seg_label_encoder.ResidualDS16Feat(layer_dims=label_encoder_dims + (num_filters, ), feat_dim=seg_out_feature_dim)
    elif label_encoder_type == 'res_ds16_feat_sw':
        label_encoder = seg_label_encoder.ResidualDS16FeatSW(layer_dims=label_encoder_dims + (num_filters, ), feat_dim=seg_out_feature_dim)
    elif label_encoder_type == 'pool_ds16':
        label_encoder = seg_label_encoder.PoolDS16()
    elif label_encoder_type == 'identity':
        label_encoder = seg_label_encoder.Identity()
    else:
        raise Exception

    residual_module = loss_residual_modules.LinearFilterSeg(init_filter_reg=optim_init_reg,
                                                            upsample_residuals=upsample_residuals,
                                                            score_act=residual_activation_fn)

    optimizer = steepestdescent.GNSteepestDescent(residual_module=residual_module, num_iter=optim_iter, detach_length=detach_length,
                                                  residual_batch_dim=1, compute_losses=True)

    seg_classifier = target_clf.LinearFilter(filter_size=filter_size, filter_initializer=initializer,
                                             filter_optimizer=optimizer, feature_extractor=clf_feature_extractor)

    refinement_layers_channels = {L: layer_channels[L] for L in refinement_layers}

    decoder = dolf_decoder.RefelixNetwork2(num_filters + 1, decoder_mdim, refinement_layers_channels,
                                           new_upsampler=True, use_bn=True)

    decoder_channels = decoder.out_feature_channels()
    if len(aux_mask_loss_layers) > 0:
        aux_layers = nn.ModuleDict()
        decoder_channels['mask_enc'] = num_filters
        for l in aux_mask_loss_layers:
            aux_layers[l] = seg_aux_layers.ConvPredictor(decoder_channels[l])
    else:
        aux_layers = None

    bbreg_decoder_layer = None
    if bb_regressor_type is None:
        bb_regressor = None
    elif bb_regressor_type == 'pool_fc_l1':
        bb_regressor = seg_aux_layers.BBRegressor(decoder_channels['layer1_dec'], (30, 52), 32, 512)
        bbreg_decoder_layer = 'layer1_dec'
    else:
        raise Exception

    # ************************************** DiMP ******************************************************
    dimp_cls_norm_scale = math.sqrt(
        1.0 / (dimp_out_feature_dim * filter_size * filter_size))
    target_cls_feature_extractor = clf_features.residual_basic_block(num_blocks=dimp_cls_feat_blocks,
                                                                     l2norm=dimp_cls_feat_norm,
                                                                     final_conv=dimp_cls_final_conv,
                                                                     norm_scale=dimp_cls_norm_scale,
                                                                     out_dim=dimp_out_feature_dim)

    target_cls_initializer = trk_initializer.FilterInitializerLinear(filter_size=filter_size,
                                                                     filter_norm=dimp_cls_init_filter_norm,
                                                                     feature_dim=dimp_out_feature_dim)

    num_dist_bins = 100
    bin_displacement = 0.1
    mask_init_factor = 3.0
    score_act = 'relu'
    act_param = None
    target_mask_act = 'sigmoid'
    target_cls_residual_module = trk_loss_residual_modules.LinearFilterLearnGen(feat_stride=1,
                                                                                init_filter_reg=optim_init_reg,
                                                                                init_gauss_sigma=dimp_init_gauss_sigma,
                                                                                num_dist_bins=num_dist_bins,
                                                                                bin_displacement=bin_displacement,
                                                                                mask_init_factor=mask_init_factor,
                                                                                score_act=score_act,
                                                                                act_param=act_param,
                                                                                mask_act=target_mask_act)

    target_cls_optimizer = steepestdescent.GNSteepestDescent(residual_module=target_cls_residual_module,
                                                             num_iter=optim_iter,
                                                             detach_length=detach_length,
                                                             residual_batch_dim=1, compute_losses=True)

    dimp_classifier = trk_target_clf.LinearFilterMetaAda(filter_size=filter_size,
                                                         filter_initializer=target_cls_initializer,
                                                         filter_optimizer=target_cls_optimizer,
                                                         feature_extractor=target_cls_feature_extractor,
                                                         pool_size=dimp_feat_pool_size,
                                                         feat_stride=dimp_feat_stride, pool_type='roi_align')

    # ****************************************************************************************************
    net = SegDolfDimpTracker(feature_extractor=backbone_net, seg_classifier=seg_classifier, decoder=decoder,
                             label_encoder=label_encoder,
                             classification_layer=classification_layer, refinement_layers=refinement_layers,
                             aux_layers=aux_layers,
                             bb_regressor=bb_regressor,
                             bbreg_decoder_layer=bbreg_decoder_layer,
                             dimp_classifier=dimp_classifier,
                             ada_dimp=True)
    return net
