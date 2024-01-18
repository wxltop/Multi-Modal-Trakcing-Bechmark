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
import ltr.models.segmentation.awesome_decoder2 as a2dec
import ltr.models.segmentation.dolf_decoder as dolf_decoder
import ltr.models.segmentation.awesome_decoder as awesome_decoder
import ltr.models.segmentation.features as seg_features
import ltr.models.backbone as backbones
import ltr.models.backbone.resnet_mrcnn as mrcnn_backbones
import ltr.models.meta.steepestdescent as steepestdescent
import ltr.models.segmentation.aux_layers as seg_aux_layers
from ltr import model_constructor
from pytracking import TensorList


class SegDolfTracker(nn.Module):
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

    def forward(self, train_imgs, test_imgs, train_masks, test_masks, num_refinement_iter=2):
        num_sequences = train_imgs.shape[1]
        num_train_frames = train_imgs.shape[0]
        num_test_frames = test_imgs.shape[0]

        im_size = test_imgs.shape[-2:]

        # Extract backbone features
        train_feat = self.extract_backbone_features(
            train_imgs.view(-1, train_imgs.shape[-3], train_imgs.shape[-2], train_imgs.shape[-1]))
        test_feat = self.extract_backbone_features(
            test_imgs.view(-1, test_imgs.shape[-3], test_imgs.shape[-2], test_imgs.shape[-1]))

        # Extract classification features
        train_feat_clf = self.extract_classification_feat(train_feat)  # seq*frames, channels, height, width
        train_feat_clf = train_feat_clf.view(num_train_frames, num_sequences, *train_feat_clf.shape[-3:])

        train_feat_clf_all = [train_feat_clf, ]

        train_mask_enc_info = self.label_encoder(train_masks, train_feat_clf)

        if isinstance(train_mask_enc_info, (tuple, list)):
            train_mask_enc = train_mask_enc_info[0]
            train_mask_sw = train_mask_enc_info[1]
        else:
            train_mask_enc = train_mask_enc_info
            train_mask_sw = None

        train_mask_enc_all = [train_mask_enc, ]
        train_mask_sw_all = None if train_mask_sw is None else [train_mask_sw, ]

        test_feat_clf = self.extract_classification_feat(test_feat)  # seq*frames, channels, height, width

        filter, filter_iter, _ = self.classifier.get_filter(train_feat_clf, (train_mask_enc, train_mask_sw))

        pred_all = []
        for i in range(num_test_frames):
            test_feat_clf_it = test_feat_clf.view(num_test_frames, num_sequences, *test_feat_clf.shape[-3:])[i:i+1, ...]
            target_scores = [self.classifier.classify(f, test_feat_clf_it) for f in filter_iter]

            test_feat_it = {k: v.view(num_test_frames, num_sequences, *v.shape[-3:])[i, ...] for k, v in test_feat.items()}
            target_scores_last_iter = target_scores[-1]
            mask_pred, decoder_feat = self.decoder(target_scores_last_iter, test_feat_it, test_imgs.shape[-2:])
            mask_pred = mask_pred.view(1, num_sequences, *mask_pred.shape[-2:])

            pred_all.append(mask_pred)
            mask_pred_clone = mask_pred.clone().detach()

            mask_pred_clone = torch.sigmoid(mask_pred_clone)

            mask_pred_enc_info = self.label_encoder(mask_pred_clone, test_feat_clf_it)
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

            ## Update
            if (i < (num_test_frames - 1)) and (num_refinement_iter > 0):
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

        pred_all = torch.cat(pred_all, dim=0)
        return pred_all


    def forward_old(self, train_imgs, test_imgs, train_masks, test_masks):
        assert train_imgs.dim() == 5 and test_imgs.dim() == 5, 'Expect 5 dimensional inputs'
        assert train_masks.dim() == 4, 'Expect 4 dimensional masks'

        num_sequences = train_imgs.shape[1]
        num_train_frames = train_imgs.shape[0]
        num_test_frames = test_imgs.shape[0]

        # Extract backbone features
        train_feat = self.extract_backbone_features(train_imgs.contiguous().view(-1, train_imgs.shape[-3], train_imgs.shape[-2], train_imgs.shape[-1]))
        test_feat = self.extract_backbone_features(test_imgs.contiguous().view(-1, test_imgs.shape[-3], test_imgs.shape[-2], test_imgs.shape[-1]))

        # Extract classification features
        train_feat_clf = self.extract_classification_feat(train_feat)       # seq*frames, channels, height, width
        test_feat_clf = self.extract_classification_feat(test_feat)         # seq*frames, channels, height, width

        if self.label_encoder is not None:
            mask_enc = self.label_encoder(train_masks.contiguous(), train_feat_clf)
            mask_enc_test = self.label_encoder(test_masks.contiguous(), test_feat_clf)
        else:
            mask_enc = train_masks.contiguous()
            mask_enc_test = None

        train_feat_clf = train_feat_clf.view(num_train_frames, num_sequences, *train_feat_clf.shape[-3:])
        filter, filter_iter, _ = self.classifier.get_filter(train_feat_clf, mask_enc)

        test_feat_clf = test_feat_clf.view(num_test_frames, num_sequences, *test_feat_clf.shape[-3:])
        target_scores = [self.classifier.classify(f, test_feat_clf) for f in filter_iter]
        # target_scores = [s.unsqueeze(dim=2) for s in target_scores]

        target_scores_last_iter = target_scores[-1]

        mask_pred, decoder_feat = self.decoder(target_scores_last_iter, test_feat, test_imgs.shape[-2:],
                                               ('layer4_dec', 'layer3_dec', 'layer2_dec', 'layer1_dec'))

        decoder_feat['mask_enc'] = target_scores_last_iter.view(-1, *target_scores_last_iter.shape[-3:])
        aux_mask_pred = {}
        for L, predictor in self.aux_layers.items():
            if L in decoder_feat.keys():
                aux_mask_pred[L] = predictor(decoder_feat[L], test_imgs.shape[-2:])

        bb_pred = None
        if self.bb_regressor is not None:
            bb_pred = self.bb_regressor(decoder_feat[self.bbreg_decoder_layer])

        if isinstance(mask_enc_test, (tuple, list)):
            mask_enc_test = mask_enc_test[0]

        if 'mask_enc_iter' in self.aux_layers.keys():
            for i, ts in enumerate(target_scores):
                aux_mask_pred['mask_enc_iter_{}'.format(i)] = \
                    self.aux_layers['mask_enc_iter'](ts.view(-1, *ts.shape[-3:]), test_imgs.shape[-2:])
        return mask_pred, target_scores, mask_enc_test, aux_mask_pred, bb_pred

    def segment_target(self, target_filter, test_feat_clf, test_feat):
        # Classification features
        assert target_filter.dim() == 5     # seq, filters, ch, h, w
        test_feat_clf = test_feat_clf.view(1, 1, *test_feat_clf.shape[-3:])

        target_scores = self.classifier.classify(target_filter, test_feat_clf)

        mask_pred, decoder_feat = self.decoder(target_scores, test_feat,
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

        decoder_feat['mask_enc'] = target_scores.view(-1, *target_scores.shape[-3:])
        aux_mask_pred = {}
        if 'mask_enc_iter' in self.aux_layers.keys():
            aux_mask_pred['mask_enc_iter'] = \
                self.aux_layers['mask_enc_iter'](target_scores.view(-1, *target_scores.shape[-3:]), (test_feat_clf.shape[-2]*16,
                                                                               test_feat_clf.shape[-1]*16))
        # Output is 1, 1, h, w
        return mask_pred, bb_pred, aux_mask_pred

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
                              classification_layer='layer3',
                              refinement_layers = ("layer4", "layer3", "layer2", "layer1",),
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
                              decoder_type='rofl',
                              cls_feat_extractor='ppm',
                              att_inter_dim=-1,
                              use_decoder_backbone_feat=True
                              ):

    # backbone
    backbone_net = backbones.resnet18(pretrained=backbone_pretrained, frozen_layers=frozen_backbone_layers)

    norm_scale = math.sqrt(1.0 / (out_feature_dim * filter_size * filter_size))

    # classifier
    layer_channels = backbone_net.out_feature_channels()

    if cls_feat_extractor == 'res_block':
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
    elif cls_feat_extractor == 'spp':
        clf_feature_extractor = seg_features.SPP(feature_dim=layer_channels[classification_layer],
                                                 inter_dim=layer_channels[classification_layer] // 4,
                                                 out_dim=out_feature_dim, use_bn=True)
    elif cls_feat_extractor == 'identity':
        clf_feature_extractor = seg_features.Identity()
        out_feature_dim = layer_channels[classification_layer]
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
    elif label_encoder_type == 'res_ds16_feat_sw':
        label_encoder = seg_label_encoder.ResidualDS16FeatSW(layer_dims=label_encoder_dims + (num_filters, ), feat_dim=out_feature_dim)
    elif label_encoder_type == 'pool_ds16':
        label_encoder = seg_label_encoder.PoolDS16()
    elif label_encoder_type == 'identity':
        label_encoder = seg_label_encoder.Identity()
    elif label_encoder_type == 'ResLabelGeneratorLabelConv':
        label_encoder = seg_label_encoder.ResLabelGeneratorLabelConv(layer_dims=label_encoder_dims + (num_filters, ), use_bn=use_bn_in_label_enc)
    elif label_encoder_type == 'res_ds16_feat_sw_att':
        label_encoder = seg_label_encoder.ResidualDS16FeatSWAtt(layer_dims=label_encoder_dims + (num_filters, ), feat_dim=out_feature_dim)
    else:
        raise Exception

    residual_module = loss_residual_modules.LinearFilterSeg(init_filter_reg=optim_init_reg,
                                                            upsample_residuals=upsample_residuals,
                                                            score_act=residual_activation_fn)

    optimizer = steepestdescent.GNSteepestDescent(residual_module=residual_module, num_iter=optim_iter, detach_length=detach_length,
                                                  residual_batch_dim=1, compute_losses=True)

    classifier = target_clf.LinearFilter(filter_size=filter_size, filter_initializer=initializer,
                                         filter_optimizer=optimizer, feature_extractor=clf_feature_extractor)

    refinement_layers_channels = {L: layer_channels[L] for L in refinement_layers}


    if decoder_type == 'dolf_decoder':
        decoder = dolf_decoder.RefelixNetwork2(num_filters, decoder_mdim, refinement_layers_channels,
                                           new_upsampler=True, use_bn=True)
    elif decoder_type == 'awesome_decoder':
        decoder = awesome_decoder.SegNetwork(num_filters, decoder_mdim, refinement_layers_channels,
                                             use_bn=True)
    elif decoder_type == 'rofl':
        decoder = dolf_decoder.RefelixNetwork2(num_filters, decoder_mdim, refinement_layers_channels,
                                               new_upsampler=True, use_bn=True,
                                               use_backbone_feat=use_decoder_backbone_feat)
    elif decoder_type == 'rofl_att':
        decoder = dolf_decoder.RefelixNetwork2Att(num_filters, decoder_mdim, refinement_layers_channels,
                                                  new_upsampler=True, use_bn=True,
                                                  att_dim=att_inter_dim)

    else:
        raise Exception

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

    net = SegDolfTracker(feature_extractor=backbone_net, classifier=classifier, decoder=decoder,
                         label_encoder=label_encoder,
                         classification_layer=classification_layer, refinement_layers=refinement_layers,
                         aux_layers=aux_layers,
                         bb_regressor=bb_regressor,
                         bbreg_decoder_layer=bbreg_decoder_layer)
    return net

@model_constructor
def steepest_descent_resnet18_finetune(net):
    # backbone


    net = SegDolfTracker(feature_extractor=net.feature_extractor, classifier=net.classifier, decoder=net.decoder,
                         label_encoder=net.label_encoder,
                         classification_layer=net.classification_layer, refinement_layers=net.refinement_layers,
                         aux_layers=net.aux_layers,
                         bb_regressor=net.bb_regressor,
                         bbreg_decoder_layer=net.bbreg_decoder_layer)
    return net


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
                              skipl = None,
                              final_reduce_factor=2,
                              cls_feat_extractor='ppm',
                              decoder_type='rofl',
                              att_inter_dim=-1,
                              dilation_factors=None,
                              use_aux_pred_for_enc=False,
                              use_final_relu=True,
                              bb_regressor_type=None,
                              filter_initializer_type='zero',
                              backbone_type='imagenet',
                              optimizer_type='GNSD',
                              filter_init_conv_dims=(),
                              filter_init_fc_dims=(),
                              filter_init_sigmoid=True):

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
    elif cls_feat_extractor == 'simple':
        clf_feature_extractor = seg_features.SegBlockSimple(feature_dim=layer_channels[classification_layer],
                                          norm_scale=norm_scale,
                                          out_dim=out_feature_dim)
    elif cls_feat_extractor == 'boring':
        clf_feature_extractor = seg_features.SegBlockBoring(feature_dim=layer_channels[classification_layer],
                                                            norm_scale=norm_scale,
                                                            out_dim=out_feature_dim)

    elif cls_feat_extractor == 'spp':
        clf_feature_extractor = seg_features.SPP(feature_dim=layer_channels[classification_layer],
                                                 inter_dim=layer_channels[classification_layer] // 4,
                                                 out_dim=out_feature_dim, use_bn=True)
    elif cls_feat_extractor == 'identity':
        clf_feature_extractor = seg_features.Identity()
        out_feature_dim = layer_channels[classification_layer]
    else:
        raise Exception

    if filter_initializer_type == 'zero':
        initializer = seg_initializer.FilterInitializerZero(filter_size=filter_size, num_filters=num_filters,
                                                            feature_dim=out_feature_dim, filter_groups=filter_groups)
    elif filter_initializer_type == 'masked_pool_sc':
        initializer = seg_initializer.FilterInitializerMaskPoolSC(conv_dims=filter_init_conv_dims,
                                                                  fc_dims=filter_init_fc_dims,
                                                                  filter_size=filter_size,
                                                                  num_filters=num_filters, feature_dim=out_feature_dim,
                                                                  use_sigmoid=filter_init_sigmoid)
    elif filter_initializer_type == 'masked_roi_pool':
        initializer = seg_initializer.FilterInitializerMaskROIPoolSC(conv_dims=filter_init_conv_dims,
                                                                     filter_size=filter_size,
                                                                     num_filters=num_filters,
                                                                     feature_dim=out_feature_dim)
    else:
        raise Exception

    if label_encoder_type == 'res_ds16':
        label_encoder = seg_label_encoder.ResidualDS16(layer_dims=label_encoder_dims + (num_filters, ))
    elif label_encoder_type == 'res_ds16_sw':
        label_encoder = seg_label_encoder.ResidualDS16SW(layer_dims=label_encoder_dims + (num_filters, ),
                                                         use_bn=use_bn_in_label_enc,
                                                         use_final_relu=use_final_relu)
    elif label_encoder_type == 'res_ds16_nosw':
        label_encoder = seg_label_encoder.ResidualDS16NoSW(layer_dims=label_encoder_dims + (num_filters, ),
                                                           use_bn=use_bn_in_label_enc,
                                                           use_final_relu=use_final_relu)
    elif label_encoder_type == 'conv1':
        label_encoder = seg_label_encoder.Conv1DS16(layer_dims=label_encoder_dims + (num_filters, ),
                                                           use_bn=use_bn_in_label_enc)
    elif label_encoder_type == 'conv1_swconv':
        label_encoder = seg_label_encoder.Conv1DS16SWConv(layer_dims=label_encoder_dims + (num_filters, ),
                                                           use_bn=use_bn_in_label_enc)
    elif label_encoder_type == 'conv1_swres':
        label_encoder = seg_label_encoder.Conv1DS16SWRes(layer_dims=label_encoder_dims + (num_filters, ),
                                                           use_bn=use_bn_in_label_enc)
    elif label_encoder_type == 'res_ds16_sw_v2':
        label_encoder = seg_label_encoder.ResidualDS16SWv2(layer_dims=label_encoder_dims + (num_filters, ),
                                                           use_bn=use_bn_in_label_enc)
    elif label_encoder_type == 'res_ds16_sw_spp':
        label_encoder = seg_label_encoder.ResidualDS16SWSPP(layer_dims=label_encoder_dims + (num_filters, ),
                                                            use_bn=use_bn_in_label_enc,
                                                            use_final_relu=use_final_relu)
    elif label_encoder_type == 'res_ds16_feat_sw':
        label_encoder = seg_label_encoder.ResidualDS16FeatSW(layer_dims=label_encoder_dims + (num_filters, ),
                                                             feat_dim=out_feature_dim,
                                                             use_bn=use_bn_in_label_enc,
                                                             use_final_relu=use_final_relu)
    elif label_encoder_type == 'res_ds16_feat_sw_v2':
        label_encoder = seg_label_encoder.ResidualDS16FeatSWv2(layer_dims=label_encoder_dims + (num_filters,),
                                                               feat_dim=out_feature_dim,
                                                               use_bn=use_bn_in_label_enc,
                                                               use_final_relu=use_final_relu)
    elif label_encoder_type == 'res_ds16_feat':
        label_encoder = seg_label_encoder.ResidualDS16Feat(layer_dims=label_encoder_dims + (num_filters, ), feat_dim=out_feature_dim)
    elif label_encoder_type == 'res_ds16_feat_d':
        label_encoder = seg_label_encoder.ResidualDS16FeatDeep1(layer_dims=label_encoder_dims + (num_filters,),
                                                                feat_dim=out_feature_dim)
    elif label_encoder_type == 'res_ds16_feat_norm':
        label_encoder = seg_label_encoder.ResidualDS16Feat(layer_dims=label_encoder_dims + (num_filters, ), feat_dim=out_feature_dim,
                                                           instance_norm=True,
                                                           norm_scale=norm_scale)
    elif label_encoder_type == 'pool_ds16':
        label_encoder = seg_label_encoder.PoolDS16()
    elif label_encoder_type == 'identity':
        label_encoder = seg_label_encoder.Identity()
    elif label_encoder_type == 'identity_sw16':
        label_encoder = seg_label_encoder.IdentityDS16SW(layer_dims=label_encoder_dims + (num_filters, ),
                                                         use_bn=use_bn_in_label_enc)
    else:
        raise Exception

    residual_module = loss_residual_modules.LinearFilterSeg(init_filter_reg=optim_init_reg, upsample_residuals=upsample_residuals,
                                                      filter_dilation_factors=dilation_factors)

    if optimizer_type == 'GNSD':
        optimizer = steepestdescent.GNSteepestDescent(residual_module=residual_module, num_iter=optim_iter, detach_length=detach_length,
                                                      residual_batch_dim=1, compute_losses=True)
    elif optimizer_type == 'none':
        optimizer = None
    else:
        raise Exception

    if dilation_factors is not None:
        assert num_filters == sum(dilation_factors.values())

    classifier = target_clf.LinearFilter(filter_size=filter_size, filter_initializer=initializer,
                                         filter_optimizer=optimizer, feature_extractor=clf_feature_extractor,
                                         filter_dilation_factors=dilation_factors)

    refinement_layers_channels = {L: layer_channels[L] for L in refinement_layers}

    if decoder_type == 'rofl':
        decoder = dolf_decoder.RefelixNetwork2(num_filters, decoder_mdim, refinement_layers_channels,
                                               new_upsampler=True, use_bn=True)
    elif decoder_type == 'rofl_enc':
        assert num_filters == 1
        decoder = dolf_decoder.RefelixNetwork2Enc(num_filters, 32, decoder_mdim, refinement_layers_channels,
                                                  new_upsampler=True, use_bn=True)
    elif decoder_type == 'rofl_att':
        decoder = dolf_decoder.RefelixNetwork2Att(num_filters, decoder_mdim, refinement_layers_channels,
                                                  new_upsampler=True, use_bn=True,
                                                  att_dim=att_inter_dim)
    elif decoder_type == 'stm_maskall':
        decoder = stm_decoder.DecoderResnet50Mask(filter_out_dim=num_filters, mdim=decoder_mdim)
    elif decoder_type == 'awesome_decoder2':
        decoder = a2dec.SegNetwork(num_filters, decoder_mdim, refinement_layers_channels,
                                              use_bn=True, skip_rrb="layer1", final_reduce_factor=1)
    elif decoder_type == 'nodecoder':
        decoder = dolf_decoder.NaiveUpsample()
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

    decoder_channels = decoder.out_feature_channels()
    bbreg_decoder_layer = None
    if bb_regressor_type is None:
        bb_regressor = None
    elif bb_regressor_type == 'pool_fc_l1':
        bb_regressor = seg_aux_layers.BBRegressor(decoder_channels['layer1_dec'], (30, 52), 32, 512)
        bbreg_decoder_layer = 'layer1_dec'
    elif bb_regressor_type == 'pool_fc_l3':
        bb_regressor = seg_aux_layers.BBRegressor(decoder_channels['layer3_dec'], (30, 52), 32, 512)
        bbreg_decoder_layer = 'layer3_dec'
    elif bb_regressor_type == 'fc_l3':
        bb_regressor = seg_aux_layers.BBRegressorFC(decoder_channels['layer3_dec'])
        bbreg_decoder_layer = 'layer3_dec'
    else:
        raise Exception

    net = SegDolfTracker(feature_extractor=backbone_net, classifier=classifier, decoder=decoder,
                         label_encoder=label_encoder,
                         bb_regressor=bb_regressor,
                         classification_layer=classification_layer, refinement_layers=refinement_layers,
                         aux_layers=aux_layers,
                         bbreg_decoder_layer=bbreg_decoder_layer)
    return net


@model_constructor
def steepest_descent_resnet101(filter_size=1, num_filters=1, optim_iter=3, optim_init_reg=0.01,
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
                               use_final_relu=True,
                               bb_regressor_type=None,
                               backbone_type='imagenet'):
    # backbone
    if backbone_type == 'imagenet':
        backbone_net = backbones.resnet50(pretrained=backbone_pretrained, frozen_layers=frozen_backbone_layers)
    elif backbone_type == 'mrcnn':
        backbone_net = mrcnn_backbones.resnet101(pretrained=False, frozen_layers=frozen_backbone_layers)
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
    elif cls_feat_extractor == 'spp':
        clf_feature_extractor = seg_features.SPP(feature_dim=layer_channels[classification_layer],
                                                 inter_dim=layer_channels[classification_layer] // 4,
                                                 out_dim=out_feature_dim, use_bn=True)
    elif cls_feat_extractor == 'identity':
        clf_feature_extractor = seg_features.Identity()
        out_feature_dim = layer_channels[classification_layer]
    else:
        raise Exception

    initializer = seg_initializer.FilterInitializerZero(filter_size=filter_size, num_filters=num_filters,
                                                        feature_dim=out_feature_dim, filter_groups=filter_groups)

    if label_encoder_type == 'res_ds16':
        label_encoder = seg_label_encoder.ResidualDS16(layer_dims=label_encoder_dims + (num_filters, ))
    elif label_encoder_type == 'res_ds16_sw':
        label_encoder = seg_label_encoder.ResidualDS16SW(layer_dims=label_encoder_dims + (num_filters, ), use_bn=use_bn_in_label_enc)
    elif label_encoder_type == 'res_ds16_sw_bgfg':
        label_encoder = seg_label_encoder.ResidualDS16SWFGBG(layer_dims=label_encoder_dims + (num_filters,),
                                                         use_bn=use_bn_in_label_enc)
        label_encoder = seg_label_encoder.ResidualDS16SW(layer_dims=label_encoder_dims + (num_filters, ),
                                                         use_bn=use_bn_in_label_enc,
                                                         use_final_relu=use_final_relu)
    elif label_encoder_type == 'res_ds16_nosw':
        label_encoder = seg_label_encoder.ResidualDS16NoSW(layer_dims=label_encoder_dims + (num_filters, ),
                                                           use_bn=use_bn_in_label_enc,
                                                           use_final_relu=use_final_relu)
    elif label_encoder_type == 'conv1':
        label_encoder = seg_label_encoder.Conv1DS16(layer_dims=label_encoder_dims + (num_filters, ),
                                                           use_bn=use_bn_in_label_enc)
    elif label_encoder_type == 'conv1_swconv':
        label_encoder = seg_label_encoder.Conv1DS16SWConv(layer_dims=label_encoder_dims + (num_filters, ),
                                                           use_bn=use_bn_in_label_enc)
    elif label_encoder_type == 'conv1_swres':
        label_encoder = seg_label_encoder.Conv1DS16SWRes(layer_dims=label_encoder_dims + (num_filters, ),
                                                           use_bn=use_bn_in_label_enc)
    elif label_encoder_type == 'res_ds16_sw_v2':
        label_encoder = seg_label_encoder.ResidualDS16SWv2(layer_dims=label_encoder_dims + (num_filters, ),
                                                           use_bn=use_bn_in_label_enc)
    elif label_encoder_type == 'res_ds16_sw_spp':
        label_encoder = seg_label_encoder.ResidualDS16SWSPP(layer_dims=label_encoder_dims + (num_filters, ),
                                                            use_bn=use_bn_in_label_enc,
                                                            use_final_relu=use_final_relu)
    elif label_encoder_type == 'res_ds16_feat_sw':
        label_encoder = seg_label_encoder.ResidualDS16FeatSW(layer_dims=label_encoder_dims + (num_filters, ),
                                                             feat_dim=out_feature_dim,
                                                             use_bn=use_bn_in_label_enc,
                                                             use_final_relu=use_final_relu)
    elif label_encoder_type == 'res_ds16_feat_sw_v2':
        label_encoder = seg_label_encoder.ResidualDS16FeatSWv2(layer_dims=label_encoder_dims + (num_filters,),
                                                               feat_dim=out_feature_dim,
                                                               use_bn=use_bn_in_label_enc,
                                                               use_final_relu=use_final_relu)
    elif label_encoder_type == 'res_ds16_feat':
        label_encoder = seg_label_encoder.ResidualDS16Feat(layer_dims=label_encoder_dims + (num_filters, ), feat_dim=out_feature_dim)
    elif label_encoder_type == 'res_ds16_feat_d':
        label_encoder = seg_label_encoder.ResidualDS16FeatDeep1(layer_dims=label_encoder_dims + (num_filters,),
                                                                feat_dim=out_feature_dim)
    elif label_encoder_type == 'res_ds16_feat_norm':
        label_encoder = seg_label_encoder.ResidualDS16Feat(layer_dims=label_encoder_dims + (num_filters, ), feat_dim=out_feature_dim,
                                                           instance_norm=True,
                                                           norm_scale=norm_scale)
    elif label_encoder_type == 'pool_ds16':
        label_encoder = seg_label_encoder.PoolDS16()
    elif label_encoder_type == 'identity':
        label_encoder = seg_label_encoder.Identity()
    elif label_encoder_type == 'res_ds16_feat_sw_att':
        label_encoder = seg_label_encoder.ResidualDS16FeatSWAtt(layer_dims=label_encoder_dims + (num_filters,),
                                                                feat_dim=out_feature_dim)
    elif label_encoder_type == 'res_d16_feat_sw':
        label_encoder = seg_label_encoder.ResidualDS16FeatSW(layer_dims=label_encoder_dims + (num_filters,),
                                                                feat_dim=out_feature_dim)
    elif label_encoder_type == 'identity_sw16':
        label_encoder = seg_label_encoder.IdentityDS16SW(layer_dims=label_encoder_dims + (num_filters, ),
                                                         use_bn=use_bn_in_label_enc)
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

    #if decoder_type == 'dolf_decoder':
    #    decoder = dolf_decoder.RefelixNetwork2(num_filters, decoder_mdim, refinement_layers_channels,
    #                                       new_upsampler=True, use_bn=True)
    #elif decoder_type == 'awesome_decoder':
    #    decoder = awesome_decoder.SegNetwork(num_filters, decoder_mdim, refinement_layers_channels,
    #                                         use_bn=True, skip_rrb=skipl)
    if decoder_type == 'rofl':
        decoder = dolf_decoder.RefelixNetwork2(num_filters, decoder_mdim, refinement_layers_channels,
                                               new_upsampler=True, use_bn=True)
    elif decoder_type == 'rofl_enc':
        assert num_filters == 1
        decoder = dolf_decoder.RefelixNetwork2Enc(num_filters, 32, decoder_mdim, refinement_layers_channels,
                                                  new_upsampler=True, use_bn=True)
    elif decoder_type == 'rofl_att':
        decoder = dolf_decoder.RefelixNetwork2Att(num_filters, decoder_mdim, refinement_layers_channels,
                                                  new_upsampler=True, use_bn=True,
                                                  att_dim=att_inter_dim)
    elif decoder_type == 'stm_maskall':
        decoder = stm_decoder.DecoderResnet50Mask(filter_out_dim=num_filters, mdim=decoder_mdim)
    elif decoder_type == 'awesome_decoder2':
        decoder = a2dec.SegNetwork(num_filters, decoder_mdim, refinement_layers_channels,
                                              use_bn=True, skip_rrb="layer1", final_reduce_factor=1)
    elif decoder_type == 'nodecoder':
        decoder = dolf_decoder.NaiveUpsample()
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

    decoder_channels = decoder.out_feature_channels()
    bbreg_decoder_layer = None
    if bb_regressor_type is None:
        bb_regressor = None
    elif bb_regressor_type == 'pool_fc_l1':
        bb_regressor = seg_aux_layers.BBRegressor(decoder_channels['layer1_dec'], (30, 52), 32, 512)
        bbreg_decoder_layer = 'layer1_dec'
    elif bb_regressor_type == 'pool_fc_l3':
        bb_regressor = seg_aux_layers.BBRegressor(decoder_channels['layer3_dec'], (30, 52), 32, 512)
        bbreg_decoder_layer = 'layer3_dec'
    elif bb_regressor_type == 'fc_l3':
        bb_regressor = seg_aux_layers.BBRegressorFC(decoder_channels['layer3_dec'])
        bbreg_decoder_layer = 'layer3_dec'
    else:
        raise Exception

    net = SegDolfTracker(feature_extractor=backbone_net, classifier=classifier, decoder=decoder,
                         label_encoder=label_encoder,
                         bb_regressor=bb_regressor,
                         classification_layer=classification_layer, refinement_layers=refinement_layers,
                         aux_layers=aux_layers,
                         bbreg_decoder_layer=bbreg_decoder_layer)
    return net

@model_constructor
def steepest_descent_resnet50_finetune(net):
    # backbone


    net = SegDolfTracker(feature_extractor=net.feature_extractor, classifier=net.classifier, decoder=net.decoder,
                         label_encoder=net.label_encoder,
                         classification_layer=net.classification_layer, refinement_layers=net.refinement_layers,
                         aux_layers=net.aux_layers,
                         bb_regressor=net.bb_regressor,
                         bbreg_decoder_layer=net.bbreg_decoder_layer)
    return net

@model_constructor
def steepest_descent_resnet101(filter_size=1, num_filters=1, optim_iter=3, optim_init_reg=0.01,
                              backbone_pretrained=False, clf_feat_blocks=1,
                              clf_feat_norm=True, final_conv=False,
                              out_feature_dim=256,
                              classification_layer='layer3',
                              refinement_layers = ("layer4", "layer3", "layer2", "layer1",),
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
                              decoder_type = 'dolf_decoder',
                              clf_feature_extractor_type = 'PPM'):
    # backbone
    backbone_net = backbones.resnet101(pretrained=backbone_pretrained, frozen_layers=frozen_backbone_layers)

    norm_scale = math.sqrt(1.0 / (out_feature_dim * filter_size * filter_size))

    # classifier
    layer_channels = backbone_net.out_feature_channels()

    if clf_feature_extractor_type == 'PPM':
        clf_feature_extractor = seg_features.SegBlockPPM_keepsize(feature_dim=layer_channels[classification_layer],
                                                              num_blocks=clf_feat_blocks, l2norm=clf_feat_norm,
                                                              final_conv=final_conv, norm_scale=norm_scale,
                                                              out_dim=out_feature_dim,
                                                              use_res_block=ppm_use_res_block)
    elif clf_feature_extractor_type == 'simple':
        clf_feature_extractor = seg_features.SegBlockSimple(feature_dim=layer_channels[classification_layer],
                                          norm_scale=norm_scale,
                                          out_dim=out_feature_dim)
    elif clf_feature_extractor_type == 'boring':
        clf_feature_extractor = seg_features.SegBlockBoring(feature_dim=layer_channels[classification_layer],
                                                            norm_scale=norm_scale,
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
    elif label_encoder_type == 'res_ds16_feat_sw':
        label_encoder = seg_label_encoder.ResidualDS16FeatSW(layer_dims=label_encoder_dims + (num_filters, ), feat_dim=out_feature_dim)
    elif label_encoder_type == 'pool_ds16':
        label_encoder = seg_label_encoder.PoolDS16()
    elif label_encoder_type == 'identity':
        label_encoder = seg_label_encoder.Identity()
    elif label_encoder_type == 'ResLabelGeneratorLabelConv':
        label_encoder = seg_label_encoder.ResLabelGeneratorLabelConv(layer_dims=label_encoder_dims + (num_filters, ), use_bn=use_bn_in_label_enc)
    elif label_encoder_type == 'res_ds16_feat_sw_att':
        label_encoder = seg_label_encoder.ResidualDS16FeatSWAtt(layer_dims=label_encoder_dims + (num_filters, ), feat_dim=out_feature_dim)
    else:
        raise Exception

    residual_module = loss_residual_modules.LinearFilterSeg(init_filter_reg=optim_init_reg,
                                                            upsample_residuals=upsample_residuals,
                                                            score_act=residual_activation_fn)

    optimizer = steepestdescent.GNSteepestDescent(residual_module=residual_module, num_iter=optim_iter, detach_length=detach_length,
                                                  residual_batch_dim=1, compute_losses=True)

    classifier = target_clf.LinearFilter(filter_size=filter_size, filter_initializer=initializer,
                                         filter_optimizer=optimizer, feature_extractor=clf_feature_extractor)

    refinement_layers_channels = {L: layer_channels[L] for L in refinement_layers}

    if decoder_type == 'dolf_decoder':
        decoder = dolf_decoder.RefelixNetwork2(num_filters, decoder_mdim, refinement_layers_channels,
                                           new_upsampler=True, use_bn=True)
    elif decoder_type == 'awesome_decoder':
        decoder = awesome_decoder.SegNetwork(num_filters, decoder_mdim, refinement_layers_channels,
                                             use_bn=True)
    else:
        raise Exception

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

    net = SegDolfTracker(feature_extractor=backbone_net, classifier=classifier, decoder=decoder,
                         label_encoder=label_encoder,
                         classification_layer=classification_layer, refinement_layers=refinement_layers,
                         aux_layers=aux_layers,
                         bb_regressor=bb_regressor,
                         bbreg_decoder_layer=bbreg_decoder_layer)
    return net
