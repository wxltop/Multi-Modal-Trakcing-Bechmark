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


class SegBoxDolfTracker(nn.Module):
    def __init__(self, feature_extractor, classifier, decoder, classification_layer, refinement_layers,
                 label_encoder=None, aux_layers=None, bb_regressor=None,
                 bbreg_decoder_layer=None, box_label_encoder=None, train_only_box_label_gen=False):
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
        self.box_label_encoder = box_label_encoder

        if aux_layers is None:
            self.aux_layers = nn.ModuleDict()
        else:
            self.aux_layers = aux_layers

        self.bb_regressor = bb_regressor
        self.bbreg_decoder_layer = bbreg_decoder_layer
        self.train_only_box_label_gen = train_only_box_label_gen

    def train(self, mode=True):

        for x in self.feature_extractor.parameters():
            x.requires_grad_(False)
        self.feature_extractor.eval()

        if mode:
            for x in self.box_label_encoder.parameters():
                x.requires_grad_(True)
            self.box_label_encoder.train()

            if self.train_only_box_label_gen:
                for x in self.classifier.parameters():
                    x.requires_grad_(False)
                self.classifier.eval()
                for x in self.label_encoder.parameters():
                    x.requires_grad_(False)
                self.label_encoder.eval()
                for x in self.decoder.parameters():
                    x.requires_grad_(False)
                self.decoder.eval()
                if not self.bb_regressor is None:
                    for x in self.bb_regressor.parameters():
                        x.requires_grad_(False)
                    self.bb_regressor.eval()
                if not self.bbreg_decoder_layer is None:
                    for x in self.bbreg_decoder_layer.parameters():
                        x.requires_grad_(False)
                    self.bbreg_decoder_layer.eval()


            else:
                for x in self.classifier.parameters():
                    x.requires_grad_(True)
                self.classifier.train()
                for x in self.label_encoder.parameters():
                    x.requires_grad_(True)
                self.label_encoder.train()
                for x in self.decoder.parameters():
                    x.requires_grad_(True)
                self.decoder.train()
                if not self.bb_regressor is None:
                    for x in self.bb_regressor.parameters():
                        x.requires_grad_(True)
                    self.bb_regressor.train()
                if not self.bbreg_decoder_layer is None:
                    for x in self.bbreg_decoder_layer.parameters():
                        x.requires_grad_(True)
                    self.bbreg_decoder_layer.train()
        else:
            for x in self.classifier.parameters():
                x.requires_grad_(False)
            self.classifier.eval()
            for x in self.label_encoder.parameters():
                x.requires_grad_(False)
            self.label_encoder.eval()
            for x in self.decoder.parameters():
                x.requires_grad_(False)
            self.decoder.eval()
            if not self.bb_regressor is None:
                for x in self.bb_regressor.parameters():
                    x.requires_grad_(False)
                self.bb_regressor.eval()
            if not self.bbreg_decoder_layer is None:
                for x in self.bbreg_decoder_layer.parameters():
                    x.requires_grad_(False)
                self.bbreg_decoder_layer.eval()

            for x in self.box_label_encoder.parameters():
                x.requires_grad_(False)
            self.box_label_encoder.eval()

    def forward(self, train_imgs, test_imgs, train_masks, test_masks, bb_train):
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

        if self.box_label_encoder is not None:
            bb_mask_enc = self.box_label_encoder(bb_train, train_feat_clf)
        else:
            mask_enc = train_masks.contiguous()
            mask_enc_test = None

        box_mask_pred, decoder_feat = self.decoder(bb_mask_enc, test_feat, test_imgs.shape[-2:],
                                               ('layer4_dec', 'layer3_dec', 'layer2_dec', 'layer1_dec'))

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
            aux_mask_pred[L] = predictor(decoder_feat[L], test_imgs.shape[-2:])

        bb_pred = None
        if self.bb_regressor is not None:
            bb_pred = self.bb_regressor(decoder_feat[self.bbreg_decoder_layer])

        if isinstance(mask_enc_test, (tuple, list)):
            mask_enc_test = mask_enc_test[0]
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
        # Output is 1, 1, h, w
        return mask_pred, bb_pred, None

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
                              box_label_encoder_dims=(1,1),
                              frozen_backbone_layers=(),
                              label_encoder_type='identity',
                              decoder_mdim=64, filter_groups=1,
                              upsample_residuals=True,
                              ppm_use_res_block=False,
                              use_bn_in_label_enc=True,
                              residual_activation_fn=None,
                              aux_mask_loss_layers=(),
                              bb_regressor_type=None,
                              box_label_encoder_type='ResLabelGeneratorLabelConvBox'):
    # backbone
    backbone_net = backbones.resnet18(pretrained=backbone_pretrained, frozen_layers=frozen_backbone_layers)

    norm_scale = math.sqrt(1.0 / (out_feature_dim * filter_size * filter_size))

    # classifier
    layer_channels = backbone_net.out_feature_channels()
    clf_feature_extractor = seg_features.SegBlockPPM_keepsize(feature_dim=layer_channels[classification_layer],
                                                              num_blocks=clf_feat_blocks, l2norm=clf_feat_norm,
                                                              final_conv=final_conv, norm_scale=norm_scale,
                                                              out_dim=out_feature_dim,
                                                              use_res_block=ppm_use_res_block)
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
    else:
        raise Exception

    if box_label_encoder_type == 'ResLabelGeneratorLabelConvBox':
        box_label_encoder = seg_label_encoder.ResLabelGeneratorLabelConvBox(layer_dims=box_label_encoder_dims + (num_filters, ), use_bn=use_bn_in_label_enc)
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

    decoder = dolf_decoder.RefelixNetwork2(num_filters, decoder_mdim, refinement_layers_channels,
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

    net = SegBoxDolfTracker(feature_extractor=backbone_net, classifier=classifier, decoder=decoder,
                         label_encoder=label_encoder,
                         classification_layer=classification_layer, refinement_layers=refinement_layers,
                         aux_layers=aux_layers,
                         bb_regressor=bb_regressor,
                         bbreg_decoder_layer=bbreg_decoder_layer,
                         box_label_encoder=box_label_encoder)
    return net

@model_constructor
def steepest_descent_resnet18_from_checkpoint(net=None, filter_size=1, num_filters=1, optim_iter=3, optim_init_reg=0.01,
                              backbone_pretrained=False, clf_feat_blocks=1,
                              clf_feat_norm=True, final_conv=False,
                              out_feature_dim=256,
                              classification_layer='layer3',
                              label_encoder_dims=(1, 1),
                              box_label_encoder_dims=(1,1),
                              decoder_mdim=64, filter_groups=1,
                              use_bn_in_label_enc=True,
                              box_label_encoder_type='ResLabelGeneratorLabelConvBox',
                              use_gauss = False,
                              train_only_box_label_gen=True):


    norm_scale = math.sqrt(1.0 / (out_feature_dim * filter_size * filter_size))


    if box_label_encoder_type == 'ResLabelGeneratorLabelConvBox':
        box_label_encoder = seg_label_encoder.ResLabelGeneratorLabelConvBox(layer_dims=box_label_encoder_dims + (num_filters, ),
                                                                            use_bn=use_bn_in_label_enc, use_gauss=use_gauss)
    else:
        raise Exception

    net = SegBoxDolfTracker(feature_extractor=net.feature_extractor, classifier=net.classifier, decoder=net.decoder,
                         label_encoder=net.label_encoder,
                         classification_layer=classification_layer, refinement_layers=net.refinement_layers,
                         bb_regressor=net.bb_regressor,
                         bbreg_decoder_layer=net.bbreg_decoder_layer,
                         box_label_encoder=box_label_encoder, train_only_box_label_gen=train_only_box_label_gen)
    return net

@model_constructor
def steepest_descent_resnet50_from_checkpoint(net=None, filter_size=1, num_filters=1, optim_iter=3, optim_init_reg=0.01,
                              backbone_pretrained=False, clf_feat_blocks=1,
                              clf_feat_norm=True, final_conv=False,
                              out_feature_dim=512,
                              classification_layer='layer3',
                              label_encoder_dims=(1, 1),
                              box_label_encoder_dims=(1,1),
                              decoder_mdim=64, filter_groups=1,
                              use_bn_in_label_enc=True,
                              box_label_encoder_type='ResLabelGeneratorLabelConvBox',
                              use_gauss = False,
                              train_only_box_label_gen=True,
                              use_final_relu=True,
                              non_default_init=True,
                              init_bn=1,
                              gauss_scale=0.25):


    norm_scale = math.sqrt(1.0 / (out_feature_dim * filter_size * filter_size))

    if box_label_encoder_type == 'ResLabelGeneratorLabelConvBox':
        box_label_encoder = seg_label_encoder.ResLabelGeneratorLabelConvBox(layer_dims=box_label_encoder_dims + (num_filters, ),
                                                                                                                                           use_bn=use_bn_in_label_enc, use_gauss=use_gauss)
    elif box_label_encoder_type == 'ResidualDS16FeatSWBoxAtt':
        box_label_encoder = seg_label_encoder.ResidualDS16FeatSWBoxAtt(feat_dim=out_feature_dim, layer_dims=box_label_encoder_dims + (num_filters,),
                                                                            use_gauss=use_gauss)
    elif box_label_encoder_type == 'ResidualDS16FeatSWBoxAttMultiBlock':
        box_label_encoder = seg_label_encoder.ResidualDS16FeatSWBoxAttMultiBlock(feat_dim=out_feature_dim,
                                                                       layer_dims=box_label_encoder_dims + (
                                                                       num_filters,),
                                                                       use_gauss=use_gauss)
    elif box_label_encoder_type == 'ResidualDS16FeatSWBoxCat':
        box_label_encoder = seg_label_encoder.ResidualDS16FeatSWBoxCat(feat_dim=out_feature_dim, layer_dims=box_label_encoder_dims + (num_filters,),
                                                                            use_gauss=use_gauss)
    elif box_label_encoder_type == 'ResidualDS16FeatSWBoxCatMultiBlock':
        box_label_encoder = seg_label_encoder.ResidualDS16FeatSWBoxCatMultiBlock(feat_dim=out_feature_dim, layer_dims=box_label_encoder_dims + (num_filters,),
                                                                            use_gauss=use_gauss, use_final_relu=use_final_relu, use_bn=use_bn_in_label_enc,
                                                                                 non_default_init=True, init_bn=init_bn, gauss_scale=gauss_scale)
    elif box_label_encoder_type == 'ResidualDS16FeatSWBoxCat2':
        box_label_encoder = seg_label_encoder.ResidualDS16FeatSWBoxCat2(feat_dim=out_feature_dim,
                                                                       layer_dims=box_label_encoder_dims + (
                                                                       num_filters,),
                                                                       use_gauss=use_gauss)
    elif box_label_encoder_type == 'ResidualDS16FeatSWBox':
        box_label_encoder = seg_label_encoder.ResidualDS16FeatSWBox(feat_dim=out_feature_dim,
            layer_dims=box_label_encoder_dims + (num_filters,),
            use_gauss=use_gauss)
    elif box_label_encoder_type == 'ResidualDS16FeatSWBoxAtt2':
        box_label_encoder = seg_label_encoder.ResidualDS16FeatSWBoxAtt2(feat_dim=out_feature_dim, layer_dims=box_label_encoder_dims + (num_filters,),
                                                                            use_gauss=use_gauss)
    else:
        raise Exception

    net = SegBoxDolfTracker(feature_extractor=net.feature_extractor, classifier=net.classifier, decoder=net.decoder,
                         label_encoder=net.label_encoder,
                         classification_layer=classification_layer, refinement_layers=net.refinement_layers,
                         bb_regressor=net.bb_regressor,
                         bbreg_decoder_layer=net.bbreg_decoder_layer,
                         box_label_encoder=box_label_encoder, train_only_box_label_gen=train_only_box_label_gen)
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
                              use_bn_in_label_enc=True):
    # backbone
    backbone_net = backbones.resnet50(pretrained=backbone_pretrained, frozen_layers=frozen_backbone_layers)

    norm_scale = math.sqrt(1.0 / (out_feature_dim * filter_size * filter_size))

    # classifier
    layer_channels = backbone_net.out_feature_channels()
    clf_feature_extractor = seg_features.SegBlockPPM_keepsize(feature_dim=layer_channels[classification_layer],
                                                              l2norm=clf_feat_norm,
                                                              norm_scale=norm_scale,
                                                              out_dim=out_feature_dim,
                                                              use_res_block=ppm_use_res_block)
    initializer = seg_initializer.FilterInitializerZero(filter_size=filter_size, num_filters=num_filters,
                                                        feature_dim=out_feature_dim, filter_groups=filter_groups)

    if label_encoder_type == 'res_ds16':
        label_encoder = seg_label_encoder.ResidualDS16(layer_dims=label_encoder_dims + (num_filters, ))
    elif label_encoder_type == 'res_ds16_sw':
        label_encoder = seg_label_encoder.ResidualDS16SW(layer_dims=label_encoder_dims + (num_filters, ), use_bn=use_bn_in_label_enc)
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
    else:
        raise Exception

    residual_module = loss_residual_modules.LinearFilterSeg(init_filter_reg=optim_init_reg, upsample_residuals=upsample_residuals)

    optimizer = steepestdescent.GNSteepestDescent(residual_module=residual_module, num_iter=optim_iter, detach_length=detach_length,
                                                  residual_batch_dim=1, compute_losses=True)

    classifier = target_clf.LinearFilter(filter_size=filter_size, filter_initializer=initializer,
                                         filter_optimizer=optimizer, feature_extractor=clf_feature_extractor)

    refinement_layers_channels = {L: layer_channels[L] for L in refinement_layers}

    decoder = dolf_decoder.RefelixNetwork2(num_filters, decoder_mdim, refinement_layers_channels,
                                           new_upsampler=True, use_bn=True)

    if len(aux_mask_loss_layers) > 0:
        aux_layers = nn.ModuleDict()
        decoder_channels = decoder.out_feature_channels()
        decoder_channels['mask_enc'] = num_filters
        for l in aux_mask_loss_layers:
            aux_layers[l] = seg_aux_layers.ConvPredictor(decoder_channels[l])
    else:
        aux_layers = None
    net = SegBoxDolfTracker(feature_extractor=backbone_net, classifier=classifier, decoder=decoder,
                         label_encoder=label_encoder,
                         classification_layer=classification_layer, refinement_layers=refinement_layers,
                         aux_layers=aux_layers)
    return net
