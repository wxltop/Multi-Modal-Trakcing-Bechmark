import math
import torch
import torch.nn as nn
from collections import OrderedDict
import ltr.models.segmentation.linear_filter as target_clf
import ltr.models.target_classifier.linear_filter as trk_target_clf
import ltr.models.target_classifier.features as clf_features
import ltr.models.segmentation.initializer as seg_initializer_pkg
import ltr.models.target_classifier.initializer as trk_initializer
import ltr.models.segmentation.label_encoder as seg_label_encoder
import ltr.models.segmentation.loss_residual_modules as loss_residual_modules
import ltr.models.target_classifier.residual_modules as trk_loss_residual_modules
import ltr.models.segmentation.stm_decoder as stm_decoder
import ltr.models.backbone as backbones
import ltr.models.meta.steepestdescent as steepestdescent
import ltr.models.segmentation.bbreg as bbreg_models
from ltr import model_constructor


class SegTracker(nn.Module):
    def __init__(self, feature_extractor, target_classifier, seg_classifier, label_encoder,
                 mask_decoder, bb_regressor):
        super().__init__()

        self.feature_extractor = feature_extractor
        self.target_classifier = target_classifier
        self.seg_classifier = seg_classifier
        self.mask_decoder = mask_decoder
        self.bb_regressor = bb_regressor

        self.output_layers = ['layer1', 'layer2', 'layer3']
        self.classification_layer = ['layer3']
        self.label_encoder = label_encoder

    def forward(self, train_imgs, test_imgs, train_masks, train_bb_crop=None, train_bb_search_area=None,
                train_search_area_bb=None,
                test_search_area_bb=None,
                test_proposals=None):
        assert train_imgs.dim() == 5 and test_imgs.dim() == 5, 'Expect 5 dimensional inputs'
        assert train_masks.dim() == 4, 'Expect 4 dimensional masks'

        # Extract backbone features
        train_feat = self.extract_backbone_features(train_imgs.view(-1, train_imgs.shape[-3], train_imgs.shape[-2], train_imgs.shape[-1]))
        test_feat = self.extract_backbone_features(test_imgs.view(-1, test_imgs.shape[-3], test_imgs.shape[-2], test_imgs.shape[-1]))

        # Classification features
        train_feat_clf = self.get_backbone_clf_feat(train_feat)
        test_feat_clf = self.get_backbone_clf_feat(test_feat)

        # run seg classifier
        train_mask_encoding = self.encode_masks(train_masks)
        train_mask_encoding = train_mask_encoding.view(train_imgs.shape[0], train_imgs.shape[1], *train_mask_encoding.shape[-3:])
        mask_encoding = self.seg_classifier(train_feat_clf, test_feat_clf, train_mask_encoding)[-1]

        target_scores_sa_all, target_scores_crop = self.target_classifier(train_feat_clf, test_feat_clf, train_bb_search_area, train_search_area_bb,
                                                                          test_search_area_bb)

        decoder_in = torch.cat((mask_encoding, target_scores_crop.unsqueeze(2)), dim=2)
        mask_pred, decoder_layers = self.mask_decoder(decoder_in, test_feat, ('m3', ))

        if self.bb_regressor is not None:
            # Get bb_regressor features
            train_feat_bbreg = [train_feat['layer3'], ]
            test_feat_bbreg = [decoder_layers['m3'], ]

            # Run the IoUNet module
            bb_delta = self.bb_regressor(train_feat_bbreg, test_feat_bbreg, train_bb_crop, test_proposals)
        else:
            bb_delta = None

        return mask_pred, target_scores_sa_all, bb_delta

    def encode_masks(self, masks):
        return self.label_encoder(masks)

    def segment_target(self, target_filter, test_feat_clf):
        target_scores = self.seg_classifier.classify(target_filter, test_feat_clf)

        # TODO fix dimensions
        target_scores_last_iter = target_scores[-1].unsqueeze(0)
        return target_scores_last_iter
        # mask_pred = self.mask_decoder(target_scores_last_iter, test_feat)
        # return mask_pred

    def get_backbone_clf_feat(self, backbone_feat):
        feat = OrderedDict({l: backbone_feat[l] for l in self.classification_layer})
        if len(self.classification_layer) == 1:
            return feat[self.classification_layer[0]]
        return feat

    def get_backbone_bbreg_feat(self, backbone_feat):
        return [backbone_feat[l] for l in ['layer3', ]]

    def extract_seg_classification_feat(self, backbone_feat):
        return self.seg_classifier.extract_classification_feat(self.get_backbone_clf_feat(backbone_feat))

    def extract_target_classification_feat(self, backbone_feat, search_area_bb):
        return self.target_classifier.extract_classification_feat(self.get_backbone_clf_feat(backbone_feat), search_area_bb)

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
def steepest_descent_resnet18(backbone_pretrained=False,
                              seg_filter_size=3, seg_num_filters=1, seg_optim_iter=5, seg_optim_init_reg=0.1,
                              seg_feat_blocks=1,
                              seg_feat_norm=True, seg_final_conv=True,
                              seg_out_feature_dim=256,
                              seg_detach_length=float('Inf'),
                              label_encoder_dims=(1, 1),
                              frozen_backbone_layers=(),
                              label_encoder_type='res_ds16',
                              decoder_mdim=64,
                              target_cls_optim_iter=5,
                              target_cls_optim_init_reg=0.1,
                              target_cls_detach_length=float('Inf'),
                              target_cls_init_filter_norm=False,
                              target_cls_feat_blocks=1,
                              target_cls_feat_norm=True,
                              target_cls_final_conv=True,
                              target_cls_out_feature_dim=256,
                              target_cls_filter_size=3,
                              target_cls_feat_pool_size=None,
                              init_gauss_sigma=None,
                              num_dist_bins=100,
                              bin_displacement=0.1,
                              mask_init_factor=3.0,
                              score_act='relu',
                              act_param=None,
                              target_mask_act='sigmoid',
                              bbr_type=None):
    # ********************* Feature extractor ***********************************
    # backbone
    backbone_net = backbones.resnet18(pretrained=backbone_pretrained, frozen_layers=frozen_backbone_layers)

    # additional head
    seg_norm_scale = math.sqrt(1.0 / (seg_out_feature_dim * seg_filter_size * seg_filter_size))
    seg_feature_extractor = clf_features.residual_basic_block(num_blocks=seg_feat_blocks, l2norm=seg_feat_norm,
                                                              final_conv=seg_final_conv, norm_scale=seg_norm_scale,
                                                              out_dim=seg_out_feature_dim)

    # ********************* Mask representation predictor ***********************************
    if label_encoder_type == 'res_ds16':
        label_encoder = seg_label_encoder.ResidualDS16(layer_dims=label_encoder_dims + (seg_num_filters, ))
    elif label_encoder_type == 'res_ds16_feat':
        label_encoder = seg_label_encoder.ResidualDS16Feat(layer_dims=label_encoder_dims + (seg_num_filters, ), feat_dim=seg_out_feature_dim)
    elif label_encoder_type == 'pool_ds16':
        label_encoder = seg_label_encoder.PoolDS16()
    else:
        raise Exception

    seg_initializer = seg_initializer_pkg.FilterInitializerZero(filter_size=seg_filter_size,
                                                                num_filters=seg_num_filters,
                                                                feature_dim=seg_out_feature_dim)

    seg_residual_module = loss_residual_modules.LinearFilterSeg(init_filter_reg=seg_optim_init_reg)

    seg_optimizer = steepestdescent.GNSteepestDescent(residual_module=seg_residual_module,
                                                      num_iter=seg_optim_iter, detach_length=seg_detach_length,
                                                      residual_batch_dim=1,
                                                      compute_losses=True)

    seg_classifier = target_clf.LinearFilter(filter_size=seg_filter_size, filter_initializer=seg_initializer,
                                             filter_optimizer=seg_optimizer, feature_extractor=seg_feature_extractor)

    # ********************* Target classifier ***********************************
    target_cls_norm_scale = math.sqrt(1.0 / (target_cls_out_feature_dim * target_cls_filter_size * target_cls_filter_size))
    target_cls_feature_extractor = clf_features.residual_basic_block(num_blocks=target_cls_feat_blocks,
                                                                     l2norm=target_cls_feat_norm,
                                                                     final_conv=target_cls_final_conv,
                                                                     norm_scale=target_cls_norm_scale,
                                                                     out_dim=target_cls_out_feature_dim)

    target_cls_initializer = trk_initializer.FilterInitializerLinear(filter_size=target_cls_filter_size,
                                                                     filter_norm=target_cls_init_filter_norm,
                                                                     feature_dim=target_cls_out_feature_dim)

    target_cls_residual_module = trk_loss_residual_modules.LinearFilterLearnGen(feat_stride=1,
                                                                                init_filter_reg=target_cls_optim_init_reg,
                                                                                init_gauss_sigma=init_gauss_sigma,
                                                                                num_dist_bins=num_dist_bins,
                                                                                bin_displacement=bin_displacement,
                                                                                mask_init_factor=mask_init_factor,
                                                                                score_act=score_act,
                                                                                act_param=act_param,
                                                                                mask_act=target_mask_act)

    target_cls_optimizer = steepestdescent.GNSteepestDescent(residual_module=target_cls_residual_module,
                                                             num_iter=target_cls_optim_iter, detach_length=target_cls_detach_length,
                                                             residual_batch_dim=1, compute_losses=True)

    target_cls_classifier = trk_target_clf.LinearFilterMetaAda(filter_size=target_cls_filter_size,
                                                               filter_initializer=target_cls_initializer,
                                                               filter_optimizer=target_cls_optimizer,
                                                               feature_extractor=target_cls_feature_extractor,
                                                               pool_size=target_cls_feat_pool_size,
                                                               feat_stride=16, pool_type='roi_align')

    # ********************* Mask decoder ***********************************
    decoder = stm_decoder.DecoderResnet18(filter_out_dim=seg_num_filters + 1, mdim=decoder_mdim)

    # ******************** BBReg ****************************************
    if bbr_type == 'bbr_mod':
        bb_regressor = bbreg_models.BBRNet(ref_input_dim=256, ref_feat_stride=16,
                                           ref_pool_sz=3, test_pool_sz=5,
                                           test_input_dim=256, test_feat_stride=8)
    else:
        bb_regressor = None

    net = SegTracker(feature_extractor=backbone_net, target_classifier=target_cls_classifier,
                     seg_classifier=seg_classifier, label_encoder=label_encoder,
                     mask_decoder=decoder, bb_regressor=bb_regressor)
    return net
