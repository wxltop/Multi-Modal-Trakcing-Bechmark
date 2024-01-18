import math
import torch
import torch.nn as nn
from collections import OrderedDict
import ltr.models.segmentation.linear_filter as target_clf
import ltr.models.target_classifier.features as clf_features
import ltr.models.segmentation.initializer as seg_initializer
import ltr.models.segmentation.label_encoder as seg_label_encoder
import ltr.models.segmentation.loss_residual_modules as loss_residual_modules
import ltr.models.segmentation.dolf_decoder as dolf_decoder
import ltr.models.backbone as backbones
import ltr.models.segmentation.box_predictor as box_predictors
import ltr.models.meta.steepestdescent as steepestdescent
import ltr.models.segmentation.aux_layers as seg_aux_layers
from ltr import model_constructor
from pytracking import TensorList


class SegBoxTracker(nn.Module):
    def __init__(self, feature_extractor, classifier, decoder, classification_layer, refinement_layers,
                 label_encoder, predictor, aux_layers=None):
        super().__init__()

        self.feature_extractor = feature_extractor
        self.classifier = classifier
        self.decoder = decoder

        # self.output_layers = ['layer1', 'layer2', 'layer3']

        self.classification_layer = (classification_layer,) if isinstance(classification_layer,
                                                                         str) else classification_layer
        self.refinement_layers = refinement_layers
        self.output_layers = sorted(list(set(self.classification_layer + self.refinement_layers)))

        self.label_encoder = label_encoder
        self.predictor = predictor

        if aux_layers is None:
            self.aux_layers = nn.ModuleDict()
        else:
            self.aux_layers = aux_layers

    def forward(self, train_imgs, test_imgs, train_bbox, num_refinement_iter=2, *args, **kwargs):
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

        train_mask_enc, train_mask_sw = self.label_encoder(train_bbox, train_feat_clf, list(train_imgs.shape[-2:]))

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
            mask_enc, decoder_feat = self.decoder(target_scores_last_iter, test_feat_it, test_imgs.shape[-2:])
            mask_enc = mask_enc.view(1, num_sequences, -1, *mask_enc.shape[-2:])

            prediction = self.predictor(mask_enc)
            output = {'bbox': prediction['bbox'], 'center_score': prediction['center_score'],
                      'mask': prediction['mask']}

            pred_all.append(output)
        return pred_all

    def segment_target(self, target_filter, test_feat_clf, test_feat):
        raise NotImplementedError

    def get_backbone_clf_feat(self, backbone_feat):
        feat = OrderedDict({l: backbone_feat[l] for l in self.classification_layer})
        if len(self.classification_layer) == 1:
            return feat[self.classification_layer[0]]
        return feat

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

    def convert_offsets_to_bbox(self, offset_pred):

        pass


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
                              decoder_mdim=64, filter_groups=1,
                              upsample_residuals=True,
                              residual_activation_fn=None,
                              use_decoder_backbone_feat=True,
                              predictor_use_bn=True,
                              predictor_inter_dim=16,
                              predictor_patch_sz=5,
                              ):
    # backbone
    backbone_net = backbones.resnet18(pretrained=backbone_pretrained, frozen_layers=frozen_backbone_layers)

    norm_scale = math.sqrt(1.0 / (out_feature_dim * filter_size * filter_size))

    # classifier
    layer_channels = backbone_net.out_feature_channels()

    clf_feature_extractor = clf_features.residual_basic_block(feature_dim=layer_channels[classification_layer],
                                                              num_blocks=clf_feat_blocks, l2norm=clf_feat_norm,
                                                              final_conv=final_conv, norm_scale=norm_scale,
                                                              out_dim=out_feature_dim)

    initializer = seg_initializer.FilterInitializerZero(filter_size=filter_size, num_filters=num_filters,
                                                        feature_dim=out_feature_dim, filter_groups=filter_groups)

    label_encoder = seg_label_encoder.ResidualDS16FeatSWBox(layer_dims=label_encoder_dims + (num_filters, ),
                                                            feat_dim=out_feature_dim, use_final_relu=True,
                                                            use_gauss=False)

    residual_module = loss_residual_modules.LinearFilterSeg(init_filter_reg=optim_init_reg,
                                                            upsample_residuals=upsample_residuals,
                                                            score_act=residual_activation_fn)

    optimizer = steepestdescent.GNSteepestDescent(residual_module=residual_module, num_iter=optim_iter, detach_length=detach_length,
                                                  residual_batch_dim=1, compute_losses=True)

    classifier = target_clf.LinearFilter(filter_size=filter_size, filter_initializer=initializer,
                                         filter_optimizer=optimizer, feature_extractor=clf_feature_extractor)

    refinement_layers_channels = {L: layer_channels[L] for L in refinement_layers}

    decoder = dolf_decoder.RefelixDecoder(num_filters, decoder_mdim, refinement_layers_channels,
                                          use_bn=True,
                                          use_backbone_feat=use_decoder_backbone_feat)

    # TODO hack
    predictor = box_predictors.Predictorv1(decoder_mdim*2, predictor_inter_dim, predictor_patch_sz,
                                           use_bn=predictor_use_bn)
    net = SegBoxTracker(feature_extractor=backbone_net, classifier=classifier, decoder=decoder,
                        label_encoder=label_encoder, predictor=predictor,
                        classification_layer=classification_layer, refinement_layers=refinement_layers)
    return net
