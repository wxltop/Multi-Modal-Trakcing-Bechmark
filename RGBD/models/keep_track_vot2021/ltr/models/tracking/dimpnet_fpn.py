import math
import torch
import torch.nn as nn
from collections import OrderedDict
from ltr.models.meta import steepestdescent
import ltr.models.target_classifier.linear_filter as target_clf
import ltr.models.target_classifier.features as clf_features
import ltr.models.target_classifier.initializer as clf_initializer
import ltr.models.target_classifier.optimizer as clf_optimizer
import ltr.models.target_classifier.probabilistic as clf_probabilistic
import ltr.models.bbreg as bbmodels
import ltr.models.backbone as backbones
import ltr.models.bbreg.fpn_iou_net as fpn_iounet
from ltr import model_constructor


class DiMPnet(nn.Module):
    """The DiMP network.
    args:
        feature_extractor:  Backbone feature extractor network. Must return a dict of feature maps
        classifier:  Target classification module.
        bb_regressor:  Bounding box regression module.
        classification_layer:  Name of the backbone feature layer to use for classification.
        bb_regressor_layer:  Names of the backbone layers to use for bounding box regression."""

    def __init__(self, feature_extractor, classifier, bb_regressor, feature_layers):
        super().__init__()

        self.feature_extractor = feature_extractor
        self.classifier = classifier
        self.bb_regressor = bb_regressor
        self.feature_layers = feature_layers

        self.output_layers = sorted([d['name'] for d in feature_layers])

    def forward(self, train_imgs, test_imgs, train_bb, test_proposals, feature_levels, *args, **kwargs):
        raise NotImplementedError

    def get_fpn_feature(self, backbone_feat, feat_level):
        feat = backbone_feat[self.feature_layers[feat_level]['name']]
        return feat

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
            layers = self.bb_regressor_layer + ['classification']
        if 'classification' not in layers:
            return self.feature_extractor(im, layers)
        backbone_layers = sorted(list(set([l for l in layers + self.classification_layer if l != 'classification'])))
        all_feat = self.feature_extractor(im, backbone_layers)
        all_feat['classification'] = self.extract_classification_feat(all_feat)
        return OrderedDict({l: all_feat[l] for l in layers})


@model_constructor
def dimpnet50_fpn(feature_layers, filter_size=1, optim_iter=5, optim_init_step=1.0, optim_init_reg=0.01,
                  backbone_pretrained=True, clf_num_blocks=1,
                  clf_feat_dim=256, clf_final_conv=True,
                  clf_feat_norm=True, init_filter_norm=False,
                  init_gauss_sigma=1.0, num_dist_bins=5, bin_displacement=1.0,
                  mask_init_factor=4.0,
                  score_act='relu', act_param=None, target_mask_act='sigmoid',
                  detach_length=float('Inf'), frozen_backbone_layers=()):
    # Backbone
    backbone_net = backbones.fpn.resnet50_fpn(pretrained_resnet=backbone_pretrained, frozen_layers=frozen_backbone_layers)

    fpn_feature_dim = 256

    # Feature normalization
    norm_scale = math.sqrt(1.0 / (clf_feat_dim * filter_size * filter_size))

    # Classifier features
    clf_feature_extractor = clf_features.residual_bottleneck(feature_dim=fpn_feature_dim // 4, out_dim=clf_feat_dim,
                                                             num_blocks=clf_num_blocks, l2norm=clf_feat_norm,
                                                             final_conv=clf_final_conv, norm_scale=norm_scale)

    # Initializer for the DiMP classifier
    initializer = clf_initializer.FilterInitializerLinear(filter_size=filter_size, filter_norm=init_filter_norm,
                                                          feature_dim=clf_feat_dim)

    # Optimizer for the DiMP classifier
    # Handle stride outside
    optimizer = clf_optimizer.DiMPSteepestDescentGN(num_iter=optim_iter, feat_stride=1,
                                                    init_step_length=optim_init_step,
                                                    init_filter_reg=optim_init_reg, init_gauss_sigma=init_gauss_sigma,
                                                    num_dist_bins=num_dist_bins,
                                                    bin_displacement=bin_displacement,
                                                    mask_init_factor=mask_init_factor,
                                                    score_act=score_act, act_param=act_param, mask_act=target_mask_act,
                                                    detach_length=detach_length)

    # The classifier module
    classifier = target_clf.LinearFilter(filter_size=filter_size, filter_initializer=initializer,
                                         filter_optimizer=optimizer, feature_extractor=clf_feature_extractor)

    # Handle stride outside
    bb_regressor = fpn_iounet.FPNIoUNetHR(pool_stride=1)

    # DiMP network
    net = DiMPnet(feature_extractor=backbone_net, classifier=classifier, bb_regressor=bb_regressor,
                  feature_layers=feature_layers)
    return net
