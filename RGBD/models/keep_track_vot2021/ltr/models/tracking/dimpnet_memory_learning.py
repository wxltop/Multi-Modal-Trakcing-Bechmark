import math
import torch
import torch.nn as nn
from collections import OrderedDict
from ltr.models.meta import steepestdescent
import ltr.models.target_classifier.fusion_filter as target_fusion_clf
import ltr.models.target_classifier.features as clf_features
import ltr.models.target_classifier.initializer as clf_initializer
import ltr.models.target_classifier.optimizer as clf_optimizer
import ltr.models.target_classifier.probabilistic as clf_probabilistic
from ltr.models.target_classifier import residual_modules
import ltr.models.bbreg as bbmodels
import ltr.models.backbone as backbones
import ltr.models.bbreg.fpn_iou_net as fpn_iounet
import ltr.models.bbreg.fpn_bbr_net as fpn_bbrnet
import ltr.models.backbone.resnet_mrcnn as resnet_mrcnn
from ltr import model_constructor

import ltr.models.memory_learning.attention as fusion_clf


class DiMPMemorynet(nn.Module):
    """The DiMP network.
    args:
        feature_extractor:  Backbone feature extractor network. Must return a dict of feature maps
        classifier:  Target classification module.
        bb_regressor:  Bounding box regression module.
        classification_layer:  Name of the backbone feature layer to use for classification.
        bb_regressor_layer:  Names of the backbone layers to use for bounding box regression."""

    def __init__(self, feature_extractor, classifier, bb_regressor, classification_layer, bb_regressor_layer):
        super().__init__()

        self.feature_extractor = feature_extractor
        self.classifier = classifier
        self.bb_regressor = bb_regressor
        self.classification_layer = [classification_layer] if isinstance(classification_layer, str) else classification_layer
        self.bb_regressor_layer = bb_regressor_layer
        self.output_layers = sorted(list(set(self.classification_layer + self.bb_regressor_layer)))


    def forward(self, train_imgs, test_imgs, train_bb, test_proposals, *args, **kwargs):
        """Runs the DiMP network the way it is applied during training.
        The forward function is ONLY used for training. Call the individual functions during tracking.
        args:
            train_imgs:  Train image samples (images, sequences, 3, H, W).
            test_imgs:  Test image samples (images, sequences, 3, H, W).
            trian_bb:  Target boxes (x,y,w,h) for the train images. Dims (images, sequences, 4).
            test_proposals:  Proposal boxes to use for the IoUNet (bb_regressor) module.
            *args, **kwargs:  These are passed to the classifier module.
        returns:
            test_scores:  Classification scores on the test samples.
            iou_pred:  Predicted IoU scores for the test_proposals."""

        assert train_imgs.dim() == 5 and test_imgs.dim() == 5, 'Expect 5 dimensional inputs'

        # Extract backbone features
        train_feat = self.extract_backbone_features(train_imgs.reshape(-1, *train_imgs.shape[-3:]))
        test_feat = self.extract_backbone_features(test_imgs.reshape(-1, *test_imgs.shape[-3:]))

        # Classification features
        train_feat_clf = self.get_backbone_clf_feat(train_feat)
        test_feat_clf = self.get_backbone_clf_feat(test_feat)

        # Run classifier module
        target_scores = self.classifier(train_feat_clf, test_feat_clf, train_bb, *args, **kwargs)

        # Get bb_regressor features
        train_feat_iou = self.get_backbone_bbreg_feat(train_feat)
        test_feat_iou = self.get_backbone_bbreg_feat(test_feat)

        # Run the IoUNet module
        iou_pred = self.bb_regressor(train_feat_iou, test_feat_iou, train_bb, test_proposals)

        return target_scores, iou_pred

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
def dimpnet50_attention_average_std_scaling(filter_size=1, optim_iter=5, optim_init_reg=0.01,
              classification_layer='layer3', feat_stride=16, backbone_pretrained=True, clf_feat_blocks=0,
              clf_feat_norm=True, init_filter_norm=False, final_conv=True,
              out_feature_dim=512, hinge_threshold=0.05, iou_input_dim=(256, 256), iou_inter_dim=(256, 256),
              activation_leak=0.0, score_act='relu', act_param=None,
              detach_length=float('Inf'), frozen_backbone_layers=()):

    # Backbone
    backbone_net = backbones.resnet50(pretrained=backbone_pretrained, frozen_layers=frozen_backbone_layers)

    # Feature normalization
    norm_scale = math.sqrt(1.0 / (out_feature_dim * filter_size * filter_size))

    # Classifier features
    if classification_layer == 'layer3':
        feature_dim = 256
    elif classification_layer == 'layer4':
        feature_dim = 512
    else:
        raise Exception

    clf_feature_extractor = clf_features.residual_bottleneck(feature_dim=feature_dim,
                                                             num_blocks=clf_feat_blocks, l2norm=clf_feat_norm,
                                                             final_conv=final_conv, norm_scale=norm_scale,
                                                             out_dim=out_feature_dim)

    # Initializer for the DiMP classifier
    initializer = clf_initializer.FilterInitializerLinear(filter_size=filter_size, filter_norm=init_filter_norm,
                                                          feature_dim=out_feature_dim)

    # Residual module that defined the online loss
    residual_module = residual_modules.LinearFilterHinge(feat_stride=feat_stride, init_filter_reg=optim_init_reg,
                                                         hinge_threshold=hinge_threshold, activation_leak=activation_leak,
                                                         score_act=score_act, act_param=act_param)

    # Construct generic optimizer module
    optimizer = steepestdescent.GNSteepestDescent(residual_module=residual_module, num_iter=optim_iter,
                                                  detach_length=detach_length, residual_batch_dim=1, compute_losses=True)

    attention_fusion_module = fusion_clf.AttentionAverageStdScalingModule(softmax_temp_init=50., train_softmax_temp=True)

    # The classifier module
    classifier = target_fusion_clf.LinearAttentionFusionFilter(filter_size=filter_size, filter_initializer=initializer,
                                                               filter_optimizer=optimizer,
                                                               feature_extractor=clf_feature_extractor,
                                                               fusion_module=attention_fusion_module)

    # Bounding box regressor
    bb_regressor = bbmodels.AtomIoUNet(input_dim=(4 * 128, 4 * 256), pred_input_dim=iou_input_dim,
                                       pred_inter_dim=iou_inter_dim)

    # DiMP network
    net = DiMPMemorynet(feature_extractor=backbone_net, classifier=classifier, bb_regressor=bb_regressor,
                        classification_layer=classification_layer, bb_regressor_layer=['layer2', 'layer3'])
    return net


@model_constructor
def dimpnet50_attention_average_std_scaling_rescale(filter_size=1, optim_iter=5, optim_init_reg=0.01,
              classification_layer='layer3', feat_stride=16, backbone_pretrained=True, clf_feat_blocks=0,
              clf_feat_norm=True, init_filter_norm=False, final_conv=True,
              out_feature_dim=512, hinge_threshold=0.05, iou_input_dim=(256, 256), iou_inter_dim=(256, 256),
              activation_leak=0.0, score_act='relu', act_param=None,
              detach_length=float('Inf'), frozen_backbone_layers=()):

    # Backbone
    backbone_net = backbones.resnet50(pretrained=backbone_pretrained, frozen_layers=frozen_backbone_layers)

    # Feature normalization
    norm_scale = math.sqrt(1.0 / (out_feature_dim * filter_size * filter_size))

    # Classifier features
    if classification_layer == 'layer3':
        feature_dim = 256
    elif classification_layer == 'layer4':
        feature_dim = 512
    else:
        raise Exception

    clf_feature_extractor = clf_features.residual_bottleneck(feature_dim=feature_dim,
                                                             num_blocks=clf_feat_blocks, l2norm=clf_feat_norm,
                                                             final_conv=final_conv, norm_scale=norm_scale,
                                                             out_dim=out_feature_dim)

    # Initializer for the DiMP classifier
    initializer = clf_initializer.FilterInitializerLinear(filter_size=filter_size, filter_norm=init_filter_norm,
                                                          feature_dim=out_feature_dim)

    # Residual module that defined the online loss
    residual_module = residual_modules.LinearFilterHinge(feat_stride=feat_stride, init_filter_reg=optim_init_reg,
                                                         hinge_threshold=hinge_threshold, activation_leak=activation_leak,
                                                         score_act=score_act, act_param=act_param)

    # Construct generic optimizer module
    optimizer = steepestdescent.GNSteepestDescent(residual_module=residual_module, num_iter=optim_iter,
                                                  detach_length=detach_length, residual_batch_dim=1, compute_losses=True)

    attention_fusion_module = fusion_clf.AttentionAverageStdScalingRescaleModule(softmax_temp_init=50., train_softmax_temp=False)

    # The classifier module
    classifier = target_fusion_clf.LinearAttentionFusionFilter(filter_size=filter_size, filter_initializer=initializer,
                                                               filter_optimizer=optimizer,
                                                               feature_extractor=clf_feature_extractor,
                                                               fusion_module=attention_fusion_module)

    # Bounding box regressor
    bb_regressor = bbmodels.AtomIoUNet(input_dim=(4 * 128, 4 * 256), pred_input_dim=iou_input_dim,
                                       pred_inter_dim=iou_inter_dim)

    # DiMP network
    net = DiMPMemorynet(feature_extractor=backbone_net, classifier=classifier, bb_regressor=bb_regressor,
                        classification_layer=classification_layer, bb_regressor_layer=['layer2', 'layer3'])
    return net


@model_constructor
def dimpnet50_attention_learn_fusion_direct(filter_size=1, optim_iter=5, optim_init_reg=0.01,
              classification_layer='layer3', feat_stride=16, backbone_pretrained=True, clf_feat_blocks=0,
              clf_feat_norm=True, init_filter_norm=False, final_conv=True,
              out_feature_dim=512, hinge_threshold=0.05, iou_input_dim=(256, 256), iou_inter_dim=(256, 256),
              activation_leak=0.0, score_act='relu', act_param=None,
              detach_length=float('Inf'), frozen_backbone_layers=()):

    # Backbone
    backbone_net = backbones.resnet50(pretrained=backbone_pretrained, frozen_layers=frozen_backbone_layers)

    # Feature normalization
    norm_scale = math.sqrt(1.0 / (out_feature_dim * filter_size * filter_size))

    # Classifier features
    if classification_layer == 'layer3':
        feature_dim = 256
    elif classification_layer == 'layer4':
        feature_dim = 512
    else:
        raise Exception

    clf_feature_extractor = clf_features.residual_bottleneck(feature_dim=feature_dim,
                                                             num_blocks=clf_feat_blocks, l2norm=clf_feat_norm,
                                                             final_conv=final_conv, norm_scale=norm_scale,
                                                             out_dim=out_feature_dim)

    # Initializer for the DiMP classifier
    initializer = clf_initializer.FilterInitializerLinear(filter_size=filter_size, filter_norm=init_filter_norm,
                                                          feature_dim=out_feature_dim)

    # Residual module that defined the online loss
    residual_module = residual_modules.LinearFilterHinge(feat_stride=feat_stride, init_filter_reg=optim_init_reg,
                                                         hinge_threshold=hinge_threshold, activation_leak=activation_leak,
                                                         score_act=score_act, act_param=act_param)

    # Construct generic optimizer module
    optimizer = steepestdescent.GNSteepestDescent(residual_module=residual_module, num_iter=optim_iter,
                                                  detach_length=detach_length, residual_batch_dim=1, compute_losses=True)

    attention_fusion_module = fusion_clf.AttentionLearnFusionDirectModule(softmax_temp_init=50., train_softmax_temp=True)

    # The classifier module
    classifier = target_fusion_clf.LinearAttentionFusionFilter(filter_size=filter_size, filter_initializer=initializer,
                                                               filter_optimizer=optimizer,
                                                               feature_extractor=clf_feature_extractor,
                                                               fusion_module=attention_fusion_module)

    # Bounding box regressor
    bb_regressor = bbmodels.AtomIoUNet(input_dim=(4 * 128, 4 * 256), pred_input_dim=iou_input_dim,
                                       pred_inter_dim=iou_inter_dim)

    # DiMP network
    net = DiMPMemorynet(feature_extractor=backbone_net, classifier=classifier, bb_regressor=bb_regressor,
                        classification_layer=classification_layer, bb_regressor_layer=['layer2', 'layer3'])
    return net


@model_constructor
def dimpnet50_attention_learn_fusion_scaling(filter_size=1, optim_iter=5, optim_init_reg=0.01,
              classification_layer='layer3', feat_stride=16, backbone_pretrained=True, clf_feat_blocks=0,
              clf_feat_norm=True, init_filter_norm=False, final_conv=True,
              out_feature_dim=512, hinge_threshold=0.05, iou_input_dim=(256, 256), iou_inter_dim=(256, 256),
              activation_leak=0.0, score_act='relu', act_param=None,
              detach_length=float('Inf'), frozen_backbone_layers=()):

    # Backbone
    backbone_net = backbones.resnet50(pretrained=backbone_pretrained, frozen_layers=frozen_backbone_layers)

    # Feature normalization
    norm_scale = math.sqrt(1.0 / (out_feature_dim * filter_size * filter_size))

    # Classifier features
    if classification_layer == 'layer3':
        feature_dim = 256
    elif classification_layer == 'layer4':
        feature_dim = 512
    else:
        raise Exception

    clf_feature_extractor = clf_features.residual_bottleneck(feature_dim=feature_dim,
                                                             num_blocks=clf_feat_blocks, l2norm=clf_feat_norm,
                                                             final_conv=final_conv, norm_scale=norm_scale,
                                                             out_dim=out_feature_dim)

    # Initializer for the DiMP classifier
    initializer = clf_initializer.FilterInitializerLinear(filter_size=filter_size, filter_norm=init_filter_norm,
                                                          feature_dim=out_feature_dim)

    # Residual module that defined the online loss
    residual_module = residual_modules.LinearFilterHinge(feat_stride=feat_stride, init_filter_reg=optim_init_reg,
                                                         hinge_threshold=hinge_threshold, activation_leak=activation_leak,
                                                         score_act=score_act, act_param=act_param)

    # Construct generic optimizer module
    optimizer = steepestdescent.GNSteepestDescent(residual_module=residual_module, num_iter=optim_iter,
                                                  detach_length=detach_length, residual_batch_dim=1, compute_losses=True)

    attention_fusion_module = fusion_clf.AttentionLearnMeanScalingModule(softmax_temp_init=50., train_softmax_temp=True)

    # The classifier module
    classifier = target_fusion_clf.LinearAttentionFusionFilter(filter_size=filter_size, filter_initializer=initializer,
                                                               filter_optimizer=optimizer,
                                                               feature_extractor=clf_feature_extractor,
                                                               fusion_module=attention_fusion_module)

    # Bounding box regressor
    bb_regressor = bbmodels.AtomIoUNet(input_dim=(4 * 128, 4 * 256), pred_input_dim=iou_input_dim,
                                       pred_inter_dim=iou_inter_dim)

    # DiMP network
    net = DiMPMemorynet(feature_extractor=backbone_net, classifier=classifier, bb_regressor=bb_regressor,
                        classification_layer=classification_layer, bb_regressor_layer=['layer2', 'layer3'])
    return net
