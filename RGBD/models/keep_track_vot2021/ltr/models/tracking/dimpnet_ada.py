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
import ltr.models.bbreg.fpn_bbr_net as fpn_bbrnet
from ltr import model_constructor


class DiMPnet(nn.Module):
    """The DiMP network.
    args:
        feature_extractor:  Backbone feature extractor network. Must return a dict of feature maps
        classifier:  Target classification module.
        bb_regressor:  Bounding box regression module.
        classification_layer:  Name of the backbone feature layer to use for classification.
        bb_regressor_layer:  Names of the backbone layers to use for bounding box regression."""

    def __init__(self, feature_extractor, classifier, bb_regressor, classification_layer, bb_regressor_layer, pool_cls_feat=True):
        super().__init__()

        self.feature_extractor = feature_extractor
        self.classifier = classifier
        self.bb_regressor = bb_regressor
        self.classification_layer = [classification_layer] if isinstance(classification_layer, str) else classification_layer
        self.bb_regressor_layer = bb_regressor_layer
        self.pool_cls_feat = pool_cls_feat
        self.output_layers = sorted(list(set(self.classification_layer + self.bb_regressor_layer)))

    def forward(self, train_imgs, test_imgs, train_bb_crop, train_bb_search_area, train_search_area_bb,
                test_search_area_bb, test_proposals, *args, **kwargs):
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
        train_feat = self.extract_backbone_features(train_imgs.view(-1, *train_imgs.shape[-3:]))
        test_feat = self.extract_backbone_features(test_imgs.view(-1, *test_imgs.shape[-3:]))

        # Classification features
        train_feat_clf = self.get_backbone_clf_feat(train_feat)
        test_feat_clf = self.get_backbone_clf_feat(test_feat)

        # Run classifier module
        if self.pool_cls_feat:
            target_scores = self.classifier(train_feat_clf, test_feat_clf, train_bb_search_area, train_search_area_bb,
                                            test_search_area_bb, *args, **kwargs)
        else:
            target_scores = self.classifier(train_feat_clf, test_feat_clf, train_bb_crop, None,
                                            None, *args, **kwargs)
        # Get bb_regressor features
        train_feat_iou = self.get_backbone_bbreg_feat(train_feat)
        test_feat_iou = self.get_backbone_bbreg_feat(test_feat)

        # Run the IoUNet module
        iou_pred = self.bb_regressor(train_feat_iou, test_feat_iou, train_bb_crop, test_proposals)

        return target_scores, iou_pred

    def get_backbone_clf_feat(self, backbone_feat):
        feat = OrderedDict({l: backbone_feat[l] for l in self.classification_layer})
        if len(self.classification_layer) == 1:
            return feat[self.classification_layer[0]]
        return feat

    def get_backbone_bbreg_feat(self, backbone_feat):
        return [backbone_feat[l] for l in self.bb_regressor_layer]

    def extract_classification_feat(self, backbone_feat, search_area_bb):
        return self.classifier.extract_classification_feat(self.get_backbone_clf_feat(backbone_feat), search_area_bb)

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
def dimpnet50_fpn_adr(input_sz=22, filter_size=4, optim_iter=5, optim_init_step=0.9, optim_init_reg=0.1,
                      classification_layer='layer3', backbone_pretrained=True,
                      clf_feat_blocks=0, final_conv=True,
                      clf_feat_norm=True, init_filter_norm=False, cls_feature_dim=512,
                      init_gauss_sigma=1.0, num_dist_bins=100, bin_displacement=0.1,
                      mask_init_factor=3.0,
                      score_act='relu', act_param=None, target_mask_act='sigmoid',
                      detach_length=float('Inf'), frozen_backbone_layers=(),
                      bb_regressor_type='iou_mod', backbone='resnet50_l3_fpn',
                      bb_regressor_layer='fpn_output3',
                      pool_cls_feat=True,
                      cls_pool_type='prpool',
                      bbr_pool_sz=(5, 7)):
    # Backbone
    if backbone == 'resnet':
        backbone_net = backbones.resnet50(pretrained=backbone_pretrained, frozen_layers=frozen_backbone_layers)
    elif backbone == 'resnet50_fpn':
        backbone_net = backbones.fpn.resnet50_fpn(pretrained_resnet=backbone_pretrained, frozen_layers=frozen_backbone_layers)
    elif backbone == 'resnet50_l3_fpn':
        backbone_net = backbones.fpn.resnet50_l3_fpn(pretrained_resnet=backbone_pretrained,
                                                  frozen_layers=frozen_backbone_layers)

    norm_scale = math.sqrt(1.0 / (cls_feature_dim * filter_size * filter_size))

    # Classifier features
    clf_feature_extractor = clf_features.residual_bottleneck(num_blocks=clf_feat_blocks, l2norm=clf_feat_norm,
                                                             final_conv=final_conv, norm_scale=norm_scale,
                                                             out_dim=cls_feature_dim)

    # Initializer for the DiMP classifier
    initializer = clf_initializer.FilterInitializerLinear(filter_size=filter_size, filter_norm=init_filter_norm,
                                                          feature_dim=cls_feature_dim)

    # Optimizer for the DiMP classifier
    if classification_layer == 'layer3':
        cls_feat_stride = 16
    else:
        raise ValueError

    optimizer = clf_optimizer.DiMPSteepestDescentGN(num_iter=optim_iter, feat_stride=1,
                                                    init_step_length=optim_init_step,
                                                    init_filter_reg=optim_init_reg, init_gauss_sigma=init_gauss_sigma,
                                                    num_dist_bins=num_dist_bins,
                                                    bin_displacement=bin_displacement,
                                                    mask_init_factor=mask_init_factor,
                                                    score_act=score_act, act_param=act_param, mask_act=target_mask_act,
                                                    detach_length=detach_length)

    # The classifier module
    classifier = target_clf.LinearFilterAda(filter_size=filter_size, filter_initializer=initializer,
                                            filter_optimizer=optimizer, feature_extractor=clf_feature_extractor,
                                            feat_stride=cls_feat_stride, pool_size=input_sz, pool_type=cls_pool_type)

    if bb_regressor_layer == 'fpn_output3':
        pool_stride = 8
    elif bb_regressor_layer == 'fpn_output2':
        pool_stride = 4
    elif bb_regressor_layer == 'fpn_output4':
        pool_stride = 16
    else:
        pool_stride = None

    # Bounding box regressor
    if bb_regressor_type == 'atom':
        bb_regressor = bbmodels.AtomIoUNet(input_dim=(4 * 128, 4 * 256))
    elif bb_regressor_type == 'iou_mod':
        bb_regressor = fpn_iounet.FPNIoUNetHR(pool_stride=pool_stride, pool_r=bbr_pool_sz[0], pool_t=bbr_pool_sz[1])
    elif bb_regressor_type == 'iou_corr':
        bb_regressor = fpn_iounet.FPNIoUNetHRCorr()
    elif bb_regressor_type == 'iou_cat':
        bb_regressor = fpn_iounet.FPNIoUNetHRCat()
    elif bb_regressor_type == 'bbr_corr':
        bb_regressor = fpn_bbrnet.FPNBBRNetHRCorr()
    elif bb_regressor_type == 'bbr_mod':
        bb_regressor = fpn_bbrnet.FPNBBRNetHR()
    elif bb_regressor_type == 'bbr_cat':
        bb_regressor = fpn_bbrnet.FPNBBRNetHRCat()
    elif bb_regressor_type == 'bbr_mod_cascade':
        bb_regressor = fpn_bbrnet.FPNBBRNetHRModCascade()
    else:
        raise ValueError

    # DiMP network
    if not isinstance(bb_regressor_layer, (list, tuple)):
        bb_regressor_layer = [bb_regressor_layer, ]

    net = DiMPnet(feature_extractor=backbone_net, classifier=classifier, bb_regressor=bb_regressor,
                  classification_layer=classification_layer, bb_regressor_layer=bb_regressor_layer,
                  pool_cls_feat=pool_cls_feat)
    return net
