import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import ltr.models.loss as ltr_losses
import ltr.models.layers.filter as filter_layer
import ltr.models.target_classifier.features as clf_features
import ltr.models.backbone as backbones
import ltr.models.layers.activation as activation
from ltr.models.layers.distance import DistanceMap
from ltr import model_constructor
from ltr.models.layers import fourdim
from ltr.models.motion.local_correlation import FunctionCorrelation, FunctionCorrelationTranspose



class LocalCorrInitializerNorm(nn.Module):
    """Initializes the Local DiCor filter through a simple conv layer.
    args:
        filter_size: spatial kernel size of filter
        feature_dim: dimensionality of input features
        filter_norm: normalize the filter before output
    """

    def __init__(self, filter_size=1):
        super().__init__()
        assert filter_size == 1

        self.filter_size = filter_size
        self.scaling = nn.Parameter(torch.ones(1))

    def forward(self, feat):
        """Initialize filter.
        feat: input features (sequences, feat_dim, H, W)
        output: initial filters (sequences, feat_dim, H, W)"""

        weights = feat / (feat*feat).mean(dim=1,keepdim=True)
        weights = self.scaling * weights
        return weights


class NoOptimizer(nn.Module):
    def forward(self, filter, feat, **kwargs):
        losses = {'train': [], 'test': []}
        return filter, losses



class LocalCorrOptDiMP(nn.Module):
    """Optimizes the DiCor filters on the reference image.
    args:
        num_iter: number of iteration recursions to run in the optimizer
        init_step_length: initial step length factor
        init_filter_reg: initialization of the filter regularization parameter
        target_sigma: standard deviation for the correlation volume label in the reference image
        test_loss: Loss to use for the test data
        min_filter_reg: an epsilon thing to avoid devide by zero
    """
    def __init__(self, num_iter=3, init_step_length=1.0, init_filter_reg=1e-2, init_gauss_sigma=1.0,
                 min_filter_reg=1e-5, num_dist_bins=10, bin_displacement=0.5,
                 score_act='relu', act_param=None, target_scaling=1.0,
                 mask_act='sigmoid', mask_init_factor=4.0, search_size=9):
        super().__init__()

        assert search_size == 9

        self.num_iter = num_iter
        self.min_filter_reg = min_filter_reg
        self.target = None

        self.target_sigma = init_gauss_sigma
        self.search_size = search_size

        self.log_step_length = nn.Parameter(math.log(init_step_length) * torch.ones(1))
        self.filter_reg = nn.Parameter(init_filter_reg * torch.ones(1))
        self.distance_map = DistanceMap(num_dist_bins, bin_displacement)


        # Distance coordinates
        d = torch.arange(num_dist_bins, dtype=torch.float32).view(1,-1,1,1) * bin_displacement
        if init_gauss_sigma == 0:
            init_gauss = torch.zeros_like(d)
            init_gauss[0,0,0,0] = 1
        else:
            init_gauss = torch.exp(-1/2 * (d / init_gauss_sigma)**2)

        self.label_map_predictor = nn.Conv2d(num_dist_bins, 1, kernel_size=1, bias=False)
        self.label_map_predictor.weight.data = init_gauss - init_gauss.min()

        mask_layers = [nn.Conv2d(num_dist_bins, 1, kernel_size=1, bias=False)]
        init_w = mask_init_factor * torch.tanh(2.0 - d)
        if mask_act == 'sigmoid':
            mask_layers.append(nn.Sigmoid())
        elif mask_act == 'linear':
            init_w = torch.sigmoid(init_w)
        else:
            raise ValueError('Unknown activation')
        self.target_mask_predictor = nn.Sequential(*mask_layers)
        self.target_mask_predictor[0].weight.data = init_w

        self.spatial_weight_predictor = nn.Conv2d(num_dist_bins, 1, kernel_size=1, bias=False)
        self.spatial_weight_predictor.weight.data.fill_(1.0)

        if score_act == 'bentpar':
            self.score_activation = activation.BentIdentPar(act_param)
            self.score_activation_deriv = activation.BentIdentParDeriv(act_param)
        elif score_act == 'relu':
            self.score_activation = activation.LeakyReluPar()
            self.score_activation_deriv = activation.LeakyReluParDeriv()
        elif score_act == 'dualrelu':
            self.score_activation = activation.DualLeakyReluPar()
            self.score_activation_deriv = activation.DualLeakyReluParDeriv()
        elif score_act == 'dualbentpar':
            self.score_activation = activation.DualBentIdentPar(act_param)
            self.score_activation_deriv = activation.DualBentIdentParDeriv(act_param)
        else:
            raise ValueError('Unknown activation')

        self.is_dual_activation = ('dual' in score_act)


    def forward(self, filter, feat, num_iter=None, compute_losses=True, test_feat=None, test_anno=None):
        """
        :param filter: initial filters
        :param feat: features from the reference image
        :param num_iter: number of iteration
        :param compute_losses: compute intermediate losses
        :param test_feat: features from the test frames (only used for computing the losses)
        :param test_anno: output correlation annotation for the test frame (for losses only)
        :return: filters and losses
        """
        if num_iter is None:
            num_iter = self.num_iter

        num_sequences = feat.shape[0]
        num_filters = feat.shape[-2]*feat.shape[-1]
        feat_sz = (feat.shape[-2], feat.shape[-1])
        feat_dim = feat.shape[-3]

        # Compute distance map
        dist_map_sz = (self.search_size, self.search_size)
        center = torch.Tensor([dist_map_sz[0] // 2, dist_map_sz[1] // 2]).to(feat.device)
        dist_map = self.distance_map(center, dist_map_sz)

        # Compute label map masks and weight
        label_map = self.label_map_predictor(dist_map).reshape(1,-1,1,1)
        target_mask = self.target_mask_predictor(dist_map).reshape(1,-1,1,1)
        spatial_weight = self.spatial_weight_predictor(dist_map).reshape(1,-1,1,1)

        step_length = torch.exp(self.log_step_length)
        reg_weight = (self.filter_reg*self.filter_reg).clamp(min=self.min_filter_reg**2)/(feat_dim**2)

        losses = {'train': [], 'train_source': [], 'train_reg': [], 'test': []}

        for i in range(num_iter):
            # Compute gradient
            scores = FunctionCorrelation(filter, feat)

            if not self.is_dual_activation:
                scores_act = self.score_activation(scores, target_mask)
                score_mask = self.score_activation_deriv(scores, target_mask)
                loss_residuals = spatial_weight * (scores_act - label_map)
                mapped_residuals = score_mask * (spatial_weight * loss_residuals)
            else:
                score_diff = scores - label_map
                loss_residuals = self.score_activation(score_diff, spatial_weight, target_mask)
                score_mask = self.score_activation_deriv(score_diff, spatial_weight, target_mask)
                mapped_residuals = score_mask * loss_residuals

            filter_grad_reg = reg_weight * filter
            filter_grad = FunctionCorrelationTranspose(mapped_residuals, feat) + \
                          filter_grad_reg

            # Map the gradient
            scores_grad = FunctionCorrelation(filter_grad, feat)
            if not self.is_dual_activation:
                scores_grad = spatial_weight * (score_mask * scores_grad)
            else:
                scores_grad = score_mask * scores_grad

            # Compute step length
            alpha_num = (filter_grad * filter_grad).sum(dim=1, keepdim=True)
            alpha_den = ((scores_grad * scores_grad).sum(dim=1, keepdim=True) + reg_weight * alpha_num).clamp(1e-8)
            alpha = alpha_num / alpha_den

            # Update filter
            filter = filter - (step_length * alpha) * filter_grad

            if compute_losses:
                losses['train_source'].append(0.5*(loss_residuals**2).sum()/num_sequences)
                losses['train_reg'].append(0.5/reg_weight.item() * (filter_grad_reg**2).sum()/num_sequences)
                losses['train'].append(losses['train_source'][-1] + losses['train_reg'][-1])

        return filter, losses



class LocalCorrClosedForm(nn.Module):
    """Optimizes the DiCor filters on the reference image.
    args:
        num_iter: number of iteration recursions to run in the optimizer
        init_step_length: initial step length factor
        init_filter_reg: initialization of the filter regularization parameter
        target_sigma: standard deviation for the correlation volume label in the reference image
        test_loss: Loss to use for the test data
        min_filter_reg: an epsilon thing to avoid devide by zero
    """
    def __init__(self, init_filter_reg=1e-2, init_gauss_sigma=1.0, test_loss=None,
                 min_filter_reg=1e-5, num_dist_bins=10, bin_displacement=0.5, search_size=9):
        super().__init__()

        assert search_size == 9

        if test_loss is None:
            test_loss = ltr_losses.LBHinge(threshold=0.05)

        self.test_loss = test_loss
        self.min_filter_reg = min_filter_reg
        self.target = None

        self.target_sigma = init_gauss_sigma
        self.search_size = search_size

        self.filter_reg = nn.Parameter(init_filter_reg * torch.ones(1))
        self.distance_map = DistanceMap(num_dist_bins, bin_displacement)

        # Distance coordinates
        d = torch.arange(num_dist_bins, dtype=torch.float32).view(1,-1,1,1) * bin_displacement
        if init_gauss_sigma == 0:
            init_gauss = torch.zeros_like(d)
            init_gauss[0,0,0,0] = 1
        else:
            init_gauss = torch.exp(-1/2 * (d / init_gauss_sigma)**2)

        self.label_map_predictor = nn.Conv2d(num_dist_bins, 1, kernel_size=1, bias=False)
        self.label_map_predictor.weight.data = init_gauss - init_gauss.min()


    def _unfold_corr(self, corr):
        win_sz = (self.search_size, self.search_size)
        dim = corr.shape[1]
        corr_unfold = F.unfold(corr, win_sz, padding=self.search_size//2).permute(0,2,1)
        return corr_unfold.reshape(*corr_unfold.shape[:2], dim, dim)


    def forward(self, feat, compute_losses=True, test_feat=None, test_anno=None):
        """
        :param filter: initial filters
        :param feat: features from the reference image
        :param compute_losses: compute intermediate losses
        :param test_feat: features from the test frames (only used for computing the losses)
        :param test_anno: output correlation annotation for the test frame (for losses only)
        :return: filters and losses
        """
        if test_anno is not None:
            test_anno = test_anno.view(1,-1,*test_anno.shape[-3:])

        num_sequences = feat.shape[0]
        num_filters = feat.shape[-2]*feat.shape[-1]
        num_samples = self.search_size**2
        feat_dim = feat.shape[-3]
        filter_sz = (1, 1)
        feat_sz = (feat.shape[-2], feat.shape[-1])

        # Compute distance map
        dist_map_sz = (self.search_size, self.search_size)
        center = torch.Tensor([dist_map_sz[0] // 2, dist_map_sz[1] // 2]).to(feat.device)
        dist_map = self.distance_map(center, dist_map_sz)

        # Compute label map masks and weight
        label_map = self.label_map_predictor(dist_map).reshape(1, 1, num_samples, 1)

        reg_weight = (self.filter_reg*self.filter_reg).clamp(min=self.min_filter_reg**2)

        corr = FunctionCorrelation(feat, feat) / feat_dim
        corr_mat = self._unfold_corr(corr)
        corr_mat += reg_weight * torch.eye(num_filters).to(feat.device).reshape(1,1,num_samples,num_samples)

        corr_mat_chol = torch.cholesky(corr_mat)

        factors = torch.cholesky_solve(label_map, corr_mat_chol)

        filter = FunctionCorrelationTranspose(factors, feat)

        losses = {'train': [], 'test': []}

        return filter, losses

    def _compute_test_loss(self, filter, feat, label, target_bb=None):
        scores = filter_layer.apply_filter(feat, filter)
        return self.test_loss(scores, label, target_bb)





class DiCorLocal(nn.Module):
    """The main DiCor module for computing the correlation volume.
    args:
        filter_initializer: initializer network
        filter_optimizer: optimizer network
        output_order_test_train: set order of the output. The feature dimension consists of the train image coordinates if
            False and the test image coordinates if True.
    output:
        correlation scores
    """
    def __init__(self, filter_initializer, filter_optimizer):
        super(DiCorLocal, self).__init__()

        self.filter_initializer = filter_initializer
        self.filter_optimizer = filter_optimizer


    def forward(self, train_feat, test_feat, **kwargs):
        filter = self.filter_initializer(train_feat)

        filter, losses = self.filter_optimizer(filter, train_feat, test_feat=test_feat, **kwargs)

        scores = FunctionCorrelation(filter, test_feat)
        # scores = FunctionCorrelation(filter, train_feat)

        return scores, losses



class DiCorLocalClosedForm(nn.Module):
    """The main DiCor module for computing the correlation volume.
    args:
        filter_initializer: initializer network
        filter_optimizer: optimizer network
        output_order_test_train: set order of the output. The feature dimension consists of the train image coordinates if
            False and the test image coordinates if True.
    output:
        correlation scores
    """
    def __init__(self, filter_optimizer):
        super(DiCorLocalClosedForm, self).__init__()

        self.filter_optimizer = filter_optimizer


    def forward(self, train_feat, test_feat, **kwargs):

        filter, losses = self.filter_optimizer(train_feat, test_feat=test_feat, **kwargs)

        scores = FunctionCorrelation(filter, test_feat)

        return scores, losses


class DiCorLocalNetSimple(nn.Module):
    """Simple example network using the DiCor module. It simply outputs the correlation scores on the test image."""
    def __init__(self, backbone_net, corr_feature_net, corr_module, feat_layer='layer3', train_feature_extractor=True):
        super(DiCorLocalNetSimple, self).__init__()

        self.backbone_net = backbone_net
        self.corr_feature_net = corr_feature_net
        self.corr_module = corr_module
        self.feat_layer = feat_layer

        if not train_feature_extractor:
            for p in self.backbone_net.parameters():
                p.requires_grad_(False)

    def forward(self, train_img, test_img, **kwargs):
        train_feat = self.extract_features(train_img)
        test_feat = self.extract_features(test_img)

        scores, losses = self.corr_module(train_feat, test_feat, **kwargs)
        return scores, losses

    def extract_features(self, img):
        return self.corr_feature_net(self.backbone_net(img, output_layers=[self.feat_layer])[self.feat_layer])



@model_constructor
def local_corr_resnet18(backbone_pretrained=False, train_feature_extractor=False, clf_feat_blocks=1, final_conv=True, clf_feat_norm=True,
                          out_feature_dim=256, filter_initializer='norm'):
    """Constructs the simple example network that utilizes the DiCor module."""
    filter_size = 1

    # backbone
    backbone_net = backbones.resnet18(pretrained=backbone_pretrained)

    norm_scale = math.sqrt(1.0 / (out_feature_dim * filter_size * filter_size))

    # classifier
    corr_feature_net = clf_features.residual_basic_block(num_blocks=clf_feat_blocks, l2norm=clf_feat_norm,
                                                              final_conv=final_conv, norm_scale=norm_scale,
                                                              out_dim=out_feature_dim)

    if filter_initializer == 'norm':
        initializer = LocalCorrInitializerNorm(filter_size=filter_size)
    else:
        raise ValueError('Unknown initializer')

    optimizer = NoOptimizer()

    corr_module = DiCorLocal(filter_initializer=initializer, filter_optimizer=optimizer)

    net = DiCorLocalNetSimple(backbone_net=backbone_net, corr_feature_net=corr_feature_net, corr_module=corr_module,
                         train_feature_extractor=train_feature_extractor)

    return net



@model_constructor
def local_dicor_dimp_resnet18(optim_iter=3, optim_init_step=1.0, optim_init_reg=0.01, backbone_pretrained=False,
                          train_feature_extractor=False, clf_feat_blocks=1, final_conv=True, clf_feat_norm=True,
                          out_feature_dim=256, filter_initializer='norm', num_dist_bins=10, bin_displacement=0.5, init_gauss_sigma=1.0, score_act='relu', act_param=None,
                 mask_act='sigmoid', mask_init_factor=4.0):
    """Constructs the simple example network that utilizes the DiCor module."""

    filter_size = 1

    # backbone
    backbone_net = backbones.resnet18(pretrained=backbone_pretrained)

    norm_scale = math.sqrt(1.0 / (out_feature_dim * filter_size * filter_size))

    # classifier
    corr_feature_net = clf_features.residual_basic_block(num_blocks=clf_feat_blocks, l2norm=clf_feat_norm,
                                                              final_conv=final_conv, norm_scale=norm_scale,
                                                              out_dim=out_feature_dim)

    if filter_initializer == 'norm':
        initializer = LocalCorrInitializerNorm()
    else:
        raise ValueError('Unknown initializer')

    optimizer = LocalCorrOptDiMP(num_iter=optim_iter, init_step_length=optim_init_step, init_filter_reg=optim_init_reg,
                              num_dist_bins=num_dist_bins, bin_displacement=bin_displacement, score_act=score_act, act_param=act_param,
                 mask_act=mask_act, mask_init_factor=mask_init_factor, init_gauss_sigma=init_gauss_sigma)

    corr_module = DiCorLocal(filter_initializer=initializer, filter_optimizer=optimizer)

    net = DiCorLocalNetSimple(backbone_net=backbone_net, corr_feature_net=corr_feature_net, corr_module=corr_module,
                         train_feature_extractor=train_feature_extractor)

    return net


@model_constructor
def local_dicor_closed_form_resnet18(optim_init_reg=0.01, test_loss=None, backbone_pretrained=False, train_feature_extractor=False, clf_feat_blocks=1, final_conv=True, clf_feat_norm=True,
                          out_feature_dim=256, num_dist_bins=10, bin_displacement=0.5, init_gauss_sigma=1.0):
    """Constructs the simple example network that utilizes the DiCor module."""
    filter_size = 1

    # backbone
    backbone_net = backbones.resnet18(pretrained=backbone_pretrained)

    norm_scale = math.sqrt(1.0 / (out_feature_dim * filter_size * filter_size))

    # classifier
    corr_feature_net = clf_features.residual_basic_block(num_blocks=clf_feat_blocks, l2norm=clf_feat_norm,
                                                              final_conv=final_conv, norm_scale=norm_scale,
                                                              out_dim=out_feature_dim)

    optimizer = LocalCorrClosedForm(init_filter_reg=optim_init_reg, test_loss=test_loss, num_dist_bins=num_dist_bins,
                               bin_displacement=bin_displacement, init_gauss_sigma=init_gauss_sigma)

    corr_module = DiCorLocalClosedForm(filter_optimizer=optimizer)

    net = DiCorLocalNetSimple(backbone_net=backbone_net, corr_feature_net=corr_feature_net, corr_module=corr_module,
                         train_feature_extractor=train_feature_extractor)

    return net
