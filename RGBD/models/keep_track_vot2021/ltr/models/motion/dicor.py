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



class CorrInitializerLinear(nn.Module):
    """Initializes the DiCor filter through a simple conv layer.
    args:
        filter_size: spatial kernel size of filter
        feature_dim: dimensionality of input features
        filter_norm: normalize the filter before output
    """

    def __init__(self, filter_size=1, feature_dim=256, filter_norm=False, conv_ksz=None):
        super().__init__()

        if conv_ksz is None:
            conv_ksz = filter_size

        self.filter_size = filter_size
        self.filter_conv = nn.Conv2d(feature_dim, feature_dim*filter_size**2, kernel_size=conv_ksz, padding=conv_ksz // 2)
        self.filter_norm = filter_norm

        # Init weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def forward(self, feat):
        """Initialize filter.
        feat: input features (sequences, feat_dim, H, W)
        output: initial filters (sequences, num_filters, feat_dim, fH, fW)"""

        feat = feat.view(-1, *feat.shape[-3:])
        weights = self.filter_conv(feat).permute(0,2,3,1).reshape(feat.shape[0],
                                                                  feat.shape[-2]*feat.shape[-1],
                                                                  feat.shape[-3],
                                                                  self.filter_size,
                                                                  self.filter_size).contiguous()
        if self.filter_norm:
            weights = weights / (weights.shape[-3] * weights.shape[-2] * weights.shape[-1])

        return weights


class CorrInitializerNorm(nn.Module):
    """Initializes the DiCor filter through a simple conv layer.
    args:
        filter_size: spatial kernel size of filter
        feature_dim: dimensionality of input features
        filter_norm: normalize the filter before output
    """

    def __init__(self, filter_size=1):
        super().__init__()

        self.filter_size = filter_size
        self.scaling = nn.Parameter(torch.ones(1))

    def forward(self, feat):
        """Initialize filter.
        feat: input features (sequences, feat_dim, H, W)
        output: initial filters (sequences, num_filters, feat_dim, fH, fW)"""

        feat = feat.view(-1, *feat.shape[-3:])
        weights = F.unfold(feat, self.filter_size, padding=self.filter_size//2)

        weights = weights / (weights*weights).sum(dim=1,keepdim=True)

        weights = self.scaling * weights.permute(0,2,1).reshape(feat.shape[0], feat.shape[-2]*feat.shape[-1],
                                                                  feat.shape[-3],
                                                                  self.filter_size,
                                                                  self.filter_size).contiguous()

        return weights


class CorrInitializerNormBg(nn.Module):
    """Initializes the DiCor filter through a simple conv layer.
    args:
        filter_size: spatial kernel size of filter
        feature_dim: dimensionality of input features
        filter_norm: normalize the filter before output
    """

    def __init__(self, filter_size=1, init_fg=1.0, init_bg=0.0):
        super().__init__()

        self.filter_size = filter_size
        self.target_fg = nn.Parameter(torch.Tensor([init_fg]))
        self.target_bg = nn.Parameter(torch.Tensor([init_bg]))

    def forward(self, feat):
        """Initialize filter.
        feat: input features (sequences, feat_dim, H, W)
        output: initial filters (sequences, num_filters, feat_dim, fH, fW)"""

        feat = feat.view(-1, *feat.shape[-3:])
        weights = F.unfold(feat, self.filter_size, padding=self.filter_size // 2)

        bg_weights = weights.mean(dim=2, keepdim=True)

        ff = (weights * weights).sum(dim=1, keepdim=True)
        bb = (bg_weights * bg_weights).sum(dim=1, keepdim=True)
        fb = (weights * bg_weights).sum(dim=1, keepdim=True)

        den = (ff*bb - fb*fb).clamp(1e-6)
        fg_scale = self.target_fg * bb - self.target_bg * fb
        bg_scale = self.target_fg * fb - self.target_bg * ff
        weights = (fg_scale * weights - bg_scale * bg_weights) / den

        weights = weights.permute(0, 2, 1).reshape(feat.shape[0], feat.shape[-2] * feat.shape[-1],
                                                                  feat.shape[-3],
                                                                  self.filter_size,
                                                                  self.filter_size).contiguous()
        return weights


class CorrOptL2SDGN(nn.Module):
    """Optimizes the DiCor filters on the reference image.
    args:
        num_iter: number of iteration recursions to run in the optimizer
        init_step_length: initial step length factor
        init_filter_reg: initialization of the filter regularization parameter
        target_sigma: standard deviation for the correlation volume label in the reference image
        test_loss: Loss to use for the test data
        min_filter_reg: an epsilon thing to avoid devide by zero
    """
    def __init__(self, num_iter=1, init_step_length=1.0, init_filter_reg=1e-2, target_sigma=1.0, test_loss=None, min_filter_reg=1e-5):
        super().__init__()

        if test_loss is None:
            test_loss = ltr_losses.LBHinge(threshold=0.05)

        self.num_iter = num_iter
        self.test_loss = test_loss
        self.min_filter_reg = min_filter_reg
        self.target_sigma = target_sigma
        self.target = None

        self.log_step_length = nn.Parameter(math.log(init_step_length) * torch.ones(1))
        self.filter_reg = nn.Parameter(init_filter_reg * torch.ones(1))


    def _create_target(self, sz, device):
        if self.target_sigma == 0:
            self.target = torch.eye(sz[0]*sz[1]).view(1,1,-1,sz[0],sz[1]).to(device)
        else:
            k0 = torch.arange(sz[0], dtype=torch.float32).view(1,1,-1,1)
            k1 = torch.arange(sz[1], dtype=torch.float32).view(1,1,1,-1)
            m0 = k0.clone().view(-1,1,1,1)
            m1 = k1.clone().view(1,-1,1,1)

            self.target = torch.exp(-0.5/self.target_sigma**2 * ((k0-m0)**2 + (k1-m1)**2)).view(1,1,-1,sz[0],sz[1]).to(device)


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
        if self.target is None or self.target.shape[-2:] != feat.shape[-2:]:
            self._create_target(feat.shape[-2:], feat.device)

        test_anno = test_anno.view(1,-1,*test_anno.shape[-3:])

        num_images = feat.shape[0]
        num_sequences = feat.shape[1] if feat.dim() == 5 else 1
        num_filters = filter.shape[1]
        filter_sz = (filter.shape[-2], filter.shape[-1])

        assert num_images == 1
        assert num_filters == feat.shape[-2]*feat.shape[-1]

        step_length = torch.exp(self.log_step_length)
        reg_weight = (self.filter_reg*self.filter_reg).clamp(min=self.min_filter_reg**2)

        losses = {'train': [], 'test': []}

        for i in range(num_iter):
            # Compute gradient
            scores = filter_layer.apply_filter(feat, filter)

            if compute_losses:
                losses['train'].append(((scores - self.target)**2).mean())
                if test_feat is not None:
                    losses['test'].append(self._compute_test_loss(filter, test_feat, test_anno))

            scores_act = scores
            residuals = scores_act - self.target
            filter_grad = filter_layer.apply_feat_transpose(feat, residuals, filter_sz, training=self.training) + \
                          reg_weight * filter

            # Map the gradient
            scores_grad = filter_layer.apply_filter(feat, filter_grad)

            # Compute step length
            alpha_num = (filter_grad * filter_grad).view(num_sequences, num_filters, -1).sum(dim=2)
            alpha_den = ((scores_grad * scores_grad).view(num_sequences, num_filters, -1).sum(dim=2) + reg_weight * alpha_num).clamp(1e-8)
            alpha = alpha_num / alpha_den

            # Update filter
            filter = filter - (step_length * alpha.view(num_sequences,num_filters,1,1,1)) * filter_grad

        if compute_losses:
            scores = filter_layer.apply_filter(feat, filter)
            losses['train'].append(((scores - self.target)**2).mean())
            if test_feat is not None:
                losses['test'].append(self._compute_test_loss(filter, test_feat, test_anno))

        return filter, losses

    def _compute_test_loss(self, filter, feat, label):
        scores = filter_layer.apply_filter(feat, filter)
        return self.test_loss(scores, label)



class CorrOptDiMP(nn.Module):
    """Optimizes the DiCor filters on the reference image.
    args:
        num_iter: number of iteration recursions to run in the optimizer
        init_step_length: initial step length factor
        init_filter_reg: initialization of the filter regularization parameter
        target_sigma: standard deviation for the correlation volume label in the reference image
        test_loss: Loss to use for the test data
        min_filter_reg: an epsilon thing to avoid devide by zero
    """
    def __init__(self, num_iter=1, init_step_length=1.0, init_filter_reg=1e-2, init_gauss_sigma=1.0, test_loss=None,
                 min_filter_reg=1e-5, num_dist_bins=10, bin_displacement=0.5, score_act='relu', act_param=None,
                 mask_act='sigmoid', mask_init_factor=4.0):
        super().__init__()

        if test_loss is None:
            test_loss = ltr_losses.LBHinge(threshold=0.05)

        self.num_iter = num_iter
        self.test_loss = test_loss
        self.min_filter_reg = min_filter_reg
        self.target = None

        self.target_sigma = init_gauss_sigma

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
        if mask_act == 'sigmoid':
            mask_layers.append(nn.Sigmoid())
            init_bias = 0.0
        elif mask_act == 'linear':
            init_bias = 0.5
        else:
            raise ValueError('Unknown activation')
        self.target_mask_predictor = nn.Sequential(*mask_layers)
        self.target_mask_predictor[0].weight.data = mask_init_factor * torch.tanh(2.0 - d) + init_bias

        self.spatial_weight_predictor = nn.Conv2d(num_dist_bins, 1, kernel_size=1, bias=False)
        self.spatial_weight_predictor.weight.data.fill_(1.0)

        if score_act == 'bentpar':
            self.score_activation = activation.BentIdentPar(act_param)
            self.score_activation_deriv = activation.BentIdentParDeriv(act_param)
        elif score_act == 'relu':
            self.score_activation = activation.LeakyReluPar()
            self.score_activation_deriv = activation.LeakyReluParDeriv()
        else:
            raise ValueError('Unknown activation')


    def _create_target(self, sz, device):
        if self.target_sigma == 0:
            self.target = torch.eye(sz[0]*sz[1]).view(1,1,-1,sz[0],sz[1]).to(device)
        else:
            k0 = torch.arange(sz[0], dtype=torch.float32).view(1,1,-1,1)
            k1 = torch.arange(sz[1], dtype=torch.float32).view(1,1,1,-1)
            m0 = k0.clone().view(-1,1,1,1)
            m1 = k1.clone().view(1,-1,1,1)

            self.target = torch.exp(-0.5/self.target_sigma**2 * ((k0-m0)**2 + (k1-m1)**2)).view(1,1,-1,sz[0],sz[1]).to(device)

    def _unfold_map(self, full_map):
        output_sz = (full_map.shape[-2] // 2 + 1, full_map.shape[-1] // 2 + 1)
        map_unfold = F.unfold(full_map, output_sz).view(output_sz[0], output_sz[1], output_sz[0], output_sz[1]).flip((2,3))
        map = map_unfold.permute(2,3,0,1).reshape(1,1,-1,output_sz[0],output_sz[1])
        return map


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
        if self.target is None or self.target.shape[-2:] != feat.shape[-2:]:
            self._create_target(feat.shape[-2:], feat.device)
        if test_anno is not None:
            test_anno = test_anno.view(1,-1,*test_anno.shape[-3:])

        num_images = feat.shape[0]
        num_sequences = feat.shape[1] if feat.dim() == 5 else 1
        num_filters = filter.shape[1]
        filter_sz = (filter.shape[-2], filter.shape[-1])
        feat_sz = (feat.shape[-2], feat.shape[-1])
        output_sz = (feat_sz[0] + (filter_sz[0]+1)%2, feat_sz[1] + (filter_sz[1]+1)%2)

        assert num_images == 1
        assert num_filters == feat.shape[-2]*feat.shape[-1]
        assert filter_sz[0] % 2 == 1 and filter_sz[1] % 2 == 1  # Assume odd kernels for now

        # Compute distance map
        dist_map_sz = (output_sz[0] * 2 - 1, output_sz[1] * 2 - 1)
        center = torch.Tensor([dist_map_sz[0] // 2, dist_map_sz[1] // 2]).to(feat.device)
        dist_map = self.distance_map(center, dist_map_sz)

        # Compute label map masks and weight
        label_map = self._unfold_map(self.label_map_predictor(dist_map))
        target_mask = self._unfold_map(self.target_mask_predictor(dist_map))
        spatial_weight = self._unfold_map(self.spatial_weight_predictor(dist_map))

        step_length = torch.exp(self.log_step_length)
        reg_weight = (self.filter_reg*self.filter_reg).clamp(min=self.min_filter_reg**2)

        losses = {'train': [], 'test': []}

        for i in range(num_iter):
            if compute_losses:
                if test_feat is not None:
                    losses['test'].append(self._compute_test_loss(filter, test_feat, test_anno))


            # Compute gradient
            scores = filter_layer.apply_filter(feat, filter)
            scores_act = self.score_activation(scores, target_mask)
            score_mask = self.score_activation_deriv(scores, target_mask)
            residuals = score_mask * ((spatial_weight * spatial_weight) * (scores_act - label_map))
            filter_grad = filter_layer.apply_feat_transpose(feat, residuals, filter_sz, training=self.training) + \
                          reg_weight * filter

            # Map the gradient
            scores_grad = filter_layer.apply_filter(feat, filter_grad)
            scores_grad = spatial_weight * (score_mask * scores_grad)

            # Compute step length
            alpha_num = (filter_grad * filter_grad).view(num_sequences, num_filters, -1).sum(dim=2)
            alpha_den = ((scores_grad * scores_grad).view(num_sequences, num_filters, -1).sum(dim=2) + reg_weight * alpha_num).clamp(1e-8)
            alpha = alpha_num / alpha_den

            # Update filter
            filter = filter - (step_length * alpha.view(num_sequences,num_filters,1,1,1)) * filter_grad

        if compute_losses:
            if test_feat is not None:
                losses['test'].append(self._compute_test_loss(filter, test_feat, test_anno))

        return filter, losses

    def _compute_test_loss(self, filter, feat, label, target_bb=None):
        scores = filter_layer.apply_filter(feat, filter)
        return self.test_loss(scores, label, target_bb)



class CorrOptDiMPQReg(nn.Module):
    """Optimizes the DiCor filters on the reference image.
    args:
        num_iter: number of iteration recursions to run in the optimizer
        init_step_length: initial step length factor
        init_filter_reg: initialization of the filter regularization parameter
        target_sigma: standard deviation for the correlation volume label in the reference image
        test_loss: Loss to use for the test data
        min_filter_reg: an epsilon thing to avoid devide by zero
    """
    def __init__(self, num_iter=1, init_step_length=1.0, init_filter_reg=1e-2, init_gauss_sigma=1.0, test_loss=None,
                 min_filter_reg=1e-5, num_dist_bins=10, bin_displacement=0.5, score_act='relu', act_param=None,
                 mask_act='sigmoid', mask_init_factor=4.0, reg_kernel_size=3, reg_inter_dim=1, reg_output_dim=1):
        super().__init__()

        if test_loss is None:
            test_loss = ltr_losses.LBHinge(threshold=0.05)

        self.num_iter = num_iter
        self.test_loss = test_loss
        self.min_filter_reg = min_filter_reg
        self.target = None

        self.target_sigma = init_gauss_sigma

        self.log_step_length = nn.Parameter(math.log(init_step_length) * torch.ones(1))
        self.filter_reg = nn.Parameter(init_filter_reg * torch.ones(1))
        self.distance_map = DistanceMap(num_dist_bins, bin_displacement)

        self.reg_layer = fourdim.SeparableConv4d(kernel_size=reg_kernel_size, inter_dim=reg_inter_dim, output_dim=reg_output_dim,
                                                 bias=False, permute_back_output=False)


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
        if mask_act == 'sigmoid':
            mask_layers.append(nn.Sigmoid())
            init_bias = 0.0
        elif mask_act == 'linear':
            init_bias = 0.5
        else:
            raise ValueError('Unknown activation')
        self.target_mask_predictor = nn.Sequential(*mask_layers)
        self.target_mask_predictor[0].weight.data = mask_init_factor * torch.tanh(2.0 - d) + init_bias

        self.spatial_weight_predictor = nn.Conv2d(num_dist_bins, 1, kernel_size=1, bias=False)
        self.spatial_weight_predictor.weight.data.fill_(1.0)

        if score_act == 'bentpar':
            self.score_activation = activation.BentIdentPar(act_param)
            self.score_activation_deriv = activation.BentIdentParDeriv(act_param)
        elif score_act == 'relu':
            self.score_activation = activation.LeakyReluPar()
            self.score_activation_deriv = activation.LeakyReluParDeriv()
        else:
            raise ValueError('Unknown activation')


    def _create_target(self, sz, device):
        if self.target_sigma == 0:
            self.target = torch.eye(sz[0]*sz[1]).view(1,1,-1,sz[0],sz[1]).to(device)
        else:
            k0 = torch.arange(sz[0], dtype=torch.float32).view(1,1,-1,1)
            k1 = torch.arange(sz[1], dtype=torch.float32).view(1,1,1,-1)
            m0 = k0.clone().view(-1,1,1,1)
            m1 = k1.clone().view(1,-1,1,1)

            self.target = torch.exp(-0.5/self.target_sigma**2 * ((k0-m0)**2 + (k1-m1)**2)).view(1,1,-1,sz[0],sz[1]).to(device)

    def _unfold_map(self, full_map):
        output_sz = (full_map.shape[-2] // 2 + 1, full_map.shape[-1] // 2 + 1)
        map_unfold = F.unfold(full_map, output_sz).view(output_sz[0], output_sz[1], output_sz[0], output_sz[1]).flip((2,3))
        map = map_unfold.permute(2,3,0,1).reshape(1,1,-1,output_sz[0],output_sz[1])
        return map


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

        def _compute_test_loss(filter, feat, label, scores=None):
            if scores is None:
                scores = filter_layer.apply_filter(feat, filter)
            return self.test_loss(scores, label)

        if num_iter is None:
            num_iter = self.num_iter

        losses = {'train_source': [], 'train_reg': [], 'train_target': [], 'train': [], 'test': []}

        if self.target is None or self.target.shape[-2:] != feat.shape[-2:]:
            self._create_target(feat.shape[-2:], feat.device)
        if test_anno is not None:
            test_anno = test_anno.view(1,-1,*test_anno.shape[-3:])

        num_images = feat.shape[0]
        num_sequences = feat.shape[1] if feat.dim() == 5 else 1
        num_filters = filter.shape[1]
        filter_sz = (filter.shape[-2], filter.shape[-1])
        feat_sz = (feat.shape[-2], feat.shape[-1])
        output_sz = (feat_sz[0] + (filter_sz[0]+1)%2, feat_sz[1] + (filter_sz[1]+1)%2)

        assert num_images == 1
        assert num_filters == feat.shape[-2]*feat.shape[-1]
        assert filter_sz[0] % 2 == 1 and filter_sz[1] % 2 == 1  # Assume odd kernels for now

        # Compute distance map
        dist_map_sz = (output_sz[0] * 2 - 1, output_sz[1] * 2 - 1)
        center = torch.Tensor([dist_map_sz[0] // 2, dist_map_sz[1] // 2]).to(feat.device)
        dist_map = self.distance_map(center, dist_map_sz)

        # Compute label map masks and weight
        label_map = self._unfold_map(self.label_map_predictor(dist_map))
        target_mask = self._unfold_map(self.target_mask_predictor(dist_map))
        spatial_weight = self._unfold_map(self.spatial_weight_predictor(dist_map))

        step_length = torch.exp(self.log_step_length)
        reg_weight = (self.filter_reg*self.filter_reg).clamp(min=self.min_filter_reg**2)


        for i in range(num_iter):
            # Compute gradient
            scores = filter_layer.apply_filter(feat, filter)
            scores_act = self.score_activation(scores, target_mask)
            score_mask = self.score_activation_deriv(scores, target_mask)
            loss_residuals = spatial_weight * (scores_act - label_map)
            mapped_residuals = score_mask * (spatial_weight * loss_residuals)
            filter_grad_reg = reg_weight * filter
            filter_grad = filter_layer.apply_feat_transpose(feat, mapped_residuals, filter_sz, training=self.training) + \
                          filter_grad_reg

            # Regularization on test features
            test_scores = filter_layer.apply_filter(test_feat, filter)
            loss_test_residuals = self.reg_layer(test_scores.reshape(-1, *feat_sz, *feat_sz))
            reg_tp_res = self.reg_layer(loss_test_residuals, transpose=True).reshape(test_scores.shape)
            filter_grad = filter_grad + filter_layer.apply_feat_transpose(test_feat, reg_tp_res, filter_sz, training=self.training)

            if compute_losses:
                if test_feat is not None:
                    losses['test'].append(_compute_test_loss(filter, test_feat, test_anno, test_scores))
                if i > 0:
                    losses['train_source'].append(0.5*(loss_residuals**2).sum()/num_sequences)
                    losses['train_reg'].append(0.5/reg_weight.item() * (filter_grad_reg**2).sum()/num_sequences)
                    losses['train_target'].append(0.5*(loss_test_residuals**2).sum()/num_sequences)
                    losses['train'].append(losses['train_source'][-1] + losses['train_reg'][-1] + losses['train_target'][-1])


            # Map the gradient
            scores_grad = filter_layer.apply_filter(feat, filter_grad)
            scores_grad = spatial_weight * (score_mask * scores_grad)

            # Hessian parts for regularization
            test_scores_grad = filter_layer.apply_filter(test_feat, filter_grad)
            loss_test_residuals_grad = self.reg_layer(test_scores_grad.reshape(-1, *feat_sz, *feat_sz))

            # Compute step length
            alpha_num = (filter_grad * filter_grad).reshape(num_sequences, -1).sum(dim=-1)
            alpha_den = ((scores_grad * scores_grad).reshape(num_sequences, -1).sum(dim=-1) +
                         (loss_test_residuals_grad * loss_test_residuals_grad).reshape(num_sequences, -1).sum(dim=-1) +
                         reg_weight * alpha_num).clamp(1e-8)
            alpha = alpha_num / alpha_den

            # Update filter
            filter = filter - (step_length * alpha.view(num_sequences,1,1,1,1)) * filter_grad

        if compute_losses:
            if test_feat is not None:
                losses['test'].append(_compute_test_loss(filter, test_feat, test_anno, test_scores))
            losses['train_source'].append(0.5*(loss_residuals**2).sum()/num_sequences)
            losses['train_reg'].append(0.5/reg_weight.item() * (filter_grad_reg**2).sum()/num_sequences)
            losses['train_target'].append(0.5*(loss_test_residuals**2).sum()/num_sequences)
            losses['train'].append(losses['train_source'][-1] + losses['train_reg'][-1] + losses['train_target'][-1])

        return filter, losses

    def _compute_test_loss(self, filter, feat, label, target_bb=None):
        scores = filter_layer.apply_filter(feat, filter)
        return self.test_loss(scores, label, target_bb)





class CorrOptDiMPUnique(nn.Module):
    """Optimizes the DiCor filters on the reference image.
    args:
        num_iter: number of iteration recursions to run in the optimizer
        init_step_length: initial step length factor
        init_filter_reg: initialization of the filter regularization parameter
        target_sigma: standard deviation for the correlation volume label in the reference image
        test_loss: Loss to use for the test data
        min_filter_reg: an epsilon thing to avoid devide by zero
    """
    def __init__(self, num_iter=1, init_step_length=1.0, init_filter_reg=1e-2, init_gauss_sigma=1.0, test_loss=None,
                 min_filter_reg=1e-5, num_dist_bins=10, bin_displacement=0.5, score_act='relu', act_param=None, bg_mask_val=0.0,
                 steplength_reg=0.0, uniqueness_weight=0.0, temperature=1.0, max_temp_scale=1.0):
        super().__init__()

        if test_loss is None:
            test_loss = ltr_losses.LBHinge(threshold=0.05)

        self.num_iter = num_iter
        self.test_loss = test_loss
        self.min_filter_reg = min_filter_reg
        self.target = None

        self.target_sigma = init_gauss_sigma

        self.log_step_length = nn.Parameter(math.log(init_step_length) * torch.ones(1))
        self.filter_reg = nn.Parameter(init_filter_reg * torch.ones(1))
        self.distance_map = DistanceMap(num_dist_bins, bin_displacement)

        self.steplength_reg = steplength_reg

        self.bg_mask_val = bg_mask_val

        self.uniqueness_weight = uniqueness_weight
        self.temerature = temperature
        self.max_temp_scale = max_temp_scale


        # Distance coordinates
        d = torch.arange(num_dist_bins, dtype=torch.float32).view(1,-1,1,1) * bin_displacement
        if init_gauss_sigma == 0:
            init_gauss = torch.zeros_like(d)
            init_gauss[0,0,0,0] = 1
        else:
            init_gauss = torch.exp(-1/2 * (d / init_gauss_sigma)**2)

        self.label_map_predictor = nn.Conv2d(num_dist_bins, 1, kernel_size=1, bias=False)
        self.label_map_predictor.weight.data = init_gauss - init_gauss.min()

        self.spatial_weight_predictor = nn.Conv2d(num_dist_bins, 1, kernel_size=1, bias=False)
        self.spatial_weight_predictor.weight.data.fill_(1.0)

        if score_act == 'bentpar':
            self.score_activation = activation.BentIdentPar(act_param)
            self.score_activation_deriv = activation.BentIdentParDeriv(act_param)
        elif score_act == 'relu':
            self.score_activation = activation.LeakyReluPar()
            self.score_activation_deriv = activation.LeakyReluParDeriv()
        elif score_act == 'none':
            self.score_activation = lambda s, tm: s
            self.score_activation_deriv = lambda s, tm: 1.0
        else:
            raise ValueError('Unknown activation')


    def _create_target(self, sz, device):
        if self.target_sigma == 0:
            self.target = torch.eye(sz[0]*sz[1]).view(1,1,-1,sz[0],sz[1]).to(device)
        else:
            k0 = torch.arange(sz[0], dtype=torch.float32).view(1,1,-1,1)
            k1 = torch.arange(sz[1], dtype=torch.float32).view(1,1,1,-1)
            m0 = k0.clone().view(-1,1,1,1)
            m1 = k1.clone().view(1,-1,1,1)

            self.target = torch.exp(-0.5/self.target_sigma**2 * ((k0-m0)**2 + (k1-m1)**2)).view(1,1,-1,sz[0],sz[1]).to(device)

    def _unfold_map(self, full_map):
        output_sz = (full_map.shape[-2] // 2 + 1, full_map.shape[-1] // 2 + 1)
        map_unfold = F.unfold(full_map, output_sz).view(output_sz[0], output_sz[1], output_sz[0], output_sz[1]).flip((2,3))
        map = map_unfold.permute(2,3,0,1).reshape(1,1,-1,output_sz[0],output_sz[1])
        return map

    def forward(self, filter, feat, num_iter=None, compute_losses=True, test_feat=None, test_anno=None):
        """
        :param filter: initial filters
        :param feat: features from the reference image
        :param num_iter: number of iteration
        :param compute_losses: compute intermediate losses
        :param test_feat: features from the test frames
        :param test_anno: output correlation annotation for the test frame (for losses only)
        :return: filters and losses
        """

        def _compute_test_loss(filter, feat, label, scores=None):
            if scores is None:
                scores = filter_layer.apply_filter(feat, filter)
            return self.test_loss(scores, label)

        def _compute_target_loss(scores=None, Ts=None, sig_exp_val=None):
            if self.uniqueness_weight == 0:
                return torch.zeros(1, device=feat.device)[0]
            if scores is not None:
                Ts = self.temerature * scores
                sig_exp_val = Ts.shape[-3]*Ts - math.log(Ts.shape[-3])
            num_sequences, num_filters = Ts.shape[1:3]
            return self.uniqueness_weight * (math.log(num_filters) + F.softplus(sig_exp_val).mean(dim=(-2, -1)) -
                    activation.logsumexp_reg(self.max_temp_scale*Ts.view(*Ts.shape[:-2], -1), dim=-1, reg=0)/self.max_temp_scale).sum() / (self.temerature * num_sequences)


        if num_iter is None:
            num_iter = self.num_iter

        losses = {'train_source': [], 'train_reg': [], 'train_target': [], 'train': [], 'test': []}

        if num_iter == 0:
            return filter, losses

        if self.target is None or self.target.shape[-2:] != feat.shape[-2:]:
            self._create_target(feat.shape[-2:], feat.device)
        if test_anno is not None:
            test_anno = test_anno.view(1,-1,*test_anno.shape[-3:])

        num_images = feat.shape[0]
        num_sequences = feat.shape[1] if feat.dim() == 5 else 1
        num_filters = filter.shape[1]
        filter_sz = (filter.shape[-2], filter.shape[-1])
        feat_sz = (feat.shape[-2], feat.shape[-1])
        output_sz = (feat_sz[0] + (filter_sz[0]+1)%2, feat_sz[1] + (filter_sz[1]+1)%2)

        assert num_images == 1
        assert num_filters == feat.shape[-2]*feat.shape[-1]
        assert filter_sz[0] % 2 == 1 and filter_sz[1] % 2 == 1  # Assume odd kernels for now

        # Compute distance map
        dist_map_sz = (output_sz[0] * 2 - 1, output_sz[1] * 2 - 1)
        center = torch.Tensor([dist_map_sz[0] // 2, dist_map_sz[1] // 2]).to(feat.device)
        dist_map = self.distance_map(center, dist_map_sz)

        # Compute label map masks and weight
        label_map = self._unfold_map(self.label_map_predictor(dist_map))
        spatial_weight = self._unfold_map(self.spatial_weight_predictor(dist_map))

        # target_mask = self._unfold_map(self.target_mask_predictor(dist_map))
        target_mask = (1.0-self.bg_mask_val)*torch.eye(num_filters, device=feat.device).view(1, 1, num_filters, *feat_sz) + self.bg_mask_val

        step_length = torch.exp(self.log_step_length)
        reg_weight = (self.filter_reg*self.filter_reg).clamp(min=self.min_filter_reg**2)

        # Just define some things
        test_scores = None
        Ts = None
        sig_exp_val = None

        for i in range(num_iter):
            if compute_losses:
                if test_feat is not None:
                    losses['test'].append(_compute_test_loss(filter, test_feat, test_anno, test_scores))
                if i > 0:
                    losses['train_source'].append(0.5*(loss_residuals**2).sum()/num_sequences)
                    losses['train_reg'].append(0.5/reg_weight.item() * (filter_grad_reg**2).sum()/num_sequences)
                    losses['train_target'].append(_compute_target_loss(Ts=Ts, sig_exp_val=sig_exp_val))
                    losses['train'].append(losses['train_source'][-1] + losses['train_reg'][-1] + losses['train_target'][-1])


            # Compute gradient
            scores = filter_layer.apply_filter(feat, filter)
            scores_act = self.score_activation(scores, target_mask)
            score_mask = self.score_activation_deriv(scores, target_mask)
            loss_residuals = spatial_weight * (scores_act - label_map)
            mapped_residuals = score_mask * (spatial_weight * loss_residuals)
            filter_grad_reg = reg_weight * filter
            filter_grad = filter_layer.apply_feat_transpose(feat, mapped_residuals, filter_sz, training=self.training) + \
                          filter_grad_reg

            alpha_den = 0
            if self.uniqueness_weight > 0:
                # Temperature scaled test scores
                test_scores = filter_layer.apply_filter(test_feat, filter)
                Ts = self.temerature * test_scores if self.temerature != 1 else test_scores

                # gradient of sum(max(s_j, 0))
                sig_exp_val = num_filters * Ts - math.log(num_filters)
                p_binary = torch.sigmoid(sig_exp_val)

                # gradient of max(s)
                p_softmax = activation.softmax_reg(self.max_temp_scale*Ts.view(*Ts.shape[:-2], -1), dim=-1, reg=0).reshape(Ts.shape)

                # Add to grad
                dLds = p_binary - p_softmax
                dLdf = filter_layer.apply_feat_transpose(test_feat, dLds, filter_sz, training=self.training)
                filter_grad = filter_grad + self.uniqueness_weight * dLdf

                # Hessian parts
                test_scores_grad = filter_layer.apply_filter(test_feat, filter_grad)

                diag = num_filters * (p_binary - p_binary*p_binary)

                # diag = diag - p_softmax

                uniqueness_hess = (diag * test_scores_grad*test_scores_grad).reshape(num_sequences, num_filters, -1).sum(dim=(1,2))

                # p_st = (p_softmax * test_scores_grad)
                # uniqueness_hess = uniqueness_hess + (p_st * p_st).reshape(num_sequences, num_filters, -1).sum(dim=2)

                uniqueness_hess = (self.uniqueness_weight * self.temerature) * uniqueness_hess

                alpha_den = uniqueness_hess


            # Map the gradient
            scores_grad = filter_layer.apply_filter(feat, filter_grad)
            scores_grad = spatial_weight * (score_mask * scores_grad)

            # Compute step length
            sum_dims = 2 if self.uniqueness_weight == 0 else (1, 2)
            alpha_num = (filter_grad * filter_grad).reshape(num_sequences, num_filters, -1).sum(dim=sum_dims)
            alpha_den = (alpha_den + (scores_grad * scores_grad).reshape(num_sequences, num_filters, -1).sum(dim=sum_dims) + reg_weight * alpha_num).clamp(1e-8)
            alpha_den = alpha_den + self.steplength_reg * alpha_num
            alpha = alpha_num / alpha_den

            # Update filter
            filter = filter - (step_length * alpha.view(num_sequences,-1,1,1,1)) * filter_grad

        if compute_losses:
            if test_feat is not None:
                losses['test'].append(_compute_test_loss(filter, test_feat, test_anno, test_scores))
            losses['train_source'].append(0.5*(loss_residuals**2).sum()/num_sequences)
            losses['train_reg'].append(0.5/reg_weight.item() * (filter_grad_reg**2).sum()/num_sequences)
            losses['train_target'].append(_compute_target_loss(Ts=Ts, sig_exp_val=sig_exp_val))
            losses['train'].append(losses['train_source'][-1] + losses['train_reg'][-1] + losses['train_target'][-1])

        return filter, losses




class CorrClosedForm(nn.Module):
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
                 min_filter_reg=1e-5, num_dist_bins=10, bin_displacement=0.5):
        super().__init__()

        if test_loss is None:
            test_loss = ltr_losses.LBHinge(threshold=0.05)

        self.test_loss = test_loss
        self.min_filter_reg = min_filter_reg
        self.target = None

        self.target_sigma = init_gauss_sigma

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


    def _create_target(self, sz, device):
        if self.target_sigma == 0:
            self.target = torch.eye(sz[0]*sz[1]).view(1,1,-1,sz[0],sz[1]).to(device)
        else:
            k0 = torch.arange(sz[0], dtype=torch.float32).view(1,1,-1,1)
            k1 = torch.arange(sz[1], dtype=torch.float32).view(1,1,1,-1)
            m0 = k0.clone().view(-1,1,1,1)
            m1 = k1.clone().view(1,-1,1,1)

            self.target = torch.exp(-0.5/self.target_sigma**2 * ((k0-m0)**2 + (k1-m1)**2)).view(1,1,-1,sz[0],sz[1]).to(device)

    def _unfold_map(self, full_map):
        output_sz = (full_map.shape[-2] // 2 + 1, full_map.shape[-1] // 2 + 1)
        map_unfold = F.unfold(full_map, output_sz).view(output_sz[0], output_sz[1], output_sz[0], output_sz[1]).flip((2,3))
        map = map_unfold.permute(2,3,0,1).reshape(1,1,-1,output_sz[0],output_sz[1])
        return map


    def forward(self, feat, compute_losses=True, test_feat=None, test_anno=None):
        """
        :param filter: initial filters
        :param feat: features from the reference image
        :param compute_losses: compute intermediate losses
        :param test_feat: features from the test frames (only used for computing the losses)
        :param test_anno: output correlation annotation for the test frame (for losses only)
        :return: filters and losses
        """
        if self.target is None or self.target.shape[-2:] != feat.shape[-2:]:
            self._create_target(feat.shape[-2:], feat.device)
        if test_anno is not None:
            test_anno = test_anno.view(1,-1,*test_anno.shape[-3:])

        num_images = feat.shape[0]
        num_sequences = feat.shape[1] if feat.dim() == 5 else 1
        num_filters = feat.shape[-2]*feat.shape[-1]
        filter_sz = (1, 1)
        feat_sz = (feat.shape[-2], feat.shape[-1])
        output_sz = (feat_sz[0] + (filter_sz[0]+1)%2, feat_sz[1] + (filter_sz[1]+1)%2)

        assert num_images == 1
        assert filter_sz[0] % 2 == 1 and filter_sz[1] % 2 == 1  # Assume odd kernels for now

        # Compute distance map
        dist_map_sz = (output_sz[0] * 2 - 1, output_sz[1] * 2 - 1)
        center = torch.Tensor([dist_map_sz[0] // 2, dist_map_sz[1] // 2]).to(feat.device)
        dist_map = self.distance_map(center, dist_map_sz)

        # Compute label map masks and weight
        label_map = self._unfold_map(self.label_map_predictor(dist_map))

        reg_weight = (self.filter_reg*self.filter_reg).clamp(min=self.min_filter_reg**2)

        feat_flat = feat.reshape(*feat.shape[:-2], -1)
        corr_mat = torch.matmul(feat_flat.transpose(-2, -1), feat_flat)
        corr_mat += reg_weight * torch.eye(num_filters).to(feat.device).reshape(1,1,num_filters,num_filters)

        corr_mat_chol = torch.cholesky(corr_mat)

        factors = torch.cholesky_solve(label_map.reshape(1,1,num_filters,num_filters), corr_mat_chol)

        filter = torch.matmul(feat_flat, factors).transpose(-2,-1).reshape(num_sequences, num_filters,-1,1,1)

        losses = {'train': [], 'test': []}

        return filter, losses

    def _compute_test_loss(self, filter, feat, label, target_bb=None):
        scores = filter_layer.apply_filter(feat, filter)
        return self.test_loss(scores, label, target_bb)



class DiCor(nn.Module):
    """The main DiCor module for computing the correlation volume.
    args:
        filter_initializer: initializer network
        filter_optimizer: optimizer network
        output_order_test_train: set order of the output. The feature dimension consists of the train image coordinates if
            False and the test image coordinates if True.
    output:
        correlation scores
    """
    def __init__(self, filter_initializer, filter_optimizer, output_order_test_train=False):
        super(DiCor, self).__init__()

        self.filter_initializer = filter_initializer
        self.filter_optimizer = filter_optimizer
        self.output_order_test_train = output_order_test_train


    def forward(self, train_feat, test_feat, **kwargs):

        train_feat = train_feat.view(1, *train_feat.shape[-4:])
        test_feat = test_feat.view(1, *test_feat.shape[-4:])

        filter = self.filter_initializer(train_feat)

        filter, losses = self.filter_optimizer(filter, train_feat, test_feat=test_feat, **kwargs)

        scores = filter_layer.apply_filter(test_feat, filter)

        if self.output_order_test_train:
            scores = scores.view(*scores.shape[:-3], *train_feat.shape[-2:], -1).permute(0,1,4,2,3).contiguous()

        return scores, losses




class DiCorClosedForm(nn.Module):
    """The main DiCor module for computing the correlation volume.
    args:
        filter_initializer: initializer network
        filter_optimizer: optimizer network
        output_order_test_train: set order of the output. The feature dimension consists of the train image coordinates if
            False and the test image coordinates if True.
    output:
        correlation scores
    """
    def __init__(self, filter_optimizer, output_order_test_train=False):
        super(DiCorClosedForm, self).__init__()

        self.filter_optimizer = filter_optimizer
        self.output_order_test_train = output_order_test_train


    def forward(self, train_feat, test_feat, **kwargs):

        train_feat = train_feat.view(1, *train_feat.shape[-4:])
        test_feat = test_feat.view(1, *test_feat.shape[-4:])

        filter, losses = self.filter_optimizer(train_feat, test_feat=test_feat, **kwargs)

        scores = filter_layer.apply_filter(test_feat, filter)

        if self.output_order_test_train:
            scores = scores.view(*scores.shape[:-3], *train_feat.shape[-2:], -1).permute(0,1,4,2,3).contiguous()

        return scores, losses


class DiCorNetSimple(nn.Module):
    """Simple example network using the DiCor module. It simply outputs the correlation scores on the test image."""
    def __init__(self, backbone_net, corr_feature_net, corr_module, feat_layer='layer3', train_feature_extractor=True):
        super(DiCorNetSimple, self).__init__()

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
def dicor_simple_resnet18(filter_size=1, optim_iter=3, optim_init_step=1.0, optim_init_reg=0.01, test_loss=None, backbone_pretrained=False,
                          train_feature_extractor=False, clf_feat_blocks=1, final_conv=True, clf_feat_norm=True,
                          out_feature_dim=256, target_sigma=1.0):
    """Constructs the simple example network that utilizes the DiCor module."""

    # backbone
    backbone_net = backbones.resnet18(pretrained=backbone_pretrained)

    norm_scale = math.sqrt(1.0 / (out_feature_dim * filter_size * filter_size))

    # classifier
    corr_feature_net = clf_features.residual_basic_block(num_blocks=clf_feat_blocks, l2norm=clf_feat_norm,
                                                              final_conv=final_conv, norm_scale=norm_scale,
                                                              out_dim=out_feature_dim)


    initializer = CorrInitializerLinear(filter_size=filter_size, feature_dim=out_feature_dim)

    optimizer = CorrOptL2SDGN(num_iter=optim_iter, init_step_length=optim_init_step, init_filter_reg=optim_init_reg,
                              test_loss=test_loss, target_sigma=target_sigma)

    corr_module = DiCor(filter_initializer=initializer, filter_optimizer=optimizer)

    net = DiCorNetSimple(backbone_net=backbone_net, corr_feature_net=corr_feature_net, corr_module=corr_module,
                         train_feature_extractor=train_feature_extractor)

    return net


@model_constructor
def dicor_dimp_resnet18(filter_size=1, optim_iter=3, optim_init_step=1.0, optim_init_reg=0.01, test_loss=None, backbone_pretrained=False,
                          train_feature_extractor=False, clf_feat_blocks=1, final_conv=True, clf_feat_norm=True,
                          out_feature_dim=256, filter_initializer='norm', num_dist_bins=10, bin_displacement=0.5, init_gauss_sigma=1.0, score_act='relu', act_param=None,
                 mask_act='sigmoid', mask_init_factor=4.0):
    """Constructs the simple example network that utilizes the DiCor module."""

    # backbone
    backbone_net = backbones.resnet18(pretrained=backbone_pretrained)

    norm_scale = math.sqrt(1.0 / (out_feature_dim * filter_size * filter_size))

    # classifier
    corr_feature_net = clf_features.residual_basic_block(num_blocks=clf_feat_blocks, l2norm=clf_feat_norm,
                                                              final_conv=final_conv, norm_scale=norm_scale,
                                                              out_dim=out_feature_dim)

    if filter_initializer == 'norm':
        initializer = CorrInitializerNorm(filter_size=filter_size)
    elif filter_initializer == 'normbg':
        initializer = CorrInitializerNormBg(filter_size=filter_size)
    elif filter_initializer == 'linear':
        initializer = CorrInitializerLinear(filter_size=filter_size, feature_dim=out_feature_dim)
    else:
        raise ValueError('Unknown initializer')

    optimizer = CorrOptDiMP(num_iter=optim_iter, init_step_length=optim_init_step, init_filter_reg=optim_init_reg,
                              test_loss=test_loss, num_dist_bins=num_dist_bins, bin_displacement=bin_displacement, score_act=score_act, act_param=act_param,
                 mask_act=mask_act, mask_init_factor=mask_init_factor, init_gauss_sigma=init_gauss_sigma)

    corr_module = DiCor(filter_initializer=initializer, filter_optimizer=optimizer)

    net = DiCorNetSimple(backbone_net=backbone_net, corr_feature_net=corr_feature_net, corr_module=corr_module,
                         train_feature_extractor=train_feature_extractor)

    return net



@model_constructor
def dicor_closed_form_resnet18(optim_init_reg=0.01, test_loss=None, backbone_pretrained=False,
                          train_feature_extractor=False, clf_feat_blocks=1, final_conv=True, clf_feat_norm=True,
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

    optimizer = CorrClosedForm(init_filter_reg=optim_init_reg, test_loss=test_loss, num_dist_bins=num_dist_bins,
                               bin_displacement=bin_displacement, init_gauss_sigma=init_gauss_sigma)

    corr_module = DiCorClosedForm(filter_optimizer=optimizer)

    net = DiCorNetSimple(backbone_net=backbone_net, corr_feature_net=corr_feature_net, corr_module=corr_module,
                         train_feature_extractor=train_feature_extractor)

    return net


@model_constructor
def dicor_dimp_unique_resnet18(filter_size=1, optim_iter=3, optim_init_step=1.0, optim_init_reg=0.01, test_loss=None, backbone_pretrained=False,
                          train_feature_extractor=False, clf_feat_blocks=1, final_conv=True, clf_feat_norm=True,
                          out_feature_dim=256, filter_initializer='norm', num_dist_bins=10, bin_displacement=0.5,
                               init_gauss_sigma=1.0, score_act='relu', act_param=None, bg_mask_val=0.0,
                 steplength_reg=0.0, uniqueness_weight=0.0, temperature=1.0,
                               max_temp_scale=1.0):
    """Constructs the simple example network that utilizes the DiCor module."""

    # backbone
    backbone_net = backbones.resnet18(pretrained=backbone_pretrained)

    norm_scale = math.sqrt(1.0 / (out_feature_dim * filter_size * filter_size))

    # classifier
    corr_feature_net = clf_features.residual_basic_block(num_blocks=clf_feat_blocks, l2norm=clf_feat_norm,
                                                              final_conv=final_conv, norm_scale=norm_scale,
                                                              out_dim=out_feature_dim)

    if filter_initializer == 'norm':
        initializer = CorrInitializerNorm(filter_size=filter_size)
    elif filter_initializer == 'normbg':
        initializer = CorrInitializerNormBg(filter_size=filter_size)
    elif filter_initializer == 'linear':
        initializer = CorrInitializerLinear(filter_size=filter_size, feature_dim=out_feature_dim)
    else:
        raise ValueError('Unknown initializer')

    optimizer = CorrOptDiMPUnique(num_iter=optim_iter, init_step_length=optim_init_step, init_filter_reg=optim_init_reg,
                              test_loss=test_loss, num_dist_bins=num_dist_bins, bin_displacement=bin_displacement, score_act=score_act, act_param=act_param,
                                init_gauss_sigma=init_gauss_sigma, bg_mask_val=bg_mask_val,
                                  steplength_reg=steplength_reg, uniqueness_weight=uniqueness_weight,
                                  temperature=temperature, max_temp_scale=max_temp_scale)

    corr_module = DiCor(filter_initializer=initializer, filter_optimizer=optimizer)

    net = DiCorNetSimple(backbone_net=backbone_net, corr_feature_net=corr_feature_net, corr_module=corr_module,
                         train_feature_extractor=train_feature_extractor)

    return net



@model_constructor
def dicor_dimp_testreg_resnet18(filter_size=1, optim_iter=3, optim_init_step=1.0, optim_init_reg=0.01, test_loss=None, backbone_pretrained=False,
                          train_feature_extractor=False, clf_feat_blocks=1, final_conv=True, clf_feat_norm=True,
                          out_feature_dim=256, filter_initializer='norm', num_dist_bins=10, bin_displacement=0.5, init_gauss_sigma=1.0, score_act='relu', act_param=None,
                 mask_act='sigmoid', mask_init_factor=4.0):
    """Constructs the simple example network that utilizes the DiCor module."""

    # backbone
    backbone_net = backbones.resnet18(pretrained=backbone_pretrained)

    norm_scale = math.sqrt(1.0 / (out_feature_dim * filter_size * filter_size))

    # classifier
    corr_feature_net = clf_features.residual_basic_block(num_blocks=clf_feat_blocks, l2norm=clf_feat_norm,
                                                              final_conv=final_conv, norm_scale=norm_scale,
                                                              out_dim=out_feature_dim)

    if filter_initializer == 'norm':
        initializer = CorrInitializerNorm(filter_size=filter_size)
    elif filter_initializer == 'normbg':
        initializer = CorrInitializerNormBg(filter_size=filter_size)
    elif filter_initializer == 'linear':
        initializer = CorrInitializerLinear(filter_size=filter_size, feature_dim=out_feature_dim)
    else:
        raise ValueError('Unknown initializer')

    optimizer = CorrOptDiMPQReg(num_iter=optim_iter, init_step_length=optim_init_step, init_filter_reg=optim_init_reg,
                              test_loss=test_loss, num_dist_bins=num_dist_bins, bin_displacement=bin_displacement, score_act=score_act, act_param=act_param,
                 mask_act=mask_act, mask_init_factor=mask_init_factor, init_gauss_sigma=init_gauss_sigma,
                                reg_kernel_size=3, reg_inter_dim=16, reg_output_dim=16)

    optimizer.reg_layer.weight1.data.normal_(0,8e-2)
    optimizer.reg_layer.weight2.data.normal_(0,8e-2)

    corr_module = DiCor(filter_initializer=initializer, filter_optimizer=optimizer)

    net = DiCorNetSimple(backbone_net=backbone_net, corr_feature_net=corr_feature_net, corr_module=corr_module,
                         train_feature_extractor=train_feature_extractor)

    return net