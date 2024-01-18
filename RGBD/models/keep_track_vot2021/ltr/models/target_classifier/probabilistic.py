import torch.nn as nn
import torch
import torch.nn.functional as F
import ltr.models.layers.filter as filter_layer
import math
from pytracking import TensorList


class KLFilter(nn.Module):
    def __init__(self, filter_size, filter_initializer, filter_optimizer=None, feature_extractor=None):
        super().__init__()

        self.filter_size = filter_size

        # Modules
        self.filter_initializer = filter_initializer
        self.filter_optimizer = filter_optimizer
        self.feature_extractor = feature_extractor

        # Init weights
        for m in self.feature_extractor.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, train_feat, test_feat, train_bb, *args, **kwargs):
        """ the bb should be 5d"""

        num_sequences = train_bb.shape[1]

        if train_feat.dim() == 5:
            train_feat = train_feat.reshape(-1, *train_feat.shape[-3:])
        if test_feat.dim() == 5:
            test_feat = test_feat.reshape(-1, *test_feat.shape[-3:])

        # Extract features
        train_feat = self.extract_classification_feat(train_feat, num_sequences)
        test_feat = self.extract_classification_feat(test_feat, num_sequences)

        # Train filter
        filter, filter_iter, losses = self.get_filter(train_feat, train_bb, *args, **kwargs)

        # Classify samples
        test_scores = [self.classify(f, test_feat) for f in filter_iter]

        return test_scores

    def extract_classification_feat(self, feat, num_sequences=None):
        if self.feature_extractor is None:
            return feat
        if num_sequences is None:
            return self.feature_extractor(feat)

        output = self.feature_extractor(feat)
        return output.reshape(-1, num_sequences, *output.shape[-3:])

    def classify(self, weights, feat):
        """Run classifier (filter) on the features (feat)."""

        scores = self.filter_optimizer.score_predictor(weights, feat)

        return scores

    def get_filter(self, feat, bb, *args, **kwargs):
        weights = self.filter_initializer(feat, bb)

        if self.filter_optimizer is not None:
            weights, weights_iter, losses = self.filter_optimizer(TensorList([weights]), feat=feat, bb=bb, *args, **kwargs)
            weights = weights[0]
            weights_iter = [w[0] for w in weights_iter]
        else:
            weights_iter = [weights]
            losses = None

        return weights, weights_iter, losses



class InterpScorePredictor(nn.Module):
    def __init__(self, feat_stride=16, init_filter_reg=1e-2, gauss_sigma=1.0, min_filter_reg=1e-3,
                 init_uni_weight=None, normalize_label=False, label_shrink=0, softmax_reg=None, label_threshold=0.0,
                 interp_method='none', interp_factor=1):
        super().__init__()

        self.feat_stride = feat_stride
        self.filter_reg = nn.Parameter(init_filter_reg * torch.ones(1))
        self.gauss_sigma = gauss_sigma
        self.min_filter_reg = min_filter_reg
        self.uni_weight = 0 if init_uni_weight is None else nn.Parameter(init_uni_weight * torch.ones(1))
        self.normalize_label = normalize_label
        self.label_shrink = label_shrink
        self.softmax_reg = softmax_reg
        self.label_threshold = label_threshold
        self.interp_method = interp_method
        self.interp_factor = interp_factor

    def get_label_density(self, center, output_sz):
        sigma = self.interp_factor * self.gauss_sigma
        center = center.reshape(center.shape[0], -1, center.shape[-1])
        k0 = torch.arange(output_sz[0], dtype=torch.float32).reshape(1, 1, -1, 1).to(center.device)
        k1 = torch.arange(output_sz[1], dtype=torch.float32).reshape(1, 1, 1, -1).to(center.device)
        g0 = torch.exp(-1.0 / (2 * sigma ** 2) * (k0 - center[:,:,0].reshape(*center.shape[:2], 1, 1)) ** 2)
        g1 = torch.exp(-1.0 / (2 * sigma ** 2) * (k1 - center[:,:,1].reshape(*center.shape[:2], 1, 1)) ** 2)
        gauss = (g0 / (2*math.pi*self.gauss_sigma**2)) * g1
        gauss = gauss * (gauss > self.label_threshold).float()
        if self.normalize_label:
            gauss /= (gauss.sum(dim=(-2,-1), keepdim=True) + 1e-8)
        label_dens = (1.0 - self.label_shrink)*((1.0 - self.uni_weight) * gauss + self.uni_weight / (output_sz[0]*output_sz[1]))
        return label_dens

    def init_data(self, weights, feat, bb, sample_weight=None, **kwargs):
        weights = weights[0]

        # Sizes
        num_images = feat.shape[0]
        num_sequences = feat.shape[1] if feat.dim() == 5 else 1
        filter_sz = (weights.shape[-2], weights.shape[-1])
        output_sz = (self.interp_factor * feat.shape[-2] + (weights.shape[-2] + 1) % 2,
                     self.interp_factor * feat.shape[-1] + (weights.shape[-1] + 1) % 2)

        # Get learnable scalars
        reg_weight = (self.filter_reg*self.filter_reg).clamp(min=self.min_filter_reg**2)

        # Compute distance map
        dmap_offset = (torch.Tensor(filter_sz).to(bb.device) % 2) / 2.0
        center = ((bb[..., :2] + bb[..., 2:] / 2) * self.interp_factor / self.feat_stride).flip((-1,)) - dmap_offset
        label_density = self.get_label_density(center, output_sz)

        # Get total sample weights
        if sample_weight is None:
            sample_weight = torch.Tensor([1.0 / num_images]).to(feat.device)
        elif isinstance(sample_weight, torch.Tensor):
            sample_weight = sample_weight.reshape(num_images, num_sequences, 1, 1)

        return label_density, sample_weight, reg_weight

    def forward(self, weights, feat, **kwargs):
        if isinstance(weights, TensorList):
            weights = weights[0]

        # Compute residuals
        scores = filter_layer.apply_filter(feat, weights)

        if self.interp_method == 'none' or self.interp_factor == 1:
            return scores

        is_even = weights[0].shape[-1] % 2 == 0
        interp_sz = (self.interp_factor*feat.shape[-2]+is_even, self.interp_factor*feat.shape[-1]+is_even)
        return F.interpolate(scores, size=interp_sz, mode=self.interp_method, align_corners=is_even)
