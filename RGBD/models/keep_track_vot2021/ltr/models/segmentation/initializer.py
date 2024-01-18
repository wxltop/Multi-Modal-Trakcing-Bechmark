import torch.nn as nn
import torch
import torch.nn.functional as F
from ltr.models.layers.blocks import conv_block
from ltr.models.target_classifier.initializer import FilterPool
from ltr.data.bounding_box_utils import masks_to_bboxes


class FilterInitializerZero(nn.Module):
    """Initializes a target classification filter with zeros.
    args:
        filter_size:  Size of the filter.
        feature_dim:  Input feature dimentionality."""

    def __init__(self, filter_size=1, num_filters=1, feature_dim=256, filter_groups=1):
        super().__init__()

        self.filter_size = (num_filters, feature_dim//filter_groups, filter_size, filter_size)

    def forward(self, feat, mask=None):
        assert feat.dim() == 5
        # num_sequences = feat.shape[1] if feat.dim() == 5 else 1
        num_sequences = feat.shape[1]

        return feat.new_zeros(num_sequences, *self.filter_size)


class FilterInitializerMaskPoolSC(nn.Module):
    """Initializes a target classification filter with zeros.
    args:
        filter_size:  Size of the filter.
        feature_dim:  Input feature dimentionality."""

    def __init__(self, conv_dims=(), fc_dims=(), filter_size=1, num_filters=1, feature_dim=256, filter_groups=1,
                 use_sigmoid=True):
        super().__init__()
        conv_layers = []

        d_in = feature_dim
        for d_out in conv_dims:
            conv_layers.append(conv_block(d_in, d_out, 3, padding=1))
            d_in = d_out
        self.conv_layers = nn.Sequential(*conv_layers)

        mask_processing = []
        mask_processing.append(conv_block(1, 8, kernel_size=3, stride=2, padding=1, batch_norm=False))
        mask_processing.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        mask_processing.append(conv_block(8, 8, kernel_size=3, stride=2, padding=1, batch_norm=False))
        mask_processing.append(conv_block(8, 1, kernel_size=3, stride=2, padding=1, batch_norm=False, relu=False))

        if use_sigmoid:
            mask_processing.append(nn.Sigmoid())
        else:
            mask_processing.append(nn.ReLU())

        self.mask_processing = nn.Sequential(*mask_processing)
        fc_layers = []
        d_in = 2 * d_in
        for d_out in fc_dims:
            fc_layers.append(conv_block(d_in, d_out, 1, padding=0))
            d_in = d_out
        fc_layers.append(conv_block(d_in, num_filters * (feature_dim // filter_groups) * filter_size ** 2,
                                    kernel_size=1, padding=0, batch_norm=False, relu=False))
        self.filter_predictor = nn.Sequential(*fc_layers)

        self.filter_size = (num_filters, feature_dim//filter_groups, filter_size, filter_size)

    def forward(self, feat, mask=None):
        assert feat.dim() == 5
        feat = feat[0, ...]
        mask = mask[0, ...]

        shape = feat.shape

        feat = feat.view(-1, *feat.shape[-3:])
        feat = self.conv_layers(feat)

        mask_target = mask.view(-1, 1, *mask.shape[-2:])
        mask_background = 1.0 - mask_target

        mod_target = self.mask_processing(mask_target)
        mod_bg = self.mask_processing(mask_background)

        feat_target = feat * mod_target
        feat_bg = feat * mod_bg

        feat = torch.cat((feat_target, feat_bg), dim=1)
        feat_pool = F.adaptive_avg_pool2d(feat, (1, 1))

        filter = self.filter_predictor(feat_pool)
        filter = filter.view(shape[0], *self.filter_size)
        # num_sequences = feat.shape[1] if feat.dim() == 5 else 1

        return filter


class FilterInitializerMaskROIPoolSC(nn.Module):
    """Initializes a target classification filter with zeros.
    args:
        filter_size:  Size of the filter.
        feature_dim:  Input feature dimentionality."""

    def __init__(self, conv_dims=(), filter_size=1, num_filters=1, feature_dim=256, filter_groups=1):
        super().__init__()
        conv_layers = []

        d_in = feature_dim
        for d_out in conv_dims:
            conv_layers.append(conv_block(d_in, d_out, 3, padding=1))
            d_in = d_out
        conv_layers.append(conv_block(d_in, num_filters * d_in, 3, padding=1))

        self.conv_layers = nn.Sequential(*conv_layers)

        mask_processing = []
        mask_processing.append(conv_block(1, 8, kernel_size=3, stride=2, padding=1, batch_norm=False))
        mask_processing.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        mask_processing.append(conv_block(8, 8, kernel_size=3, stride=2, padding=1, batch_norm=False))
        mask_processing.append(conv_block(8, 1, kernel_size=3, stride=2, padding=1, batch_norm=False, relu=True))

        self.mask_processing = nn.Sequential(*mask_processing)

        self.pool_layer = FilterPool(filter_size=filter_size, feature_stride=16)

        self.filter_size = (num_filters, feature_dim//filter_groups, filter_size, filter_size)

    def forward(self, feat, mask=None):
        assert feat.dim() == 5
        feat = feat[0, ...]
        mask = mask[0, ...]

        shape = feat.shape

        feat = feat.view(-1, *feat.shape[-3:])
        feat = self.conv_layers(feat)

        mask_target = mask.view(-1, 1, *mask.shape[-2:])
        bb = masks_to_bboxes(mask_target, fmt='t')
        mod_target = self.mask_processing(mask_target)

        feat_target = feat * mod_target

        feat_pool = self.pool_layer(feat_target, bb)

        filter = feat_pool.view(shape[0], *self.filter_size)
        # num_sequences = feat.shape[1] if feat.dim() == 5 else 1

        return filter
