import torch
import torch.nn as nn
import torch.nn.functional as F


class SegInitializerNormBg(nn.Module):
    """Initializes the segmentation filter through a simple conv layer.
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

    def forward(self, feat, fg_mask, bg_mask=None):
        """Initialize filter.
        feat: input features (images, sequences, feat_dim, H, W)
        fg_mask: foreground mask (images, sequences, H, W)
        bg_mask: background mask (images, sequences, H, W)
        output: initial filters (sequences, feat_dim, fH, fW)"""

        if bg_mask is None:
            bg_mask = 1 - fg_mask

        feat = feat.view(-1, *feat.shape[-3:])
        feat_weights = F.unfold(feat, self.filter_size, padding=self.filter_size // 2)
        feat_weights = feat_weights.view(*feat.shape[:2], *feat_weights.shape[1:])

        fg_mask = fg_mask.view(*fg_mask.shape[:2], 1, -1)
        bg_mask = fg_mask.view(*bg_mask.shape[:2], 1, -1)

        # Sum spatially and over images
        fg_weights = (feat_weights * fg_mask).sum(dim=(0,3)) / fg_mask.sum(dim=(0,3))
        bg_weights = (feat_weights * bg_mask).sum(dim=(0,3)) / bg_mask.sum(dim=(0,3))

        ff = (fg_weights * fg_weights).sum(dim=1, keepdim=True)
        bb = (bg_weights * bg_weights).sum(dim=1, keepdim=True)
        fb = (fg_weights * bg_weights).sum(dim=1, keepdim=True)

        den = (ff*bb - fb*fb).clamp(1e-6)
        fg_scale = self.target_fg * bb - self.target_bg * fb
        bg_scale = self.target_fg * fb - self.target_bg * ff
        weights = (fg_scale * fg_weights - bg_scale * bg_weights) / den

        weights = weights.reshape(feat.shape[1], feat.shape[-3], self.filter_size, self.filter_size)
        return weights