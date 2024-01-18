import torch.nn as nn
import torch
from torch.nn import functional as F


def cross_correlation_loss(x):
    """Should be 3-dimensional"""

    batch_size = x.shape[0]
    feat_dim = x.shape[1]

    corr = torch.matmul(x, x.permute(0,2,1))
    mask = 1.0 - torch.eye(feat_dim, device=x.device).unsqueeze(0)

    loss = (mask * corr.abs()).mean()

    return loss


def normalized_cross_correlation_loss(x, eps=1e-5):
    """Should be 3-dimensional"""

    batch_size = x.shape[0]
    feat_dim = x.shape[1]

    corr = torch.matmul(x, x.permute(0,2,1))
    std = torch.sum(x*x, dim=-1).sqrt()
    norm_corr = (corr.abs() / (std.view(std.shape[0],-1,1) + eps)) / (std.view(std.shape[0],1,-1) + eps)

    loss = norm_corr.mean() - 1.0 / feat_dim

    return loss


class NormalizedCrossCorrelationLoss(nn.Module):
    """
    """
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        """Should be 3-dimensional"""
        return normalized_cross_correlation_loss(x, self.eps)