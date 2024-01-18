import torch
import torch.nn as nn
import torch.nn.functional as F
import ltr.models.layers.filter as filter_layer
from ltr.models.target_classifier.initializer import FilterPool
import math


def conv(dims, kernel_size=1, stride=1, padding=None, dilation=1, batchnorm=True, output_linear=False):
    layers = []
    if padding is None:
        padding = kernel_size // 2
    for l in range(len(dims)-1):
        layers.append(nn.Conv2d(dims[l], dims[l+1], kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=True))
        if output_linear and l == len(dims) - 2:
            break
        if batchnorm:
            layers.append(nn.BatchNorm2d(dims[l+1]))
        layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


class FilterVariancePredictor(nn.Module):
    def __init__(self, feature_dim, filter_size, feature_stride=16, batchnorm=True):
        super().__init__()
        d = feature_dim
        self.feature_pre_layer = conv([d, d], kernel_size=3, batchnorm=batchnorm)
        self.post_layer = conv([2*d, d, d, d], kernel_size=3, batchnorm=batchnorm, output_linear=True)
        self.filter_pool = FilterPool(filter_size, feature_stride)

    def forward(self, weights, feat, bb, sample_weight=None):
        feat_pool = self.filter_pool(feat.reshape(-1, *feat.shape[-3:]), bb)
        feat_pool = self.feature_pre_layer(feat_pool).reshape(feat.shape[0], -1, *feat_pool.shape[-3:])

        if sample_weight is None:
            feat_pool = feat_pool.mean(dim=0)
        else:
            feat_pool = torch.sum(sample_weight.reshape(-1,1,1,1,1) * feat_pool, dim=0)

        x = torch.cat((weights, feat_pool), dim=1)

        log_var = self.post_layer(x)
        return torch.exp(log_var)


class ResponsePredictor(nn.Module):
    def __init__(self, hidden_dims, kernel_size=1):
        super().__init__()
        dim_config = [2] + list(hidden_dims) + [1]
        self.layers = conv(dim_config, kernel_size=kernel_size, batchnorm=False, output_linear=True)

    def forward(self, response_mean, response_var):
        x = torch.stack((response_mean.reshape(-1, *response_mean.shape[-2:]),
                         response_var.reshape(-1, *response_var.shape[-2:])), dim=1)
        final_response = self.layers(x).reshape(response_mean.shape)
        return final_response
