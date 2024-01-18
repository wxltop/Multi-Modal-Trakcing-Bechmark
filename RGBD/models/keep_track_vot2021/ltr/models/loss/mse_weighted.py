import torch
import torch.nn as nn
from torch.nn import functional as F


class MSEWeighted(nn.Module):
    def __init__(self, inverse_weight=True, eps=0.0):
        super().__init__()
        self.eps = eps
        self.inverse_weight = inverse_weight

    def forward(self, scores, gt, weight):
        if self.inverse_weight:
            loss = (scores - gt)**2 * (1 / (weight + self.eps))
        else:
            loss = (scores - gt)**2 * weight

        return loss.mean()
