import math
import torch
import torch.nn as nn
from torch.nn import functional as F


class GaussNLL(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, mean, var, target):
        mse = F.mse_loss(mean, target, reduction='none')
        L = torch.mean(mse / var + torch.log(var))
        return L