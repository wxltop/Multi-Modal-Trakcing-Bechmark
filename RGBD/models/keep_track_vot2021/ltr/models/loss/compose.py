import torch
import torch.nn as nn
from torch.nn import functional as F


identity = lambda x: x

class ComposeLoss(nn.Module):
    def __init__(self, losses, p_losses, transforms_input=None, transforms_target=None):
        """
        Simple loss module that computes weighted combination of a list of losses
        :param losses:
        :param p_losses:
        :param transforms_input:
        :param transforms_target:
        """
        super().__init__()
        self.losses = losses

        # Normalize
        p_total = sum(p_losses)
        self.p_losses = [float(x) / p_total for x in p_losses]

        if transforms_input is None:
            transforms_input = [None for _ in self.losses]
        if transforms_target is None:
            transforms_target = [None for _ in self.losses]

        self.transforms_input = [t if t is not None else identity for t in transforms_input]
        self.transforms_target = [t if t is not None else identity for t in transforms_target]

    def forward(self, input, target):
        l = 0.
        for loss, p, t_i, t_t in zip(self.losses, self.p_losses, self.transforms_input, self.transforms_target):
            l += p * loss(t_i(input), t_t(target))
        return l
