import math
import torch
import torch.nn as nn
from torch.nn import functional as F


class BCEWithLogitsLoss(nn.Module):
    def __init__(self, ignore_index=-100):
        super().__init__()
        self.ignore_index = ignore_index
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, input, target):
        input = input.view(-1)
        target = target.view(-1)

        ids = target != self.ignore_index
        return self.bce_loss(input[ids], target[ids])


class MeanIou(nn.Module):
    def __init__(self, class_names, ignore_index=None):
        super().__init__()
        self.class_names = class_names
        self.ignore_index = ignore_index

    def forward(self, input, target, class_map):
        assert input.dim() == 4 and target.dim() == 4

        for input_i, target_i, map_i in zip(input, target, class_map):
            for id, class_name in map_i.items():
                pass
        shape = input.shape
        input = input.view()
        pass