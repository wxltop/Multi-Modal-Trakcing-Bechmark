import math
import torch
import torch.nn as nn
from torch.nn import functional as F

from pytracking import dcf, TensorList


class PeakClassificationLoss(nn.Module):

    def __init__(self, enable_sigmoid=True):
        super().__init__()
        self.enable_sigmoid = enable_sigmoid

        self.loss = nn.CrossEntropyLoss()

    def forward(self, peak_scores, peak_coords, train_ys):
        train_ys = train_ys[1].reshape(-1,1,23,23)
        val, coord = dcf.max2d(train_ys)
        # dist_to_gth = torch.tensor([torch.min(torch.sum((c - pc) ** 2, dim=1)) for c, pc in zip(coord, peak_coords)])
        gth_peak_idx = torch.tensor([torch.argmin(torch.sum((c - pc)**2, dim=1)) for c, pc in zip(coord, peak_coords)])

        losses = torch.cat([self.loss(pc.view(1, -1), gi.view(1,)) for pc, gi in zip(peak_scores, gth_peak_idx)])
        loss_mean = torch.mean(losses[losses>0])
        return loss_mean


class PeakClassificationLossV2(nn.Module):

    def __init__(self, enable_sigmoid=True):
        super().__init__()
        self.enable_sigmoid = enable_sigmoid

        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, peak_scores, peak_coords, train_ys):
        train_ys = train_ys[1].reshape(-1,1,23,23)
        val, coord = dcf.max2d(train_ys)
        # dist_to_gth = torch.tensor([torch.min(torch.sum((c - pc) ** 2, dim=1)) for c, pc in zip(coord, peak_coords)])
        # gth_peak_idx = torch.tensor([torch.argmin(torch.sum((c - pc)**2, dim=1)) for c, pc in zip(coord, peak_coords)])
        # torch.nn.functional.one_hot(target)

        losses = []
        for v, c, pc, ps in zip(val, coord, peak_coords, peak_scores):
            num_peaks = pc.shape[0]
            dists = torch.sqrt(torch.sum((c - pc)**2, dim=1).float())
            gth_peak_idx = torch.argmin(torch.sum((c - pc) ** 2, dim=1))
            if dists[gth_peak_idx] <= 2 and v > 0:
                labels = F.one_hot(gth_peak_idx, num_peaks).view(ps.shape)
            else:
                labels = torch.zeros_like(ps)
            losses.append(self.loss(ps, labels.float()))

        loss_mean = torch.mean(torch.stack(losses))
        return loss_mean
