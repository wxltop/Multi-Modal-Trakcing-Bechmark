import math
import torch
import torch.nn as nn
from torch.nn import functional as F


class BinThsOverlapLoss(nn.Module):

    def __init__(self, overlap_pos_ths=0.5, overlap_neg_ths=0.0, enable_sigmoid=True):
        super().__init__()
        self.overlap_pos_ths = overlap_pos_ths
        self.overlap_neg_ths = overlap_neg_ths
        self.enable_sigmoid = enable_sigmoid

        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, scores, overlap):
        scores = scores.view(-1)
        overlap = overlap.view(-1)
        labels = torch.zeros_like(overlap)
        labels[overlap > self.overlap_pos_ths] = 1.

        mask = (overlap > self.overlap_pos_ths) | (overlap <= self.overlap_neg_ths)

        loss = self.loss(scores[mask], labels[mask])

        if self.enable_sigmoid:
            probs = torch.sigmoid(scores)
        else:
            probs = torch.clamp(scores, min=0., max=1.)


        stats = {'MaskedLabels': (labels[mask] == 0),
                 'MaskedProbs': 1 - probs[mask],
                 'Probs': probs}

        return loss, stats

