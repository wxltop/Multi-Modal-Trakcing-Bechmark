import torch
import torch.nn as nn
import torch.nn.functional as F

from pytracking import dcf, TensorList
from ltr.models.layers.distance import DistanceValueEncoder
from ltr import model_constructor
from ltr.models.loss.peak_prediction_loss import PeakClassificationLoss


EPS = 0.01


@model_constructor
def learned_peak_predictor_v1(peak_th=0.05, ks=5):
    net = PeakPredictorV1(peak_th=peak_th, ks=ks)
    return net

@model_constructor
def learned_peak_predictor_v2(peak_th=0.05, ks=5):
    net = PeakPredictorV2(peak_th=peak_th, ks=ks)
    return net


def find_local_maxima(scores, th, ks):
    ndims = scores.ndim

    if ndims == 2:
        scores = scores.view(1, 1, scores.shape[0], scores.shape[1])

    scores_max = F.max_pool2d(scores, kernel_size=ks, stride=1, padding=ks//2)

    peak_mask = (scores == scores_max) & (scores > th)
    coords = torch.nonzero(peak_mask)
    intensities = scores[peak_mask]

    # Highest peak first
    idx_maxsort = torch.argsort(-intensities)
    coords = coords[idx_maxsort]
    intensities = intensities[idx_maxsort]

    if ndims == 4:

        coords_batch, intensities_batch, = TensorList(), TensorList()
        for i in range(scores.shape[0]):
            mask = (coords[:, 0] == i)
            coords_batch.append(coords[mask, 2:])
            intensities_batch.append(intensities[mask])
    else:
        coords_batch = coords[:, 2:]
        intensities_batch = intensities

    return coords_batch, intensities_batch


class PeakPredictorV1(nn.Module):
    def __init__(self, peak_th=0.05, ks=5):
        super().__init__()
        self.peak_th = peak_th
        self.ks = ks

        self.mlp_layers = [

            nn.Conv1d(in_channels=2, out_channels=8, kernel_size=1, bias=True),
            nn.BatchNorm1d(num_features=8),
            nn.ReLU(),
            nn.Conv1d(in_channels=8, out_channels=8, kernel_size=1, bias=True),
            nn.BatchNorm1d(num_features=8),
            nn.ReLU(),
            nn.Conv1d(in_channels=8, out_channels=1, kernel_size=1, bias=True),
        ]

        self.mlp = nn.Sequential(*self.mlp_layers)

    def compute_peak_probabilities(self, score_old, score_cur, anno_label_old):
        target_score = torch.stack([score_old, score_cur])
        preds, coords, scores = self.forward(target_score, anno_label_old)
        peak_probs = torch.sigmoid(preds[0]).view(-1)
        peak_coords = coords[0]
        peak_scores = scores[0]
        return peak_probs, peak_coords, peak_scores

    def forward(self, target_scores, anno_label, **kwargs):
        train_y_old = anno_label[0].reshape(-1, 1, 23, 23)

        scores_old = target_scores[0].reshape(-1, 1, 23, 23)
        scores_cur = target_scores[1].reshape(-1, 1, 23, 23)

        peak_coords_old, peak_scores_old = find_local_maxima(scores_old, th=self.peak_th, ks=self.ks)
        peak_coords_cur, peak_scores_cur = find_local_maxima(scores_cur, th=self.peak_th, ks=self.ks)

        val, coord = dcf.max2d(train_y_old)
        # dist_to_gth = torch.tensor([torch.min(torch.sum((c - pc) ** 2, dim=1)) for c, pc in zip(coord, peak_coords_old)])
        gth_peak_idx = torch.tensor([torch.argmin(torch.sum((c - pc) ** 2, dim=1)) for c, pc in zip(coord, peak_coords_old)])

        v_old = TensorList([el[idx] - el for idx, el in zip(gth_peak_idx, peak_coords_old)])
        f_old = TensorList([torch.sum(s.unsqueeze(1)*v, dim=0) for s, v in zip(peak_scores_old, v_old)])

        v_cur = TensorList([el[:, None, :] - el[None, :, :] for el in peak_coords_cur])
        f_cur = TensorList([torch.sum(s.view(1,-1,1)*v, dim=1) for s, v in zip(peak_scores_cur, v_cur)])

        err = TensorList([torch.sqrt(torch.sum((old - cur)**2, dim=1)) for old, cur in zip(f_old, f_cur)])

        # e_old = TensorList([torch.cat([f, s[0].view(1,)], dim=0) for f, s in zip(f_old, peak_scores_old)])
        # e_cur = TensorList([torch.cat([f, s.view(-1, 1)], dim=1) for f, s in zip(f_cur, peak_scores_cur)])

        peak_feats_tl = TensorList([torch.stack([e, s]) for e, s in zip(err, peak_scores_cur)])
        peak_feats = torch.cat(peak_feats_tl, dim=1).permute(1, 0)
        batch_ids = torch.cat([i*torch.ones_like(e) for i, e in enumerate(err)]).long()

        # Normalize features at input? Or translate distance of graph embedding into Probability?
        match_pred = self.mlp(peak_feats.unsqueeze(2).float()).reshape(-1)

        match_pred_tl = TensorList([match_pred[batch_ids == i] for i in range(scores_old.shape[0])])

        return match_pred_tl, peak_coords_cur, peak_scores_cur


class PeakEmbedding(nn.Module):
    def __init__(self, in_channels=4, out_channels=8):
        super().__init__()
        self.mlp_base_embedding_layers = [
            nn.Conv1d(in_channels=in_channels, out_channels=8, kernel_size=1, bias=True),
            nn.BatchNorm1d(num_features=8),
            nn.ReLU(),
            nn.Conv1d(in_channels=8, out_channels=8, kernel_size=1, bias=True),
            nn.BatchNorm1d(num_features=8),
            nn.ReLU(),
        ]

        self.mlp_feature_embedding_layers = [
            nn.Conv1d(in_channels=8, out_channels=out_channels, kernel_size=1, bias=True)
        ]

        self.mlp_embedding_weights_layers = [
            nn.Conv1d(in_channels=8, out_channels=1, kernel_size=1, bias=True)
        ]

        self.mlp_base_embedding = nn.Sequential(*self.mlp_base_embedding_layers)
        self.mlp_feature_embedding = nn.Sequential(*self.mlp_feature_embedding_layers)
        self.mlp_embedding_weights = nn.Sequential(*self.mlp_embedding_weights_layers)

    def forward(self, x):
        base_feat = self.mlp_base_embedding(x)
        feat_embedding = self.mlp_feature_embedding(base_feat)
        feat_embedding_weights = self.mlp_embedding_weights(base_feat)
        return feat_embedding_weights, feat_embedding


class PeakDiscriminator(nn.Module):
    def __init__(self, in_channels=1+2*8):
        super().__init__()
        self.mlp_layers = [
            nn.Conv1d(in_channels=in_channels, out_channels=32, kernel_size=1, bias=True),
            nn.BatchNorm1d(num_features=32),
            nn.ReLU(),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=1, bias=True),
            nn.BatchNorm1d(num_features=64),
            nn.ReLU(),
            nn.Conv1d(in_channels=64, out_channels=1, kernel_size=1, bias=True)
        ]

        self.mlp = nn.Sequential(*self.mlp_layers)

    def forward(self, x):
        return self.mlp(x)


class PeakPredictorV2(nn.Module):
    def __init__(self, peak_th=0.05, ks=5):
        super().__init__()
        self.peak_th = peak_th
        self.ks = ks

        self.peak_embedding = PeakEmbedding()
        self.peak_disc = PeakDiscriminator()

    def compute_peak_probabilities(self, score_old, score_cur, anno_label_old):
        target_score = torch.stack([score_old, score_cur])
        preds, coords, scores = self.forward(target_score, anno_label_old)
        peak_probs = torch.sigmoid(preds[0]).view(-1)
        peak_coords = coords[0]
        peak_scores = scores[0]
        return peak_probs, peak_coords, peak_scores


    def forward(self, target_scores, anno_label, **kwargs):
        train_y_old = anno_label[0].reshape(-1, 1, 23, 23)

        scores_old = target_scores[0].reshape(-1, 1, 23, 23)
        scores_cur = target_scores[1].reshape(-1, 1, 23, 23)

        # extract img_coords and height of each peak
        peak_coords_old, peak_scores_old = find_local_maxima(scores_old, th=self.peak_th, ks=self.ks)
        peak_coords_cur, peak_scores_cur = find_local_maxima(scores_cur, th=self.peak_th, ks=self.ks)

        # check which of the peaks is the correct one (closest to annotation)
        val, coord = dcf.max2d(train_y_old)
        # dist_to_gth = torch.tensor([torch.min(torch.sum((c - pc) ** 2, dim=1)) for c, pc in zip(coord, peak_coords_old)])
        gth_peak_idx = torch.tensor([torch.argmin(torch.sum((c - pc) ** 2, dim=1)) for c, pc in zip(coord, peak_coords_old)])

        # compute difference between all peaks.
        v_old = TensorList([el[idx] - el for idx, el in zip(gth_peak_idx, peak_coords_old)])
        v_cur = TensorList([el[:, None, :] - el[None, :, :] for el in peak_coords_cur])

        # prepare height features for computing the embedding
        s_center_old = TensorList([s[idx].repeat(s.shape[0]) for idx, s in zip(gth_peak_idx, peak_scores_old)])
        s_leaves_old = TensorList([s for s in peak_scores_old])

        s_leaves_cur = TensorList([s.view(1, -1).repeat(s.shape[0], 1) for s in peak_scores_cur])
        s_center_cur = TensorList([s.view(-1, 1).repeat(1, s.shape[0]) for s in peak_scores_cur])

        # build features for previous frame and selected peak
        vss_old_tl = TensorList([torch.cat([v.float(), sc.unsqueeze(1).float(), sl.unsqueeze(1).float()], dim=1)
                                 for v, sc, sl in zip(v_old, s_center_old, s_leaves_old)])

        vss_cur_tl = TensorList([torch.cat([v.float(), sc.unsqueeze(2).float(), sl.unsqueeze(2).float()], dim=2)
                                 for v, sc, sl in zip(v_cur, s_center_cur, s_leaves_cur)])

        vss_old = torch.cat(vss_old_tl)
        vss_cur = torch.cat([v.reshape(-1, 4) for v in vss_cur_tl])

        batch_ids_old = torch.cat([i * torch.ones(v.shape[0]) for i, v in enumerate(vss_old_tl)]).long()
        batch_ids_cur = torch.cat([i * torch.ones(v.shape[0] * v.shape[1]) for i, v in enumerate(vss_cur_tl)]).long()

        embed_old_weights, embed_old_feats = self.peak_embedding(vss_old.unsqueeze(2))
        embed_cur_weights, embed_cur_feats = self.peak_embedding(vss_cur.unsqueeze(2))


        embed_old_feats_tl = TensorList([embed_old_feats[batch_ids_old == i].squeeze(2) for i in range(len(vss_old_tl))])
        try:
            embed_cur_feats_tl = TensorList([embed_cur_feats[batch_ids_cur == i].reshape(vss_cur_tl[i].shape[0], vss_cur_tl[i].shape[1], -1) for i in range(len(vss_cur_tl))])
        except:
            breakpoint()

        weight_old_tl = TensorList([torch.softmax(embed_old_weights[batch_ids_old == i], dim=0).squeeze(2) for i in range(len(vss_old_tl))])
        weight_cur_tl = TensorList([torch.softmax(embed_cur_weights[batch_ids_cur == i].reshape(vss_cur_tl[i].shape[:2]), dim=1)
                                    for i in range(len(vss_cur_tl))])

        embed_old_tl = TensorList([torch.sum(w*f, dim=0) for w, f in zip(weight_old_tl, embed_old_feats_tl)])
        embed_cur_tl = TensorList([torch.sum(w.unsqueeze(2)*f, dim=1) for w, f in zip(weight_cur_tl, embed_cur_feats_tl)])


        feats_tl = TensorList([torch.cat([s.view(-1, 1).float(), cur, old.view(1, -1).repeat(cur.shape[0], 1)], dim=1)
                               for s, cur, old in zip(peak_scores_cur, embed_cur_tl, embed_old_tl)])

        feats = torch.cat(feats_tl)
        batch_ids = torch.cat([i*torch.ones(f.shape[0]) for i, f in enumerate(feats_tl)])

        match_pred = self.peak_disc(feats.unsqueeze(2)).squeeze(2)

        match_pred_tl = TensorList([match_pred[batch_ids == i] for i in range(len(feats_tl))])

        return match_pred_tl, peak_coords_cur, peak_scores_cur



# if __name__ == '__main__':
#     import numpy as np
#     import pandas as pd
#     frame_old = np.array(pd.read_csv('../layers/hand-09-frame-44.csv', index_col=None, header=None).values[::-1])
#     frame_cur = np.array(pd.read_csv('../layers/hand-09-frame-45.csv', index_col=None, header=None).values[::-1])
#
#     frame_old = torch.tensor(frame_old).reshape(1, 1, 23, 23)
#     frame_cur = torch.tensor(frame_cur).reshape(1, 1, 23, 23)
#
#
#     scores_old = torch.cat([frame_old, frame_old], dim=0)
#     scores_cur = torch.cat([frame_cur, frame_old], dim=0)
#     scores = torch.stack([scores_old, scores_cur])
#
#     anno_label = torch.zeros_like(scores)
#     anno_label[:, :, :, 18, 11] = 1.
#
#     net = PeakPredictorV2(peak_th=0.05)
#     # loss = PeakClassificationLoss()
#
#     peak_scores, peak_coords = net(scores, anno_label)
#
#     # l = loss(peak_scores, peak_coords, scores_old)



