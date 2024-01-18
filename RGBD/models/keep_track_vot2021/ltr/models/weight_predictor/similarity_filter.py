import torch.nn as nn
import torch
from torch.nn import init
import torch.nn.functional as F
from torch.nn.parameter import Parameter

import ltr.models.layers.filter as filter_layer
import ltr.models.layers.activation as activation
from ltr.models.layers.distance import DistanceValueEncoder, DistanceMap
from ltr.models.weight_predictor import features

import math
from pytracking import TensorList



class LinearSimilarityFilter(nn.Module):
    def __init__(self, filter_dim=2048, similarity_feature_dim=64, weight_scaling_activation="sigmoid"):
        super().__init__()
        self.out_dim = similarity_feature_dim
        self.filter_dim = filter_dim
        self.weight_scaling_activation = weight_scaling_activation
        self.conv = nn.Conv2d(self.filter_dim, self.out_dim, kernel_size=3, padding=1)
        self.a = nn.Parameter(-1*torch.ones(1))
        self.b = nn.Parameter(torch.zeros(1))
        self.c = nn.Parameter(torch.zeros(1))

    def forward(self, feat, nseq=1, **kwargs):
        """"""
        nframes, nseq, cin, w, h = feat.size()
        feat = feat.reshape(nframes*nseq, cin, w, h)
        # n, cin, w, h = feat.size()
        # nframes = n // nseq
        z = self.extract_similarity_features(feat, nseq) # (nframes, nseq, C, W, H)
        similarity = self.compute_cosine_similarity(z)   # (nseq, nframes*W*H, nframes*W*H)
        weight_flat = self.compute_weights(similarity)   # (nseq, nframes*W*H)
        weight = weight_flat.view(nseq, nframes, w, h).permute(1, 0, 2, 3).contiguous()  # (nframes, nseq, W, H)

        return weight

    def extract_similarity_features(self, feat, nseq=1):
        n, cin, w, h = feat.size()
        nframes = n // nseq
        z = self.conv(feat)  # (nframes*nseq, C, W, H)
        return z.view(nframes, nseq, self.out_dim, w, h)  # (nframes, nseq, C, W, H)

    def compute_cosine_similarity(self, feat):
        nframes, nseq, c, w, h = feat.size()

        z_reshuffle = feat.permute(1, 2, 0, 3, 4)  # (Anframes, Anseg, AC, AW, AH) -> (Anseq, AC, Anframes, AW, AH)
        z = z_reshuffle.reshape(nseq, c, nframes * w * h) # merge dimensions into one patch dimension

        norm = torch.sqrt(torch.einsum('bjk,bjk->bk', z, z)).view(nseq, 1, nframes * w * h)
        znorm = z/norm

        return torch.einsum('bij,bik->bjk', znorm, znorm)

    def compute_weights(self, sim):
        # feat: (nseq, nframes*W*H, nframes*W*H)
        # learn threshold and scaling for weights
        # a, b, c = self.a, self.b, self.c
        a, b, c = self.a, self.b, self.c
        # a, b, c = 1, 0.0, 0
        # print(a,b,c)


        patch_weight = a*torch.mean(F.relu(sim - b), dim=1) + c
        # patch_weight = torch.mean(F.relu((1 - F.relu(sim)) - b), dim=1)

        # patch_weight = self.a*torch.pow(t, 5) + self.c
        pmin = torch.min(patch_weight, dim=1)[0].view(-1, 1)
        pmax = torch.max(patch_weight, dim=1)[0].view(-1, 1)

        patch_weight_uniform = (patch_weight - pmin)/(pmax - pmin)

        # Normalize sum of weights
        if self.weight_scaling_activation == "softmax":
            weight = F.softmax(patch_weight_uniform, dim=1)
        elif self.weight_scaling_activation == "sigmoid":
            weight = torch.sigmoid(patch_weight_uniform)
        else:
            raise NotImplementedError()

        return weight


# class SimilarityWeightPredictor(nn.Module):
#     def __init__(self, hidden_dim=32, feat_dim=512, kernel_size=1, feature_list=None,
#                  softmax_temp_init=1, use_batch_norm=True, num_bins=16, feat_stride=16):
#         super().__init__()
#         self.a = nn.Parameter(softmax_temp_init*torch.ones(1))
#         self.hidden_dim = hidden_dim
#         self.feat_dim = feat_dim
#         self.feature_list = feature_list
#         self.kernel_size = kernel_size
#         self.use_batch_norm = use_batch_norm
#         self.num_bins = num_bins
#         self.feat_stride = feat_stride
#
#         if self.feature_list is None:
#             self.feature_list = ['proj_test_label_on_mem']
#
#         self.use_raw_train_feats = True if 'raw_train_features' in self.feature_list else False
#
#         self._build_model()
#
#     def _build_model(self):
#         alpha = len(self.feature_list) - 1 if self.use_raw_train_feats else len(self.feature_list)
#         beta = self.feat_dim // 8 if self.use_raw_train_feats else 0
#
#
#         self.d_value_encoder = DistanceValueEncoder(self.num_bins)
#         self.dist_map = DistanceMap(self.num_bins, 1.0)
#
#         self.conv_layer_1 = torch.nn.Conv2d(alpha*self.num_bins, alpha*self.hidden_dim, kernel_size=self.kernel_size, padding=self.kernel_size // 2)
#         self.bn_layer_1 = torch.nn.BatchNorm2d(alpha*self.hidden_dim)
#
#         self.conv_layer_2 = torch.nn.Conv2d(alpha*self.hidden_dim, alpha*2*self.hidden_dim, kernel_size=self.kernel_size, padding=self.kernel_size // 2)
#         self.bn_layer_2 = torch.nn.BatchNorm2d(alpha*2*self.hidden_dim)
#
#         self.conv_layer_3 = torch.nn.Conv2d(alpha*2*self.hidden_dim + beta, alpha*2*self.hidden_dim + beta, kernel_size=self.kernel_size, padding=self.kernel_size // 2)
#         self.bn_layer_3 = torch.nn.BatchNorm2d(alpha*2*self.hidden_dim + beta)
#
#         self.conv_layer_4 = torch.nn.Conv2d(alpha*2*self.hidden_dim + beta, 1,
#                                             kernel_size=self.kernel_size, padding=self.kernel_size // 2)
#
#         if self.use_raw_train_feats:
#             self.conv_layer_1b = torch.nn.Conv2d(self.feat_dim, self.feat_dim // 4, kernel_size=self.kernel_size, padding=self.kernel_size // 2)
#             self.bn_layer_1b = torch.nn.BatchNorm2d(self.feat_dim // 4)
#
#             self.conv_layer_2b = torch.nn.Conv2d(self.feat_dim // 4, self.feat_dim // 8, kernel_size=self.kernel_size, padding=self.kernel_size // 2)
#             self.bn_layer_2b = torch.nn.BatchNorm2d(self.feat_dim // 8)
#
#     def forward(self, test_feat, train_feat, test_label, train_label, train_bb, **kwargs):
#         test_label = test_label[0].unsqueeze(0)
#
#         sim = self.cosine_similarity(train_feat, test_feat)
#         feats_dict, _ = self.prepare_features(sim, train_label, test_label, train_feat, test_feat, train_bb, self.feature_list)
#         w = self.compute_weights(feats_dict, train_label)
#
#         return w
#
#     def cosine_similarity(self, train_feat, test_feat):
#         tr_nframes, nseq, c, w, h = train_feat.size()
#         te_nframes, _, _, _, _ = test_feat.size()
#
#         train_reshuffle = train_feat.permute(1, 2, 0, 3, 4)  # (nframes, nseq, C, W, H) -> (nseq, C, nframes, W, H)
#         test_reshuffle = test_feat.permute(1, 2, 0, 3, 4)  # (nframes, nseq, C, W, H) -> (nseq, C, nframes, W, H)
#         train_reshuffle = train_reshuffle.reshape(nseq, c, tr_nframes * w * h)  # merge dimensions into one patch dimension
#         test_reshuffle = test_reshuffle.reshape(nseq, c, te_nframes * w * h)  # merge dimensions into one patch dimension
#
#         train_norm = torch.sqrt(
#             torch.einsum('bij,bij->bj', train_reshuffle, train_reshuffle)
#         ).view(nseq, 1, tr_nframes * w * h)
#
#         test_norm = torch.sqrt(
#             torch.einsum('bij,bij->bj', test_reshuffle, test_reshuffle)
#         ).view(nseq, 1, te_nframes * w * h)
#
#         train_normalized = train_reshuffle / train_norm  # (nseq, C, tr_nframes*W*H)
#         test_normalized = test_reshuffle / test_norm  # (nseq, C, te_nframes*W*H)
#
#         return torch.einsum('bij,bik->bjk', test_normalized, train_normalized)  # (nseq, te_nframes*w*h, tr_nframes*w*h)
#
#     def project_test_label_on_memory(self, sim, test_label, train_label):
#         # sim  (nseq, te_nframes*W*H, tr_nframes*W*H)
#         # test_label_reshuffle (nseq, te_nframes*W*H, 1)
#         p = F.softmax(self.a * sim, dim=1)  # (nseq, te_nframes*w*h, tr_nframes*w*h)
#
#         tr_nframes, _, _, _ = train_label.size()
#         te_nframes, nseq, w, h = test_label.size()
#         test_label_reshuffle = test_label.permute(1, 0, 2, 3)  # (nframes, nseq, W, H) -> (nseq, nframes, W, H)
#         test_label_reshuffle = test_label_reshuffle.reshape(nseq, te_nframes*w*h, 1)  # (nseq, nframes*W*H, 1) merge dimensions into one patch dimension
#
#         proj_test_label_on_mem = torch.einsum('bij,bik->bkj', test_label_reshuffle, p)  # (nseq, nframes*w*h, 1)
#         proj_test_label_on_mem_reshuffle = proj_test_label_on_mem.reshape(nseq, tr_nframes, w, h, 1).permute(1, 0, 4, 2, 3)
#         proj_test_label_on_mem_reshuffle = proj_test_label_on_mem_reshuffle.reshape(tr_nframes*nseq, 1, w, h)
#
#         return proj_test_label_on_mem_reshuffle
#
#     def project_memory_on_test_label(self, sim, test_label, train_label, num_mem_frames=50):
#         # sim  (nseq, te_nframes*W*H, tr_nframes*W*H)
#         # test_label_reshuffle (nseq, te_nframes*W*H, 1)
#
#         tr_nframes, _, _, _ = train_label.size()
#         te_nframes, nseq, w, h = test_label.size()
#
#         p = F.softmax(1*sim[:, :, :num_mem_frames*w*h], dim=2)  # (nseq, te_nframes*w*h, tr_nframes*w*h)
#
#         train_label_reshuffle = train_label.permute(1, 0, 2, 3)  # (nframes, nseq, W, H) -> (nseq, nframes, W, H)
#
#         train_label_reshuffle = train_label_reshuffle[:, :num_mem_frames, :, :]
#
#         train_label_reshuffle = train_label_reshuffle.reshape(nseq, -1, 1)  # (nseq, nframes*W*H, 1) merge dimensions into one patch dimension
#
#         proj_mem_on_test_label = torch.einsum('bij,bki->bkj', train_label_reshuffle, p)  # (nseq, nframes*w*h, 1)
#         proj_mem_on_test_label_reshuffle = proj_mem_on_test_label.reshape(nseq, te_nframes, w, h, 1).permute(1, 0, 4, 2, 3)
#         proj_mem_on_test_label_reshuffle = proj_mem_on_test_label_reshuffle.reshape(te_nframes*nseq, 1, w, h)
#
#         return proj_mem_on_test_label_reshuffle
#
#     def raw_train_labels(self, train_label):
#         nframes, nseq, w, h = train_label.size()
#         train_label_reshuffle = train_label.reshape(nframes*nseq, 1, h, w)
#
#         return train_label_reshuffle
#
#     def raw_train_features(self, train_feat):
#         nframes, nseq, c, w, h = train_feat.size()
#         train_feat_reshuffle = train_feat.reshape(nframes * nseq, c, w, h)
#
#         return train_feat_reshuffle
#
#     def compute_bbox_center_distance_map(self, train_bb, size):
#         center = ((train_bb[..., :2] + train_bb[..., 2:] / 2) / self.feat_stride).flip((-1,))  # - dmap_offset
#         return self.dist_map(center, size)
#
#     def prepare_features(self, sim, train_label, test_label, train_feat, test_feat, train_bb, feature_list,
#                          debug_feature_list=None):
#         if debug_feature_list is None: debug_feature_list = []
#
#         _, _, _, w, h = train_feat.size()
#
#         train_label = F.interpolate(train_label, size=(w, h), mode='bilinear')
#         test_label = F.interpolate(test_label, size=(w, h), mode='bilinear')
#
#         feats_dict = {}
#
#         for feature in feature_list:
#             if 'raw_train_features' == feature:
#                 feats_dict['raw_train_features'] = self.raw_train_features(train_feat)  # (nframes*nseq, 512, w, h)
#
#             elif 'bbox_distance_map' == feature:
#                 self.dist_map.bin_displacement = math.sqrt(2)*w/(self.num_bins - 1)
#                 feats_dict['bbox_distance_map'] = self.compute_bbox_center_distance_map(train_bb, (w, h))
#
#             elif 'proj_test_label_on_mem' == feature:
#                 inp = self.project_test_label_on_memory(sim, test_label, train_label)  # (nframes*nseq, 1, w, h)
#                 enc = self.d_value_encoder(F.tanh(inp))
#                 feats_dict['proj_test_label_on_mem'] = enc
#
#             else:
#                 raise NotImplementedError('Feature {} does not exist!'.format(feature))
#
#         debug_feats_dict = {}
#
#         if len(debug_feature_list) > 0:
#             for feature in debug_feature_list:
#                 if 'proj_all_mem_on_test_label' == feature:
#                     debug_feats_dict['proj_all_mem_on_test_label'] = self.project_memory_on_test_label(sim, test_label, train_label) # (nframes*nseq, 1, w, h)
#
#                 elif 'proj_gth_mem_on_test_label' == feature:
#                     debug_feats_dict['proj_gth_mem_on_test_label'] = self.project_memory_on_test_label(sim, test_label, train_label, 15) # (nframes*nseq, 1, w, h)
#
#                 else:
#                     raise NotImplementedError('Debug Feature {} does not exist!'.format(feature))
#
#         return feats_dict, debug_feats_dict
#
#     def run_simple_model(self, inp):
#
#         h1 = self.conv_layer_1(inp)
#         bn1 = self.bn_layer_1(h1) if self.use_batch_norm else h1
#         a1 = F.relu(bn1)
#
#         h2 = self.conv_layer_2(a1)
#         bn2 = self.bn_layer_2(h2) if self.use_batch_norm else h2
#         a2 = F.relu(bn2)
#
#         h3 = self.conv_layer_3(a2)
#         bn3 = self.bn_layer_3(h3) if self.use_batch_norm else h3
#         a3 = F.relu(bn3)
#
#         hout = self.conv_layer_4(a3)
#         weights = F.softplus(hout)
#
#         return weights
#
#     def run_two_branch_model(self, inp, feats):
#         # first branch
#         h1a = self.conv_layer_1(inp)
#         bn1a = self.bn_layer_1(h1a) if self.use_batch_norm else h1a
#         a1a = F.relu(bn1a)
#
#         h2a = self.conv_layer_2(a1a)
#         bn2a = self.bn_layer_2(h2a) if self.use_batch_norm else h2a
#         a2a = F.relu(bn2a)
#
#         # second branch
#         h1b = self.conv_layer_1b(feats)
#         bn1b = self.bn_layer_1b(h1b) if self.use_batch_norm else h1b
#         a1b = F.relu(bn1b)
#
#         h2b = self.conv_layer_2b(a1b)
#         bn2b = self.bn_layer_2b(h2b) if self.use_batch_norm else h2b
#         a2b = F.relu(bn2b)
#
#         # concatenate both branches
#         h3 = self.conv_layer_3(torch.cat([a2a, a2b], dim=1))
#         bn3 = self.bn_layer_3(h3) if self.use_batch_norm else h3
#         a3 = F.relu(bn3)
#         hout = self.conv_layer_4(a3)
#
#         weights = F.softplus(hout)
#
#         return weights
#
#     def compute_weights(self, feats_dict, train_label):
#         nframes, nseq, w, h = train_label.size()
#
#         keys = sorted(list(feats_dict.keys()))
#         inp_list = list([feats_dict[key] for key in keys if key != 'raw_train_features'])
#
#         inp = torch.cat(inp_list, dim=1)  # (nframes*nseq, c, w, h)
#         inp = F.interpolate(inp, size=(w, h), mode='bilinear')
#
#         if self.use_raw_train_feats:
#             feats = F.interpolate(feats_dict["raw_train_features"], size=(w, h), mode='bilinear')
#             weights = self.run_two_branch_model(inp, feats)
#         else:
#             weights = self.run_simple_model(inp)
#
#         weights = weights.reshape(nframes, nseq, w, h)
#
#         return weights


class SimilarityWeightPredictor(nn.Module):
    def __init__(
            self, hidden_dim=32, feat_dim=512, kernel_size=1, feature_list=None, softmax_temp_init=1,
            use_batch_norm=True, num_bins=16, feat_stride=16, out_dim=16, out_size=(23, 23), feat_size=(22, 22)
    ):
        super().__init__()
        self.softmax_temp_init = softmax_temp_init
        self.hidden_dim = hidden_dim
        self.feat_dim = feat_dim
        self.feature_list = feature_list
        self.kernel_size = kernel_size
        self.use_batch_norm = use_batch_norm
        self.num_bins = num_bins
        self.feat_stride = feat_stride
        self.out_dim = out_dim
        self.out_size = out_size
        self.feat_size = feat_size

        self.input_dim = 0
        self.num_feats = len(self.feature_list)
        self.feature_module_dict = dict()

        if self.feature_list is None:
            self.feature_list = ['proj_test_label_on_mem']

        self.feature_module_dict = self.build_features(self.feature_list)
        self.input_dim = sum([feat.feat_dim for feat in self.feature_module_dict.values()])
        self._build_model()

    def _build_model(self):
        layers = []
        layers.append(nn.Conv2d(self.input_dim, self.num_feats * self.hidden_dim, kernel_size=self.kernel_size,
                                padding=self.kernel_size // 2))
        if self.use_batch_norm:
            layers.append(nn.BatchNorm2d(self.num_feats * self.hidden_dim))

        layers.append(nn.ReLU())
        layers.append(nn.Conv2d(self.num_feats * self.hidden_dim, self.num_feats * self.hidden_dim,
                                kernel_size=self.kernel_size, padding=self.kernel_size // 2))
        if self.use_batch_norm:
            layers.append(nn.BatchNorm2d(self.num_feats * self.hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Conv2d(self.num_feats * self.hidden_dim, 1, kernel_size=self.kernel_size,
                                padding=self.kernel_size // 2))
        layers.append(nn.Softplus())
        self.add_module('model', nn.Sequential(*layers))

    def build_features(self, feature_list):
        args = dict(
            out_dim=self.out_dim,
            out_size=self.out_size,
            kernel_size=self.kernel_size,
            use_batch_norm=self.use_batch_norm,
            feat_dim=self.feat_dim,
            feat_stride=self.feat_stride,
            feat_size=self.feat_size,
            softmax_temp_init=self.softmax_temp_init,
        )

        feature_module_dict = dict()

        for feat_name in feature_list:
            feat = getattr(features, feat_name)(**args)
            feat.to('cuda')

            feature_module_dict[feat_name] = feat

        return feature_module_dict


    def forward(self, test_feat, train_feat, test_label, train_label, train_bb, train_certainty, **kwargs):
        test_label = test_label[0].unsqueeze(0)
        sim = self.cosine_similarity(train_feat, test_feat)
        feats_dict = self.prepare_features(sim, train_label, test_label, train_feat, test_feat, train_bb,
                                           train_certainty, self.feature_list)
        w = self.compute_weights(feats_dict, train_label)

        return w

    def cosine_similarity(self, train_feat, test_feat):
        tr_nframes, nseq, c, w, h = train_feat.size()
        te_nframes, _, _, _, _ = test_feat.size()

        train_reshuffle = train_feat.permute(1, 2, 0, 3, 4)  # (nframes, nseq, C, W, H) -> (nseq, C, nframes, W, H)
        test_reshuffle = test_feat.permute(1, 2, 0, 3, 4)  # (nframes, nseq, C, W, H) -> (nseq, C, nframes, W, H)
        train_reshuffle = train_reshuffle.reshape(nseq, c, tr_nframes * w * h)  # merge dimensions into one patch dimension
        test_reshuffle = test_reshuffle.reshape(nseq, c, te_nframes * w * h)  # merge dimensions into one patch dimension

        train_norm = torch.sqrt(
            torch.einsum('bij,bij->bj', train_reshuffle, train_reshuffle)
        ).view(nseq, 1, tr_nframes * w * h)

        test_norm = torch.sqrt(
            torch.einsum('bij,bij->bj', test_reshuffle, test_reshuffle)
        ).view(nseq, 1, te_nframes * w * h)

        train_normalized = train_reshuffle / train_norm  # (nseq, C, tr_nframes*W*H)
        test_normalized = test_reshuffle / test_norm  # (nseq, C, te_nframes*W*H)

        return torch.einsum('bij,bik->bjk', test_normalized, train_normalized)  # (nseq, te_nframes*w*h, tr_nframes*w*h)

    def prepare_features(self, sim, train_label, test_label, train_feat, test_feat, train_bb, train_certainty,
                         feature_list):

        feats_value_dict = {}

        for feat_name in feature_list:
            feats_value_dict[feat_name] = self.feature_module_dict[feat_name](train_label=train_label, sim=sim,
                                                                              test_label=test_label, train_bb=train_bb,
                                                                              test_feat=test_feat, train_feat=train_feat,
                                                                              train_certainty=train_certainty)

        return feats_value_dict


    def compute_weights(self, feat_value_dict, train_label):
        nframes, nseq, w, h = train_label.size()

        features = sorted(list(feat_value_dict.keys()))
        inp_list = list([feat_value_dict[key] for key in features])
        inp = torch.cat(inp_list, dim=1)
        weights = self.model(inp)

        return weights.reshape(nframes, nseq, w, h)
