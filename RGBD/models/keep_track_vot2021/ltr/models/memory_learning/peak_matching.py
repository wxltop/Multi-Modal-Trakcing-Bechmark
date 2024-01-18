import torch
import torch.utils.checkpoint

from torch import nn

import time
import logging
import re

from pathlib import Path
from collections import OrderedDict
from copy import deepcopy, copy
from abc import ABCMeta, abstractmethod

from ltr import model_constructor
import ltr.models.backbone as backbones


class BaseModel(nn.Module, metaclass=ABCMeta):
    """
    What the child model is expect to declare:
        default_conf: dictionary of the default configuration of the model.
        It overwrites base_default_conf in BaseModel, and it is overwritten by
        the user-provided configuration passed to __init__.
        Configurations can be nested.
        required_data_keys: list of expected keys in the input data dictionary.
        strict_conf (optional): boolean. If false, BaseModel does not raise
        an error when the user provides an unknown configuration entry.
        _init(self, conf): initialization method, where conf is the final
        configuration object (also accessible with `self.conf`). Accessing
        unkown configuration entries will raise an error.
        _forward(self, data): method that returns a dictionary of batched
        prediction tensors based on a dictionary of batched input data tensors.
        loss(self, pred, data): method that returns a dictionary of losses,
        computed from model predictions and input data. Each loss is a batch
        of scalars, i.e. a torch.Tensor of shape (B,).
        The total loss to be optimized has the key `'total'`.
        metrics(self, pred, data): method that returns a dictionary of metrics,
        each as a batch of scalars.
    """
    base_default_conf = {
        'name': None,
        'trainable': True,  # if false: do not optimize this model parameters
        'freeze_batch_normalization': False,  # use test-time statistics
    }
    default_conf = {}
    # required_data_keys = []
    # strict_conf = True

    def __init__(self, conf=None):
        """Perform some logic and call the _init method of the child model."""
        super().__init__()
        self.conf = {}
        self.conf.update(**self.base_default_conf)
        self.conf.update(**self.default_conf)
        if conf is not None:
            self.conf.update(**conf)

        self.required_data_keys = copy(self.required_data_keys)
        # default_conf = OmegaConf.merge(
        #         OmegaConf.create(self.base_default_conf),
        #         OmegaConf.create(self.default_conf))
        # if self.strict_conf:
        #     OmegaConf.set_struct(default_conf, True)
        # if isinstance(conf, dict):
        #     conf = OmegaConf.create(conf)
        # self.conf = conf = OmegaConf.merge(default_conf, conf)
        # OmegaConf.set_readonly(conf, True)
        # OmegaConf.set_struct(conf, True)
        # self.required_data_keys = copy(self.required_data_keys)

        if not self.conf['trainable']:
            for p in self.parameters():
                p.requires_grad = False

    def train(self, mode=True):
        super().train(mode)

        def freeze_bn(module):
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
        if self.conf['freeze_batch_normalization']:
            self.apply(freeze_bn)

        return self

    def forward(self, data):
        """Check the data and call the _forward method of the child model."""
        for key in self.required_data_keys:
            assert key in data, 'Missing key {} in data'.format(key)
        return self._forward(data)

    @abstractmethod
    def _forward(self, data):
        """To be implemented by the child class."""
        raise NotImplementedError



def MLP(channels, do_bn=True):
    n = len(channels)
    layers = []
    for i in range(1, n):
        layers.append(
            nn.Conv1d(channels[i - 1], channels[i], kernel_size=1, bias=True))
        if i < (n-1):
            if do_bn:
                layers.append(nn.BatchNorm1d(channels[i]))
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


def normalize_keypoints(kpts, shape_or_size):
    if isinstance(shape_or_size, (tuple, list)):
        # it's a shape
        h, w = shape_or_size[-2:]
        size = kpts.new_tensor([[w, h]])
    else:
        # it's a size
        assert isinstance(shape_or_size, torch.Tensor)
        size = shape_or_size.to(kpts)
    c = size / 2
    f = size.max(1, keepdim=True).values * 0.7  # somehow we used 0.7 for SG
    return (kpts - c[:, None, :]) / f[:, None, :]


class KeypointEncoder(nn.Module):
    def __init__(self, feature_dim, layers):
        super().__init__()
        self.encoder = MLP([3] + list(layers) + [feature_dim])
        nn.init.constant_(self.encoder[-1].bias, 0.0)

    def forward(self, kpts, scores):
        inputs = [kpts.transpose(1, 2), scores.unsqueeze(1)]
        return self.encoder(torch.cat(inputs, dim=1))


def attention(query, key, value):
    dim = query.shape[1]
    scores = torch.einsum('bdhn,bdhm->bhnm', query, key) / dim**.5
    prob = torch.nn.functional.softmax(scores, dim=-1)
    return torch.einsum('bhnm,bdhm->bdhn', prob, value), prob


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model):
        super().__init__()
        assert d_model % h == 0
        self.dim = d_model // h
        self.h = h
        self.merge = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.proj = nn.ModuleList([deepcopy(self.merge) for _ in range(3)])

    def forward(self, query, key, value):
        b = query.size(0)
        query, key, value = [l(x).view(b, self.dim, self.h, -1)
                             for l, x in zip(self.proj, (query, key, value))]
        x, _ = attention(query, key, value)
        # self.prob.append(prob)
        return self.merge(x.contiguous().view(b, self.dim*self.h, -1))


class AttentionalPropagation(nn.Module):
    def __init__(self, num_dim, num_heads):
        super().__init__()
        self.attn = MultiHeadedAttention(num_heads, num_dim)
        self.mlp = MLP([num_dim*2, num_dim*2, num_dim])
        nn.init.constant_(self.mlp[-1].bias, 0.0)

    def forward(self, x, source):
        message = self.attn(x, source, source)
        return self.mlp(torch.cat([x, message], dim=1))


class GNNLayer(nn.Module):
    def __init__(self, feature_dim, layer_type):
        super().__init__()
        self.update = AttentionalPropagation(feature_dim, 4)
        assert layer_type in ['cross', 'self']
        self.type = layer_type

    def forward(self, desc0, desc1):
        if self.type == 'cross':
            src0, src1 = desc1, desc0
        elif self.type == 'self':
            src0, src1 = desc0, desc1
        else:
            raise ValueError(self.type)
        delta0, delta1 = self.update(desc0, src0), self.update(desc1, src1)
        desc0, desc1 = (desc0 + delta0), (desc1 + delta1)
        return desc0, desc1


class AttentionalGNN(nn.Module):
    def __init__(self, feature_dim, layer_types, checkpointed=False):
        super().__init__()
        self.checkpointed = checkpointed
        self.layers = nn.ModuleList([
            GNNLayer(feature_dim, layer_type) for layer_type in layer_types])

    def forward(self, desc0, desc1):
        for layer in self.layers:
            if self.checkpointed:
                desc0, desc1 = torch.utils.checkpoint.checkpoint(
                        layer, desc0, desc1, preserve_rng_state=False)
            else:
                desc0, desc1 = layer(desc0, desc1)
        return desc0, desc1


def log_sinkhorn_iterations(Z, log_mu, log_nu, iters):
    u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
    for _ in range(iters):
        u = log_mu - torch.logsumexp(Z + v.unsqueeze(1), dim=2)
        v = log_nu - torch.logsumexp(Z + u.unsqueeze(2), dim=1)
    return Z + u.unsqueeze(2) + v.unsqueeze(1)


def log_optimal_transport(scores, alpha, iters):
    b, m, n = scores.shape
    one = scores.new_tensor(1)
    ms, ns = (m*one).to(scores), (n*one).to(scores)

    bins0 = alpha.expand(b, m, 1)
    bins1 = alpha.expand(b, 1, n)
    alpha = alpha.expand(b, 1, 1)

    couplings = torch.cat([torch.cat([scores, bins0], -1),
                           torch.cat([bins1, alpha], -1)], 1)

    norm = - (ms + ns).log()
    log_mu = torch.cat([norm.expand(m), ns.log()[None] + norm])
    log_nu = torch.cat([norm.expand(n), ms.log()[None] + norm])
    log_mu, log_nu = log_mu[None].expand(b, -1), log_nu[None].expand(b, -1)

    Z = log_sinkhorn_iterations(couplings, log_mu, log_nu, iters)
    Z = Z - norm  # multiply probabilities by M+N
    return Z


def log_double_softmax(scores, bin_score):
    b, m, n = scores.shape
    bin_ = bin_score[None, None, None]
    scores0 = torch.cat([scores, bin_.expand(b, m, 1)], 2)
    scores1 = torch.cat([scores, bin_.expand(b, 1, n)], 1)
    scores0 = torch.nn.functional.log_softmax(scores0, 2)
    scores1 = torch.nn.functional.log_softmax(scores1, 1)
    scores = scores.new_full((b, m+1, n+1), 0)
    scores[:, :m, :n] = (scores0[:, :, :n] + scores1[:, :m, :]) / 2
    scores[:, :-1, -1] = scores0[:, :, -1]
    scores[:, -1, :-1] = scores1[:, -1, :]
    return scores


def arange_like(x, dim):
    return x.new_ones(x.shape[dim]).cumsum(0) - 1  # traceable in 1.1


class SuperGlue(BaseModel):
    default_conf = {
        'input_dim': 256,
        'descriptor_dim': 256,
        'bottleneck_dim': None,
        'weights': 'indoor',
        'keypoint_encoder': [32, 64, 128, 256],
        'GNN_layers': ['self', 'cross'] * 9,
        'output_normalization': 'sinkhorn',
        'num_sinkhorn_iterations': 50,
        'filter_threshold': 0.2,
        'checkpointed': False,
        'loss': {
            'nll_weight': 1.,
            'nll_balancing': 0.5,
            'reward_weight': 0.,
            'bottleneck_l2_weight': 0.,
        },
    }
    required_data_keys = [
            'keypoints0', 'keypoints1',
            'descriptors0', 'descriptors1',
            'keypoint_scores0', 'keypoint_scores1']

    def __init__(self, conf=None):
        super().__init__(conf=conf)
        if self.conf['bottleneck_dim'] is not None:
            self.bottleneck_down = nn.Conv1d(
                self.conf['input_dim'], self.conf['bottleneck_dim'],
                kernel_size=1, bias=True)
            self.bottleneck_up = nn.Conv1d(
                self.conf['bottleneck_dim'], self.conf['input_dim'],
                kernel_size=1, bias=True)
            nn.init.constant_(self.bottleneck_down.bias, 0.0)
            nn.init.constant_(self.bottleneck_up.bias, 0.0)

        if self.conf['input_dim'] != self.conf['descriptor_dim']:
            self.input_proj = nn.Conv1d(
                self.conf['input_dim'], self.conf['descriptor_dim'],
                kernel_size=1, bias=True)
            nn.init.constant_(self.input_proj.bias, 0.0)

        self.kenc = KeypointEncoder(self.conf['descriptor_dim'], self.conf['keypoint_encoder'])

        if not self.conf['skip_gnn']:
            self.gnn = AttentionalGNN(
                self.conf['descriptor_dim'], self.conf['GNN_layers'], self.conf['checkpointed'])

        self.final_proj = nn.Conv1d(
            self.conf['descriptor_dim'], self.conf['descriptor_dim'], kernel_size=1, bias=True)
        nn.init.constant_(self.final_proj.bias, 0.0)
        nn.init.orthogonal_(self.final_proj.weight, gain=1)

        bin_score = torch.nn.Parameter(torch.tensor(0.0))
        self.register_parameter('bin_score', bin_score)

        # if self.conf['weights']:
        #     assert self.conf['weights'] in ['indoor', 'outdoor']
        #     path = Path(DATA_PATH, 'weights/superglue')
        #     path = path / 'superglue_{}.pth'.format(self.conf['weights'])
        #     state_dict = torch.load(path)
        #     state_dict = {  # remain compatible with original weights
        #         re.sub(r'(gnn\.layers\.[0-9]+\.)', r'\1update.', k): v
        #         for k, v in state_dict.items()}
        #     self.load_state_dict(state_dict, strict=True)
        #     logging.info(f'Loading SuperGlue trained for {conf.weights}.')

    def _forward(self, data):
        pred = {}
        desc0, desc1 = data['descriptors0'], data['descriptors1']
        kpts0, kpts1 = data['keypoints0'], data['keypoints1']

        if kpts0.shape[1] == 0 or kpts1.shape[1] == 0:  # no keypoints
            shape0, shape1 = kpts0.shape[:-1], kpts1.shape[:-1]
            return {
                'matches0': kpts0.new_full(shape0, -1, dtype=torch.int),
                'matches1': kpts1.new_full(shape1, -1, dtype=torch.int),
                'match_scores0': kpts0.new_zeros(shape0),
                'match_scores1': kpts1.new_zeros(shape1),
            }

        if self.conf['bottleneck_dim'] is not None:
            pred['down_descriptors0'] = desc0 = self.bottleneck_down(desc0)
            pred['down_descriptors1'] = desc1 = self.bottleneck_down(desc1)
            desc0 = self.bottleneck_up(desc0)
            desc1 = self.bottleneck_up(desc1)
            desc0 = nn.functional.normalize(desc0, p=2, dim=1)
            desc1 = nn.functional.normalize(desc1, p=2, dim=1)
            pred['bottleneck_descriptors0'] = desc0
            pred['bottleneck_descriptors1'] = desc1
            if self.conf['loss']['nll_weight'] == 0:
                desc0 = desc0.detach()
                desc1 = desc1.detach()

        if self.conf['input_dim'] != self.conf['descriptor_dim']:
            desc0 = self.input_proj(desc0)
            desc1 = self.input_proj(desc1)

        kpts0 = normalize_keypoints(kpts0, data['image_size0'])
        kpts1 = normalize_keypoints(kpts1, data['image_size1'])

        # if not torch.all(kpts0 >= -1): print('kpts0[kpts0 < -1]', kpts0[kpts0 < -1])
        # if not torch.all(kpts0 <= 1): print('kpts0[kpts0 > 1]', kpts0[kpts0 > 1])
        # if not torch.all(kpts1 >= -1): print('kpts1[kpts1 < -1]', kpts1[kpts1 < -1])
        # if not torch.all(kpts1 <= 1): print('kpts1[kpts1 > 1]', kpts1[kpts1 > 1])

        desc0 = desc0 + self.kenc(kpts0, data['keypoint_scores0'])
        desc1 = desc1 + self.kenc(kpts1, data['keypoint_scores1'])

        if not self.conf['skip_gnn']:
            desc0, desc1 = self.gnn(desc0, desc1)

        mdesc0, mdesc1 = self.final_proj(desc0), self.final_proj(desc1)

        scores = torch.einsum('bdn,bdm->bnm', mdesc0, mdesc1)
        scores = scores / self.conf['descriptor_dim']**.5

        if self.conf['output_normalization'] == 'sinkhorn':
            scores = log_optimal_transport(scores, self.bin_score, iters=self.conf['num_sinkhorn_iterations'])
        elif self.conf['output_normalization'] == 'double_softmax':
            scores = log_double_softmax(scores, self.bin_score)
        else:
            raise ValueError(self.conf['output_normalization'])

        max0, max1 = scores[:, :-1, :-1].max(2), scores[:, :-1, :-1].max(1)
        m0, m1 = max0.indices, max1.indices
        mutual0 = arange_like(m0, 1)[None] == m1.gather(1, m0)
        mutual1 = arange_like(m1, 1)[None] == m0.gather(1, m1)
        zero = scores.new_tensor(0)
        mscores0 = torch.where(mutual0, max0.values.exp(), zero)
        mscores1 = torch.where(mutual1, mscores0.gather(1, m1), zero)
        valid0 = mutual0 & (mscores0 > self.conf['filter_threshold'])
        valid1 = mutual1 & valid0.gather(1, m1)
        m0 = torch.where(valid0, m0, m0.new_tensor(-1))
        m1 = torch.where(valid1, m1, m1.new_tensor(-1))

        return {
            **pred,
            'log_assignment': scores,
            'matches0': m0,
            'matches1': m1,
            'match_scores0': mscores0,
            'match_scores1': mscores1,
            'bin_score': self.bin_score
        }

    # def loss(self, pred, data):
    #     losses = {'total': 0}
    #
    #     positive = data['gt_assignment'].float()
    #     num_pos = torch.max(positive.sum((1, 2)), positive.new_tensor(1))
    #     neg0 = (data['gt_matches0'] == -1).float()
    #     neg1 = (data['gt_matches1'] == -1).float()
    #     num_neg = torch.max(neg0.sum(1) + neg1.sum(1), neg0.new_tensor(1))
    #
    #     log_assignment = pred['log_assignment']
    #     nll_pos = -(log_assignment[:, :-1, :-1]*positive).sum((1, 2))
    #     nll_pos /= num_pos
    #     nll_neg0 = -(log_assignment[:, :-1, -1]*neg0).sum(1)
    #     nll_neg1 = -(log_assignment[:, -1, :-1]*neg1).sum(1)
    #     nll_neg = (nll_neg0 + nll_neg1) / num_neg
    #     nll = (self.conf['loss']['nll_balancing'] * nll_pos + (1 - self.conf['loss']['nll_balancing']) * nll_neg)
    #
    #     losses['assignment_nll'] = nll
    #
    #     if self.conf['loss']['nll_weight'] > 0:
    #         losses['total'] = nll*self.conf['loss']['nll_weight']
    #
    #     if self.conf['loss']['reward_weight'] > 0:
    #         reward = data['match_reward']
    #         prob = log_assignment[:, :-1, :-1].exp()
    #         reward_loss = - torch.sum(prob * reward, (1, 2))
    #         norm = torch.sum(torch.clamp(reward, min=0), (1, 2))
    #         reward_loss /= torch.clamp(norm, min=1)
    #         losses['expected_match_reward'] = reward_loss
    #         losses['total'] += self.conf['loss']['reward_weight'] * reward_loss
    #
    #     # Some statistics
    #     losses['num_matchable'] = num_pos
    #     losses['num_unmatchable'] = num_neg
    #     losses['sinkhorn_norm'] = log_assignment.exp()[:, :-1].sum(2).mean(1)
    #     losses['bin_score'] = self.bin_score[None]
    #
    #     if self.conf['loss']['bottleneck_l2_weight'] > 0:
    #         assert self.conf['bottleneck_dim'] is not None
    #         l2_0 = torch.sum((data['descriptors0'] - pred['bottleneck_descriptors0'])**2, 1)
    #         l2_1 = torch.sum((data['descriptors1'] - pred['bottleneck_descriptors1'])**2, 1)
    #         l2 = (l2_0.mean(-1) + l2_1.mean(-1)) / 2
    #         losses['bottleneck_l2'] = l2
    #         losses['total'] += l2*self.conf['loss']['bottleneck_l2_weight']
    #
    #     return losses


class DescriptorExtractor(nn.Module):
    def __init__(self, backbone_feat_dim, descriptor_dim, kernel_size=4):
        super().__init__()

        self.conv = nn.Conv2d(in_channels=backbone_feat_dim, out_channels=descriptor_dim, kernel_size=kernel_size,
                              padding=kernel_size//2, bias=True)

    def forward(self, x, coords):
        feats =  self.conv(x)
        assert torch.all(coords >= 0) and torch.all(coords < feats.shape[3])
        desc = feats[torch.arange(x.shape[0]).unsqueeze(1), :, coords[:, :, 0].long(), coords[:, :, 1].long()]
        return desc.permute(0,2,1)

    def get_descriptors(self, x, coords):
        if coords.ndim == 2:
            coords = coords.unsqueeze(0)

        feats = self.conv(x)
        assert torch.all(coords >= 0) and torch.all(coords < feats.shape[3])
        desc = feats[torch.arange(x.shape[0]).unsqueeze(1), :, coords[:, :, 0].long(), coords[:, :, 1].long()]
        return desc.permute(0, 2, 1)


class PeakMatchingNetwork(nn.Module):
    def __init__(self, feature_extractor, classification_layer, descriptor_extractor, matcher):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.classification_layer = classification_layer
        self.output_layers = sorted(list(set(self.classification_layer)))

        self.descriptor_extractor = descriptor_extractor
        self.matcher = matcher


    def forward(self, img_cropped0, img_cropped1, peak_tsm_coords0, peak_tsm_coords1, peak_img_coords0, peak_img_coords1,
                peak_scores0, peak_scores1, img_shape0, img_shape1, **kwargs):


        # Extract backbone features
        frame_feat0 = self.extract_backbone_features(img_cropped0.reshape(-1, *img_cropped0.shape[-3:]))
        frame_feat1 = self.extract_backbone_features(img_cropped1.reshape(-1, *img_cropped1.shape[-3:]))

        # Classification features
        frame_feat_clf0 = self.get_backbone_clf_feat(frame_feat0)
        frame_feat_clf1 = self.get_backbone_clf_feat(frame_feat1)

        descriptors0 = self.descriptor_extractor(frame_feat_clf0, peak_tsm_coords0[0])
        descriptors1 = self.descriptor_extractor(frame_feat_clf1, peak_tsm_coords1[0])

        data = {
            'descriptors0': descriptors0,
            'descriptors1': descriptors1,
            'keypoints0': peak_img_coords0[0],
            'keypoints1': peak_img_coords1[0],
            'keypoint_scores0': peak_scores0[0],
            'keypoint_scores1': peak_scores1[0],
            'image_size0': img_shape0[0],
            'image_size1': img_shape1[0],
        }

        pred = self.matcher(data)

        return pred

    def extract_backbone_features(self, im, layers=None):
        if layers is None:
            layers = self.output_layers
        return self.feature_extractor(im, layers)

    def get_backbone_clf_feat(self, backbone_feat):
        feat = OrderedDict({l: backbone_feat[l] for l in self.classification_layer})
        if len(self.classification_layer) == 1:
            return feat[self.classification_layer[0]]
        return feat



@model_constructor
def peak_matching_net(backbone_pretrained=True, classification_layer=None, frozen_backbone_layers=(), skip_gnn=False):
    if classification_layer is None:
        classification_layer = ['layer3']
    # Backbone
    backbone_net = backbones.resnet50(pretrained=backbone_pretrained, frozen_layers=frozen_backbone_layers)
    descriptor_extractor = DescriptorExtractor(backbone_feat_dim=1024, descriptor_dim=256, kernel_size=4)

    matcher = SuperGlue(conf={'skip_gnn': skip_gnn})

    net = PeakMatchingNetwork(feature_extractor=backbone_net, classification_layer=classification_layer,
                              descriptor_extractor=descriptor_extractor, matcher=matcher)

    return net

