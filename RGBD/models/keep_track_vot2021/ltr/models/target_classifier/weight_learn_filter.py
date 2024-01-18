import torch.nn as nn
import torch
import ltr.models.layers.filter as filter_layer
import math
from pytracking import TensorList



class LinearFilterWeightLearn(nn.Module):
    def __init__(self, filter_size, filter_initializer, filter_optimizer=None, feature_extractor=None, weight_predictor=None):
        super().__init__()

        self.filter_size = filter_size

        # Modules
        self.filter_initializer = filter_initializer
        self.filter_optimizer = filter_optimizer
        self.feature_extractor = feature_extractor
        self.weight_predictor = weight_predictor

        # Init weights
        for m in self.feature_extractor.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, train_feat, test_feat, train_bb, **kwargs):
        """ the bb should be 5d"""

        num_sequences = train_bb.shape[1]

        if train_feat.dim() == 5:
            train_feat = train_feat.view(-1, *train_feat.shape[-3:])
        if test_feat.dim() == 5:
            test_feat = test_feat.view(-1, *test_feat.shape[-3:])

        # Extract features
        train_feat_clf = self.extract_classification_feat(train_feat, num_sequences)
        test_feat_clf = self.extract_classification_feat(test_feat, num_sequences)

        sample_weight = self.weight_predictor(train_feat_clf, num_sequences)
        # sample_weight = self.weight_predictor(train_feat, num_sequences)

        # Train filter
        filter, filter_iter, _ = self.get_filter(train_feat_clf, train_bb, sample_weight=sample_weight, **kwargs)

        # Classify samples
        test_scores = [self.classify(f, test_feat_clf) for f in filter_iter]

        return test_scores

    def extract_classification_feat(self, feat, num_sequences=None):
        if self.feature_extractor is None:
            return feat
        if num_sequences is None:
            return self.feature_extractor(feat)

        output = self.feature_extractor(feat)
        return output.view(-1, num_sequences, *output.shape[-3:])

    def classify(self, weights, feat):
        """Run classifier (filter) on the features (feat)."""

        scores = filter_layer.apply_filter(feat, weights)

        return scores

    def get_filter(self, feat, bb, *args, **kwargs):
        weights = self.filter_initializer(feat, bb)

        if self.filter_optimizer is not None:
            weights, weights_iter, losses = self.filter_optimizer(TensorList([weights]), feat=feat, bb=bb, *args, **kwargs)
            weights = weights[0]
            weights_iter = [w[0] for w in weights_iter]
        else:
            weights_iter = [weights]
            losses = None

        return weights, weights_iter, losses


class LinearFilterSimilarityWeightLearn(nn.Module):
    def __init__(self, filter_size, filter_initializer, filter_optimizer=None, feature_extractor=None, weight_predictor=None):
        super().__init__()

        self.filter_size = filter_size

        # Modules
        self.filter_initializer = filter_initializer
        self.filter_optimizer = filter_optimizer
        self.feature_extractor = feature_extractor
        self.weight_predictor = weight_predictor

        # Init weights
        for m in self.feature_extractor.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, train_feat, test_feat, train_bb, **kwargs):
        """ the bb should be 5d"""

        num_sequences = train_bb.shape[1]
        print(list(kwargs.keys()))

        if train_feat.dim() == 5:
            train_feat = train_feat.view(-1, *train_feat.shape[-3:])
        if test_feat.dim() == 5:
            test_feat = test_feat.view(-1, *test_feat.shape[-3:])

        # Extract features
        train_feat_clf = self.extract_classification_feat(train_feat, num_sequences)
        test_feat_clf = self.extract_classification_feat(test_feat, num_sequences)

        # Train filter
        filter, filter_iter, _ = self.get_filter(train_feat_clf, train_bb, **kwargs)

        # Classify samples
        test_scores = [self.classify(f, test_feat_clf) for f in filter_iter]

        sample_weight = self.weight_predictor(test_feat_clf, train_feat_clf, test_scores[-1], **kwargs)

        filter, filter_iter, _ = self.get_filter(train_feat_clf, train_bb, sample_weight=sample_weight, **kwargs)

        test_scores = [self.classify(f, test_feat_clf) for f in filter_iter]

        return test_scores

    def extract_classification_feat(self, feat, num_sequences=None):
        if self.feature_extractor is None:
            return feat
        if num_sequences is None:
            return self.feature_extractor(feat)

        output = self.feature_extractor(feat)
        return output.view(-1, num_sequences, *output.shape[-3:])

    def classify(self, weights, feat):
        """Run classifier (filter) on the features (feat)."""

        scores = filter_layer.apply_filter(feat, weights)

        return scores

    def get_filter(self, feat, bb, *args, **kwargs):
        weights = self.filter_initializer(feat, bb)

        if self.filter_optimizer is not None:
            weights, weights_iter, losses = self.filter_optimizer(TensorList([weights]), feat=feat, bb=bb, *args, **kwargs)
            weights = weights[0]
            weights_iter = [w[0] for w in weights_iter]
        else:
            weights_iter = [weights]
            losses = None

        return weights, weights_iter, losses
