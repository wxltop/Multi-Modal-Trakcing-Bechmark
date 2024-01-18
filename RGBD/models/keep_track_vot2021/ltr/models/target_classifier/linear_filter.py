import torch.nn as nn
import torch
import torchvision.ops as tv_ops
import ltr.models.layers.filter as filter_layer
import math
from pytracking import TensorList
from ltr.external.PreciseRoIPooling.pytorch.prroi_pool.functional import prroi_pool2d


class LinearFilter(nn.Module):
    """Target classification filter module.
    args:
        filter_size:  Size of filter (int).
        filter_initialize:  Filter initializer module.
        filter_optimizer:  Filter optimizer module.
        feature_extractor:  Feature extractor module applied to the input backbone features."""

    def __init__(self, filter_size, filter_initializer, filter_optimizer=None, feature_extractor=None):
        super().__init__()

        self.filter_size = filter_size

        # Modules
        self.filter_initializer = filter_initializer
        self.filter_optimizer = filter_optimizer
        self.feature_extractor = feature_extractor

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

    def forward(self, train_feat, test_feat, train_bb, *args, **kwargs):
        """Learns a target classification filter based on the train samples and return the resulting classification
        scores on the test samples.
        The forward function is ONLY used for training. Call the individual functions during tracking.
        args:
            train_feat:  Backbone features for the train samples (4 or 5 dims).
            test_feat:  Backbone features for the test samples (4 or 5 dims).
            trian_bb:  Target boxes (x,y,w,h) for the train samples in image coordinates. Dims (images, sequences, 4).
            *args, **kwargs:  These are passed to the optimizer module.
        returns:
            test_scores:  Classification scores on the test samples."""

        assert train_bb.dim() == 3

        num_sequences = train_bb.shape[1]

        if train_feat.dim() == 5:
            train_feat = train_feat.reshape(-1, *train_feat.shape[-3:])
        if test_feat.dim() == 5:
            test_feat = test_feat.reshape(-1, *test_feat.shape[-3:])

        # Extract features
        train_feat = self.extract_classification_feat(train_feat, num_sequences)
        test_feat = self.extract_classification_feat(test_feat, num_sequences)

        # Train filter
        filter, filter_iter, losses = self.get_filter(train_feat, train_bb, *args, **kwargs)

        # Classify samples using all return filters
        test_scores = [self.classify(f, test_feat) for f in filter_iter]

        return test_scores

    def extract_classification_feat(self, feat, num_sequences=None):
        """Extract classification features based on the input backbone features."""
        if self.feature_extractor is None:
            return feat
        if num_sequences is None:
            return self.feature_extractor(feat)

        output = self.feature_extractor(feat)
        return output.reshape(-1, num_sequences, *output.shape[-3:])

    def classify(self, weights, feat):
        """Run classifier (filter) on the features (feat)."""
        scores = filter_layer.apply_filter(feat, weights)
        return scores

    def get_filter(self, feat, bb, *args, **kwargs):
        """Outputs the learned filter based on the input features (feat) and target boxes (bb) by running the
        filter initializer and optimizer. Note that [] denotes an optional dimension.
        args:
            feat:  Input feature maps. Dims (images_in_sequence, [sequences], feat_dim, H, W).
            bb:  Target bounding boxes (x, y, w, h) in the image img_coords. Dims (images_in_sequence, [sequences], 4).
            *args, **kwargs:  These are passed to the optimizer module.
        returns:
            weights:  The final oprimized weights. Dims (sequences, feat_dim, wH, wW).
            weight_iterates:  The weights computed in each iteration (including initial input and final output).
            losses:  Train losses."""
        
        weights = self.filter_initializer(feat, bb)

        if self.filter_optimizer is not None:
            weights, weights_iter, losses = self.filter_optimizer(weights, feat=feat, bb=bb, *args, **kwargs)
        else:
            weights_iter = [weights]
            losses = None

        return weights, weights_iter, losses

    def train_classifier(self, backbone_feat, bb):
        num_sequences = bb.shape[1]

        if backbone_feat.dim() == 5:
            backbone_feat = backbone_feat.reshape(-1, *backbone_feat.shape[-3:])

        # Extract features
        train_feat = self.extract_classification_feat(backbone_feat, num_sequences)

        # Get filters from each iteration
        final_filter, _, train_losses = self.get_filter(train_feat, bb)
        return final_filter, train_losses

    def track_frame(self, filter_weights, backbone_feat):
        if backbone_feat.dim() == 5:
            num_sequences = backbone_feat.shape[1]
            backbone_feat = backbone_feat.reshape(-1, *backbone_feat.shape[-3:])
        else:
            num_sequences = None

        test_feat = self.extract_classification_feat(backbone_feat, num_sequences)

        scores = filter_layer.apply_filter(test_feat, filter_weights)

        return scores


class LinearGaussianFilter(nn.Module):
    """Target classification filter module.
    args:
        filter_size:  Size of filter (int).
        filter_initialize:  Filter initializer module.
        filter_optimizer:  Filter optimizer module.
        feature_extractor:  Feature extractor module applied to the input backbone features."""

    def __init__(self, filter_size, filter_initializer, filter_optimizer=None, feature_extractor=None,
                 filter_variance_estimator=None, final_predictor=None):
        super().__init__()

        self.filter_size = filter_size

        # Modules
        self.filter_initializer = filter_initializer
        self.filter_optimizer = filter_optimizer
        self.feature_extractor = feature_extractor
        self.filter_variance_estimator = filter_variance_estimator
        self.final_predictor = final_predictor

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

    def forward(self, train_feat, test_feat, train_bb, *args, **kwargs):
        """Learns a target classification filter based on the train samples and return the resulting classification
        scores on the test samples.
        The forward function is ONLY used for training. Call the individual functions during tracking.
        args:
            train_feat:  Backbone features for the train samples (4 or 5 dims).
            test_feat:  Backbone features for the test samples (4 or 5 dims).
            trian_bb:  Target boxes (x,y,w,h) for the train samples in image coordinates. Dims (images, sequences, 4).
            *args, **kwargs:  These are passed to the optimizer module.
        returns:
            test_scores:  Classification scores on the test samples."""

        assert train_bb.dim() == 3

        num_sequences = train_bb.shape[1]

        if train_feat.dim() == 5:
            train_feat = train_feat.reshape(-1, *train_feat.shape[-3:])
        if test_feat.dim() == 5:
            test_feat = test_feat.reshape(-1, *test_feat.shape[-3:])

        # Extract features
        train_feat = self.extract_classification_feat(train_feat, num_sequences)
        test_feat = self.extract_classification_feat(test_feat, num_sequences)

        # Train filter
        filter, filter_iter, losses = self.get_filter(train_feat, train_bb, *args, **kwargs)

        # Classify samples using all return filters
        test_scores = [self.classify(f, test_feat) for f in filter_iter]

        # Get variance and final scores
        sample_weight = args[0] if args else kwargs.get('sample_weight')
        filter_var = self.get_filter_variance(filter, train_feat, train_bb, sample_weight)
        scores_var = self.get_response_variance(filter_var, test_feat)
        final_scores = self.final_response(test_scores[-1], scores_var)

        return test_scores, scores_var, final_scores

    def extract_classification_feat(self, feat, num_sequences=None):
        """Extract classification features based on the input backbone features."""
        if self.feature_extractor is None:
            return feat
        if num_sequences is None:
            return self.feature_extractor(feat)

        output = self.feature_extractor(feat)
        return output.reshape(-1, num_sequences, *output.shape[-3:])

    def classify(self, weights, feat, weights_var=None):
        """Run classifier (filter) on the features (feat)."""

        scores = filter_layer.apply_filter(feat, weights)

        if weights_var is not None:
            scores_var = self.get_response_variance(weights_var, feat)
            scores = self.final_response(scores, scores_var)

        return scores

    def get_filter(self, feat, bb, *args, **kwargs):
        """Outputs the learned filter based on the input features (feat) and target boxes (bb) by running the
        filter initializer and optimizer. Note that [] denotes an optional dimension.
        args:
            feat:  Input feature maps. Dims (images_in_sequence, [sequences], feat_dim, H, W).
            bb:  Target bounding boxes (x, y, w, h) in the image img_coords. Dims (images_in_sequence, [sequences], 4).
            *args, **kwargs:  These are passed to the optimizer module.
        returns:
            weights:  The final oprimized weights. Dims (sequences, feat_dim, wH, wW).
            weight_iterates:  The weights computed in each iteration (including initial input and final output).
            losses:  Train losses."""

        weights = self.filter_initializer(feat, bb)

        if self.filter_optimizer is not None:
            weights, weights_iter, losses = self.filter_optimizer(weights, feat=feat, bb=bb, *args, **kwargs)
        else:
            weights_iter = [weights]
            losses = None

        return weights, weights_iter, losses

    def get_filter_variance(self, weights, train_feat, train_bb, sample_weight=None):
        return self.filter_variance_estimator(weights, train_feat, train_bb, sample_weight)

    def get_response_variance(self, weights_var, feat):
        return filter_layer.apply_filter(feat * feat, weights_var)

    def final_response(self, response_mean, response_var):
        return self.final_predictor(response_mean, response_var)

    def train_classifier(self, backbone_feat, bb):
        num_sequences = bb.shape[1]

        if backbone_feat.dim() == 5:
            backbone_feat = backbone_feat.reshape(-1, *backbone_feat.shape[-3:])

        # Extract features
        train_feat = self.extract_classification_feat(backbone_feat, num_sequences)

        # Get filters from each iteration
        final_filter, _, train_losses = self.get_filter(train_feat, bb)
        return final_filter, train_losses

    def track_frame(self, filter_weights, backbone_feat):
        if backbone_feat.dim() == 5:
            num_sequences = backbone_feat.shape[1]
            backbone_feat = backbone_feat.reshape(-1, *backbone_feat.shape[-3:])
        else:
            num_sequences = None

        test_feat = self.extract_classification_feat(backbone_feat, num_sequences)

        scores = filter_layer.apply_filter(test_feat, filter_weights)

        return scores


class LinearFilterMeta(nn.Module):
    def __init__(self, filter_size, filter_initializer, filter_optimizer=None, feature_extractor=None):
        super().__init__()

        self.filter_size = filter_size

        # Modules
        self.filter_initializer = filter_initializer
        self.filter_optimizer = filter_optimizer
        self.feature_extractor = feature_extractor

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

    def forward(self, train_feat, test_feat, train_bb, *args, **kwargs):
        """ the bb should be 5d"""

        num_sequences = train_bb.shape[1]

        if train_feat.dim() == 5:
            train_feat = train_feat.reshape(-1, *train_feat.shape[-3:])
        if test_feat.dim() == 5:
            test_feat = test_feat.reshape(-1, *test_feat.shape[-3:])

        # Extract features
        train_feat = self.extract_classification_feat(train_feat, num_sequences)
        test_feat = self.extract_classification_feat(test_feat, num_sequences)

        # Train filter
        filter, filter_iter, _ = self.get_filter(train_feat, train_bb, *args, **kwargs)

        # Classify samples
        test_scores = [self.classify(f, test_feat) for f in filter_iter]

        return test_scores

    def extract_classification_feat(self, feat, num_sequences=None):
        if self.feature_extractor is None:
            return feat
        if num_sequences is None:
            return self.feature_extractor(feat)

        output = self.feature_extractor(feat)
        return output.reshape(-1, num_sequences, *output.shape[-3:])

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



class LinearFilterMetaAda(nn.Module):
    def __init__(self, filter_size, filter_initializer, filter_optimizer=None, feature_extractor=None,
                 pool_size=22, feat_stride=16, pool_type='roi_align'):
        super().__init__()

        self.filter_size = filter_size

        # Modules
        self.filter_initializer = filter_initializer
        self.filter_optimizer = filter_optimizer
        self.feature_extractor = feature_extractor

        if not isinstance(pool_size, (list, tuple)):
            self.pool_size = (int(pool_size), int(pool_size))
        else:
            self.pool_size = [int(p) for p in pool_size]

        self.feat_stride = feat_stride
        self.pool_type = pool_type

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

    def forward(self, train_feat, test_feat, train_bb, train_search_area_bb, test_search_area_bb, *args, **kwargs):
        """ the bb should be 5d"""
        assert train_bb.dim() == 3

        num_sequences = train_bb.shape[1]

        if train_feat.dim() == 5:
            train_feat = train_feat.reshape(-1, *train_feat.shape[-3:])
        if test_feat.dim() == 5:
            test_feat = test_feat.reshape(-1, *test_feat.shape[-3:])

        orig_feat_sz = (test_feat.shape[-1], test_feat.shape[-2])

        # Extract features
        train_feat = self.extract_classification_feat(train_feat, train_search_area_bb, num_sequences)
        test_feat = self.extract_classification_feat(test_feat, test_search_area_bb, num_sequences)

        if train_search_area_bb is None:
            norm_factor = 1.0 / self.feat_stride
        else:
            norm_factor = train_feat.shape[-1] / train_search_area_bb[..., 2:3]

        train_bb_feat = train_bb * norm_factor

        # Train filter
        filter, filter_iter, _ = self.get_filter(train_feat, train_bb_feat, *args, **kwargs)

        # Classify samples
        test_scores = [self.classify(f, test_feat) for f in filter_iter]

        # Convert last iter scores to crop co-ordinates
        test_score_last_iter = test_scores[-1]

        test_score_last_iter_resamp = self.resample_scores(test_score_last_iter, test_search_area_bb, orig_feat_sz)
        return test_scores, test_score_last_iter_resamp

    def resample_scores(self, scores, test_search_area_bb, out_sz):
        crop_h = out_sz[1] * self.feat_stride
        crop_w = out_sz[0] * self.feat_stride

        roi = torch.tensor([0.0, 0.0, crop_w, crop_h], dtype=test_search_area_bb.dtype,
                           device=test_search_area_bb.device).reshape(1, 1, 4)
        roi = roi.repeat(test_search_area_bb.shape[0], test_search_area_bb.shape[1], 1)

        roi[:, :, :2] -= test_search_area_bb[:, :, :2]
        roi[:, :, [0, 2]] = (roi[:, :, [0, 2]] / test_search_area_bb[:, :, 2:3]) * self.pool_size[0]
        roi[:, :, [1, 3]] = (roi[:, :, [1, 3]] / test_search_area_bb[:, :, 3:]) * self.pool_size[1]

        roi[:, :, :2] -= 0.5
        scores_shape = scores.shape
        scores = scores.reshape(-1, 1, *scores_shape[-2:])
        roi = self._bb_to_roi(roi.reshape(-1, 4), scores)

        if self.pool_type == 'prpool':
            scores_pooled = prroi_pool2d(scores, roi, out_sz[1], out_sz[0], 1.0)
        elif self.pool_type == 'roi_align':
            scores_pooled = tv_ops.roi_align(scores, roi, (out_sz[1], out_sz[0]), 1.0)
        else:
            raise ValueError

        return scores_pooled.reshape(*scores_shape[:2], *scores_pooled.shape[-2:])

    def pool_features(self, feat, search_area_bb=None):
        if search_area_bb is None:
            return feat

        search_area_bb = search_area_bb.reshape(-1, 4).clone()
        search_area_bb[:, 0:2] -= self.feat_stride * 0.5

        roi = self._bb_to_roi(search_area_bb, feat)

        # self.pool_size = (1, 1)
        if self.pool_type == 'prpool':
            feat_pooled = prroi_pool2d(feat, roi, self.pool_size[1], self.pool_size[0], 1.0 / self.feat_stride)
        elif self.pool_type == 'roi_align':
            feat_pooled = tv_ops.roi_align(feat, roi, (self.pool_size[1], self.pool_size[0]), 1.0 / self.feat_stride)
        else:
            raise ValueError

        return feat_pooled

    def _bb_to_roi(self, bb, feat):
        # Add batch_index to rois
        batch_size = feat.shape[0]
        batch_index = torch.arange(batch_size, dtype=torch.float32).reshape(-1, 1).to(feat.device)

        # input proposals2 is in format xywh, convert it to x0y0x1y1 format
        bb_xyxy = torch.cat((bb[:, 0:2], bb[:, 0:2] + bb[:, 2:4]), dim=1)

        # Add batch index
        roi = torch.cat((batch_index.reshape(batch_size, -1), bb_xyxy), dim=1)
        return roi

    def extract_classification_feat(self, feat, search_area_bb=None, num_sequences=None):
        # Pool before feature extraction
        feat_pooled = self.pool_features(feat, search_area_bb)

        if self.feature_extractor is None:
            return feat_pooled

        if num_sequences is None:
            return self.feature_extractor(feat_pooled)

        output = self.feature_extractor(feat_pooled)
        return output.reshape(-1, num_sequences, *output.shape[-3:])

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


class LinearFilterAda(nn.Module):
    """Target classification filter module.
    args:
        filter_size:  Size of filter (int).
        filter_initialize:  Filter initializer module.
        filter_optimizer:  Filter optimizer module.
        feature_extractor:  Feature extractor module applied to the input backbone features."""

    def __init__(self, filter_size, filter_initializer, filter_optimizer=None, feature_extractor=None,
                 pool_size=22, feat_stride=16, pool_type='prpool'):
        super().__init__()

        self.filter_size = filter_size

        # Modules
        self.filter_initializer = filter_initializer
        self.filter_optimizer = filter_optimizer
        self.feature_extractor = feature_extractor
        self.pool_size = pool_size
        if isinstance(self.pool_size, int):
            self.pool_size = (self.pool_size, self.pool_size)

        self.feat_stride = feat_stride
        self.pool_type = pool_type

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

    def forward(self, train_feat, test_feat, train_bb, train_search_bb, test_search_bb, *args, **kwargs):
        """Learns a target classification filter based on the train samples and return the resulting classification
        scores on the test samples.
        The forward function is ONLY used for training. Call the individual functions during tracking.
        args:
            train_feat:  Backbone features for the train samples (4 or 5 dims).
            test_feat:  Backbone features for the test samples (4 or 5 dims).
            trian_bb:  Target boxes (x,y,w,h) for the train samples in image coordinates. Dims (images, sequences, 4).
            *args, **kwargs:  These are passed to the optimizer module.
        returns:
            test_scores:  Classification scores on the test samples."""

        assert train_bb.dim() == 3

        num_sequences = train_bb.shape[1]

        if train_feat.dim() == 5:
            train_feat = train_feat.reshape(-1, *train_feat.shape[-3:])
        if test_feat.dim() == 5:
            test_feat = test_feat.reshape(-1, *test_feat.shape[-3:])

        # Extract features
        train_feat = self.extract_classification_feat(train_feat, train_search_bb, num_sequences)
        test_feat = self.extract_classification_feat(test_feat, test_search_bb, num_sequences)

        # Train filter
        # Normalize the train_bb so that it's in feature coordinates
        if train_search_bb is None:
            norm_factor = 1.0 / self.feat_stride
        else:
            norm_factor = train_feat.shape[-1] / train_search_bb[..., 2:3]
        train_bb_feat = train_bb * norm_factor
        filter, filter_iter, losses = self.get_filter(train_feat, train_bb_feat, *args, **kwargs)

        # Classify samples using all return filters
        test_scores = [self.classify(f, test_feat) for f in filter_iter]

        return test_scores

    def pool_features(self, feat, search_bb=None):
        if search_bb is None:
            return feat

        search_bb = search_bb.reshape(-1, 4).clone()
        search_bb[:, 0:2] -= self.feat_stride*0.5

        # Add batch_index to rois
        batch_size = feat.shape[0]
        batch_index = torch.arange(batch_size, dtype=torch.float32).reshape(-1, 1).to(feat.device)

        # input proposals2 is in format xywh, convert it to x0y0x1y1 format
        bb_xyxy = torch.cat((search_bb[:, 0:2], search_bb[:, 0:2] + search_bb[:, 2:4]), dim=1)

        # Add batch index
        roi = torch.cat((batch_index.reshape(batch_size, -1), bb_xyxy), dim=1)

        # self.pool_size = (1, 1)
        if self.pool_type == 'prpool':
            feat_pooled = prroi_pool2d(feat, roi, self.pool_size[1], self.pool_size[0], 1.0 / self.feat_stride)
        elif self.pool_type == 'roi_align':
            feat_pooled = tv_ops.roi_align(feat, roi, (self.pool_size[1], self.pool_size[0]), 1.0 / self.feat_stride)
        else:
            raise ValueError

        return feat_pooled

    def extract_classification_feat(self, feat, search_bb=None, num_sequences=None):
        """Extract classification features based on the input backbone features."""
        if self.feature_extractor is None:
            raise NotImplementedError

        if num_sequences is None:
            feat_cls = self.feature_extractor(feat)
            return self.pool_features(feat_cls, search_bb)

        output = self.feature_extractor(feat)
        output = self.pool_features(output, search_bb)
        return output.reshape(-1, num_sequences, *output.shape[-3:])

    def classify(self, weights, feat):
        """Run classifier (filter) on the features (feat)."""

        scores = filter_layer.apply_filter(feat, weights)

        return scores

    def get_filter(self, feat, bb, *args, **kwargs):
        """Outputs the learned filter based on the input features (feat) and target boxes (bb) by running the
        filter initializer and optimizer. Note that [] denotes an optional dimension.
        args:
            feat:  Input feature maps. Dims (images_in_sequence, [sequences], feat_dim, H, W).
            bb:  Target bounding boxes (x, y, w, h) in the image img_coords. Dims (images_in_sequence, [sequences], 4).
            *args, **kwargs:  These are passed to the optimizer module.
        returns:
            weights:  The final oprimized weights. Dims (sequences, feat_dim, wH, wW).
            weight_iterates:  The weights computed in each iteration (including initial input and final output).
            losses:  Train losses."""

        weights = self.filter_initializer(feat, bb)

        if self.filter_optimizer is not None:
            weights, weights_iter, losses = self.filter_optimizer(weights, feat=feat, bb=bb, *args, **kwargs)
        else:
            weights_iter = [weights]
            losses = None

        return weights, weights_iter, losses

    def train_classifier(self, backbone_feat, bb, search_area_bb):
        num_sequences = bb.shape[1]

        if backbone_feat.dim() == 5:
            backbone_feat = backbone_feat.reshape(-1, *backbone_feat.shape[-3:])

        # Extract features
        train_feat = self.extract_classification_feat(backbone_feat, search_area_bb, num_sequences)

        # Get filters from each iteration
        final_filter, _, train_losses = self.get_filter(train_feat, bb)
        return final_filter, train_losses

    def track_frame(self, filter_weights, backbone_feat, search_area_bb):
        if backbone_feat.dim() == 5:
            num_sequences = backbone_feat.shape[1]
            backbone_feat = backbone_feat.reshape(-1, *backbone_feat.shape[-3:])
        else:
            num_sequences = None

        test_feat = self.extract_classification_feat(backbone_feat, search_area_bb, num_sequences)

        scores = filter_layer.apply_filter(test_feat, filter_weights)

        return scores
