import torch.nn as nn
import torch
import torch.nn.functional as F


class IsOccludedLoss(nn.Module):
    def __init__(self, pos_weight=1.0, return_per_sequence=False):
        super(IsOccludedLoss, self).__init__()
        self.pos_weight = pos_weight
        self.return_per_sequence = return_per_sequence

    def forward(self, prediction, label, valid_sample=None):
        prediction = prediction.view(-1)
        label = label.view(-1)

        if valid_sample is not None:
            valid_sample = valid_sample.float().view(-1)

            prediction_accuracy = F.binary_cross_entropy(prediction, label.float(), reduction='none')
            prediction_accuracy[label == 1] *= self.pos_weight

            prediction_accuracy = prediction_accuracy * valid_sample

            if not self.return_per_sequence:
                num_valid_samples = valid_sample.sum()
                if num_valid_samples > 0:
                    prediction_accuracy = prediction_accuracy.sum() / num_valid_samples
                else:
                    prediction_accuracy = 0.0
        else:
            prediction_accuracy = F.binary_cross_entropy(prediction, label, reduction='none')
            prediction_accuracy[label == 1] *= self.pos_weight

            if not self.return_per_sequence:
                prediction_accuracy = prediction_accuracy.mean()

        return prediction_accuracy


class IsTargetCellLoss(nn.Module):
    def __init__(self, return_per_sequence=False, use_with_logits=True):
        super(IsTargetCellLoss, self).__init__()
        self.return_per_sequence = return_per_sequence
        self.use_with_logits = use_with_logits

    def forward(self, prediction, label, target_bb=None, valid_samples=None):
        score_shape = label.shape[-2:]

        prediction = prediction.view(-1, score_shape[0], score_shape[1])
        label = label.view(-1, score_shape[0], score_shape[1])

        if valid_samples is not None:
            valid_samples = valid_samples.float().view(-1)

            if self.use_with_logits:
                prediction_accuracy_persample = F.binary_cross_entropy_with_logits(prediction, label, reduction='none')
            else:
                prediction_accuracy_persample = F.binary_cross_entropy(prediction, label, reduction='none')

            prediction_accuracy = prediction_accuracy_persample.mean((-2, -1))
            prediction_accuracy = prediction_accuracy * valid_samples

            if not self.return_per_sequence:
                num_valid_samples = valid_samples.sum()
                if num_valid_samples > 0:
                    prediction_accuracy = prediction_accuracy.sum() / num_valid_samples
                else:
                    prediction_accuracy = 0.0 * prediction_accuracy.sum()
        else:
            if self.use_with_logits:
                prediction_accuracy = F.binary_cross_entropy_with_logits(prediction, label, reduction='none')
            else:
                prediction_accuracy = F.binary_cross_entropy(prediction, label, reduction='none')

            if self.return_per_sequence:
                prediction_accuracy = prediction_accuracy.mean((-2, -1))
            else:
                prediction_accuracy = prediction_accuracy.mean()

        return prediction_accuracy


class IsTargetCellLossBinary(nn.Module):
    def __init__(self, thresh_pos=0.5, thresh_neg=0.1):
        super(IsTargetCellLossBinary, self).__init__()
        self.thresh_pos = thresh_pos
        self.thresh_neg = thresh_neg

    def forward(self, prediction, label, valid_sample=None):
        score_shape = label.shape[-2:]

        prediction = prediction.view(-1, score_shape[0], score_shape[1])
        label = label.view(-1, score_shape[0], score_shape[1])
        label_bin = (label > self.thresh_pos).float()

        uncertain_sample = (label > self.thresh_neg).float() * (label <= self.thresh_pos).float()
        if valid_sample is not None:
            valid_sample = valid_sample.float().view(-1, 1, 1) * (1.0 - uncertain_sample)

            prediction_accuracy_persample = F.binary_cross_entropy_with_logits(prediction, label_bin, reduction='none')
            num_valid_samples = valid_sample.sum()
            if num_valid_samples > 0:
                prediction_accuracy = (valid_sample * prediction_accuracy_persample).sum() / num_valid_samples
            else:
                prediction_accuracy = 1.0
        else:
            raise NotImplementedError
            prediction_accuracy = F.binary_cross_entropy_with_logits(prediction, label)

        return prediction_accuracy


class IsDistractorCellLossBinary(nn.Module):
    def __init__(self, thresh_pos=0.5, thresh_neg=0.1, pos_weight=None):
        super(IsDistractorCellLossBinary, self).__init__()
        self.thresh_pos = thresh_pos
        self.thresh_neg = thresh_neg
        self.pos_weight = pos_weight

    def forward(self, prediction, label, dimp_prediction, valid_sample=None):
        score_shape = label.shape[-2:]

        prediction = prediction.view(-1, score_shape[0], score_shape[1])
        dimp_prediction = dimp_prediction.view(-1, score_shape[0], score_shape[1])
        label = label.view(-1, score_shape[0], score_shape[1])

        is_distractor = (label < 0.01).float() * (dimp_prediction > self.thresh_pos).float()

        uncertain_sample = (dimp_prediction > self.thresh_neg).float() * (dimp_prediction <= self.thresh_pos).float()
        if valid_sample is not None:
            valid_sample = valid_sample.float().view(-1, 1, 1) * (1.0 - uncertain_sample)

            prediction_accuracy_persample = F.binary_cross_entropy_with_logits(prediction, is_distractor, reduction='none')

            if self.pos_weight is not None:
                prediction_accuracy_persample[dimp_prediction > self.thresh_pos] *= self.pos_weight
            num_valid_samples = valid_sample.sum()
            if num_valid_samples > 0:
                prediction_accuracy = (valid_sample * prediction_accuracy_persample).sum() / num_valid_samples
            else:
                prediction_accuracy = 1.0
        else:
            raise NotImplementedError
            prediction_accuracy = F.binary_cross_entropy_with_logits(prediction, label)

        return prediction_accuracy


class TargetMaskPredictionLoss(nn.Module):
    def __init__(self):
        super(TargetMaskPredictionLoss, self).__init__()

    def forward(self, prediction, prev_label, label, valid_sample=None):
        # Prediction: n x h x w x 1 x h x w
        # Prev label: n x 1 x h x w
        # Label: n x 1 x h x w
        # Valid_samples: n

        # Take out target samples
        score_shape = label.shape[-2:]

        # prediction = prediction.view(-1, score_shape[0] * score_shape[1], score_shape[0], score_shape[1])
        prediction = prediction.view(-1, score_shape[0], score_shape[1])
        prev_label = prev_label.view(-1, score_shape[0], score_shape[1])

        valid_sample = valid_sample.view(-1, 1, 1)
        target_cells = ((prev_label > 0.5) * valid_sample).view(-1)

        prediction_target_cells = prediction[target_cells, ...]

        # TODO is it making a copy?
        label = label.view(-1, 1, 1, score_shape[0], score_shape[1]).repeat(1, score_shape[0], score_shape[1], 1, 1).view(-1, score_shape[0], score_shape[1])
        label_target_cells = label[target_cells, ...]

        # Calculate L2 loss
        loss = F.mse_loss(prediction_target_cells, label_target_cells)

        return loss


class BackgroundMaskPredictionLoss(nn.Module):
    def __init__(self, max_energy=None, ignore_thresh=0):
        super(BackgroundMaskPredictionLoss, self).__init__()
        self.max_energy = max_energy
        self.ignore_thresh = ignore_thresh

    def forward(self, prediction, prev_label, label, valid_sample=None):
        # Prediction: n x h x w x 1 x h x w
        # Prev label: n x 1 x h x w
        # Label: n x 1 x h x w
        # Valid_samples: n

        # Take out target samples
        score_shape = label.shape[-2:]

        # prediction = prediction.view(-1, score_shape[0] * score_shape[1], score_shape[0], score_shape[1])
        prediction = prediction.view(-1, score_shape[0] * score_shape[1], score_shape[0], score_shape[1])
        prev_label = prev_label.view(-1, score_shape[0] * score_shape[1])

        valid_sample = valid_sample.view(-1, 1)

        background_cells = ((prev_label < 0.1) * valid_sample).view(-1, score_shape[0] * score_shape[1], 1, 1).float()

        background_score = prediction * background_cells

        label = label.view(-1, 1, score_shape[0], score_shape[1])
        background_energy = ((background_score * (label < 0.1).float() * (background_score > self.ignore_thresh).float())**2).sum(dim=(2, 3))

        # Clip energy if needed
        if self.max_energy is not None:
            background_energy = background_energy.clamp(0, self.max_energy)

        num_samples = background_cells.sum()

        if num_samples > 0:
            background_energy = background_energy.sum() / num_samples
        else:
            background_energy = 0

        # Target energy
        target_energy = ((background_score * (label > 0.5).float())**2).sum(dim=(2, 3))

        if num_samples > 0:
            target_energy = target_energy.sum() / num_samples
        else:
            target_energy = 0

        return -background_energy, target_energy
