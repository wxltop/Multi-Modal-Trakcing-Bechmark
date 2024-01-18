import torch.nn as nn
import torch
import torch.nn.functional as F


class TrackingClassificationAccuracy(nn.Module):
    def __init__(self, threshold, neg_threshold=None):
        super(TrackingClassificationAccuracy, self).__init__()
        self.threshold = threshold

        if neg_threshold is None:
            neg_threshold = threshold
        self.neg_threshold = neg_threshold

    def forward(self, prediction, label, valid_samples=None):
        prediction_reshaped = prediction.view(-1, prediction.shape[-2] * prediction.shape[-1])
        label_reshaped = label.view(-1, label.shape[-2] * label.shape[-1])

        prediction_max_val, argmax_id = prediction_reshaped.max(dim=1)
        label_max_val, _ = label_reshaped.max(dim=1)

        label_val_at_peak = label_reshaped[torch.arange(len(argmax_id)), argmax_id]
        label_val_at_peak = torch.max(label_val_at_peak, torch.zeros_like(label_val_at_peak))

        prediction_correct = ((label_val_at_peak >= self.threshold) & (label_max_val > 0.25)) | ((label_val_at_peak < self.neg_threshold) & (label_max_val < 0.25))

        if valid_samples is not None:
            valid_samples = valid_samples.float().view(-1)

            num_valid_samples = valid_samples.sum()
            if num_valid_samples > 0:
                prediction_accuracy = (valid_samples * prediction_correct.float()).sum() / num_valid_samples
            else:
                prediction_accuracy = 1.0
        else:
            prediction_accuracy = prediction_correct.float().mean()

        return prediction_accuracy, prediction_correct.float()


class TrackingCertainityPredictionLossAndAccuracy(nn.Module):
    def __init__(self, threshold):
        super(TrackingCertainityPredictionLossAndAccuracy, self).__init__()
        self.threshold = threshold

    def forward(self, score_prediction, label, confidence_prediction, valid_sample=None):
        prediction_reshaped = score_prediction.view(-1, score_prediction.shape[-2] * score_prediction.shape[-1])
        label_reshaped = label.view(-1, label.shape[-2] * label.shape[-1])

        prediction_max_val, argmax_id = prediction_reshaped.max(dim=1)
        label_max_val, _ = label_reshaped.max(dim=1)

        label_val_at_peak = label_reshaped[torch.arange(len(argmax_id)), argmax_id]
        label_val_at_peak = torch.max(label_val_at_peak, torch.zeros_like(label_val_at_peak))

        prediction_correct = ((label_val_at_peak >= self.threshold) & (label_max_val > 0.25)) | ((label_val_at_peak < self.threshold) & (label_max_val < 0.25))
        prediction_correct = prediction_correct.float()

        confidence_prediction = confidence_prediction.view(-1)

        if valid_sample is not None:
            valid_sample = valid_sample.float().view(-1)

            num_valid_samples = valid_sample.sum()
            if num_valid_samples > 0:
                conf_pred_loss = F.binary_cross_entropy_with_logits(confidence_prediction, prediction_correct, reduction='none')
                conf_pred_loss = (conf_pred_loss * valid_sample).sum() / num_valid_samples

                conf_pred_acc = ((confidence_prediction > 0.0).float() * prediction_correct.float() * valid_sample).sum() / num_valid_samples
            else:
                conf_pred_loss = 0.0
                conf_pred_acc = 1.0
        else:
            conf_pred_loss = F.binary_cross_entropy_with_logits(confidence_prediction, prediction_correct)
            conf_pred_acc = ((confidence_prediction > 0.0).float() * prediction_correct.float()).mean()

        return conf_pred_loss, conf_pred_acc
