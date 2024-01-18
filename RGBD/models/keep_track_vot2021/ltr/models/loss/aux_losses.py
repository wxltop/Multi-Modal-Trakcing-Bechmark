import math
import torch
import torch.nn as nn
from torch.nn import functional as F


class SemanticClassifierLoss(nn.Module):
    def __init__(self, class_dict):
        super().__init__()
        self.class_dict = class_dict
        self.class_to_id_map = {d:{class_name: i for i, class_name in enumerate(class_list)} for d, class_list in self.class_dict.items()}
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')

    def forward(self, predictions, target_classes, datasets, valid=None):
        # Currently we assume that all frames in the sequence have the same class.
        # Thus target classes and datasets are 1d lists
        loss_total = 0
        num_elements = 0
        for cur_dataset, pred_score in predictions.items():
            seq_in_cur_dataset = [cur_dataset == d for d in datasets]
            target_classes_cur_dataset = [t for t, vs in zip(target_classes, seq_in_cur_dataset) if vs]

            labels_cur_dataset = torch.tensor([self.class_to_id_map[cur_dataset][c] for c in target_classes_cur_dataset]).view(1, -1)
            labels_cur_dataset = labels_cur_dataset.repeat(pred_score.shape[0], 1).view(-1).to(pred_score.device)

            seq_in_cur_dataset_t = torch.tensor(seq_in_cur_dataset, dtype=torch.bool, device=pred_score.device)
            predictions_cur_dataset = pred_score[:, seq_in_cur_dataset_t, :]

            # Remove invalid frames
            valid_cur_dataset = valid[:, seq_in_cur_dataset_t].view(-1)

            valid_predictions = predictions_cur_dataset.view(-1, predictions_cur_dataset.shape[-1])[valid_cur_dataset, :]
            valid_labels = labels_cur_dataset[valid_cur_dataset]

            if valid_labels.numel() > 0:
                loss = self.ce_loss(valid_predictions, valid_labels.long())
                loss_total += loss.sum()
                num_elements += loss.numel()

        return loss_total / (num_elements + 1e-12)


class SemanticClassifierAcc(nn.Module):
    def __init__(self, class_dict):
        super().__init__()
        self.class_dict = class_dict
        self.class_to_id_map = {d:{class_name: i for i, class_name in enumerate(class_list)} for d, class_list in self.class_dict.items()}

    def forward(self, predictions, target_classes, datasets, valid=None):
        # Currently we assume that all frames in the sequence have the same class.
        # Thus target classes and datasets are 1d lists
        num_correct_pred = 0
        num_elements = 0
        for cur_dataset, pred_score in predictions.items():
            seq_in_cur_dataset = [cur_dataset == d for d in datasets]
            target_classes_cur_dataset = [t for t, vs in zip(target_classes, seq_in_cur_dataset) if vs]

            labels_cur_dataset = torch.tensor(
                [self.class_to_id_map[cur_dataset][c] for c in target_classes_cur_dataset]).view(1, -1)
            labels_cur_dataset = labels_cur_dataset.repeat(pred_score.shape[0], 1).view(-1).to(pred_score.device)

            seq_in_cur_dataset_t = torch.tensor(seq_in_cur_dataset, dtype=torch.bool, device=pred_score.device)
            predictions_cur_dataset = pred_score[:, seq_in_cur_dataset_t, :]

            # Remove invalid frames
            valid_cur_dataset = valid[:, seq_in_cur_dataset_t].view(-1)

            valid_predictions = predictions_cur_dataset.view(-1, predictions_cur_dataset.shape[-1])[valid_cur_dataset,
                                :]
            valid_labels = labels_cur_dataset[valid_cur_dataset]

            if valid_labels.numel() > 0:
                pred_label = valid_predictions.argmax(dim=1)
                num_correct_pred += (pred_label == valid_labels).sum()
                num_elements += pred_label.numel()

        if num_elements > 0:
            return num_correct_pred.float() / num_elements
        else:
            return 0


