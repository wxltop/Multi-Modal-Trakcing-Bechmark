from . import BaseActor
import torch
import torch.nn.functional as F
from ltr.data.processing_utils import iou_gen
import torch.nn as nn


class CPActor(BaseActor):
    """Actor for training the DiMP network."""
    def __init__(self, net, objective):
        super().__init__(net, objective)

    def __call__(self, data):
        """
        args:
            data - The input data.

        returns:
            loss    - the training loss
            stats  -  dict containing detailed losses
        """
        # Run network
        # for key, val in data.items():
        #     try:
        #         print(key, val.shape)
        #     except:
        #         print(key, val)

        certainty_pred = self.net(**data)

        # Classification losses for the different optimization iterations
        clf_loss_test, clf_loss_stats = self.objective['certainty_pred'](certainty_pred, data['overlap'])

        # Total loss
        loss = clf_loss_test

        # Log stats
        stats = {'Loss/total': loss.item()}
        raw_stats = {}

        for key, val in clf_loss_stats.items():
            if 'Masked' in key:
                raw_stats[key] = val
            elif 'Probs' in key and 'Masked' not in key:
                if 'seq_name' in data:
                    raw_stats['FrameProbs/' + data['seq_name'][0]] = val
                    raw_stats['FrameIoU/' + data['seq_name'][0]] = data['overlap']
            else:
                stats['Loss/' + key] = val

        return loss, stats, raw_stats
