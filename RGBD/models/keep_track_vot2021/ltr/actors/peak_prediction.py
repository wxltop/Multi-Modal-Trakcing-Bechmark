from . import BaseActor
import torch
import torch.nn.functional as F
from ltr.data.processing_utils import iou_gen
import torch.nn as nn


class PPActor(BaseActor):
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

        preds = self.net(**data)

        # Classification losses for the different optimization iterations
        losses = self.objective['peak_pred'](**data, **preds)


        # Total loss
        loss = losses['total'].mean()

        # Log stats
        stats = {
            'Loss/total': loss.item(),
            'Loss/nll_pos': losses['nll_pos'].mean().item(),
            'Loss/nll_neg': losses['nll_neg'].mean().item(),
            'Loss/num_matchable': losses['num_matchable'].mean().item(),
            'Loss/num_unmatchable': losses['num_unmatchable'].mean().item(),
            'Loss/sinkhorn_norm': losses['sinkhorn_norm'].mean().item(),
            'Loss/bin_score': losses['bin_score'].item(),
        }

        if hasattr(self.objective['peak_pred'], 'metrics'):
            metrics = self.objective['peak_pred'].metrics(**data, **preds)

            for key, val in metrics.items():
                stats[key] = torch.mean(val[~torch.isnan(val)]).item()


        # raw_stats = {}

        # for key, val in clf_loss_stats.items():
        #     if 'Masked' in key:
        #         raw_stats[key] = val
        #     elif 'Probs' in key and 'Masked' not in key:
        #         if 'seq_name' in data:
        #             raw_stats['FrameProbs/' + data['seq_name'][0]] = val
        #             raw_stats['FrameIoU/' + data['seq_name'][0]] = data['overlap']
        #     else:
        #         stats['Loss/' + key] = val

        return loss, stats, None
