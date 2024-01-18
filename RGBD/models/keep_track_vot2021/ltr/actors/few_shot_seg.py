from . import BaseActor
import torch
import torch.nn as nn


class FSSActor(BaseActor):
    def __init__(self, net, objective, loss_weight=None, disable_all_bn=False):
        super().__init__(net, objective)
        if loss_weight is None:
            loss_weight = {'segm': 1.0}
        self.loss_weight = loss_weight

        self.disable_all_bn = disable_all_bn

    def train(self, mode=True):
        """ Set whether the network is in train mode.
        args:
            mode (True) - Bool specifying whether in training mode.
        """
        self.net.train(mode)

        if self.disable_all_bn:
            for m in self.net.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()

    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'train_images', 'test_images', 'train_anno',
                    'test_proposals', 'proposal_iou', 'test_label', 'train_masks' and 'test_masks'
        returns:
            loss    - the training loss
            stats  -  dict containing detailed losses
        """
        # Run network
        segm_pred = self.net(train_imgs=data['train_images'], test_imgs=data['test_images'],
                             train_masks=data['train_masks'], test_masks=data['test_masks'])

        gt_segm = data['test_masks'].view(-1, 1, *data['test_masks'].shape[-2:])

        loss_segm = self.loss_weight['segm'] * self.objective['segm'](segm_pred.view(gt_segm.shape), gt_segm)

        # Total loss
        loss = loss_segm

        if torch.isinf(loss) or torch.isnan(loss):
            raise Exception('ERROR: Loss was nan or inf!!!')

        # Log stats
        stats = {'Loss/total': loss.item(),
                 'Loss/segm': loss_segm.item()}

        return loss, stats

