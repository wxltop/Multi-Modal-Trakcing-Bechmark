from . import BaseActor
from ltr.data.processing_utils import iou_gen


class AtomActor(BaseActor):
    """ Actor for training the IoU-Net in ATOM"""
    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'train_images', 'test_images', 'train_anno',
                    'test_proposals' and 'proposal_iou'.

        returns:
            loss    - the training loss
            stats   - dict containing detailed losses
        """
        # Run network to obtain IoU prediction for each proposal in 'test_proposals'
        iou_pred = self.net(data['train_images'], data['test_images'], data['train_anno'], data['test_proposals'])

        iou_pred = iou_pred.view(-1, iou_pred.shape[2])
        iou_gt = data['proposal_iou'].view(-1, data['proposal_iou'].shape[2])

        # Compute loss
        loss = self.objective(iou_pred, iou_gt)

        # Return training stats
        stats = {'Loss/total': loss.item(),
                 'Loss/iou': loss.item()}

        return loss, stats


class Atomv2Actor(BaseActor):
    """ Actor for training the IoU-Net in ATOM"""
    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'train_images', 'test_images', 'train_anno',
                    'test_proposals' and 'proposal_iou'.

        returns:
            loss    - the training loss
            states  -  dict containing detailed losses
        """
        # Run network to obtain IoU prediction for each proposal in 'test_proposals'
        iou_pred = self.net(data['train_images'], data['test_images'], data['train_anno'], data['test_proposals'])

        iou_pred = iou_pred.view(-1, iou_pred.shape[2])
        iou_gt = data['proposal_iou'].view(-1, data['proposal_iou'].shape[2])

        proposal_density = data['proposal_density'].view(-1, data['proposal_density'].shape[2])
        gt_density = data['gt_density'].view(-1, data['gt_density'].shape[2])

        # Compute loss
        loss = self.objective(iou_pred, iou_gt, proposal_density)

        # Return training stats
        stats = {'Loss/total': loss.item(),
                 'Loss/iou': loss.item()}

        return loss, stats

class AtomBBKLActor(BaseActor):
    """ Actor for training the IoU-Net in ATOM with BBKL"""
    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'train_images', 'test_images', 'train_anno',
                    'test_proposals' and 'proposal_iou'.

        returns:
            loss    - the training loss
            states  -  dict containing detailed losses
        """
        # Run network to obtain IoU prediction for each proposal in 'test_proposals'
        bb_scores = self.net(data['train_images'], data['test_images'], data['train_anno'], data['test_proposals'])

        bb_scores = bb_scores.view(-1, bb_scores.shape[2])
        proposal_density = data['proposal_density'].view(-1, data['proposal_density'].shape[2])
        gt_density = data['gt_density'].view(-1, data['gt_density'].shape[2])

        # Compute loss
        loss = self.objective(bb_scores, sample_density=proposal_density, gt_density=gt_density, mc_dim=1)

        # Return training stats
        stats = {'Loss/total': loss.item(),
                 'Loss/bb_ce': loss.item()}

        return loss, stats


class AtomHNActor(BaseActor):
    """ Actor for training the IoU-Net in ATOM"""
    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'train_images', 'test_images', 'train_anno',
                    'test_proposals' and 'proposal_iou'.

        returns:
            loss    - the training loss
            states  -  dict containing detailed losses
        """
        # Extract backbone features
        train_imgs = data['train_images']
        test_imgs = data['test_images']

        train_anno = data['train_anno']
        test_proposals = data['test_proposals']

        num_sequences = train_imgs.shape[-4]
        num_train_images = train_imgs.shape[0] if train_imgs.dim() == 5 else 1
        num_test_images = test_imgs.shape[0] if test_imgs.dim() == 5 else 1

        train_feat = self.net.extract_backbone_features(
            train_imgs.view(-1, train_imgs.shape[-3], train_imgs.shape[-2], train_imgs.shape[-1]))
        test_feat = self.net.extract_backbone_features(
            test_imgs.view(-1, test_imgs.shape[-3], test_imgs.shape[-2], test_imgs.shape[-1]))

        train_feat_iou = [feat.view(num_train_images, num_sequences, feat.shape[-3], feat.shape[-2], feat.shape[-1])
                          for feat in train_feat.values()]
        test_feat_iou = [feat for feat in test_feat.values()]

        # Get attention vector
        # Extract first train sample
        train_feat_iou = [f[0, ...] for f in train_feat_iou]
        train_anno = train_anno[0, ...]

        filter = self.net.bb_regressor.get_filter(train_feat_iou, train_anno)

        iou_feat = self.net.bb_regressor.get_iou_feat(test_feat_iou)

        # Repeating for now to keep it simple. Modify predict_iou if this is expensive
        filter = [f.view(1, num_sequences, -1).repeat(num_test_images, 1, 1).view(
            num_sequences * num_test_images, -1) for f in filter]

        # Optimize boxes
        test_proposals = test_proposals.view(num_sequences * num_test_images, -1, 4)

        filter_c = [f.clone().detach() for f in filter]
        iou_feat_c = [f.clone().detach() for f in iou_feat]

        test_proposals_optim, _ = self.net.bb_regressor.optimize_boxes(filter_c, iou_feat_c,
                                                                       test_proposals, step_length=1, num_iterations=5)

        # find cases where bb_regressor failed
        optim_iou = iou_gen(test_proposals_optim, data['test_anno'].view(-1, 1, 4))*2 - 1

        hard_neg = optim_iou < data['proposal_iou'].view(optim_iou.shape)

        test_proposals[hard_neg] = test_proposals_optim[hard_neg]

        iou_gt = data['proposal_iou'].view(-1, data['proposal_iou'].shape[2])
        iou_gt[hard_neg] = optim_iou[hard_neg]
        iou_pred = self.net.bb_regressor.predict_iou(filter, iou_feat, test_proposals)

        # Compute loss
        loss = self.objective(iou_pred, iou_gt)

        # Return training stats
        stats = {'Loss/total': loss.item(),
                 'Loss/iou': loss.item()}

        return loss, stats