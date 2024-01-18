from . import BaseActor
import torch
import torch.nn.functional as F
import time
from pytracking.utils.visdom import Visdom


class SimpleCorrActor(BaseActor):
    def __init__(self, net, objective=None, loss_weight=None, visdom_info=None):
        super().__init__(net, objective)
        if loss_weight is None:
            loss_weight = {'corr_test': 1.0,
                           'corr_train': 1.0}
        self.loss_weight = loss_weight

        self.pause_mode = False

        if visdom_info is not None:
            self.visdom = Visdom(ui_info={'handler': self.visdom_ui_handler, 'win_id': 'Scores UI'},
                                 visdom_info=visdom_info)
        else:
            self.visdom = None


    def visdom_ui_handler(self, data):
        if data['event_type'] == 'KeyPress':
            if data['key'] == 'n':
                self.pause_mode = False


    def __call__(self, data):
        # Run network
        scores, losses = self.net(data['train_images'], data['test_images'], test_anno=data['test_label'], compute_losses=True)

        # Compute loss
        # loss_corr_test = (self.loss_weight['corr_test'] / len(losses['test'])) * sum(losses['test'])
        # loss = loss_corr_test

        loss = torch.zeros(1)

        if self.visdom is not None:
            im_train = 255*(data['train_images'][0]/5+0.5)
            im_test = 255*(data['test_images'][0]/5+0.5)
            # self.visdom.register(im_train, 'image', title='Train image')
            self.visdom.register(im_test, 'image', title='Test image')
            self.visdom.register((im_train, scores.shape[-2:]), 'cost_volume_ui', title='Scores UI')
            self.visdom.register(scores[0].view(*scores.shape[-2:], *scores.shape[-2:]).clamp(0), 'cost_volume', title='Scores')
            # self.visdom.register(torch.stack(losses['train']), 'lineplot', title='Total loss')
            # loss_names = ['train_source', 'train_reg', 'train_target']
            # self.visdom.register({k: torch.stack(losses[k]) for k in loss_names}, 'lineplot', title='Loss terms')


            self.pause_mode = True
            while self.pause_mode:
                time.sleep(0.1)

        stats = {'Loss/total': loss.item()}

        return loss, stats



class LocalCorrActor(BaseActor):
    def __init__(self, net, objective=None, loss_weight=None, visdom_info=None):
        super().__init__(net, objective)
        if loss_weight is None:
            loss_weight = {'corr_test': 1.0,
                           'corr_train': 1.0}
        self.loss_weight = loss_weight

        self.pause_mode = False

        if visdom_info is not None:
            self.visdom = Visdom(ui_info={'handler': self.visdom_ui_handler, 'win_id': 'Scores UI'},
                                 visdom_info=visdom_info)
        else:
            self.visdom = None


    def visdom_ui_handler(self, data):
        if data['event_type'] == 'KeyPress':
            if data['key'] == 'n':
                self.pause_mode = False


    def __call__(self, data):
        # Run network
        scores, losses = self.net(data['train_images'], data['test_images'], test_anno=data['test_label'], compute_losses=True)

        # Compute loss
        # loss_corr_test = (self.loss_weight['corr_test'] / len(losses['test'])) * sum(losses['test'])
        # loss = loss_corr_test

        loss = torch.zeros(1)

        if self.visdom is not None:
            im_train = 255*(data['train_images'][0]/5+0.5)
            im_test = 255*(data['test_images'][0]/5+0.5)
            # self.visdom.register(im_train, 'image', title='Train image')
            self.visdom.register(im_test, 'image', title='Test image')
            self.visdom.register((im_train, scores.shape[-2:]), 'cost_volume_ui', title='Scores UI')
            self.visdom.register(scores[0].permute(1,2,0).reshape(*scores.shape[-2:], 9, 9).clamp(0), 'cost_volume', title='Scores')
            self.visdom.register(torch.stack(losses['train']), 'lineplot', title='Total loss')
            loss_names = ['train_source', 'train_reg']
            self.visdom.register({k: torch.stack(losses[k]) for k in loss_names}, 'lineplot', title='Loss terms')


            self.pause_mode = True
            while self.pause_mode:
                time.sleep(0.1)

        stats = {'Loss/total': loss.item()}

        return loss, stats