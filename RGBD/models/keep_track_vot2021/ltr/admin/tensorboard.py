import os
import numpy as np
from collections import OrderedDict
try:
    from torch.utils.tensorboard import SummaryWriter
except:
    print('WARNING: You are using tensorboardX instead sis you have a too old pytorch version.')
    from tensorboardX import SummaryWriter


class TensorboardWriter:
    def __init__(self, directory, loader_names):
        self.directory = directory
        self.writer = OrderedDict({name: SummaryWriter(os.path.join(self.directory, name)) for name in loader_names})

    def write_info(self, module_name, script_name, description):
        tb_info_writer = SummaryWriter(os.path.join(self.directory, 'info'))
        tb_info_writer.add_text('Modulet_name', module_name)
        tb_info_writer.add_text('Script_name', script_name)
        tb_info_writer.add_text('Description', description)
        tb_info_writer.close()

    def write_epoch(self, stats: OrderedDict, epoch: int, ind=-1):
        for loader_name, loader_stats in stats.items():
            if loader_stats is None:
                continue
            for var_name, val in loader_stats.items():
                if hasattr(val, 'history') and getattr(val, 'has_new_data', True):
                    self.writer[loader_name].add_scalar(var_name, val.history[ind], epoch)

    def write_epoch_precision_recall(self, raw_stats, epoch):
        for loader_name, loader_stats in raw_stats.items():
            if loader_stats is None:
                continue
            if 'MaskedProbs' in loader_stats.keys() and 'MaskedLabels' in loader_stats.keys():
                probs = np.hstack([l.cpu().detach().numpy() for l in loader_stats['MaskedProbs']])
                labels = np.hstack([l.cpu().detach().numpy() for l in loader_stats['MaskedLabels']])

                print(probs.shape, labels.shape)

                self.writer[loader_name].add_pr_curve('certainty-prediction', labels=labels, predictions=probs,
                                                      global_step=epoch)


    def write_epoch_certainty_scores_per_seq(self, raw_stats, epoch=None):
        for loader_name, loader_stats in raw_stats.items():
            if loader_stats is None:
                continue
            for key in loader_stats.keys():
                if 'FrameProbs' in key:
                    loader_name_probs = loader_name + '_probs_' + str(epoch)
                    loader_name_iou = loader_name + '_iou_' + str(epoch)

                    if loader_name_probs not in self.writer:
                        self.writer[loader_name_probs] = SummaryWriter(os.path.join(self.directory, loader_name_probs))
                    if loader_name_iou not in self.writer:
                        self.writer[loader_name_iou] = SummaryWriter(os.path.join(self.directory, loader_name_iou))

                    seq_name = key.split('/')[1]
                    probs = np.hstack([l.cpu().detach().numpy() for l in loader_stats['FrameProbs/' + seq_name]]).reshape(-1)
                    overlaps = np.hstack([l.cpu().detach().numpy() for l in loader_stats['FrameIoU/' + seq_name]]).reshape(-1)

                    for i in range(probs.shape[0]):
                        # self.writer[loader_name].add_scalars('Seqs/'+seq_name, {'Probs': probs[i], 'IoU': overlaps[i]}, i)
                        self.writer[loader_name_probs].add_scalar('Seqs/' + seq_name, probs[i], i)
                        self.writer[loader_name_iou].add_scalar('Seqs/' + seq_name, overlaps[i], i)
