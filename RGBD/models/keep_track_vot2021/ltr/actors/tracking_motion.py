from . import BaseActor
import torch
from ltr.utils.motion_tracking import DiMPScoreJittering, DiMPScoreJitteringState
import torch.nn.functional as F
from pytracking.utils.visdom import Visdom
import time
import random


class MotionTrackerActor(BaseActor):
    def visdom_ui_handler(self, data):
        if data['event_type'] == 'KeyPress':
            if data['key'] == 'p':
                self.pause_mode = not self.pause_mode

            elif data['key'] == 'ArrowRight' and self.pause_mode:
                self.step = True

    def __init__(self, net, objective, loss_weight=None, dimp_jitter_params=None, visdom_info=None):
        super().__init__(net, objective)
        self.loss_weight = loss_weight

        if dimp_jitter_params is not None:
            self.dimp_jitter_fn = DiMPScoreJittering(dimp_jitter_params['p_zero'],
                                                     dimp_jitter_params['distractor_ratio'],
                                                     dimp_jitter_params['p_distractor'],
                                                     dimp_jitter_params['max_distractor_enhance_factor'],
                                                     dimp_jitter_params['min_distractor_enhance_factor'])
        else:
            self.dimp_jitter_fn = None

        self.pause_mode = False
        self.step = False

        if visdom_info is not None:
            self.visdom = Visdom(ui_info={'handler': self.visdom_ui_handler, 'win_id':  'Test Cur images'}, visdom_info=visdom_info)
        else:
            self.visdom = None

    def __call__(self, data):
        while True:
            if not self.pause_mode:
                break
            elif self.step:
                self.step = False
                break
            else:
                time.sleep(0.1)

        data['test_images'] = data['test_images'].view(2, -1, data['test_images'].shape[1],
                                                       data['test_images'].shape[2], data['test_images'].shape[3],
                                                       data['test_images'].shape[4])

        data['test_anno'] = data['test_anno'].view(2, -1, data['test_anno'].shape[1],
                                                   data['test_anno'].shape[2])

        data['test_label'] = data['test_label'].view(2, -1, data['test_label'].shape[1],
                                                     data['test_label'].shape[2], data['test_label'].shape[3])

        data['test_proposals'] = data['test_proposals'].view(2, -1, data['test_proposals'].shape[-3],
                                                             data['test_proposals'].shape[-2],
                                                             data['test_proposals'].shape[-1])
        data['test_proposals'] = data['test_proposals'][1, ...]

        data['proposal_iou'] = data['proposal_iou'].view(2, -1, data['proposal_iou'].shape[-2], data['proposal_iou'].shape[-1])
        data['proposal_iou'] = data['proposal_iou'][1, ...]
        response_preds, iou_pred, dimp_pred, motion_pred = self.net(data['train_images'], data['test_images'],
                                                                    data['train_anno'],
                                                                    data['test_proposals'],
                                                                    data['train_label'],
                                                                    test_ref_label=data['test_label'][0, ...],
                                                                    test_ref_anno=data['test_anno'][0, ...],
                                                                    dimp_postprocess_fn=self.dimp_jitter_fn,
                                                                    test_label=data['test_label'][1, ...]
                                                                    )

        clf_loss_test = self.objective['test_clf'](response_preds, data['test_label'][1, ...], data['test_anno'][1, ...])
        clf_loss_test_w = self.loss_weight['test_clf'] * clf_loss_test

        zero_tensor = torch.zeros_like(clf_loss_test_w)

        dimp_pred = [d.view(data['test_label'][1, ...].shape) for d in dimp_pred]
        dimp_loss_test = [self.objective['dimp_clf'](d, data['test_label'][1, ...], data['test_anno'][1, ...])
                          for d in dimp_pred]
        dimp_loss_test_final = dimp_loss_test[-1]
        dimp_loss_test_init = dimp_loss_test[0]

        if len(dimp_loss_test) < 3:
            dimp_loss_test_iter = zero_tensor
        else:
            dimp_loss_test_iter = sum(dimp_loss_test[1:-1])

        dimp_loss_test_init_w = self.loss_weight['dimp_clf_init'] * dimp_loss_test_init
        dimp_loss_test_final_w = self.loss_weight['dimp_clf_final'] * dimp_loss_test_final
        dimp_loss_test_iter_w = self.loss_weight['dimp_clf_iter'] * dimp_loss_test_iter

        motion_pred_loss_test = zero_tensor
        if 'motion_clf' in self.loss_weight.keys():
            motion_pred_loss_test = self.objective['motion_clf'](motion_pred, data['test_label'][1, ...],
                                                                 data['test_anno'][1, ...])
        motion_pred_loss_test_w = self.loss_weight['motion_clf'] * motion_pred_loss_test

        if 'target_absent_test_frame' in data.keys():
            data['target_absent_test_frame'] = data['target_absent_test_frame'].view(2, -1,
                                                                                     data['target_absent_test_frame'].shape[
                                                                                       1])
        else:
            data['target_absent_test_frame'] = data['is_distractor_test_frame'].view(2, -1,
                                                                                     data[
                                                                                         'is_distractor_test_frame'].shape[
                                                                                         1])
        target_absent = data['target_absent_test_frame'][1, ...].view(-1)

        if iou_pred is not None:
            iou_pred_valid = iou_pred.view(-1, iou_pred.shape[2])[target_absent == 0, :]
            iou_gt_valid = data['proposal_iou'].view(-1, data['proposal_iou'].shape[2])[target_absent == 0, :]

            # IoU loss
            loss_iou = self.objective['iou'](iou_pred_valid, iou_gt_valid)
        else:
            loss_iou = zero_tensor

        loss_iou_w = self.loss_weight['iou'] * loss_iou

        loss = loss_iou_w + clf_loss_test_w + dimp_loss_test_init_w + dimp_loss_test_iter_w + dimp_loss_test_final_w \
               + motion_pred_loss_test_w

        test_clf_acc = zero_tensor
        if 'clf_acc' in self.objective.keys():
            test_clf_acc = self.objective['clf_acc'](response_preds, data['test_label'][1, ...])

        dimp_clf_acc = zero_tensor
        if 'clf_acc' in self.objective.keys():
            dimp_clf_acc = self.objective['clf_acc'](dimp_pred[-1], data['test_label'][1, ...])

        motion_clf_acc = zero_tensor
        if 'clf_acc' in self.objective.keys() and motion_pred is not None:
            motion_clf_acc = self.objective['clf_acc'](motion_pred, data['test_label'][1, ...])

        stats = {'Loss/total': loss.item(),
                 'Loss/iou': loss_iou_w.item(),
                 'Loss/test_clf': clf_loss_test_w.item(),
                 'Loss/motion_clf': motion_pred_loss_test_w.item(),
                 'Loss/dimp_clf_init': dimp_loss_test_init_w.item(),
                 'Loss/dimp_clf_iter': dimp_loss_test_iter_w.item(),
                 'Loss/dimp_clf_final': dimp_loss_test_final_w.item(),
                 'Loss/raw/iou': loss_iou.item(),
                 'Loss/raw/test_clf': clf_loss_test.item(),
                 'Loss/raw/motion_clf': motion_pred_loss_test.item(),
                 'Loss/raw/dimp_clf_init': dimp_loss_test_init.item(),
                 'Loss/raw/dimp_clf_iter': dimp_loss_test_iter.item(),
                 'Loss/raw/dimp_clf_final': dimp_loss_test_final.item(),
                 'Loss/raw/test_clf_acc': test_clf_acc.item(),
                 'Loss/raw/motion_clf_acc': motion_clf_acc.item(),
                 'Loss/raw/dimp_clf_acc': dimp_clf_acc.item()}

        if self.visdom is not None:
            self.visdom.register(data['test_images'][0, ...], 'image_list', title='Test Prev images')
            self.visdom.register(data['test_images'][1, ...], 'image_list', title='Test Cur images')
            self.visdom.register(data['test_label'][1, ...], 'heatmap_list', title='Label')

        return loss, stats


class MotionSequenceTrackerActor(BaseActor):
    def visdom_ui_handler(self, data):
        if data['event_type'] == 'KeyPress':
            if data['key'] == 'p':
                self.pause_mode = not self.pause_mode

            elif data['key'] == 'ArrowRight' and self.pause_mode:
                self.step = True

    def __init__(self, net, objective, loss_weight=None, visdom_info=None, dimp_jitter_params=None, dropout_params=None):
        super().__init__(net, objective)
        self.loss_weight = loss_weight

        self.dimp_jitter_params = dimp_jitter_params

        if dropout_params is None:
            self.dropout_params = {}
        else:
            self.dropout_params = dropout_params

        if dimp_jitter_params is None:
            self.dimp_jitter_fn = None
        else:
            dimp_jitter_type = dimp_jitter_params.get('mode', 'default')
            self.dimp_jitter_type = dimp_jitter_type

            if dimp_jitter_type == 'default':
                self.dimp_jitter_fn = DiMPScoreJittering(dimp_jitter_params['p_zero'],
                                                         dimp_jitter_params['distractor_ratio'],
                                                         dimp_jitter_params['p_distractor'],
                                                         dimp_jitter_params['max_distractor_enhance_factor'],
                                                         dimp_jitter_params['min_distractor_enhance_factor'])
            elif dimp_jitter_type == 'sequence':
                self.dimp_jitter_fn = DiMPScoreJitteringState(dimp_jitter_params['num_sequence'],
                                                              dimp_jitter_params['p_distractor_lo'],
                                                              dimp_jitter_params['p_distractor_hi'],
                                                              dimp_jitter_params['p_dimp_fail'])

        self.pause_mode = False
        self.step = False

        # TODO set it somewhere
        self.device = torch.device("cuda:0")

        if visdom_info is not None:
            self.visdom = Visdom(ui_info={'handler': self.visdom_ui_handler, 'win_id': 'Test images'},
                                 visdom_info=visdom_info)
        else:
            self.visdom = None

    def __call__(self, data):
        sequence_length = data['test_images'].shape[0]
        num_sequences = data['test_images'].shape[1]

        valid_samples = data['test_valid_image'].to(self.device)
        test_visibility = data['test_visible_ratio'].to(self.device)

        # Init tracker
        train_images = data['train_images'].to(self.device)
        train_anno = data['train_anno'].to(self.device)
        train_label = data['train_label'].to(self.device)
        dimp_filters = self.net.train_classifier(train_images, train_anno)

        # Track in the first frame
        test_image_cur = data['test_images'][0, ...].to(self.device)
        backbone_feat_prev = self.net.extract_backbone_features(test_image_cur)
        backbone_feat_prev = backbone_feat_prev[self.net.classification_layer]
        backbone_feat_prev = backbone_feat_prev.view(1, num_sequences, -1,
                                                     backbone_feat_prev.shape[-2], backbone_feat_prev.shape[-1])

        dimp_scores_prev = self.net.dimp_classifier.track_frame(dimp_filters, backbone_feat_prev)

        # Remove last row and col
        dimp_scores_prev = dimp_scores_prev[:, :, :-1, :-1].contiguous()

        label_prev = data['test_label'][0:1, ...].to(self.device)
        label_prev = label_prev[:, :, :-1, :-1].contiguous()

        anno_prev = data['test_anno'][0:1, ...].to(self.device)

        state_prev = None

        is_valid_prev = valid_samples[0, :].view(1, -1, 1, 1).byte()

        zero_tensor = torch.zeros(1).to(self.device)

        clf_loss_test = 0
        dimp_loss_test = 0
        motion_pred_loss_test = 0
        test_clf_acc = 0
        dimp_clf_acc = 0
        motion_clf_acc = 0

        test_seq_all_correct = torch.ones(num_sequences).to(self.device)
        dimp_seq_all_correct = torch.ones(num_sequences).to(self.device)

        is_target_loss = zero_tensor.clone()
        is_target_after_prop_loss = zero_tensor.clone()

        for i in range(1, sequence_length):
            test_image_cur = data['test_images'][i, ...].to(self.device)
            test_label_cur = data['test_label'][i:i+1, ...].to(self.device)
            test_label_cur = test_label_cur[:, :, :-1, :-1].contiguous()

            test_anno_cur = data['test_anno'][i:i + 1, ...].to(self.device)

            backbone_feat_cur = self.net.extract_backbone_features(test_image_cur)
            backbone_feat_cur = backbone_feat_cur[self.net.classification_layer]
            backbone_feat_cur = backbone_feat_cur.view(1, num_sequences, -1,
                                                       backbone_feat_cur.shape[-2], backbone_feat_cur.shape[-1])

            if self.dropout_params.get('use_dropout', False) and random.random() < self.dropout_params.get('p_dropout', -1):
                backbone_feat_cur = F.dropout(backbone_feat_cur, p=0.2, inplace=True)

            dimp_scores_cur = self.net.dimp_classifier.track_frame(dimp_filters, backbone_feat_cur)
            dimp_scores_cur = dimp_scores_cur[:, :, :-1, :-1].contiguous()

            jitter_info = None
            if self.dimp_jitter_fn is not None:
                if self.dimp_jitter_type == "default":
                    dimp_scores_cur = self.dimp_jitter_fn(dimp_scores_cur, test_label_cur.clone())
                elif self.dimp_jitter_type == "sequence":
                    if i > 5:
                        dimp_scores_cur, jitter_info = self.dimp_jitter_fn(dimp_scores_cur, test_label_cur.clone())

                        if not self.dimp_jitter_params.get('jitter_cv', False):
                            jitter_info = None

            predictor_input_data = {'input1': backbone_feat_prev, 'input2': backbone_feat_cur,
                                    'label_prev': label_prev, 'anno_prev': anno_prev,
                                    'dimp_score_prev': dimp_scores_prev, 'dimp_score_cur': dimp_scores_cur,
                                    'state_prev': state_prev,
                                    'jitter_info': jitter_info}

            predictor_output = self.net.predictor(predictor_input_data)

            predicted_resp = predictor_output['response']
            state_prev = predictor_output['state_cur']
            aux_data = predictor_output['auxiliary_outputs']

            if 'motion_response' in predictor_output.keys():
                motion_pred = predictor_output['motion_response']
            elif 'motion_response' in aux_data.keys():
                motion_pred = aux_data['motion_response']
            else:
                motion_pred = None

            is_valid = valid_samples[i, :].view(1, -1, 1, 1).byte()
            uncertain_frame = (test_visibility[i, :].view(1, -1, 1, 1) < 0.75) * (test_visibility[i, :].view(1, -1, 1, 1) > 0.25)

            is_valid = is_valid * (1 - uncertain_frame)

            clf_loss_test_new = self.objective['test_clf'](predicted_resp, test_label_cur,
                                                           test_anno_cur, valid_samples=is_valid)
            clf_loss_test += clf_loss_test_new

            dimp_loss_test_new = self.objective['dimp_clf'](dimp_scores_cur, test_label_cur,
                                                            test_anno_cur, valid_samples=is_valid)
            dimp_loss_test += dimp_loss_test_new

            motion_pred_loss_test_new = 0
            if 'motion_clf' in self.loss_weight.keys():
                motion_pred_loss_test_new = self.objective['motion_clf'](motion_pred, test_label_cur,
                                                                         test_anno_cur, valid_samples=is_valid)
            motion_pred_loss_test += motion_pred_loss_test_new

            if 'is_target' in aux_data and 'is_target' in self.loss_weight.keys() and 'is_target' in self.objective.keys():
                is_target_loss_new = self.objective['is_target'](aux_data['is_target'], label_prev, is_valid_prev)
                is_target_loss += is_target_loss_new

            if 'is_target_after_prop' in aux_data and 'is_target_after_prop' in self.loss_weight.keys() and 'is_target' in self.objective.keys():
                is_target_after_prop_loss_new = self.objective['is_target'](aux_data['is_target_after_prop'],
                                                                            test_label_cur, is_valid)
                is_target_after_prop_loss += is_target_after_prop_loss_new

            test_clf_acc_new, test_pred_correct = self.objective['clf_acc'](predicted_resp, test_label_cur, valid_sample=is_valid)
            test_clf_acc += test_clf_acc_new

            test_seq_all_correct = test_seq_all_correct * (test_pred_correct.long() | (1 - is_valid).long()).float()

            dimp_clf_acc_new, dimp_pred_correct = self.objective['clf_acc'](dimp_scores_cur, test_label_cur, valid_sample=is_valid)
            dimp_clf_acc += dimp_clf_acc_new

            dimp_seq_all_correct = dimp_seq_all_correct * (dimp_pred_correct.long() | (1 - is_valid).long()).float()

            if motion_pred is not None:
                motion_clf_acc_new = self.objective['clf_acc'](motion_pred, test_label_cur, valid_sample=is_valid)
                motion_clf_acc += motion_clf_acc_new

            if self.visdom is not None:
                self.visdom.register(test_image_cur, 'image_list', title='Test images')
                self.visdom.register(test_label_cur, 'heatmap_list', title='Label')
                # self.visdom.register(motion_pred, 'heatmap_list', title='Motion Pred')
                self.visdom.register(dimp_scores_cur, 'heatmap_list', title='DiMP Pred')
                self.visdom.register(predicted_resp, 'heatmap_list', title='Fusion Pred')

                while True:
                    if not self.pause_mode:
                        break
                    elif self.step:
                        self.step = False
                        break
                    else:
                        time.sleep(0.1)

            backbone_feat_prev = backbone_feat_cur.clone()
            dimp_scores_prev = dimp_scores_cur.clone()
            label_prev = test_label_cur.clone()
            is_valid_prev = is_valid.clone()

        clf_loss_test /= (sequence_length - 1)
        dimp_loss_test /= (sequence_length - 1)
        motion_pred_loss_test /= (sequence_length - 1)
        test_clf_acc /= (sequence_length - 1)
        dimp_clf_acc /= (sequence_length - 1)
        motion_clf_acc /= (sequence_length - 1)

        is_target_loss /= (sequence_length - 1)
        is_target_after_prop_loss /= (sequence_length - 1)

        test_seq_clf_acc = test_seq_all_correct.mean()
        dimp_seq_clf_acc = dimp_seq_all_correct.mean()

        clf_loss_test_w = self.loss_weight['test_clf'] * clf_loss_test
        dimp_loss_test_w = self.loss_weight.get('dimp_clf', 0.0) * dimp_loss_test

        motion_pred_loss_test_w = 0
        if 'motion_clf' in self.loss_weight.keys():
            motion_pred_loss_test_w = self.loss_weight['motion_clf'] * motion_pred_loss_test

        is_target_loss_w = self.loss_weight.get('is_target', 0.0) * is_target_loss
        is_target_after_prop_loss_w = self.loss_weight.get('is_target_after_prop', 0.0) * is_target_after_prop_loss

        loss = clf_loss_test_w + dimp_loss_test_w + motion_pred_loss_test_w + \
               is_target_loss_w + is_target_after_prop_loss_w

        stats = {'Loss/total': loss.item(),
                 'Loss/test_clf': clf_loss_test_w.item(),
                 'Loss/dimp_clf': dimp_loss_test_w.item(),
                 'Loss/raw/test_clf': clf_loss_test.item(),
                 'Loss/raw/dimp_clf': dimp_loss_test.item(),
                 'Loss/raw/test_clf_acc': test_clf_acc.item(),
                 'Loss/raw/dimp_clf_acc': dimp_clf_acc.item(),
                 'Loss/raw/is_target': is_target_loss.item(),
                 'Loss/raw/is_target_after_prop': is_target_after_prop_loss.item(),
                 'Loss/raw/test_seq_acc': test_seq_clf_acc.item(),
                 'Loss/raw/dimp_seq_acc': dimp_seq_clf_acc.item(),
                 }

        if 'motion_clf' in self.loss_weight.keys():
            stats['Loss/motion_clf'] = motion_pred_loss_test_w.item()
            stats['Loss/raw/motion_clf'] = motion_pred_loss_test.item()
            stats['Loss/raw/motion_clf_acc'] = motion_clf_acc.item()

        return loss, stats


class MotionSequenceTrackerActorv2(BaseActor):
    def visdom_ui_handler(self, data):
        if data['event_type'] == 'KeyPress':
            if data['key'] == 'p':
                self.pause_mode = not self.pause_mode

            elif data['key'] == 'ArrowRight' and self.pause_mode:
                self.step = True

    def __init__(self, net, objective, loss_weight=None, visdom_info=None, dimp_jitter_params=None):
        super().__init__(net, objective)
        self.loss_weight = loss_weight

        if dimp_jitter_params is not None:
            self.dimp_jitter_fn = DiMPScoreJittering(dimp_jitter_params['p_zero'],
                                                     dimp_jitter_params['distractor_ratio'],
                                                     dimp_jitter_params['p_distractor'],
                                                     dimp_jitter_params['max_distractor_enhance_factor'],
                                                     dimp_jitter_params['min_distractor_enhance_factor'])
        else:
            self.dimp_jitter_fn = None

        self.pause_mode = False
        self.step = False

        # TODO set it somewhere
        self.device = torch.device("cuda:0")

        if visdom_info is not None:
            self.visdom = Visdom(ui_info={'handler': self.visdom_ui_handler, 'win_id': 'Test images'},
                                 visdom_info=visdom_info)
        else:
            self.visdom = None

    def __call__(self, data):
        sequence_length = data['test_images'].shape[0]
        num_sequences = data['test_images'].shape[1]

        valid_samples = data['test_valid_image'].to(self.device)
        test_visibility = data['test_visible_ratio'].to(self.device)

        # Init tracker
        train_images = data['train_images'].to(self.device)
        train_anno = data['train_anno'].to(self.device)
        train_label = data['train_label'].to(self.device)
        dimp_filters = self.net.train_classifier(train_images, train_anno)

        # add an extra dim to filters so that seq is along first dim
        # Track in the first frame
        test_image_cur = data['test_images'][0:1, ...].to(self.device)
        backbone_feat_prev = self.net.extract_backbone_features(test_image_cur)
        backbone_feat_prev = backbone_feat_prev[self.net.classification_layer]
        backbone_feat_prev = backbone_feat_prev.view(1, num_sequences, -1,
                                                     backbone_feat_prev.shape[-2], backbone_feat_prev.shape[-1])

        dimp_scores_prev = self.net.dimp_classifier.track_frame(dimp_filters, backbone_feat_prev)
        dimp_filters = dimp_filters.unsqueeze(0)

        # Remove last row and col
        dimp_scores_prev = dimp_scores_prev[:, :, :-1, :-1].contiguous()

        label_prev = data['test_label'][0:1, ...].to(self.device)
        label_prev = label_prev[:, :, :-1, :-1].contiguous()

        anno_prev = data['test_anno'][0:1, ...].to(self.device)

        state_prev = None

        is_valid_prev = valid_samples[0, :].view(1, -1, 1, 1).byte()

        zero_tensor = torch.zeros(1).to(self.device)

        clf_loss_test = 0
        dimp_loss_test = 0
        motion_pred_loss_test = 0
        test_clf_acc = 0
        dimp_clf_acc = 0
        motion_clf_acc = 0
        dimp_weight_mean = 0

        is_target_loss = zero_tensor.clone()
        is_target_after_prop_loss = zero_tensor.clone()
        target_mask_loss = zero_tensor.clone()
        bg_mask_bg_loss = zero_tensor.clone()
        bg_mask_target_loss = zero_tensor.clone()

        dimp_conf_pred_loss = 0
        dimp_conf_pred_acc = 0
        motion_conf_pred_loss = 0
        motion_conf_pred_acc = 0

        for i in range(1, sequence_length):
            test_image_cur = data['test_images'][i:i+1, ...].to(self.device)
            test_label_cur = data['test_label'][i:i+1, ...].to(self.device)
            test_label_cur = test_label_cur[:, :, :-1, :-1].contiguous()

            test_anno_cur = data['test_anno'][i:i + 1, ...].to(self.device)

            predicted_resp, dimp_scores_cur, backbone_feat_cur, state_prev, aux_is_target, aux_is_target_after_prop = self.net(
                test_image_cur, dimp_filters,
                test_label_cur,
                backbone_feat_prev,
                label_prev,
                anno_prev, dimp_scores_prev,
                state_prev,
                self.dimp_jitter_fn)

            motion_pred = None

            is_valid = valid_samples[i, :].view(1, -1, 1, 1).byte()
            uncertain_frame = (test_visibility[i, :].view(1, -1, 1, 1) < 0.75) * (test_visibility[i, :].view(1, -1, 1, 1) > 0.25)

            is_valid = is_valid * (1 - uncertain_frame)

            clf_loss_test_new = self.objective['test_clf'](predicted_resp, test_label_cur,
                                                           test_anno_cur, valid_samples=is_valid)
            clf_loss_test += clf_loss_test_new

            dimp_loss_test_new = self.objective['dimp_clf'](dimp_scores_cur, test_label_cur,
                                                            test_anno_cur, valid_samples=is_valid)
            dimp_loss_test += dimp_loss_test_new

            motion_pred_loss_test_new = 0
            if 'motion_clf' in self.loss_weight.keys():
                motion_pred_loss_test_new = self.objective['motion_clf'](motion_pred, test_label_cur,
                                                                         test_anno_cur, valid_samples=is_valid)
            motion_pred_loss_test += motion_pred_loss_test_new

            if aux_is_target is not None and 'is_target' in self.loss_weight.keys():
                is_target_loss_new = self.objective['is_target'](aux_is_target, label_prev, is_valid_prev)
                is_target_loss += is_target_loss_new

            if aux_is_target_after_prop is not None and 'is_target_after_prop' in self.loss_weight.keys():
                is_target_after_prop_loss_new = self.objective['is_target'](aux_is_target_after_prop,
                                                                            test_label_cur, is_valid)
                is_target_after_prop_loss += is_target_after_prop_loss_new

            # Get prediction accuracies
            test_clf_acc_new = self.objective['clf_acc'](predicted_resp, test_label_cur, valid_sample=is_valid)
            test_clf_acc += test_clf_acc_new

            dimp_clf_acc_new = self.objective['clf_acc'](dimp_scores_cur, test_label_cur, valid_sample=is_valid)
            dimp_clf_acc += dimp_clf_acc_new

            if motion_pred is not None:
                motion_clf_acc_new = self.objective['clf_acc'](motion_pred, test_label_cur, valid_sample=is_valid)
                motion_clf_acc += motion_clf_acc_new

            if self.visdom is not None:
                self.visdom.register(test_image_cur, 'image_list', title='Test images')
                self.visdom.register(test_label_cur, 'heatmap_list', title='Label')
                # self.visdom.register(motion_pred, 'heatmap_list', title='Motion Pred')
                self.visdom.register(dimp_scores_cur, 'heatmap_list', title='DiMP Pred')
                self.visdom.register(predicted_resp, 'heatmap_list', title='Fusion Pred')

                while True:
                    if not self.pause_mode:
                        break
                    elif self.step:
                        self.step = False
                        break
                    else:
                        time.sleep(0.1)

            backbone_feat_prev = backbone_feat_cur.clone()
            dimp_scores_prev = dimp_scores_cur.clone()
            label_prev = test_label_cur.clone()
            is_valid_prev = is_valid.clone()

        clf_loss_test /= (sequence_length - 1)
        dimp_loss_test /= (sequence_length - 1)
        motion_pred_loss_test /= (sequence_length - 1)
        test_clf_acc /= (sequence_length - 1)
        dimp_clf_acc /= (sequence_length - 1)
        motion_clf_acc /= (sequence_length - 1)
        dimp_weight_mean /= (sequence_length - 1)

        dimp_conf_pred_acc /= (sequence_length - 1)
        motion_conf_pred_acc /= (sequence_length - 1)
        dimp_conf_pred_loss /= (sequence_length - 1)
        motion_conf_pred_loss /= (sequence_length - 1)

        is_target_loss /= (sequence_length - 1)
        is_target_after_prop_loss /= (sequence_length - 1)
        target_mask_loss /= (sequence_length - 1)
        bg_mask_bg_loss /= (sequence_length - 1)
        bg_mask_target_loss /= (sequence_length - 1)

        clf_loss_test_w = self.loss_weight['test_clf'] * clf_loss_test
        dimp_loss_test_w = self.loss_weight.get('dimp_clf', 0.0) * dimp_loss_test

        dimp_conf_pred_loss_w = self.loss_weight.get('dimp_conf_pred', 0.0) * dimp_conf_pred_loss
        motion_conf_pred_loss_w = self.loss_weight.get('motion_conf_pred', 0.0) * motion_conf_pred_loss

        motion_pred_loss_test_w = 0
        if 'motion_clf' in self.loss_weight.keys():
            motion_pred_loss_test_w = self.loss_weight['motion_clf'] * motion_pred_loss_test

        dimp_weight_mean_w = self.loss_weight.get('dimp_weight_reg', 0.0) * dimp_weight_mean

        is_target_loss_w = self.loss_weight['is_target'] * is_target_loss
        is_target_after_prop_loss_w = self.loss_weight['is_target_after_prop'] * is_target_after_prop_loss
        target_mask_loss_w = self.loss_weight['target_mask'] * target_mask_loss
        bg_mask_bg_loss_w = self.loss_weight['bg_mask_bg'] * bg_mask_bg_loss
        bg_mask_target_loss_w = self.loss_weight['bg_mask_target'] * bg_mask_target_loss

        loss = clf_loss_test_w + dimp_loss_test_w + motion_pred_loss_test_w - dimp_weight_mean_w + \
               dimp_conf_pred_loss_w + motion_conf_pred_loss_w + is_target_loss_w + is_target_after_prop_loss_w + \
               target_mask_loss_w + bg_mask_bg_loss_w + bg_mask_target_loss_w

        stats = {'Loss/total': loss.item(),
                 'Loss/test_clf': clf_loss_test_w.item(),
                 'Loss/dimp_clf': dimp_loss_test_w.item(),
                 'Loss/raw/test_clf': clf_loss_test.item(),
                 'Loss/raw/dimp_clf': dimp_loss_test.item(),
                 'Loss/raw/test_clf_acc': test_clf_acc.item(),
                 'Loss/raw/dimp_clf_acc': dimp_clf_acc.item(),
                 'Loss/raw/is_target': is_target_loss.item(),
                 'Loss/raw/is_target_after_prop': is_target_after_prop_loss.item(),
                 #'Loss/raw/target_mask': target_mask_loss.item(),
                 #'Loss/raw/bg_mask_bg': bg_mask_bg_loss.item(),
                 #'Loss/raw/bg_mask_target': bg_mask_target_loss.item(),
                 }

        if 'motion_clf' in self.loss_weight.keys():
            stats['Loss/motion_clf'] = motion_pred_loss_test_w.item()
            stats['Loss/raw/motion_clf'] = motion_pred_loss_test.item()
            stats['Loss/raw/motion_clf_acc'] = motion_clf_acc.item()

        return loss, stats


class MotionSequenceTrackerActorv3(BaseActor):
    def __init__(self, net, objective, loss_weight=None, dimp_jitter_params=None):
        super().__init__(net, objective)
        self.loss_weight = loss_weight
        self.dimp_jitter_params = dimp_jitter_params

        if dimp_jitter_params is None:
            self.dimp_jitter_fn = None
        else:
            self.dimp_jitter_fn = DiMPScoreJittering(dimp_jitter_params['p_zero'],
                                                     dimp_jitter_params['distractor_ratio'],
                                                     dimp_jitter_params['p_distractor'],
                                                     dimp_jitter_params['max_distractor_enhance_factor'],
                                                     dimp_jitter_params['min_distractor_enhance_factor'])


        # TODO set it somewhere
        self.device = torch.device("cuda:0")

    def __call__(self, data):
        sequence_length = data['test_images'].shape[0]
        num_sequences = data['test_images'].shape[1]

        valid_samples = data['test_valid_image'].to(self.device)
        test_visibility = data['test_visible_ratio'].to(self.device)

        # Init tracker
        train_images = data['train_images'].to(self.device)
        train_anno = data['train_anno'].to(self.device)
        dimp_filters = self.net.train_classifier(train_images, train_anno)

        # Track in the first frame
        test_image_cur = data['test_images'][0, ...].to(self.device)
        backbone_feat_prev_all = self.net.extract_backbone_features(test_image_cur)
        backbone_feat_prev = backbone_feat_prev_all[self.net.classification_layer]
        backbone_feat_prev = backbone_feat_prev.view(1, num_sequences, -1,
                                                     backbone_feat_prev.shape[-2], backbone_feat_prev.shape[-1])

        if self.net.motion_feat_extractor is not None:
            motion_feat_prev = self.net.motion_feat_extractor(backbone_feat_prev_all).view(1, num_sequences, -1,
                                                                                           backbone_feat_prev.shape[-2],
                                                                                           backbone_feat_prev.shape[-1])
        else:
            motion_feat_prev = backbone_feat_prev

        dimp_scores_prev = self.net.dimp_classifier.track_frame(dimp_filters, backbone_feat_prev)

        # Remove last row and col
        dimp_scores_prev = dimp_scores_prev[:, :, :-1, :-1].contiguous()

        label_prev = data['test_label'][0:1, ...].to(self.device)
        label_prev = label_prev[:, :, :-1, :-1].contiguous()

        anno_prev = data['test_anno'][0:1, ...].to(self.device)
        state_prev = None

        is_valid_prev = valid_samples[0, :].view(1, -1, 1, 1).byte()

        clf_loss_test_all = torch.zeros(num_sequences, sequence_length - 1).to(self.device)
        clf_loss_test_orig_all = torch.zeros(num_sequences, sequence_length - 1).to(self.device)
        dimp_loss_test_all = torch.zeros(num_sequences, sequence_length - 1).to(self.device)
        test_clf_acc = 0
        dimp_clf_acc = 0

        test_tracked_correct = torch.zeros(num_sequences, sequence_length - 1).long().to(self.device)
        test_seq_all_correct = torch.ones(num_sequences).to(self.device)
        dimp_seq_all_correct = torch.ones(num_sequences).to(self.device)

        is_target_loss_all = torch.zeros(num_sequences, sequence_length - 1).to(self.device)
        is_target_after_prop_loss_all = torch.zeros(num_sequences, sequence_length - 1).to(self.device)

        for i in range(1, sequence_length):
            test_image_cur = data['test_images'][i, ...].to(self.device)
            test_label_cur = data['test_label'][i:i+1, ...].to(self.device)
            test_label_cur = test_label_cur[:, :, :-1, :-1].contiguous()

            test_anno_cur = data['test_anno'][i:i + 1, ...].to(self.device)

            backbone_feat_cur_all = self.net.extract_backbone_features(test_image_cur)
            backbone_feat_cur = backbone_feat_cur_all[self.net.classification_layer]
            backbone_feat_cur = backbone_feat_cur.view(1, num_sequences, -1,
                                                       backbone_feat_cur.shape[-2], backbone_feat_cur.shape[-1])

            if self.net.motion_feat_extractor is not None:
                motion_feat_cur = self.net.motion_feat_extractor(backbone_feat_cur_all).view(1, num_sequences, -1,
                                                                                             backbone_feat_cur.shape[-2],
                                                                                             backbone_feat_cur.shape[-1])
            else:
                motion_feat_cur = backbone_feat_cur

            dimp_scores_cur = self.net.dimp_classifier.track_frame(dimp_filters, backbone_feat_cur)
            dimp_scores_cur = dimp_scores_cur[:, :, :-1, :-1].contiguous()

            jitter_info = None
            if self.dimp_jitter_fn is not None:
                dimp_scores_cur = self.dimp_jitter_fn(dimp_scores_cur, test_label_cur.clone())

            predictor_input_data = {'input1': motion_feat_prev, 'input2': motion_feat_cur,
                                    'label_prev': label_prev, 'anno_prev': anno_prev,
                                    'dimp_score_prev': dimp_scores_prev, 'dimp_score_cur': dimp_scores_cur,
                                    'state_prev': state_prev,
                                    'jitter_info': jitter_info}

            predictor_output = self.net.predictor(predictor_input_data)

            predicted_resp = predictor_output['response']
            state_prev = predictor_output['state_cur']
            aux_data = predictor_output['auxiliary_outputs']

            is_valid = valid_samples[i, :].view(1, -1, 1, 1).byte()
            uncertain_frame = (test_visibility[i, :].view(1, -1, 1, 1) < 0.75) * (test_visibility[i, :].view(1, -1, 1, 1) > 0.25)

            is_valid = is_valid * ~uncertain_frame

            clf_loss_test_new = self.objective['test_clf'](predicted_resp, test_label_cur,
                                                           test_anno_cur, valid_samples=is_valid)
            clf_loss_test_all[:, i - 1] = clf_loss_test_new.squeeze()

            dimp_loss_test_new = self.objective['dimp_clf'](dimp_scores_cur, test_label_cur,
                                                            test_anno_cur, valid_samples=is_valid)
            dimp_loss_test_all[:, i - 1] = dimp_loss_test_new.squeeze()

            if 'fused_score_orig' in aux_data and 'test_clf_orig' in self.loss_weight.keys():
                aux_data['fused_score_orig'] = aux_data['fused_score_orig'].view(test_label_cur.shape)
                clf_loss_test_orig_new = self.objective['test_clf'](aux_data['fused_score_orig'], test_label_cur, test_anno_cur,  valid_samples=is_valid)
                clf_loss_test_orig_all[:, i - 1] = clf_loss_test_orig_new.squeeze()

            if 'is_target' in aux_data and 'is_target' in self.loss_weight.keys() and 'is_target' in self.objective.keys():
                is_target_loss_new = self.objective['is_target'](aux_data['is_target'], label_prev, is_valid_prev)
                is_target_loss_all[:, i - 1] = is_target_loss_new

            if 'is_target_after_prop' in aux_data and 'is_target_after_prop' in self.loss_weight.keys() and 'is_target' in self.objective.keys():
                is_target_after_prop_loss_new = self.objective['is_target'](aux_data['is_target_after_prop'],
                                                                            test_label_cur, is_valid)
                is_target_after_prop_loss_all[:, i - 1] = is_target_after_prop_loss_new

            test_clf_acc_new, test_pred_correct = self.objective['clf_acc'](predicted_resp, test_label_cur, valid_samples=is_valid)
            test_clf_acc += test_clf_acc_new

            test_seq_all_correct = test_seq_all_correct * (test_pred_correct.long() | (1 - is_valid).long()).float()
            test_tracked_correct[:, i - 1] = test_pred_correct

            dimp_clf_acc_new, dimp_pred_correct = self.objective['clf_acc'](dimp_scores_cur, test_label_cur, valid_samples=is_valid)
            dimp_clf_acc += dimp_clf_acc_new

            dimp_seq_all_correct = dimp_seq_all_correct * (dimp_pred_correct.long() | (1 - is_valid).long()).float()

            motion_feat_prev = motion_feat_cur.clone()
            dimp_scores_prev = dimp_scores_cur.clone()
            label_prev = test_label_cur.clone()
            is_valid_prev = is_valid.clone()

        clf_loss_test = clf_loss_test_all.mean()
        clf_loss_test_orig = clf_loss_test_orig_all.mean()
        dimp_loss_test = dimp_loss_test_all.mean()
        is_target_loss = is_target_loss_all.mean()
        is_target_after_prop_loss = is_target_after_prop_loss_all.mean()

        test_clf_acc /= (sequence_length - 1)
        dimp_clf_acc /= (sequence_length - 1)
        clf_loss_test_orig /= (sequence_length - 1)

        test_seq_clf_acc = test_seq_all_correct.mean()
        dimp_seq_clf_acc = dimp_seq_all_correct.mean()

        clf_loss_test_w = self.loss_weight['test_clf'] * clf_loss_test
        clf_loss_test_orig_w = self.loss_weight['test_clf_orig'] * clf_loss_test_orig
        dimp_loss_test_w = self.loss_weight.get('dimp_clf', 0.0) * dimp_loss_test

        is_target_loss_w = self.loss_weight.get('is_target', 0.0) * is_target_loss
        is_target_after_prop_loss_w = self.loss_weight.get('is_target_after_prop', 0.0) * is_target_after_prop_loss

        loss = clf_loss_test_w + dimp_loss_test_w + is_target_loss_w + is_target_after_prop_loss_w + clf_loss_test_orig_w

        stats = {'Loss/total': loss.item(),
                 'Loss/test_clf': clf_loss_test_w.item(),
                 'Loss/dimp_clf': dimp_loss_test_w.item(),
                 'Loss/raw/test_clf': clf_loss_test.item(),
                 'Loss/raw/test_clf_orig': clf_loss_test_orig.item(),
                 'Loss/raw/dimp_clf': dimp_loss_test.item(),
                 'Loss/raw/test_clf_acc': test_clf_acc.item(),
                 'Loss/raw/dimp_clf_acc': dimp_clf_acc.item(),
                 'Loss/raw/is_target': is_target_loss.item(),
                 'Loss/raw/is_target_after_prop': is_target_after_prop_loss.item(),
                 'Loss/raw/test_seq_acc': test_seq_clf_acc.item(),
                 'Loss/raw/dimp_seq_acc': dimp_seq_clf_acc.item(),
                 }

        return loss, stats


class KYSActor(BaseActor):
    def __init__(self, net, objective, loss_weight=None, dimp_jitter_params=None):
        super().__init__(net, objective)
        self.loss_weight = loss_weight
        self.dimp_jitter_params = dimp_jitter_params

        if dimp_jitter_params is None:
            self.dimp_jitter_fn = None
        else:
            self.dimp_jitter_fn = DiMPScoreJittering(dimp_jitter_params['p_zero'],
                                                     dimp_jitter_params['distractor_ratio'],
                                                     dimp_jitter_params['p_distractor'],
                                                     dimp_jitter_params['max_distractor_enhance_factor'],
                                                     dimp_jitter_params['min_distractor_enhance_factor'])


        # TODO set it somewhere
        self.device = torch.device("cuda:0")

    def __call__(self, data):
        sequence_length = data['test_images'].shape[0]
        num_sequences = data['test_images'].shape[1]

        valid_samples = data['test_valid_image'].to(self.device)
        test_visibility = data['test_visible_ratio'].to(self.device)

        # Initialize loss variables
        clf_loss_test_all = torch.zeros(num_sequences, sequence_length - 1).to(self.device)
        clf_loss_test_orig_all = torch.zeros(num_sequences, sequence_length - 1).to(self.device)
        dimp_loss_test_all = torch.zeros(num_sequences, sequence_length - 1).to(self.device)
        test_clf_acc = 0
        dimp_clf_acc = 0

        test_tracked_correct = torch.zeros(num_sequences, sequence_length - 1).long().to(self.device)
        test_seq_all_correct = torch.ones(num_sequences).to(self.device)
        dimp_seq_all_correct = torch.ones(num_sequences).to(self.device)

        is_target_loss_all = torch.zeros(num_sequences, sequence_length - 1).to(self.device)
        is_target_after_prop_loss_all = torch.zeros(num_sequences, sequence_length - 1).to(self.device)

        # Initialize target model using the training frames
        train_images = data['train_images'].to(self.device)
        train_anno = data['train_anno'].to(self.device)
        dimp_filters = self.net.train_classifier(train_images, train_anno)

        # Track in the first test frame
        test_image_cur = data['test_images'][0, ...].to(self.device)
        backbone_feat_prev_all = self.net.extract_backbone_features(test_image_cur)
        backbone_feat_prev = backbone_feat_prev_all[self.net.classification_layer]
        backbone_feat_prev = backbone_feat_prev.view(1, num_sequences, -1,
                                                     backbone_feat_prev.shape[-2], backbone_feat_prev.shape[-1])

        if self.net.motion_feat_extractor is not None:
            motion_feat_prev = self.net.motion_feat_extractor(backbone_feat_prev_all).view(1, num_sequences, -1,
                                                                                           backbone_feat_prev.shape[-2],
                                                                                           backbone_feat_prev.shape[-1])
        else:
            motion_feat_prev = backbone_feat_prev

        dimp_scores_prev = self.net.dimp_classifier.track_frame(dimp_filters, backbone_feat_prev)

        # Remove last row and col (added due to even kernel size in the target model)
        dimp_scores_prev = dimp_scores_prev[:, :, :-1, :-1].contiguous()

        # Set previous frame information
        label_prev = data['test_label'][0:1, ...].to(self.device)
        label_prev = label_prev[:, :, :-1, :-1].contiguous()

        anno_prev = data['test_anno'][0:1, ...].to(self.device)
        state_prev = None

        is_valid_prev = valid_samples[0, :].view(1, -1, 1, 1).byte()

        # Loop over the sequence
        for i in range(1, sequence_length):
            test_image_cur = data['test_images'][i, ...].to(self.device)
            test_label_cur = data['test_label'][i:i+1, ...].to(self.device)
            test_label_cur = test_label_cur[:, :, :-1, :-1].contiguous()

            test_anno_cur = data['test_anno'][i:i + 1, ...].to(self.device)

            # Extract features
            backbone_feat_cur_all = self.net.extract_backbone_features(test_image_cur)
            backbone_feat_cur = backbone_feat_cur_all[self.net.classification_layer]
            backbone_feat_cur = backbone_feat_cur.view(1, num_sequences, -1,
                                                       backbone_feat_cur.shape[-2], backbone_feat_cur.shape[-1])

            if self.net.motion_feat_extractor is not None:
                motion_feat_cur = self.net.motion_feat_extractor(backbone_feat_cur_all).view(1, num_sequences, -1,
                                                                                             backbone_feat_cur.shape[-2],
                                                                                             backbone_feat_cur.shape[-1])
            else:
                motion_feat_cur = backbone_feat_cur

            # Run target model
            dimp_scores_cur = self.net.dimp_classifier.track_frame(dimp_filters, backbone_feat_cur)
            dimp_scores_cur = dimp_scores_cur[:, :, :-1, :-1].contiguous()

            # Jitter target model output for augmentation
            jitter_info = None
            if self.dimp_jitter_fn is not None:
                dimp_scores_cur = self.dimp_jitter_fn(dimp_scores_cur, test_label_cur.clone())

            # Input target model output along with previous frame information to the predictor
            predictor_input_data = {'input1': motion_feat_prev, 'input2': motion_feat_cur,
                                    'label_prev': label_prev, 'anno_prev': anno_prev,
                                    'dimp_score_prev': dimp_scores_prev, 'dimp_score_cur': dimp_scores_cur,
                                    'state_prev': state_prev,
                                    'jitter_info': jitter_info}

            predictor_output = self.net.predictor(predictor_input_data)

            predicted_resp = predictor_output['response']
            state_prev = predictor_output['state_cur']
            aux_data = predictor_output['auxiliary_outputs']

            is_valid = valid_samples[i, :].view(1, -1, 1, 1).byte()
            uncertain_frame = (test_visibility[i, :].view(1, -1, 1, 1) < 0.75) * (test_visibility[i, :].view(1, -1, 1, 1) > 0.25)

            is_valid = is_valid * ~uncertain_frame

            # Calculate losses
            clf_loss_test_new = self.objective['test_clf'](predicted_resp, test_label_cur,
                                                           test_anno_cur, valid_samples=is_valid)
            clf_loss_test_all[:, i - 1] = clf_loss_test_new.squeeze()

            dimp_loss_test_new = self.objective['dimp_clf'](dimp_scores_cur, test_label_cur,
                                                            test_anno_cur, valid_samples=is_valid)
            dimp_loss_test_all[:, i - 1] = dimp_loss_test_new.squeeze()

            if 'fused_score_orig' in aux_data and 'test_clf_orig' in self.loss_weight.keys():
                aux_data['fused_score_orig'] = aux_data['fused_score_orig'].view(test_label_cur.shape)
                clf_loss_test_orig_new = self.objective['test_clf'](aux_data['fused_score_orig'], test_label_cur, test_anno_cur,  valid_samples=is_valid)
                clf_loss_test_orig_all[:, i - 1] = clf_loss_test_orig_new.squeeze()

            if 'is_target' in aux_data and 'is_target' in self.loss_weight.keys() and 'is_target' in self.objective.keys():
                is_target_loss_new = self.objective['is_target'](aux_data['is_target'], label_prev, is_valid_prev)
                is_target_loss_all[:, i - 1] = is_target_loss_new

            if 'is_target_after_prop' in aux_data and 'is_target_after_prop' in self.loss_weight.keys() and 'is_target' in self.objective.keys():
                is_target_after_prop_loss_new = self.objective['is_target'](aux_data['is_target_after_prop'],
                                                                            test_label_cur, is_valid)
                is_target_after_prop_loss_all[:, i - 1] = is_target_after_prop_loss_new

            test_clf_acc_new, test_pred_correct = self.objective['clf_acc'](predicted_resp, test_label_cur, valid_samples=is_valid)
            test_clf_acc += test_clf_acc_new

            test_seq_all_correct = test_seq_all_correct * (test_pred_correct.long() | (1 - is_valid).long()).float()
            test_tracked_correct[:, i - 1] = test_pred_correct

            dimp_clf_acc_new, dimp_pred_correct = self.objective['clf_acc'](dimp_scores_cur, test_label_cur, valid_samples=is_valid)
            dimp_clf_acc += dimp_clf_acc_new

            dimp_seq_all_correct = dimp_seq_all_correct * (dimp_pred_correct.long() | (1 - is_valid).long()).float()

            motion_feat_prev = motion_feat_cur.clone()
            dimp_scores_prev = dimp_scores_cur.clone()
            label_prev = test_label_cur.clone()
            is_valid_prev = is_valid.clone()

        # Compute average loss over the sequence
        clf_loss_test = clf_loss_test_all.mean()
        clf_loss_test_orig = clf_loss_test_orig_all.mean()
        dimp_loss_test = dimp_loss_test_all.mean()
        is_target_loss = is_target_loss_all.mean()
        is_target_after_prop_loss = is_target_after_prop_loss_all.mean()

        test_clf_acc /= (sequence_length - 1)
        dimp_clf_acc /= (sequence_length - 1)
        clf_loss_test_orig /= (sequence_length - 1)

        test_seq_clf_acc = test_seq_all_correct.mean()
        dimp_seq_clf_acc = dimp_seq_all_correct.mean()

        clf_loss_test_w = self.loss_weight['test_clf'] * clf_loss_test
        clf_loss_test_orig_w = self.loss_weight['test_clf_orig'] * clf_loss_test_orig
        dimp_loss_test_w = self.loss_weight.get('dimp_clf', 0.0) * dimp_loss_test

        is_target_loss_w = self.loss_weight.get('is_target', 0.0) * is_target_loss
        is_target_after_prop_loss_w = self.loss_weight.get('is_target_after_prop', 0.0) * is_target_after_prop_loss

        loss = clf_loss_test_w + dimp_loss_test_w + is_target_loss_w + is_target_after_prop_loss_w + clf_loss_test_orig_w

        stats = {'Loss/total': loss.item(),
                 'Loss/test_clf': clf_loss_test_w.item(),
                 'Loss/dimp_clf': dimp_loss_test_w.item(),
                 'Loss/raw/test_clf': clf_loss_test.item(),
                 'Loss/raw/test_clf_orig': clf_loss_test_orig.item(),
                 'Loss/raw/dimp_clf': dimp_loss_test.item(),
                 'Loss/raw/test_clf_acc': test_clf_acc.item(),
                 'Loss/raw/dimp_clf_acc': dimp_clf_acc.item(),
                 'Loss/raw/is_target': is_target_loss.item(),
                 'Loss/raw/is_target_after_prop': is_target_after_prop_loss.item(),
                 'Loss/raw/test_seq_acc': test_seq_clf_acc.item(),
                 'Loss/raw/dimp_seq_acc': dimp_seq_clf_acc.item(),
                 }

        return loss, stats


class PrevStateData:
    def __init__(self, prev_frame_gap):
        self.num_prev_frames = len(prev_frame_gap)
        self.prev_frame_gap = prev_frame_gap

        self.max_list_len = prev_frame_gap[-1]
        self.feat_list = []
        self.state_list = []

    def insert_data(self, frame_number, feat, state):
        self.feat_list.append(feat)
        self.state_list.append(state)

        if len(self.feat_list) > self.max_list_len:
            self.feat_list.pop(0)
            self.state_list.pop(0)

    def get_data(self):
        feat_out = []
        state_out = []
        idx = 0
        f_num = 0

        for i in range(len(self.feat_list)):
            if f_num >= self.prev_frame_gap[idx]:
                feat_out.append(self.feat_list[-(i + 1)])
                state_out.append(self.state_list[-(i + 1)])
                idx += 1

                if idx == len(self.prev_frame_gap):
                    return feat_out, state_out
            f_num += 1

        for i in range(idx, len(self.prev_frame_gap)):
            feat_out.append(self.feat_list[0])
            state_out.append(self.state_list[0])

        return feat_out, state_out


class MotionSequenceTrackerActorv4(BaseActor):
    def visdom_ui_handler(self, data):
        if data['event_type'] == 'KeyPress':
            if data['key'] == 'p':
                self.pause_mode = not self.pause_mode

            elif data['key'] == 'ArrowRight' and self.pause_mode:
                self.step = True

    def __init__(self, net, objective, num_prev_frames, max_prev_frame_gap,
                 loss_weight=None, visdom_info=None, dimp_jitter_params=None,
                 dropout_params=None, loss_compute_params=None):
        super().__init__(net, objective)

        self.num_prev_frames = num_prev_frames
        self.max_prev_frame_gap = max_prev_frame_gap
        self.loss_weight = loss_weight

        self.dimp_jitter_params = dimp_jitter_params

        if loss_compute_params is None:
            self.loss_compute_params = {'mode': 'normal'}
        else:
            self.loss_compute_params = loss_compute_params

        if dropout_params is None:
            self.dropout_params = {}
        else:
            self.dropout_params = dropout_params

        if dimp_jitter_params is None:
            self.dimp_jitter_fn = None
        else:
            dimp_jitter_type = dimp_jitter_params.get('mode', 'default')
            self.dimp_jitter_type = dimp_jitter_type

            if dimp_jitter_type == 'default':
                self.dimp_jitter_fn = DiMPScoreJittering(dimp_jitter_params['p_zero'],
                                                         dimp_jitter_params['distractor_ratio'],
                                                         dimp_jitter_params['p_distractor'],
                                                         dimp_jitter_params['max_distractor_enhance_factor'],
                                                         dimp_jitter_params['min_distractor_enhance_factor'])
            elif dimp_jitter_type == 'sequence':
                self.dimp_jitter_fn = DiMPScoreJitteringState(dimp_jitter_params['num_sequence'],
                                                              dimp_jitter_params['p_distractor_lo'],
                                                              dimp_jitter_params['p_distractor_hi'],
                                                              dimp_jitter_params['p_dimp_fail'])

        self.pause_mode = False
        self.step = False

        # TODO set it somewhere
        self.device = torch.device("cuda:0")

        if visdom_info is not None:
            self.visdom = Visdom(ui_info={'handler': self.visdom_ui_handler, 'win_id': 'Test images'},
                                 visdom_info=visdom_info)
        else:
            self.visdom = None

    def __call__(self, data):
        sequence_length = data['test_images'].shape[0]
        num_sequences = data['test_images'].shape[1]

        valid_samples = data['test_valid_image'].to(self.device)
        test_visibility = data['test_visible_ratio'].to(self.device)

        # Init tracker
        train_images = data['train_images'].to(self.device)
        train_anno = data['train_anno'].to(self.device)
        train_label = data['train_label'].to(self.device)
        dimp_filters = self.net.train_classifier(train_images, train_anno)

        # Track in the first frame
        test_image_cur = data['test_images'][0, ...].to(self.device)
        backbone_feat_prev = self.net.extract_backbone_features(test_image_cur)
        backbone_feat_prev = backbone_feat_prev[self.net.classification_layer]
        backbone_feat_prev = backbone_feat_prev.view(1, num_sequences, -1,
                                                     backbone_feat_prev.shape[-2], backbone_feat_prev.shape[-1])

        dimp_scores_prev = self.net.dimp_classifier.track_frame(dimp_filters, backbone_feat_prev)

        # Remove last row and col
        dimp_scores_prev = dimp_scores_prev[:, :, :-1, :-1].contiguous()

        label_prev = data['test_label'][0:1, ...].to(self.device)
        label_prev = label_prev[:, :, :-1, :-1].contiguous()

        anno_prev = data['test_anno'][0:1, ...].to(self.device)

        state_prev = None

        prev_frame_gaps = [0]

        for pf in range(1, self.num_prev_frames):
            prev_frame_gaps.append(prev_frame_gaps[-1] + self.max_prev_frame_gap)

        prev_state_handler = PrevStateData(prev_frame_gaps)
        prev_state_handler.insert_data(0, backbone_feat_prev, state_prev)

        is_valid_prev = valid_samples[0, :].view(1, -1, 1, 1).byte()

        zero_tensor = torch.zeros(1).to(self.device)

        clf_loss_test_all = torch.zeros(num_sequences, sequence_length - 1).to(self.device)
        dimp_loss_test_all = torch.zeros(num_sequences, sequence_length - 1).to(self.device)
        test_clf_acc = 0
        dimp_clf_acc = 0

        test_tracked_correct = torch.zeros(num_sequences, sequence_length - 1).long().to(self.device)
        test_seq_all_correct = torch.ones(num_sequences).to(self.device)
        dimp_seq_all_correct = torch.ones(num_sequences).to(self.device)

        is_target_loss_all = torch.zeros(num_sequences, sequence_length - 1).to(self.device)
        is_target_after_prop_loss_all = torch.zeros(num_sequences, sequence_length - 1).to(self.device)

        for i in range(1, sequence_length):
            test_image_cur = data['test_images'][i, ...].to(self.device)
            test_label_cur = data['test_label'][i:i+1, ...].to(self.device)
            test_label_cur = test_label_cur[:, :, :-1, :-1].contiguous()

            test_anno_cur = data['test_anno'][i:i + 1, ...].to(self.device)

            backbone_feat_cur = self.net.extract_backbone_features(test_image_cur)
            backbone_feat_cur = backbone_feat_cur[self.net.classification_layer]
            backbone_feat_cur = backbone_feat_cur.view(1, num_sequences, -1,
                                                       backbone_feat_cur.shape[-2], backbone_feat_cur.shape[-1])

            if self.dropout_params.get('use_dropout', False) and random.random() < self.dropout_params.get('p_dropout', -1):
                backbone_feat_cur = F.dropout(backbone_feat_cur, p=0.2, inplace=True)

            dimp_scores_cur = self.net.dimp_classifier.track_frame(dimp_filters, backbone_feat_cur)
            dimp_scores_cur = dimp_scores_cur[:, :, :-1, :-1].contiguous()

            jitter_info = None
            if self.dimp_jitter_fn is not None:
                if self.dimp_jitter_type == "default":
                    dimp_scores_cur = self.dimp_jitter_fn(dimp_scores_cur, test_label_cur.clone())
                elif self.dimp_jitter_type == "sequence":
                    if i > 5:
                        dimp_scores_cur, jitter_info = self.dimp_jitter_fn(dimp_scores_cur, test_label_cur.clone())

                        if not self.dimp_jitter_params.get('jitter_cv', False):
                            jitter_info = None

            backbone_feat_prev, state_prev = prev_state_handler.get_data()
            predictor_input_data = {'input1': backbone_feat_prev, 'input2': backbone_feat_cur,
                                    'label_prev': label_prev, 'anno_prev': anno_prev,
                                    'dimp_score_prev': dimp_scores_prev, 'dimp_score_cur': dimp_scores_cur,
                                    'state_prev': state_prev,
                                    'jitter_info': jitter_info}

            predictor_output = self.net.predictor(predictor_input_data)

            predicted_resp = predictor_output['response']
            state_prev = predictor_output['state_cur']
            aux_data = predictor_output['auxiliary_outputs']

            is_valid = valid_samples[i, :].view(1, -1, 1, 1).byte()
            uncertain_frame = (test_visibility[i, :].view(1, -1, 1, 1) < 0.75) * (test_visibility[i, :].view(1, -1, 1, 1) > 0.25)

            is_valid = is_valid * (1 - uncertain_frame)

            clf_loss_test_new = self.objective['test_clf'](predicted_resp, test_label_cur,
                                                           test_anno_cur, valid_samples=is_valid)
            clf_loss_test_all[:, i - 1] = clf_loss_test_new.squeeze()

            dimp_loss_test_new = self.objective['dimp_clf'](dimp_scores_cur, test_label_cur,
                                                            test_anno_cur, valid_samples=is_valid)
            dimp_loss_test_all[:, i - 1] = dimp_loss_test_new.squeeze()

            if 'is_target' in aux_data and 'is_target' in self.loss_weight.keys() and 'is_target' in self.objective.keys():
                is_target_loss_new = self.objective['is_target'](aux_data['is_target'], label_prev, is_valid_prev)
                is_target_loss_all[:, i - 1] = is_target_loss_new

            if 'is_target_after_prop' in aux_data and 'is_target_after_prop' in self.loss_weight.keys() and 'is_target' in self.objective.keys():
                is_target_after_prop_loss_new = [self.objective['is_target'](i_t, test_label_cur, is_valid)
                                                 for i_t in aux_data['is_target_after_prop']]

                is_target_after_prop_loss_new = sum(is_target_after_prop_loss_new) / len(is_target_after_prop_loss_new)
                is_target_after_prop_loss_all[:, i - 1] = is_target_after_prop_loss_new

            test_clf_acc_new, test_pred_correct = self.objective['clf_acc'](predicted_resp, test_label_cur, valid_sample=is_valid)
            test_clf_acc += test_clf_acc_new

            test_seq_all_correct = test_seq_all_correct * (test_pred_correct.long() | (1 - is_valid).long()).float()
            test_tracked_correct[:, i - 1] = test_pred_correct

            dimp_clf_acc_new, dimp_pred_correct = self.objective['clf_acc'](dimp_scores_cur, test_label_cur, valid_sample=is_valid)
            dimp_clf_acc += dimp_clf_acc_new

            dimp_seq_all_correct = dimp_seq_all_correct * (dimp_pred_correct.long() | (1 - is_valid).long()).float()

            prev_state_handler.insert_data(i, backbone_feat_cur.clone(), state_prev)
            dimp_scores_prev = dimp_scores_cur.clone()
            label_prev = test_label_cur.clone()
            is_valid_prev = is_valid.clone()

        if self.loss_compute_params['mode'] == 'normal':
            clf_loss_test = clf_loss_test_all.mean()
            dimp_loss_test = dimp_loss_test_all.mean()
            is_target_loss = is_target_loss_all.mean()
            is_target_after_prop_loss = is_target_after_prop_loss_all.mean()
        elif self.loss_compute_params['mode'] == 'early_stop':
            min_val, min_idx = test_tracked_correct.min(dim=1)

            clf_loss_test = clf_loss_test_all.mean() * 0.0
            dimp_loss_test = dimp_loss_test_all.mean() * 0.0
            is_target_loss = is_target_loss_all.mean() * 0.0
            is_target_after_prop_loss = is_target_after_prop_loss_all.mean() * 0.0

            num_failed_sequences = 0
            for i in range(min_val.numel()):
                # Ignore sequences with no failure
                if min_val[i] == 0:
                    num_failed_sequences += 1
                    clf_loss_test += clf_loss_test_all[i, :min_idx[i]+1].mean()
                    dimp_loss_test += dimp_loss_test_all[i, :min_idx[i] + 1].mean()
                    is_target_loss += is_target_loss_all[i, :min_idx[i] + 1].mean()
                    is_target_after_prop_loss += is_target_after_prop_loss_all[i, :min_idx[i] + 1].mean()

            clf_loss_test /= num_sequences
            dimp_loss_test /= num_sequences
            is_target_loss /= num_sequences
            is_target_after_prop_loss /= num_sequences

        test_clf_acc /= (sequence_length - 1)
        dimp_clf_acc /= (sequence_length - 1)

        test_seq_clf_acc = test_seq_all_correct.mean()
        dimp_seq_clf_acc = dimp_seq_all_correct.mean()

        clf_loss_test_w = self.loss_weight['test_clf'] * clf_loss_test
        dimp_loss_test_w = self.loss_weight.get('dimp_clf', 0.0) * dimp_loss_test

        is_target_loss_w = self.loss_weight.get('is_target', 0.0) * is_target_loss
        is_target_after_prop_loss_w = self.loss_weight.get('is_target_after_prop', 0.0) * is_target_after_prop_loss

        loss = clf_loss_test_w + dimp_loss_test_w + is_target_loss_w + is_target_after_prop_loss_w

        stats = {'Loss/total': loss.item(),
                 'Loss/test_clf': clf_loss_test_w.item(),
                 'Loss/dimp_clf': dimp_loss_test_w.item(),
                 'Loss/raw/test_clf': clf_loss_test.item(),
                 'Loss/raw/dimp_clf': dimp_loss_test.item(),
                 'Loss/raw/test_clf_acc': test_clf_acc.item(),
                 'Loss/raw/dimp_clf_acc': dimp_clf_acc.item(),
                 'Loss/raw/is_target': is_target_loss.item(),
                 'Loss/raw/is_target_after_prop': is_target_after_prop_loss.item(),
                 'Loss/raw/test_seq_acc': test_seq_clf_acc.item(),
                 'Loss/raw/dimp_seq_acc': dimp_seq_clf_acc.item(),
                 }

        return loss, stats
