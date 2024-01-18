# from pytracking.tracker.base import BaseTracker
# import torch
# import torch.nn.functional as F
# import math
# import time
# from pytracking import dcf, TensorList
# from pytracking.features.preprocessing import numpy_to_torch
# from pytracking.utils.plotting import show_tensor, plot_graph
# from pytracking.features.preprocessing import sample_patch_multiscale, sample_patch_transformed
# from pytracking.features import augmentation
# import ltr.data.bounding_box_utils as bbutils
# from ltr.models.target_classifier.initializer import FilterInitializerZero
# from ltr.models.layers import activation
# from util.misc import (NestedTensor, nested_tensor_from_tensor,
#                        nested_tensor_from_tensor_2,
#                        accuracy)
# import sys
# import os 

# import numpy as np
# import cv2

# from vot_path import base_path
# import torch
# import numpy as np
# import math
# print(os.path.abspath(os.getcwd()))
# sys.path.append(os.path.join('/ssd3/lz/MM2022/ACM-MM2022/LTMU/DiMP_LTMU/meta_updater'))
# sys.path.append(os.path.join('/ssd3/lz/MM2022/ACM-MM2022/LTMU/utils/metric_net'))
# from tcNet import tclstm
# from tcopt import tcopts
# from metric_model import ft_net
# from torch.autograd import Variable
# from me_sample_generator import *

# # Dimp
# import argparse
# from pytracking.libs.tensorlist import TensorList
# from pytracking.utils.plotting import show_tensor
# from pytracking.features.preprocessing import numpy_to_torch
# env_path = os.path.join(os.path.dirname(__file__))
# if env_path not in sys.path:
#     sys.path.append(env_path)
# from pytracking.evaluation import Tracker
# from PIL import Image
# Image.MAX_IMAGE_PIXELS = 1000000000
# from tracking_utils import compute_iou, show_res, process_regions
# import tensorflow as tf

# sys.path.append('/ssd3/lz/MM2022/ACM-MM2022/LTMU/DiMP_LTMU/Global_tracker')
# # Global Tracker
# import Global_Track._init_paths
# import neuron.data as data
# from Global_tracker import *
# sys.path.append('/ssd3/lz/MM2022/ACM-MM2022/LTMU/DiMP_LTMU/pyMDNet/modules')
# sys.path.append('/ssd3/lz/MM2022/ACM-MM2022/LTMU/DiMP_LTMU/pyMDNet/tracking')
#  #pymdnet
# from pyMDNet.modules.model import *
# sys.path.insert(0, '/ssd3/lz/MM2022/ACM-MM2022/LTMU/DiMP_LTMU/pyMDNet')
# from pyMDNet.modules.model import MDNet, BCELoss, set_optimizer
# from pyMDNet.modules.sample_generator import SampleGenerator
# from pyMDNet.modules.utils import overlap_ratio
# from pyMDNet.tracking.data_prov import RegionExtractor
# from pyMDNet.tracking.run_tracker import *
# from bbreg import BBRegressor
# from gen_config import gen_config
# opts = yaml.safe_load(open('pyMDNet/tracking/options.yaml','r'))
# #from custom import Custom
# #from tools.test import *
# from tensorflow import set_random_seed
# import random

# class DiMP_LTMU(BaseTracker):

#     #multiobj_mode = 'parallel'

#     def initialize_features(self):
#         if not getattr(self, 'features_initialized', False):
#             self.params.net.initialize()
#         self.features_initialized = True


#     def initialize(self, image, info: dict) -> dict:

#         # Initialize some stuff
#         self.frame_num = 1
#         if not self.params.has('device'):
#             self.params.device = 'cuda' if self.params.use_gpu else 'cpu'
        
    
#         # Initialize network
#         self.initialize_features()

#         '''
#         add from LTMU
#         '''
#         self.i = 0
#         self.t_id = 0
#         tfconfig = tf.ConfigProto()
#         tfconfig.gpu_options.per_process_gpu_memory_fraction = 0.3
#         self.sess = tf.Session(config=tfconfig)
#         '''
#         state == LTMU: init_gt1
#         state1 == LTMU: init_gt
#         '''
#         state = info['init_bbox']
#         state1 = [state[1], state[0], state[1]+state[3], state[0]+state[2]]
#         self.last_gt = state1
#         self.pos_meta = torch.Tensor([state1[1] + (state1[3] - 1)/2, state1[0] + (state1[2] - 1)/2])
#         self.target_sz_meta = torch.Tensor([state[3], state[2]])
#         self.init_pymdnet(image, state)
#         self.Golbal_Track_init(image, state)
#         self.tc_init("dimp_mu_1")
#         self.metric_init(image, np.array(state))
#         self.dis_record = []
#         self.state_record = []
#         self.rv_record = []
#         self.all_map = []
#         self.count = 0

#         local_state1, self.score_map, update, self.score_max, dis = self.local_track(image)
        
#         '''
#         add from LTMU
#         '''

#         # The DiMP network
#         self.net = self.params.net

#         # Time initialization
#         tic = time.time()

#         # Convert image
#         im = numpy_to_torch(image) # HxWx6 -> 6 * H * W

#         # Get target position and size
#         self.pos = torch.Tensor([state[1] + (state[3] - 1)/2, state[0] + (state[2] - 1)/2])
#         self.target_sz = torch.Tensor([state[3], state[2]])

#         # Get object id
#         self.object_id = info.get('object_ids', [None])[0]
#         self.id_str = '' if self.object_id is None else ' {}'.format(self.object_id)

#         # Set sizes
#         self.image_sz = torch.Tensor([im.shape[2], im.shape[3]])
#         sz = self.params.image_sample_size
#         sz = torch.Tensor([sz, sz] if isinstance(sz, int) else sz)
#         if self.params.get('use_image_aspect_ratio', False):
#             sz = self.image_sz * sz.prod().sqrt() / self.image_sz.prod().sqrt()
#             stride = self.params.get('feature_stride', 32)
#             sz = torch.round(sz / stride) * stride
#         self.img_sample_sz = sz
#         self.img_support_sz = self.img_sample_sz

#         # Set search area
#         search_area = torch.prod(self.target_sz * self.params.search_area_scale).item()
#         self.target_scale =  math.sqrt(search_area) / self.img_sample_sz.prod().sqrt()

#         # Target size in base scale
#         self.base_target_sz = self.target_sz / self.target_scale

#         # Setup scale factors
#         if not self.params.has('scale_factors'):
#             self.params.scale_factors = torch.ones(1)
#         elif isinstance(self.params.scale_factors, (list, tuple)):
#             self.params.scale_factors = torch.Tensor(self.params.scale_factors)

#         # Setup scale bounds
#         self.min_scale_factor = torch.max(10 / self.base_target_sz)
#         self.max_scale_factor = torch.min(self.image_sz / self.base_target_sz) 

#         # Extract and transform sample
#         init_backbone_feat = self.generate_init_samples(im)

#         # Initialize classifier
#         self.init_classifier(init_backbone_feat)

#         # Initialize IoUNet
#         if self.params.get('use_iou_net', True):
#             self.init_iou_net(init_backbone_feat)

#         out = {'time': time.time() - tic}
#         return out

#     '''
#         add from LTMU
#     '''
#     def get_first_state(self):
#         return self.score_map, self.score_max
    

#     '''
#         add from LTMU
#     '''
#     def Golbal_Track_init(self, image, init_box):
#         init_box = [init_box[0], init_box[1], init_box[0]+init_box[2], init_box[1]+init_box[3]]
#         cfg_file = 'Global_Track/configs/qg_rcnn_r50_fpn.py'
#         ckp_file = 'Global_Track/checkpoints/qg_rcnn_r50_fpn_coco_got10k_lasot.pth'
#         transforms = data.BasicPairTransforms(train=False)
#         self.Global_Tracker = GlobalTrack(
#             cfg_file, ckp_file, transforms,
#             name_suffix='qg_rcnn_r50_fpn')
#         self.Global_Tracker.init(image, init_box)


#     '''
#         add from LTMU
#     '''
#     def Global_Track_eval(self, image, num):
#         # xywh
#         results = self.Global_Tracker.update(image)
#         index = np.argsort(results[:, -1])[::-1]
#         max_index = index[:num]
#         can_boxes = results[max_index][:, :4]
#         can_boxes = np.array([can_boxes[:, 0], can_boxes[:, 1], can_boxes[:, 2]-can_boxes[:, 0], can_boxes[:, 3]-can_boxes[:, 1]]).transpose()
#         return can_boxes
    


#     '''
#         add from LTMU
#     '''
#     def init_pymdnet(self, image, init_bbox):
#         target_bbox = np.array(init_bbox)
#         self.last_result = target_bbox
#         self.pymodel = MDNet('./pyMDNet/models/mdnet_imagenet_vid.pth')
#         if opts['use_gpu']:
#             self.pymodel = self.pymodel.cuda()
#         self.pymodel.set_learnable_params(opts['ft_layers'])

#         # Init criterion and optimizer
#         self.criterion = BCELoss()
#         init_optimizer = set_optimizer(self.pymodel, opts['lr_init'], opts['lr_mult'])
#         self.update_optimizer = set_optimizer(self.pymodel, opts['lr_update'], opts['lr_mult'])

#         tic = time.time()

#         # Draw pos/neg samples
#         pos_examples = SampleGenerator('gaussian', image.size, opts['trans_pos'], opts['scale_pos'])(
#             target_bbox, opts['n_pos_init'], opts['overlap_pos_init'])

#         neg_examples = np.concatenate([
#             SampleGenerator('uniform', image.size, opts['trans_neg_init'], opts['scale_neg_init'])(
#                 target_bbox, int(opts['n_neg_init'] * 0.5), opts['overlap_neg_init']),
#             SampleGenerator('whole', image.size)(
#                 target_bbox, int(opts['n_neg_init'] * 0.5), opts['overlap_neg_init'])])
#         neg_examples = np.random.permutation(neg_examples)

#         # Extract pos/neg features
#         pos_feats = forward_samples(self.pymodel, image, pos_examples, opts)
#         neg_feats = forward_samples(self.pymodel, image, neg_examples, opts)
#         self.feat_dim = pos_feats.size(-1)

#         # Initial training
#         train(self.pymodel, self.criterion, init_optimizer, pos_feats, neg_feats, opts['maxiter_init'], opts=opts)
#         del init_optimizer, neg_feats
#         torch.cuda.empty_cache()

#         # Train bbox regressor
#         bbreg_examples = SampleGenerator('uniform', image.size, opts['trans_bbreg'], opts['scale_bbreg'],
#                                          opts['aspect_bbreg'])(
#             target_bbox, opts['n_bbreg'], opts['overlap_bbreg'])
#         bbreg_feats = forward_samples(self.pymodel, image, bbreg_examples, opts)
#         self.bbreg = BBRegressor(image.size)
#         self.bbreg.train(bbreg_feats, bbreg_examples, target_bbox)
#         del bbreg_feats
#         torch.cuda.empty_cache()
#         # Init sample generators
#         self.sample_generator = SampleGenerator('gaussian', image.size, opts['trans'], opts['scale'])
#         self.pos_generator = SampleGenerator('gaussian', image.size, opts['trans_pos'], opts['scale_pos'])
#         self.neg_generator = SampleGenerator('uniform', image.size, opts['trans_neg'], opts['scale_neg'])

#         # Init pos/neg features for update
#         neg_examples = self.neg_generator(target_bbox, opts['n_neg_update'], opts['overlap_neg_init'])
#         neg_feats = forward_samples(self.pymodel, image, neg_examples, opts)
#         self.pos_feats_all = [pos_feats]
#         self.neg_feats_all = [neg_feats]

#         spf_total = time.time() - tic
    

#     '''
#         add from LTMU
#     '''
#     def pymdnet_eval(self, image, samples):
#         sample_scores = forward_samples(self.pymodel, image, samples, out_layer='fc6', opts=opts)
#         return sample_scores[:, 1][:].cpu().numpy()


#     '''
#         add from LTMU
#     '''
#     def local_track(self, image):
#         state, score_map, test_x, scale_ind, sample_pos, sample_scales, flag, s = track_updater(image)
#         update_score = 0
#         update_flag = flag not in ['not_found', 'uncertain']
#         update = update_flag
#         max_score = max(score_map.flatten())
#         self.all_map.append(score_map)
#         local_state = np.array(state).reshape((1, 4))
#         ap_dis = self.metric_eval(image, local_state, self.anchor_feature)
#         self.dis_record.append(ap_dis.data.cpu().numpy()[0])
#         h = image.shape[0]
#         w = image.shape[1]
#         self.state_record.append([local_state[0][0] / w, local_state[0][1] / h,
#                                   (local_state[0][0] + local_state[0][2]) / w,
#                                   (local_state[0][1] + local_state[0][3]) / h])
#         self.rv_record.append(max_score)
#         if len(self.state_record) >= tcopts['time_steps']:
#             dis = np.array(self.dis_record[-tcopts["time_steps"]:]).reshape((tcopts["time_steps"], 1))
#             rv = np.array(self.rv_record[-tcopts["time_steps"]:]).reshape((tcopts["time_steps"], 1))
#             state_tc = np.array(self.state_record[-tcopts["time_steps"]:])
#             map_input = np.array(self.all_map[-tcopts["time_steps"]:])
#             map_input = np.reshape(map_input, [tcopts['time_steps'], 1, 19, 19])
#             map_input = map_input.transpose((0, 2, 3, 1))
#             X_input = np.concatenate((state_tc, rv, dis), axis=1)
#             logits = self.sess.run(self.logits,
#                                                feed_dict={self.X_input: np.expand_dims(X_input, axis=0),
#                                                           self.maps: map_input})
#             update = logits[0][0] < logits[0][1]
#             update_score = logits[0][1]

#         hard_negative = (flag == 'hard_negative')
#         learning_rate = getattr(self.params, 'hard_negative_learning_rate', None) if hard_negative else None

#         if update:
#             # Get train sample
#             train_x = test_x[scale_ind:scale_ind+1, ...]

#             # Create target_box and label for spatial sample
#             target_box = self.get_iounet_box(self.pos, self.target_sz,
#                                                            sample_pos[scale_ind, :], sample_scales[scale_ind])

#             # Update the classifier model
#             self.update_classifier(train_x, target_box, learning_rate, s[scale_ind,...])
#         self.last_gt = [state[1], state[0], state[1]+state[3], state[0]+state[2]]
#         return state, score_map, update, max_score, ap_dis.data.cpu().numpy()[0], flag, update_score
    

#     '''
#         change according to LTMU: dimp.py
#     '''
#     def track_updater(self, image) -> dict:
#         self.debug_info = {}

#         self.frame_num += 1
#         self.debug_info['frame_num'] = self.frame_num

#         # Convert image

#         im = numpy_to_torch(image)

#         # ------- LOCALIZATION ------- #

#         # Extract backbone features
#         backbone_feat, sample_coords, im_patches = self.extract_backbone_features(im, self.get_centered_sample_pos(),
#                                                                       self.target_scale * self.params.scale_factors,
#                                                                       self.img_sample_sz)
#         # Extract classification features
#         test_x = self.get_classification_features(backbone_feat)

#         # Location of sample
#         sample_pos, sample_scales = self.get_sample_location(sample_coords)

#         # Compute classification scores
#         scores_raw = self.classify_target(test_x)

#         # Localize the target
#         translation_vec, scale_ind, s, flag = self.localize_target(scores_raw, sample_pos, sample_scales)
#         new_pos = sample_pos[scale_ind,:] + translation_vec

#         # Update position and scale
#         if flag != 'not_found':
#             if self.params.get('use_iou_net', True):
#                 update_scale_flag = self.params.get('update_scale_when_uncertain', True) or flag != 'uncertain'
#                 if self.params.get('use_classifier', True):
#                     self.update_state(new_pos)
#                 self.refine_target_box(backbone_feat, sample_pos[scale_ind,:], sample_scales[scale_ind], scale_ind, update_scale_flag)
#             elif self.params.get('use_classifier', True):
#                 self.update_state(new_pos, sample_scales[scale_ind])


#         # ------- UPDATE ------- #

#         # update_flag = flag not in ['not_found', 'uncertain']
#         # hard_negative = (flag == 'hard_negative')
#         # learning_rate = self.params.get('hard_negative_learning_rate', None) if hard_negative else None

#         # if update_flag and self.params.get('update_classifier', False):
#         #     # Get train sample
#         #     train_x = test_x[scale_ind:scale_ind+1, ...]

#         #     # Create target_box and label for spatial sample
#         #     target_box = self.get_iounet_box(self.pos, self.target_sz, sample_pos[scale_ind,:], sample_scales[scale_ind])

#         #     # Update the classifier model
#         #     self.update_classifier(train_x, target_box, learning_rate, s[scale_ind,...])

#         # Set the pos of the tracker to iounet pos
#         if self.params.get('use_iou_net', True) and flag != 'not_found' and hasattr(self, 'pos_iounet'):
#             self.pos = self.pos_iounet.clone()

#         score_map = s[scale_ind, ...]
#         max_score = torch.max(score_map).item()
#         self.debug_info['max_score'] = max_score

#         # Visualize and set debug info
#         # self.search_area_box = torch.cat((sample_coords[scale_ind,[1,0]], sample_coords[scale_ind,[3,2]] - sample_coords[scale_ind,[1,0]] - 1))
#         # self.debug_info['flag' + self.id_str] = flag
#         # self.debug_info['max_score' + self.id_str] = max_score
#         if self.visdom is not None:
#             self.visdom.register(score_map, 'heatmap', 2, 'Score Map' + self.id_str)
#             self.visdom.register(self.debug_info, 'info_dict', 1, 'Status')
#         elif self.params.debug >= 2:
#             show_tensor(score_map, 5, title='Max score = {:.2f}'.format(max_score))

#         # Compute output bounding box
#         new_state = torch.cat((self.pos[[1,0]] - (self.target_sz[[1,0]]-1)/2, self.target_sz[[1,0]]))

#         # if self.params.get('output_not_found_box', False) and flag == 'not_found':
#         #     output_state = [-1, -1, -1, -1]
#         # else:
#         #     output_state = new_state.tolist()


#         out = {'target_bbox': new_state.tolist(), 'score_map': score_map}
#         return new_state.tolist(), score_map.cpu().data.numpy(), test_x, scale_ind, sample_pos, sample_scales, flag, s
#         # # out = {'target_bbox': output_state, 'confidence': max_score}
#         # # return out
#         # return new_state.tolist(), score_map.cpu().data.numpy(), test_x, scale_ind, sample_pos, sample_scales, flag, s

#     '''
#         add from LTMU
#     '''
#     def tracking(self, image):
#         self.i += 1
#         mask = None
#         candidate_bboxes = None
#         # state, pyscore = self.pymdnet_track(image)
#         # self.last_gt = [state[1], state[0], state[1] + state[3], state[0] + state[2]]
#         self.pos = torch.FloatTensor(
#             [(self.last_gt[0] + self.last_gt[2] - 1) / 2, (self.last_gt[1] + self.last_gt[3] - 1) / 2])
#         self.target_sz = torch.FloatTensor(
#             [(self.last_gt[2] - self.last_gt[0]), (self.last_gt[3] - self.last_gt[1])])
#         tic = time.time()
#         local_state, self.score_map, update, local_score, dis, flag, update_score = self.local_track(image)

#         md_score = self.pymdnet_eval(image, np.array(local_state).reshape([-1, 4]))[0]
#         self.score_max = md_score

#         if md_score > 0 and flag == 'normal':
#             self.flag = 'found'
#             # if self.p.use_mask: # don't use mask,change to flase
#             #     self.siamstate['target_pos'] = self.local_Tracker.pos.numpy()[::-1]
#             #     self.siamstate['target_sz'] = self.local_Tracker.target_sz.numpy()[::-1]
#             #     siamscore, mask = self.siammask_track(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
#             #     self.local_Tracker.pos = torch.FloatTensor(self.siamstate['target_pos'][::-1].copy())
#             #     self.local_Tracker.target_sz = torch.FloatTensor(self.siamstate['target_sz'][::-1].copy())
#             #     local_state = torch.cat((self.local_Tracker.pos[[1, 0]] - (self.local_Tracker.target_sz[[1, 0]] - 1) / 2,
#             #                             self.local_Tracker.target_sz[[1, 0]])).data.cpu().numpy()
#             self.last_gt = np.array(
#                 [local_state[1], local_state[0], local_state[1] + local_state[3], local_state[0] + local_state[2]])
#         elif md_score < 0 or flag == 'not_found':
#             self.count += 1
#             self.flag = 'not_found'
#             candidate_bboxes = self.Global_Track_eval(image, 10)
#             candidate_scores = self.pymdnet_eval(image, candidate_bboxes)
#             max_id = np.argmax(candidate_scores)
#             if candidate_scores[max_id] > 0:
#                 redet_bboxes = candidate_bboxes[max_id]
#                 if self.count >= 5:
#                     self.last_gt = np.array([redet_bboxes[1], redet_bboxes[0], redet_bboxes[1] + redet_bboxes[3],
#                                              redet_bboxes[2] + redet_bboxes[0]])
#                     self.pos = torch.FloatTensor(
#                         [(self.last_gt[0] + self.last_gt[2] - 1) / 2, (self.last_gt[1] + self.last_gt[3] - 1) / 2])
#                     self.target_sz = torch.FloatTensor(
#                         [(self.last_gt[2] - self.last_gt[0]), (self.last_gt[3] - self.last_gt[1])])
#                     self.score_max = candidate_scores[max_id]
#                     self.count = 0
#         if update:
#             self.collect_samples_pymdnet(image)

#         self.pymdnet_long_term_update()

#         width = self.last_gt[3] - self.last_gt[1]
#         height = self.last_gt[2] - self.last_gt[0]
#         toc = time.time() - tic
#         print(toc)
#         # if self.flag == 'found' and self.score_max > 0:
#         #     confidence_score = 0.99
#         # elif self.flag == 'not_found':
#         #     confidence_score = 0.0
#         # else:
#         #     confidence_score = np.clip((local_score+np.arctan(0.2*self.score_max)/math.pi+0.5)/2, 0, 1)
#         confidence_score = np.clip((local_score + np.arctan(0.2 * self.score_max) / math.pi + 0.5) / 2, 0, 1)
#         if self.p.visualization:
#             show_res(cv2.cvtColor(image, cv2.COLOR_RGB2BGR), np.array(self.last_gt, dtype=np.int32), '2',
#                      groundtruth=self.groundtruth, update=update_score, can_bboxes=candidate_bboxes,
#                      frame_id=self.i, tracker_score=md_score, mask=mask)

#         return [float(self.last_gt[1]), float(self.last_gt[0]), float(width),
#                 float(height)], self.score_map, 0, confidence_score, 0



#     def get_sample_location(self, sample_coord):
#         """Get the location of the extracted sample."""
#         sample_coord = sample_coord.float()
#         sample_pos = 0.5*(sample_coord[:,:2] + sample_coord[:,2:] - 1)
#         sample_scales = ((sample_coord[:,2:] - sample_coord[:,:2]) / self.img_sample_sz).prod(dim=1).sqrt()
#         return sample_pos, sample_scales

#     def get_centered_sample_pos(self):
#         """Get the center position for the new sample. Make sure the target is correctly centered."""
#         return self.pos + ((self.feature_sz + self.kernel_size) % 2) * self.target_scale * \
#                self.img_support_sz / (2*self.feature_sz)

#     def classify_target(self, sample_x: TensorList):
#         """Classify target by applying the DiMP filter."""
#         with torch.no_grad():
#             scores = self.net.classifier.classify(self.target_filter, sample_x)
#         return scores

#     def localize_target(self, scores, sample_scales):
#         """Run the target localization."""

#         scores = scores.squeeze(1)

#         if getattr(self.params, 'advanced_localization', False):
#             return self.localize_advanced(scores, sample_scales)

#         # Get maximum
#         score_sz = torch.Tensor(list(scores.shape[-2:]))
#         score_center = (score_sz - 1)/2
#         max_score, max_disp = dcf.max2d(scores)
#         _, scale_ind = torch.max(max_score, dim=0)
#         max_disp = max_disp[scale_ind,...].float().cpu().view(-1)
#         target_disp = max_disp - score_center

#         # Compute translation vector and scale change factor
#         translation_vec = target_disp * (self.img_support_sz / self.feature_sz) * sample_scales[scale_ind]

#         return translation_vec, scale_ind, scores, None


#     def localize_advanced(self, scores, sample_scales):
#         """Run the target advanced localization (as in ATOM)."""

#         sz = scores.shape[-2:]
#         score_sz = torch.Tensor(list(sz))
#         score_center = (score_sz - 1)/2

#         scores_hn = scores
#         if self.output_window is not None and getattr(self.params, 'perform_hn_without_windowing', False):
#             scores_hn = scores.clone()
#             scores *= self.output_window

#         max_score1, max_disp1 = dcf.max2d(scores)
#         _, scale_ind = torch.max(max_score1, dim=0)
#         sample_scale = sample_scales[scale_ind]
#         max_score1 = max_score1[scale_ind]
#         max_disp1 = max_disp1[scale_ind,...].float().cpu().view(-1)
#         target_disp1 = max_disp1 - score_center
#         translation_vec1 = target_disp1 * (self.img_support_sz / self.feature_sz) * sample_scale

#         if max_score1.item() < self.params.target_not_found_threshold:
#             return translation_vec1, scale_ind, scores_hn, 'not_found'

#         # Mask out target neighborhood
#         target_neigh_sz = self.params.target_neighborhood_scale * (self.target_sz / sample_scale) * (self.feature_sz / self.img_support_sz)

#         tneigh_top = max(round(max_disp1[0].item() - target_neigh_sz[0].item() / 2), 0)
#         tneigh_bottom = min(round(max_disp1[0].item() + target_neigh_sz[0].item() / 2 + 1), sz[0])
#         tneigh_left = max(round(max_disp1[1].item() - target_neigh_sz[1].item() / 2), 0)
#         tneigh_right = min(round(max_disp1[1].item() + target_neigh_sz[1].item() / 2 + 1), sz[1])
#         scores_masked = scores_hn[scale_ind:scale_ind + 1, ...].clone()
#         scores_masked[...,tneigh_top:tneigh_bottom,tneigh_left:tneigh_right] = 0

#         # Find new maximum
#         max_score2, max_disp2 = dcf.max2d(scores_masked)
#         max_disp2 = max_disp2.float().cpu().view(-1)
#         target_disp2 = max_disp2 - score_center
#         translation_vec2 = target_disp2 * (self.img_support_sz / self.feature_sz) * sample_scale

#         # Handle the different cases
#         if max_score2 > self.params.distractor_threshold * max_score1:
#             disp_norm1 = torch.sqrt(torch.sum(target_disp1**2))
#             disp_norm2 = torch.sqrt(torch.sum(target_disp2**2))
#             disp_threshold = self.params.dispalcement_scale * math.sqrt(sz[0] * sz[1]) / 2

#             if disp_norm2 > disp_threshold and disp_norm1 < disp_threshold:
#                 return translation_vec1, scale_ind, scores_hn, 'hard_negative'
#             if disp_norm2 < disp_threshold and disp_norm1 > disp_threshold:
#                 return translation_vec2, scale_ind, scores_hn, 'hard_negative'
#             if disp_norm2 > disp_threshold and disp_norm1 > disp_threshold:
#                 return translation_vec1, scale_ind, scores_hn, 'uncertain'

#             # If also the distractor is close, return with highest score
#             return translation_vec1, scale_ind, scores_hn, 'uncertain'

#         if max_score2 > self.params.hard_negative_threshold * max_score1 and max_score2 > self.params.target_not_found_threshold:
#             return translation_vec1, scale_ind, scores_hn, 'hard_negative'

#         return translation_vec1, scale_ind, scores_hn, 'normal'

#     def extract_backbone_features(self, im: torch.Tensor, pos: torch.Tensor, scales, sz: torch.Tensor):
#         im_patches, patch_coords = sample_patch_multiscale(im, pos, scales, sz, getattr(self.params, 'border_mode', 'replicate'))
#         with torch.no_grad():
#             backbone_feat = self.net.extract_backbone(im_patches)
#         return backbone_feat, patch_coords

#     def get_classification_features(self, backbone_feat):
#         with torch.no_grad():
#             return self.net.extract_classification_feat(backbone_feat)

#     def get_iou_backbone_features(self, backbone_feat):
#         return self.net.get_backbone_bbreg_feat(backbone_feat)

#     def get_iou_features(self, backbone_feat):
#         with torch.no_grad():
#             return self.net.bb_regressor.get_iou_feat(self.get_iou_backbone_features(backbone_feat))

#     def get_iou_modulation(self, iou_backbone_feat, target_boxes):
#         with torch.no_grad():
#             return self.net.bb_regressor.get_modulation(iou_backbone_feat, target_boxes)


#     def generate_init_samples(self, im: torch.Tensor) -> TensorList:
#         """Perform data augmentation to generate initial training samples."""

#         if getattr(self.params, 'border_mode', 'replicate') == 'inside':
#             # Get new sample size if forced inside the image
#             im_sz = torch.Tensor([im.shape[2], im.shape[3]])
#             sample_sz = self.target_scale * self.img_sample_sz
#             shrink_factor = (sample_sz.float() / im_sz).max().clamp(1)
#             sample_sz = (sample_sz.float() / shrink_factor)
#             self.init_sample_scale = (sample_sz / self.img_sample_sz).prod().sqrt()
#             tl = self.pos - (sample_sz - 1) / 2
#             br = self.pos + sample_sz / 2 + 1
#             global_shift = - ((-tl).clamp(0) - (br - im_sz).clamp(0)) / self.init_sample_scale
#         else:
#             self.init_sample_scale = self.target_scale
#             global_shift = torch.zeros(2)

#         self.init_sample_pos = self.pos.round()

#         # Compute augmentation size
#         aug_expansion_factor = getattr(self.params, 'augmentation_expansion_factor', None)
#         aug_expansion_sz = self.img_sample_sz.clone()
#         aug_output_sz = None
#         if aug_expansion_factor is not None and aug_expansion_factor != 1:
#             aug_expansion_sz = (self.img_sample_sz * aug_expansion_factor).long()
#             aug_expansion_sz += (aug_expansion_sz - self.img_sample_sz.long()) % 2
#             aug_expansion_sz = aug_expansion_sz.float()
#             aug_output_sz = self.img_sample_sz.long().tolist()

#         # Random shift for each sample
#         get_rand_shift = lambda: None
#         random_shift_factor = getattr(self.params, 'random_shift_factor', 0)
#         if random_shift_factor > 0:
#             get_rand_shift = lambda: ((torch.rand(2) - 0.5) * self.img_sample_sz * random_shift_factor + global_shift).long().tolist()

#         # Always put identity transformation first, since it is the unaugmented sample that is always used
#         self.transforms = [augmentation.Identity(aug_output_sz, global_shift.long().tolist())]

#         augs = self.params.augmentation if getattr(self.params, 'use_augmentation', True) else {}

#         # Add all augmentations
#         if 'shift' in augs:
#             self.transforms.extend([augmentation.Translation(shift, aug_output_sz, global_shift.long().tolist()) for shift in augs['shift']])
#         if 'relativeshift' in augs:
#             get_absolute = lambda shift: (torch.Tensor(shift) * self.img_sample_sz/2).long().tolist()
#             self.transforms.extend([augmentation.Translation(get_absolute(shift), aug_output_sz, global_shift.long().tolist()) for shift in augs['relativeshift']])
#         if 'fliplr' in augs and augs['fliplr']:
#             self.transforms.append(augmentation.FlipHorizontal(aug_output_sz, get_rand_shift()))
#         if 'blur' in augs:
#             self.transforms.extend([augmentation.Blur(sigma, aug_output_sz, get_rand_shift()) for sigma in augs['blur']])
#         if 'scale' in augs:
#             self.transforms.extend([augmentation.Scale(scale_factor, aug_output_sz, get_rand_shift()) for scale_factor in augs['scale']])
#         if 'rotate' in augs:
#             self.transforms.extend([augmentation.Rotate(angle, aug_output_sz, get_rand_shift()) for angle in augs['rotate']])

#         # Extract augmented image patches
#         im_patches = sample_patch_transformed(im, self.init_sample_pos, self.init_sample_scale, aug_expansion_sz, self.transforms)

#         # Extract initial backbone features
#         with torch.no_grad():
#             init_backbone_feat = self.net.extract_backbone(im_patches)

#         return init_backbone_feat

#     def init_target_boxes(self):
#         """Get the target bounding boxes for the initial augmented samples."""
#         self.classifier_target_box = self.get_iounet_box(self.pos, self.target_sz, self.init_sample_pos, self.init_sample_scale)
#         init_target_boxes = TensorList()
#         for T in self.transforms:
#             init_target_boxes.append(self.classifier_target_box + torch.Tensor([T.shift[1], T.shift[0], 0, 0]))
#         init_target_boxes = torch.cat(init_target_boxes.view(1, 4), 0).to(self.params.device)
#         self.target_boxes = init_target_boxes.new_zeros(self.params.sample_memory_size, 4)
#         self.target_boxes[:init_target_boxes.shape[0],:] = init_target_boxes
#         return init_target_boxes

#     def init_memory(self, train_x: TensorList):
#         # Initialize first-frame spatial training samples
#         self.num_init_samples = train_x.size(0)
#         init_sample_weights = TensorList([x.new_ones(1) / x.shape[0] for x in train_x])

#         # Sample counters and weights for spatial
#         self.num_stored_samples = self.num_init_samples.copy()
#         self.previous_replace_ind = [None] * len(self.num_stored_samples)
#         self.sample_weights = TensorList([x.new_zeros(self.params.sample_memory_size) for x in train_x])
#         for sw, init_sw, num in zip(self.sample_weights, init_sample_weights, self.num_init_samples):
#             sw[:num] = init_sw

#         # Initialize memory
#         self.training_samples = TensorList(
#             [x.new_zeros(self.params.sample_memory_size, x.shape[1], x.shape[2], x.shape[3]) for x in train_x])

#         for ts, x in zip(self.training_samples, train_x):
#             ts[:x.shape[0],...] = x


#     def update_memory(self, sample_x: TensorList, target_box, learning_rate = None):
#         # Update weights and get replace ind
#         replace_ind = self.update_sample_weights(self.sample_weights, self.previous_replace_ind, self.num_stored_samples, self.num_init_samples, learning_rate)
#         self.previous_replace_ind = replace_ind

#         # Update sample and label memory
#         for train_samp, x, ind in zip(self.training_samples, sample_x, replace_ind):
#             train_samp[ind:ind+1,...] = x

#         # Update bb memory
#         self.target_boxes[replace_ind[0],:] = target_box

#         self.num_stored_samples += 1


#     def update_sample_weights(self, sample_weights, previous_replace_ind, num_stored_samples, num_init_samples, learning_rate = None):
#         # Update weights and get index to replace
#         replace_ind = []
#         for sw, prev_ind, num_samp, num_init in zip(sample_weights, previous_replace_ind, num_stored_samples, num_init_samples):
#             lr = learning_rate
#             if lr is None:
#                 lr = self.params.learning_rate

#             init_samp_weight = getattr(self.params, 'init_samples_minimum_weight', None)
#             if init_samp_weight == 0:
#                 init_samp_weight = None
#             s_ind = 0 if init_samp_weight is None else num_init

#             if num_samp == 0 or lr == 1:
#                 sw[:] = 0
#                 sw[0] = 1
#                 r_ind = 0
#             else:
#                 # Get index to replace
#                 if num_samp < sw.shape[0]:
#                     r_ind = num_samp
#                 else:
#                     _, r_ind = torch.min(sw[s_ind:], 0)
#                     r_ind = r_ind.item() + s_ind

#                 # Update weights
#                 if prev_ind is None:
#                     sw /= 1 - lr
#                     sw[r_ind] = lr
#                 else:
#                     sw[r_ind] = sw[prev_ind] / (1 - lr)

#             sw /= sw.sum()
#             if init_samp_weight is not None and sw[:num_init].sum() < init_samp_weight:
#                 sw /= init_samp_weight + sw[num_init:].sum()
#                 sw[:num_init] = init_samp_weight / num_init

#             replace_ind.append(r_ind)

#         return replace_ind

#     def update_state(self, new_pos, new_scale = None):
#         # Update scale
#         if new_scale is not None:
#             self.target_scale = new_scale.clamp(self.min_scale_factor, self.max_scale_factor)
#             self.target_sz = self.base_target_sz * self.target_scale

#         # Update pos
#         inside_ratio = getattr(self.params, 'target_inside_ratio', 0.2)
#         inside_offset = (inside_ratio - 0.5) * self.target_sz
#         self.pos = torch.max(torch.min(new_pos, self.image_sz - inside_offset), inside_offset)


#     def get_iounet_box(self, pos, sz, sample_pos, sample_scale):
#         """All inputs in original image coordinates.
#         Generates a box in the cropped image sample reference frame, in the format used by the IoUNet."""
#         box_center = (pos - sample_pos) / sample_scale + (self.img_sample_sz - 1) / 2
#         box_sz = sz / sample_scale
#         target_ul = box_center - (box_sz - 1) / 2
#         return torch.cat([target_ul.flip((0,)), box_sz.flip((0,))])


#     def init_iou_net(self, backbone_feat):
#         # Setup IoU net and objective
#         for p in self.net.bb_regressor.parameters():
#             p.requires_grad = False

#         # Get target boxes for the different augmentations
#         self.classifier_target_box = self.get_iounet_box(self.pos, self.target_sz, self.init_sample_pos, self.init_sample_scale)
#         target_boxes = TensorList()
#         if self.params.iounet_augmentation:
#             for T in self.transforms:
#                 if not isinstance(T, (augmentation.Identity, augmentation.Translation, augmentation.FlipHorizontal, augmentation.FlipVertical, augmentation.Blur)):
#                     break
#                 target_boxes.append(self.classifier_target_box + torch.Tensor([T.shift[1], T.shift[0], 0, 0]))
#         else:
#             target_boxes.append(self.classifier_target_box + torch.Tensor([self.transforms[0].shift[1], self.transforms[0].shift[0], 0, 0]))
#         target_boxes = torch.cat(target_boxes.view(1,4), 0).to(self.params.device)

#         # Get iou features
#         iou_backbone_feat = self.get_iou_backbone_features(backbone_feat)

#         # Remove other augmentations such as rotation
#         iou_backbone_feat = TensorList([x[:target_boxes.shape[0],...] for x in iou_backbone_feat])

#         # Get modulation vector
#         self.iou_modulation = self.get_iou_modulation(iou_backbone_feat, target_boxes)
#         self.iou_modulation = TensorList([x.detach().mean(0) for x in self.iou_modulation])


#     def init_classifier(self, init_backbone_feat):
#         # Get classification features
#         x = self.get_classification_features(init_backbone_feat)

#         # Add the dropout augmentation here, since it requires extraction of the classification features
#         if 'dropout' in self.params.augmentation and getattr(self.params, 'use_augmentation', True):
#             num, prob = self.params.augmentation['dropout']
#             self.transforms.extend(self.transforms[:1]*num)
#             x = torch.cat([x, F.dropout2d(x[0:1,...].expand(num,-1,-1,-1), p=prob, training=True)])

#         # Set feature size and other related sizes
#         self.feature_sz = torch.Tensor(list(x.shape[-2:]))
#         ksz = self.net.classifier.filter_size
#         self.kernel_size = torch.Tensor([ksz, ksz] if isinstance(ksz, (int, float)) else ksz)
#         self.output_sz = self.feature_sz + (self.kernel_size + 1)%2

#         # Construct output window
#         self.output_window = None
#         if getattr(self.params, 'window_output', False):
#             if getattr(self.params, 'use_clipped_window', False):
#                 self.output_window = dcf.hann2d_clipped(self.output_sz.long(), self.output_sz.long()*self.params.effective_search_area / self.params.search_area_scale, centered=False).to(self.params.device)
#             else:
#                 self.output_window = dcf.hann2d(self.output_sz.long(), centered=True).to(self.params.device)
#             self.output_window = self.output_window.squeeze(0)

#         # Get target boxes for the different augmentations
#         target_boxes = self.init_target_boxes()

#         # Set number of iterations
#         plot_loss = self.params.debug > 0
#         num_iter = getattr(self.params, 'net_opt_iter', None)

#         # Get target filter by running the discriminative model prediction module
#         with torch.no_grad():
#             self.target_filter, _, losses = self.net.classifier.get_filter(x, target_boxes, num_iter=num_iter,
#                                                                            compute_losses=plot_loss)

#         # Init memory
#         if getattr(self.params, 'update_classifier', True):
#             self.init_memory(TensorList([x]))

#         if plot_loss:
#             if isinstance(losses, dict):
#                 losses = losses['train']
#             self.losses = torch.cat(losses)
#             if self.visdom is not None:
#                 self.visdom.register((self.losses, torch.arange(self.losses.numel())), 'lineplot', 3, 'Training Loss')
#             elif self.params.debug >= 3:
#                 plot_graph(self.losses, 10, title='Training loss')


#     def update_classifier(self, train_x, target_box, learning_rate=None, scores=None):
#         # Set flags and learning rate
#         hard_negative_flag = learning_rate is not None
#         if learning_rate is None:
#             learning_rate = self.params.learning_rate

#         # Update the tracker memory
#         self.update_memory(TensorList([train_x]), target_box, learning_rate)

#         # Decide the number of iterations to run
#         num_iter = 0
#         low_score_th = getattr(self.params, 'low_score_opt_threshold', None)
#         if hard_negative_flag:
#             num_iter = getattr(self.params, 'net_opt_hn_iter', None)
#         elif low_score_th is not None and low_score_th > scores.max().item():
#             num_iter = getattr(self.params, 'net_opt_low_iter', None)
#         elif (self.frame_num - 1) % self.params.train_skipping == 0:
#             num_iter = getattr(self.params, 'net_opt_update_iter', None)

#         plot_loss = self.params.debug > 0

#         if num_iter > 0:
#             # Get inputs for the DiMP filter optimizer module
#             samples = self.training_samples[0][:self.num_stored_samples[0],...]
#             target_boxes = self.target_boxes[:self.num_stored_samples[0],:].clone()
#             sample_weights = self.sample_weights[0][:self.num_stored_samples[0]]

#             # Run the filter optimizer module
#             with torch.no_grad():
#                 self.target_filter, _, losses = self.net.classifier.filter_optimizer(self.target_filter, samples, target_boxes,
#                                                                                      sample_weight=sample_weights,
#                                                                                      num_iter=num_iter,
#                                                                                      compute_losses=plot_loss)

#             if plot_loss:
#                 if isinstance(losses, dict):
#                     losses = losses['train']
#                 self.losses = torch.cat((self.losses, torch.cat(losses)))
#                 if self.visdom is not None:
#                     self.visdom.register((self.losses, torch.arange(self.losses.numel())), 'lineplot', 3, 'Training Loss')
#                 elif self.params.debug >= 3:
#                     plot_graph(self.losses, 10, title='Training loss')

#     def refine_target_box(self, backbone_feat, sample_pos, sample_scale, scale_ind, update_scale = True):
#         """Run the ATOM IoUNet to refine the target bounding box."""

#         # Initial box for refinement
#         init_box = self.get_iounet_box(self.pos, self.target_sz, sample_pos, sample_scale)

#         # Extract features from the relevant scale
#         iou_features = self.get_iou_features(backbone_feat)
#         iou_features = TensorList([x[scale_ind:scale_ind+1,...] for x in iou_features])

#         # Generate random initial boxes
#         init_boxes = init_box.view(1,4).clone()
#         if self.params.num_init_random_boxes > 0:
#             square_box_sz = init_box[2:].prod().sqrt()
#             rand_factor = square_box_sz * torch.cat([self.params.box_jitter_pos * torch.ones(2), self.params.box_jitter_sz * torch.ones(2)])

#             minimal_edge_size = init_box[2:].min()/3
#             rand_bb = (torch.rand(self.params.num_init_random_boxes, 4) - 0.5) * rand_factor
#             new_sz = (init_box[2:] + rand_bb[:,2:]).clamp(minimal_edge_size)
#             new_center = (init_box[:2] + init_box[2:]/2) + rand_bb[:,:2]
#             init_boxes = torch.cat([new_center - new_sz/2, new_sz], 1)
#             init_boxes = torch.cat([init_box.view(1,4), init_boxes])

#         # Optimize the boxes
#         output_boxes, output_iou = self.optimize_boxes(iou_features, init_boxes)

#         # Remove weird boxes
#         output_boxes[:, 2:].clamp_(1)
#         aspect_ratio = output_boxes[:,2] / output_boxes[:,3]
#         keep_ind = (aspect_ratio < self.params.maximal_aspect_ratio) * (aspect_ratio > 1/self.params.maximal_aspect_ratio)
#         output_boxes = output_boxes[keep_ind,:]
#         output_iou = output_iou[keep_ind]

#         # If no box found
#         if output_boxes.shape[0] == 0:
#             return

#         # Predict box
#         k = getattr(self.params, 'iounet_k', 5)
#         topk = min(k, output_boxes.shape[0])
#         _, inds = torch.topk(output_iou, topk)
#         predicted_box = output_boxes[inds, :].mean(0)
#         predicted_iou = output_iou.view(-1, 1)[inds, :].mean(0)

#         # Get new position and size
#         new_pos = predicted_box[:2] + predicted_box[2:] / 2
#         new_pos = (new_pos.flip((0,)) - (self.img_sample_sz - 1) / 2) * sample_scale + sample_pos
#         new_target_sz = predicted_box[2:].flip((0,)) * sample_scale
#         new_scale = torch.sqrt(new_target_sz.prod() / self.base_target_sz.prod())

#         self.pos_iounet = new_pos.clone()

#         if getattr(self.params, 'use_iounet_pos_for_learning', True):
#             self.pos = new_pos.clone()

#         self.target_sz = new_target_sz

#         if update_scale:
#             self.target_scale = new_scale


#     def optimize_boxes(self, iou_features, init_boxes):
#         # Optimize iounet boxes
#         output_boxes = init_boxes.view(1, -1, 4).to(self.params.device)
#         step_length = self.params.box_refinement_step_length
#         if isinstance(step_length, (tuple, list)):
#             step_length = torch.Tensor([step_length[0], step_length[0], step_length[1], step_length[1]], device=self.params.device).view(1,1,4)

#         for i_ in range(self.params.box_refinement_iter):
#             # forward pass
#             bb_init = output_boxes.clone().detach()
#             bb_init.requires_grad = True

#             outputs = self.net.bb_regressor.predict_iou(self.iou_modulation, iou_features, bb_init)

#             if isinstance(outputs, (list, tuple)):
#                 outputs = outputs[0]

#             outputs.backward(gradient = torch.ones_like(outputs))

#             # Update proposal
#             output_boxes = bb_init + step_length * bb_init.grad * bb_init[:, :, 2:].repeat(1, 1, 2)
#             output_boxes.detach_()

#             step_length *= self.params.box_refinement_step_decay

#         return output_boxes.view(-1,4).cpu(), outputs.detach().view(-1).cpu()