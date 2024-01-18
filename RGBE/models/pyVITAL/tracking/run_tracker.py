import numpy as np
import os
import sys
import time
import argparse
import yaml, json
from PIL import Image

import matplotlib.pyplot as plt

import torch
import torch.utils.data as data
import torch.optim as optim

sys.path.insert(0, '.')
from modules.model import MDNet, BCELoss, set_optimizer
from modules.sample_generator import SampleGenerator
from modules.utils import overlap_ratio
from data_prov import RegionExtractor
from bbreg import BBRegressor
from gen_config import gen_config

sys.path.insert(0,'./gnet')
from gnet.g_init import NetG, set_optimizer_g
from gnet.g_pretrain import *

opts = yaml.safe_load(open('./tracking/options.yaml','r'))


def forward_samples(model, image_vis, image_event, samples, out_layer='conv3'):
    model.eval()
    extractor = RegionExtractor(image_vis, image_event, samples, opts)
    for i, regions in enumerate(extractor):
        regions_vis   = regions[0]
        regions_event = regions[1] 

        with torch.no_grad():
            feat_vis, feat_event = model(regions_vis.cuda(), regions_event.cuda(), out_layer=out_layer)
        if i==0:
            feats_vis = feat_vis.detach().clone() 
            feats_event = feat_event.detach().clone()
        else:
            feats_vis = torch.cat((feats_vis, feat_vis.detach().clone()), 0)
            feats_event = torch.cat((feats_event, feat_event.detach().clone()), 0)

    return feats_vis, feats_event 


def train(model, model_g, criterion, optimizer, pos_feats_vis, pos_feats_event, neg_feats_vis, neg_feats_event, maxiter, in_layer='fc4'):
    model.train()

    batch_pos = opts['batch_pos']
    batch_neg = opts['batch_neg']
    batch_test = opts['batch_test']
    batch_neg_cand = max(opts['batch_neg_cand'], batch_neg)

    pos_idx = np.random.permutation(pos_feats_vis.size(0))
    neg_idx = np.random.permutation(neg_feats_vis.size(0))
    while(len(pos_idx) < batch_pos * maxiter):
        pos_idx = np.concatenate([pos_idx, np.random.permutation(pos_feats_vis.size(0))])
    while(len(neg_idx) < batch_neg_cand * maxiter):
        neg_idx = np.concatenate([neg_idx, np.random.permutation(neg_feats_vis.size(0))])
    pos_pointer = 0
    neg_pointer = 0

    for i in range(maxiter):

        # select pos idx
        pos_next    = pos_pointer + batch_pos
        pos_cur_idx = pos_idx[pos_pointer:pos_next]
        pos_cur_idx = pos_feats_vis.new(pos_cur_idx).long()
        pos_pointer = pos_next

        # select neg idx
        neg_next = neg_pointer + batch_neg_cand
        neg_cur_idx = neg_idx[neg_pointer:neg_next]
        neg_cur_idx = neg_feats_vis.new(neg_cur_idx).long()
        neg_pointer = neg_next

        # create batch
        batch_pos_feats_vis   = pos_feats_vis[pos_cur_idx]
        batch_pos_feats_event = pos_feats_event[pos_cur_idx]
        if model_g is not None:
            batch_asdn_feats_vis = pos_feats_vis.index_select(0, pos_cur_idx)
            batch_asdn_feats_event = pos_feats_event.index_select(0, pos_cur_idx)

        batch_neg_feats_vis = neg_feats_vis[neg_cur_idx]
        batch_neg_feats_event = neg_feats_event[neg_cur_idx]

        # hard negative mining
        if batch_neg_cand > batch_neg:
            model.eval()
            for start in range(0, batch_neg_cand, batch_test):
                end = min(start + batch_test, batch_neg_cand)
                with torch.no_grad():
                    score, _ = model(batch_neg_feats_vis[start:end], batch_neg_feats_event[start:end], in_layer=in_layer)
                if start==0:
                    neg_cand_score = score.detach()[:, 1].clone()
                else:
                    neg_cand_score = torch.cat((neg_cand_score, score.detach()[:, 1].clone()), 0)

            _, top_idx = neg_cand_score.topk(batch_neg)
            batch_neg_feats_vis = batch_neg_feats_vis[top_idx]
            batch_neg_feats_event = batch_neg_feats_event[top_idx]
            model.train()

        if model_g is not None:
            model_g.eval()
            batch_asdn_feats = batch_asdn_feats_vis + batch_asdn_feats_event 
            res_asdn = model_g(batch_asdn_feats)
            model_g.train()
            num = res_asdn.size(0)
            mask_asdn = torch.ones(num, 512, 3, 3)
            res_asdn = res_asdn.view(num, 3, 3)
            for i in range(num):
                feat_ = res_asdn[i, :, :]
                featlist = feat_.view(1, 9).squeeze()
                feat_list = featlist.detach().cpu().numpy()
                idlist = feat_list.argsort()
                idxlist = idlist[:3]

                for k in range(len(idxlist)):
                    idx = idxlist[k]
                    row = idx // 3
                    col = idx % 3
                    mask_asdn[:, :, col, row] = 0
            mask_asdn = mask_asdn.view(mask_asdn.size(0), -1)
            if opts['use_gpu']:
                batch_asdn_feats = batch_asdn_feats.cuda()
                mask_asdn = mask_asdn.cuda()

            batch_asdn_feats_vis = batch_asdn_feats_vis * mask_asdn
            batch_asdn_feats_event = batch_asdn_feats_event * mask_asdn

        # forward
        if model_g is None:
            pos_score, _ = model(batch_pos_feats_vis, batch_pos_feats_event, in_layer=in_layer)
        else:
            pos_score, _ = model(batch_asdn_feats_vis, batch_asdn_feats_event, in_layer=in_layer)
        
        neg_score, _ = model(batch_neg_feats_vis, batch_neg_feats_event, in_layer=in_layer)

        # optimize
        loss = criterion(pos_score, neg_score)
        model.zero_grad()
        loss.backward()
        if 'grad_clip' in opts:
            torch.nn.utils.clip_grad_norm_(model.parameters(), opts['grad_clip'])
        optimizer.step()

        if model_g is not None:
            start = time.time()
            prob_k = torch.zeros(9)
            for k in range(9):
                row = k // 3
                col = k % 3

                model.eval()
                batch_vis = batch_pos_feats_vis.view(batch_pos, 512, 3, 3)
                batch_vis[:, :, col, row] = 0
                batch_vis = batch_vis.view(batch_vis.size(0), -1)

                batch_event = batch_pos_feats_event.view(batch_pos, 512, 3, 3)
                batch_event[:, :, col, row] = 0
                batch_event = batch_event.view(batch_event.size(0), -1)

                if opts['use_gpu']:
                    batch_event = batch_event.cuda()
                    batch_vis = batch_vis.cuda()

                prob = model(batch_vis, batch_event, in_layer='fc4', out_layer='fc6_softmax')[:, 1]
                model.train()

                prob_k[k] = prob.sum()

            _, idx = torch.min(prob_k, 0)
            idx = idx.item()
            row = idx // 3
            col = idx % 3

            optimizer_g = set_optimizer_g(model_g)
            labels = torch.ones(batch_pos, 1, 3, 3)
            labels[:, :, col, row] = 0

            batch_pos_feats_vis = batch_pos_feats_vis.view(batch_pos_feats_vis.size(0), -1)
            batch_pos_feats_event = batch_pos_feats_event.view(batch_pos_feats_event.size(0), -1)
            res = model_g(batch_pos_feats_vis + batch_pos_feats_event)
            labels = labels.view(batch_pos, -1)
            criterion_g = torch.nn.MSELoss(reduction='mean')
            loss_g_2 = criterion_g(res.float(), labels.cuda().float())
            model_g.zero_grad()
            loss_g_2.backward()
            optimizer_g.step()

            end = time.time()
            print('asdn objective %.3f, %.2f s' % (loss_g_2, end - start))


def run_vital(vis_img_list, event_img_list, init_bbox, gt=None, savefig_dir='', display=False):

    # Init bbox
    target_bbox = np.array(init_bbox)
    result = np.zeros((len(vis_img_list), 4))
    result_bb = np.zeros((len(vis_img_list), 4))
    result[0] = target_bbox
    result_bb[0] = target_bbox

    if gt is not None:
        overlap = np.zeros(len(vis_img_list))
        overlap[0] = 1

    # Init model
    model = MDNet(opts['model_path'])
    model_g = NetG()
    if opts['use_gpu']:
        model = model.cuda()
        model_g = model_g.cuda()

    # Init criterion and optimizer 
    criterion = BCELoss()
    criterion_g = torch.nn.MSELoss(reduction='mean')
    model.set_learnable_params(opts['ft_layers'])
    model_g.set_learnable_params(opts['ft_layers'])
    init_optimizer = set_optimizer(model, opts['lr_init'], opts['lr_mult'])
    update_optimizer = set_optimizer(model, opts['lr_update'], opts['lr_mult'])

    tic = time.time()
    # Load first image
    vis_image   = Image.open(vis_img_list[0]).convert('RGB')
    event_image = Image.open(event_img_list[0]).convert('RGB')

    # Draw pos/neg samples
    pos_examples = SampleGenerator('gaussian', vis_image.size, opts['trans_pos'], opts['scale_pos'])(
                        target_bbox, opts['n_pos_init'], opts['overlap_pos_init'])

    neg_examples = np.concatenate([
                    SampleGenerator('uniform', vis_image.size, opts['trans_neg_init'], opts['scale_neg_init'])(
                        target_bbox, int(opts['n_neg_init'] * 0.5), opts['overlap_neg_init']),
                    SampleGenerator('whole', vis_image.size)(
                        target_bbox, int(opts['n_neg_init'] * 0.5), opts['overlap_neg_init'])])
    neg_examples = np.random.permutation(neg_examples)

    # Extract pos/neg features
    pos_feats_vis, pos_feats_event = forward_samples(model, vis_image, event_image, pos_examples)
    neg_feats_vis, neg_feats_event = forward_samples(model, vis_image, event_image, neg_examples)

    # Initial training
    train(model, None, criterion, init_optimizer, pos_feats_vis, pos_feats_event, neg_feats_vis, neg_feats_event, opts['maxiter_init'])
    del init_optimizer, neg_feats_vis, neg_feats_event
    torch.cuda.empty_cache()
    g_pretrain(model, model_g, criterion_g, pos_feats_vis + pos_feats_event)
    torch.cuda.empty_cache()

    # Train bbox regressor
    bbreg_examples = SampleGenerator('uniform', vis_image.size, opts['trans_bbreg'], opts['scale_bbreg'], opts['aspect_bbreg'])(
                        target_bbox, opts['n_bbreg'], opts['overlap_bbreg'])
    bbreg_feats_vis, bbreg_feats_event = forward_samples(model, vis_image, event_image, bbreg_examples)
    bbreg = BBRegressor(vis_image.size)
    bbreg.train(bbreg_feats_vis + bbreg_feats_event, bbreg_examples, target_bbox)
    del bbreg_feats_vis, bbreg_feats_event
    torch.cuda.empty_cache()

    # Init sample generators for update
    sample_generator = SampleGenerator('gaussian', vis_image.size, opts['trans'], opts['scale'])
    pos_generator    = SampleGenerator('gaussian', vis_image.size, opts['trans_pos'], opts['scale_pos'])
    neg_generator    = SampleGenerator('uniform', vis_image.size, opts['trans_neg'], opts['scale_neg'])

    # Init pos/neg features for update
    neg_examples = neg_generator(target_bbox, opts['n_neg_update'], opts['overlap_neg_init'])
    neg_feats_vis, neg_feats_event = forward_samples(model, vis_image, event_image, neg_examples)
    pos_feats_all_vis = [pos_feats_vis]
    neg_feats_all_vis = [neg_feats_vis]
    pos_feats_all_event = [pos_feats_event]
    neg_feats_all_event = [neg_feats_event]

    spf_total = time.time() - tic

    # Display
    savefig = savefig_dir != ''
    if display or savefig:
        dpi = 80.0
        figsize = (vis_image.size[0] / dpi, vis_image.size[1] / dpi)

        fig = plt.figure(frameon=False, figsize=figsize, dpi=dpi)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        im = ax.imshow(vis_image, aspect='auto')

        if gt is not None:
            gt_rect = plt.Rectangle(tuple(gt[0, :2]), gt[0, 2], gt[0, 3],
                                    linewidth=3, edgecolor="#00ff00", zorder=1, fill=False)
            ax.add_patch(gt_rect)

        rect = plt.Rectangle(tuple(result_bb[0, :2]), result_bb[0, 2], result_bb[0, 3],
                             linewidth=3, edgecolor="#ff0000", zorder=1, fill=False)
        ax.add_patch(rect)

        if display:
            plt.pause(.01)
            plt.draw()
        if savefig:
            fig.savefig(os.path.join(savefig_dir, '0000.jpg'), dpi=dpi)

    # Main loop
    for i in range(1, len(vis_img_list)):

        tic = time.time()
        # Load image
        vis_image = Image.open(vis_img_list[i]).convert('RGB')
        event_image = Image.open(event_img_list[i]).convert('RGB')

        # Estimate target bbox
        samples = sample_generator(target_bbox, opts['n_samples'])
        sample_scores, _ = forward_samples(model, vis_image, event_image, samples, out_layer='fc6')

        top_scores, top_idx = sample_scores[:, 1].topk(5)
        top_idx = top_idx.cpu()
        target_score = top_scores.mean()
        target_bbox = samples[top_idx]
        if top_idx.shape[0] > 1:
            target_bbox = target_bbox.mean(axis=0)
        success = target_score > 0
        
        # Expand search area at failure
        if success:
            sample_generator.set_trans(opts['trans'])
        else:
            sample_generator.expand_trans(opts['trans_limit'])

        # Bbox regression
        if success:
            bbreg_samples = samples[top_idx]
            if top_idx.shape[0] == 1:
                bbreg_samples = bbreg_samples[None,:]
            bbreg_feats_vis, bbreg_feats_event = forward_samples(model, vis_image, event_image, bbreg_samples)
            bbreg_samples = bbreg.predict(bbreg_feats_vis + bbreg_feats_event, bbreg_samples)
            bbreg_bbox = bbreg_samples.mean(axis=0)
        else:
            bbreg_bbox = target_bbox

        # Save result
        result[i]    = target_bbox
        result_bb[i] = bbreg_bbox

        # Data collect
        if success:
            pos_examples = pos_generator(target_bbox, opts['n_pos_update'], opts['overlap_pos_update'])
            pos_feats_vis, pos_feats_event = forward_samples(model, vis_image, event_image, pos_examples)
            pos_feats_all_vis.append(pos_feats_vis) 
            pos_feats_all_event.append(pos_feats_event) 

            if len(pos_feats_all_vis) > opts['n_frames_long']:
                del pos_feats_all_vis[0] 
                del pos_feats_all_event[0]

            neg_examples = neg_generator(target_bbox, opts['n_neg_update'], opts['overlap_neg_update'])
            neg_feats_vis, neg_feats_event = forward_samples(model, vis_image, event_image, neg_examples)
            neg_feats_all_vis.append(neg_feats_vis)
            neg_feats_all_event.append(neg_feats_event)
            if len(neg_feats_all_vis) > opts['n_frames_short']:
                del neg_feats_all_vis[0]
                del neg_feats_all_event[0]

        # Short term update
        if not success:
            nframes = min(opts['n_frames_short'], len(pos_feats_all_vis))
            pos_data_vis = torch.cat(pos_feats_all_vis[-nframes:], 0)
            neg_data_vis = torch.cat(neg_feats_all_vis, 0)
            pos_data_event = torch.cat(pos_feats_all_event[-nframes:], 0)
            neg_data_event = torch.cat(neg_feats_all_event, 0)

            train(model, None, criterion, update_optimizer, pos_data_vis, pos_data_event, neg_data_vis, neg_data_event, opts['maxiter_update'])

        # Long term update
        elif i % opts['long_interval'] == 0:
            pos_data_vis = torch.cat(pos_feats_all_vis, 0)
            neg_data_vis = torch.cat(neg_feats_all_vis, 0)
            pos_data_event = torch.cat(pos_feats_all_event, 0)
            neg_data_event = torch.cat(neg_feats_all_event, 0)

            train(model, model_g, criterion, update_optimizer, pos_data_vis, pos_data_event, neg_data_vis, neg_data_event, opts['maxiter_update'])

        torch.cuda.empty_cache()
        spf = time.time() - tic
        spf_total += spf

        # Display
        if display or savefig:
            im.set_data(vis_image)

            if gt is not None:
                gt_rect.set_xy(gt[i, :2])
                gt_rect.set_width(gt[i, 2])
                gt_rect.set_height(gt[i, 3])

            rect.set_xy(result_bb[i, :2])
            rect.set_width(result_bb[i, 2])
            rect.set_height(result_bb[i, 3])

            if display:
                plt.pause(.01)
                plt.draw()
            if savefig:
                fig.savefig(os.path.join(savefig_dir, '{:04d}.jpg'.format(i)), dpi=dpi)

        if gt is None:
            print('Frame {:d}/{:d}, Score {:.3f}, Time {:.3f}'
                .format(i + 1, len(vis_img_list), target_score, spf))
        else:
            overlap[i] = overlap_ratio(gt[i], result_bb[i])[0]
            print('Frame {:d}/{:d}, Overlap {:.3f}, Score {:.3f}, Time {:.3f}'
                .format(i + 1, len(vis_img_list), overlap[i], target_score, spf))

    if gt is not None:
        print('meanIOU: {:.3f}'.format(overlap.mean()))
    fps = len(vis_img_list) / spf_total
    return result, result_bb, fps


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--seq', default='', help='input seq')
    parser.add_argument('-j', '--json', default='', help='input json')
    parser.add_argument('-f', '--savefig', action='store_true')
    parser.add_argument('-d', '--display', action='store_true')

    # args = parser.parse_args()
    # assert args.seq != '' or args.json != ''

    np.random.seed(0)
    torch.manual_seed(0)

    # # Generate sequence config
    # img_list, init_bbox, gt, savefig_dir, display, result_path = gen_config(args)

    seq_home = '/home/wangxiao/Documents/rgb_event_tracking_benchmark/visEvent_dataset/test'
    result_home = './results'
    result_dir = os.path.join(result_home)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    videofiles = os.listdir(seq_home) 
    for vidIDX in range(len(videofiles)): 
        
        seq_name = videofiles[vidIDX] 
        vis_img_dir = os.path.join(seq_home, seq_name, 'vis_imgs')
        event_img_dir = os.path.join(seq_home, seq_name, 'event_imgs')
        gt_path = os.path.join(seq_home, seq_name, 'groundtruth.txt')

        vis_img_list = os.listdir(vis_img_dir)
        vis_img_list.sort()
        vis_img_list = [os.path.join(vis_img_dir, x) for x in vis_img_list]

        event_img_list = os.listdir(event_img_dir)
        event_img_list.sort()
        event_img_list = [os.path.join(event_img_dir, x) for x in event_img_list]

        with open(gt_path) as f:
            gt = np.loadtxt((x.replace('\t',',') for x in f), delimiter=',')
        init_bbox = gt[0]

        # Run tracker
        result, result_bb, fps = run_vital(vis_img_list, event_img_list, init_bbox, gt=gt, display=False)  

        # # Save result
        # res = {}
        # res['res'] = result_bb.round().tolist()
        # res['type'] = 'rect'
        # res['fps'] = fps
        # json.dump(res, open(result_path, 'w'), indent=2)




