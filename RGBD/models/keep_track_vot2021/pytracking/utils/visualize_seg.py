import sys
import numpy as np
from pathlib import Path
from shutil import copy2
from PIL import Image
import os
from os.path import isfile, join
import glob
import cv2

davis_palette = np.repeat(np.expand_dims(np.arange(0,256), 1), 3, 1).astype(np.uint8)
davis_palette[:22, :] = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                         [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                         [64, 0, 0], [191, 0, 0], [64, 128, 0], [191, 128, 0],
                         [64, 0, 128], [191, 0, 128], [64, 128, 128], [191, 128, 128],
                         [0, 64, 0], [128, 64, 0], [0, 191, 0], [128, 191, 0],
                         [0, 64, 128], [128, 64, 128]]


def indexed_to_rgb(array, color_palette=None):

    if color_palette is None:
        color_palette = davis_palette

    im = Image.fromarray(array.squeeze(), 'P')
    im.putpalette(color_palette.ravel())
    im = np.array(im.convert('RGB'))
    return im

def remap_colors(im):

    palette = np.array([[0,0,0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1]], dtype=np.uint8)
    im = indexed_to_rgb(im, palette * 200)
    return im

def prepare_labels(seq_name, dst_path, label_path, image_path, sz):

    dpath = dst_path / seq_name
    dpath.mkdir(exist_ok=True, parents=True)

    label_dst=os.listdir(label_path / seq_name)
    frames=[x.split('.')[0] for x in label_dst]
    alpha = 0.4
    print(frames)
    for frame in frames:
        src_file = label_path / seq_name / ("%s.png" % frame)
        jpg_file = image_path / seq_name / ("%s.jpg" % frame)
        dst_file = dpath / ("%s.jpg" % frame)

        mask = np.array(Image.open(src_file))
        im = remap_colors(mask)
        print(im.shape)
        jpeg = np.array(Image.open(jpg_file))
        foreground = jpeg * alpha + (1 - alpha) * im
        jpeg[mask>0,:] = foreground[mask>0,:]
        if not sz is None:
            jpeg = cv2.resize(jpeg, dsize=sz, interpolation=cv2.INTER_CUBIC)
        Image.fromarray(jpeg.astype(np.uint8)).save(dst_file, "JPEG", quality=80)

def bbox_to_mask(bbox, l, mask):
    #mask = np.zeros(sz)
    #for i, bb in enumerate(bbox):
    x1, y1, w, h = bbox#list(map(int, bb))
    x1 = int(x1+0.5)
    y1 = int(y1+0.5)
    h = int(h+0.5)
    w = int(w+0.5)
    mask[y1:(y1+h), x1:(x1+w)] = l
    return mask


def mask_to_bb(gt_mask):
    labels = np.unique(gt_mask)
    mask = np.zeros(gt_mask.shape, dtype=np.uint8)
    for l in labels:
        m = gt_mask == l
        mx = m.sum(axis=-2).nonzero()[0]
        my = m.sum(axis=-1).nonzero()[0]
        bb = [mx.min(), my.min(), mx.max()-mx.min(), my.max()-my.min()]# if (len(mx) > 0 and len(my) > 0) else [0, 0, 0, 0]
        mask = bbox_to_mask(bb, l, mask)

    return mask

def prepare_labels_box(seq_name, dst_path, label_path, image_path, gt_anno_path, sz):

    dpath = dst_path / seq_name
    dpath.mkdir(exist_ok=True, parents=True)

    label_dst=os.listdir(label_path / seq_name)
    frames=[x.split('.')[0] for x in label_dst]
    frames.sort()

    alpha = 0.4
    gt_mask = np.array(Image.open( gt_anno_path / seq_name / ("%s.png" % frames[0])))
    gt_bbox_mask = mask_to_bb(gt_mask)

    jpg_file = image_path / seq_name / ("%s.jpg" % frames[0])
    jpeg = np.array(Image.open(jpg_file))
    im = remap_colors(gt_bbox_mask)
    foreground = jpeg * alpha + (1 - alpha) * im
    jpeg[gt_bbox_mask > 0, :] = foreground[gt_bbox_mask > 0, :]

    dst_file = dpath / ("box_init%s.jpg" % frames[0])
    if not sz is None:
        jpeg = cv2.resize(jpeg, dsize=sz, interpolation=cv2.INTER_CUBIC)
    Image.fromarray(jpeg.astype(np.uint8)).save(dst_file, "JPEG", quality=80)

    for frame in frames:
        src_file = label_path / seq_name / ("%s.png" % frame)
        jpg_file = image_path / seq_name / ("%s.jpg" % frame)
        dst_file = dpath / ("%s.jpg" % frame)

        mask = np.array(Image.open(src_file))

        im = remap_colors(mask)
        print(im.shape)
        jpeg = np.array(Image.open(jpg_file))
        foreground = jpeg * alpha + (1 - alpha) * im
        jpeg[mask>0,:] = foreground[mask>0,:]
        if not sz is None:
            jpeg = cv2.resize(jpeg, dsize=sz, interpolation=cv2.INTER_CUBIC)
        Image.fromarray(jpeg.astype(np.uint8)).save(dst_file, "JPEG", quality=80)

if __name__ == '__main__':
    #super_awesome_net_000
    #92c46be756
    #/home/felja34/data/datasets/YouTubeVOS/2019/train_all_frames/JPEGImages
#b05faf54f7
    #sequences = ['ff14721af5', '00f88c4f0a',
    #             '3f99366076', '9c693f291b', '39bce09d8d', '67e397b1f2', '188cb4e03d',
    #             '6a75316e99', '7fb4f14054', '8e2e5af6a8', '9a38b8e463', '9ce299a510',
    #             '9fd2d2782b', '30fe0ed0ce', '45dc90f558', '63a68c6741', '77bec90101']
    sequences = ['bike-packing', 'blackswan', 'bmx-trees', 'drift-chicane', 'drift-straight', 'dance-twirl', 'goat', 'kite-surf',
                'soapbox', 'motocross-jump', 'judo', 'horsejump-high']
    sz = (854, 480)
    for s in sequences:
        results_path = Path("/home/felja34/data/davis_results/dolf_test1_lr2_nfilt32_ytvosjj_1_3_sw_mem_r2_det_fix_nobn_res50_final_conv_flip_aug_dv_lovashz_bugflip_mrcnnweights_ftvid_l234_mgpu_ft3_onlydv_lr2_ep15_var_000")

        #results_path=Path("/home/felja34/data/segmentation_results/dimp_vos_box/dolf_awesome_net_swfeat_nobn_mrcnn50_box_box_cat_evenmorecoco_lessaug2_2block_nobn_paper_075")
        #results_path = Path("/home/felja34/data/ytvos18_sub2")
        experiment = "super_awesome_net_000"#"super_awesome_net_006"
        seq_name = s
        dst_path = Path("/home/felja34/data/vis_LW2L_supplement2/")
        label_path = results_path
        #image_path =Path("/home/felja34/data/datasets/YouTubeVOS/2019/train_all_frames/JPEGImages") #Path("/home/felja34/data/datasets/DAVIS/JPEGImages/480p/")
        #image_path = Path("/home/felja34/data/datasets/YouTubeVOS/2018/valid_all_frames/JPEGImages")
        #gt_anno_path = Path('/home/felja34/data/datasets/YouTubeVOS/2018/valid/Annotations/')
        #gt_anno_path = Path('/home/felja34/data/datasets/DAVIS/Annotations/480p')
        image_path = Path("/home/felja34/data/datasets/DAVIS/JPEGImages/480p/")
        #prepare_labels_box(seq_name=seq_name, dst_path=dst_path, label_path=label_path, image_path=image_path,
        #                   gt_anno_path=gt_anno_path, sz=None)

        prepare_labels(seq_name, dst_path, label_path, image_path, sz)