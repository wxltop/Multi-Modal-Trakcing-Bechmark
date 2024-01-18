# import torch
# ckpt_path = '/home/lz/PycharmProjects/MixFormer/models/mixformer_cvt_22k.pth.tar'
# ckpt = torch.load(ckpt_path, map_location='cpu')
# # print("debug")
# net_params = ckpt['net']
# state = {}
# for name, value in net_params.items():
#     if 'score_branch' in name or 'box_head' in name:
#         continue
#     new_name = name[9:]
#     state[new_name] = value
# ckpt['net'] = state
# torch.save(ckpt, '/home/lz/PycharmProjects/MixFormer/models//mixformer_cvt_22k_no_prefix.pth.tar')

# import pandas
# import os
# import numpy as np
# root = '/home/lz/Videos/DepthTrackTraining'
# seqs = [i for i in os.listdir(root) if os.path.isdir(os.path.join(root, i))]
# for seq in seqs:
#     print(seq)
#     bb_anno_file = os.path.join('/home/lz/Videos/DepthTrackTraining/{}'.format(seq), "groundtruth.txt")
#     gt = pandas.read_csv(bb_anno_file, delimiter=',', header=None, dtype=np.float32, na_filter=False, low_memory=False).values
#     tensorgt = torch.tensor(gt)
#     print("-----------------------------")
#     # print(gt)


# def read_fun(filename):
#     with open(filename, 'r') as f:
#         content = f.read().splitlines()
#     return content
#
# def write_fun(filename, str_list):
#     with open(filename, 'w') as w:
#         for i in str_list:
#             w.write(i+'\n')
#
# import os
# import shutil
# from os.path import join, isdir
#
# rgbdepth_root = '/home/lz/Videos/VOT2022RGBD/sequences'
# dst_root = '/home/lz/Videos/VOT2022D'
# seqs = [i for i in os.listdir(rgbdepth_root) if isdir(join(rgbdepth_root, i))]
# for seq in seqs[1:]:
#     print(f"===================> Now copy {seq} <===================")
#     seq_save_root = join(dst_root, seq)
#     if not os.path.exists(seq_save_root):
#         os.makedirs(seq_save_root)
#
#     src_seq_root = join(rgbdepth_root, seq)
#     seq_color = join(src_seq_root, 'depth')
#     shutil.copytree(seq_color, join(seq_save_root, 'depth'))
#
#     files = [i for i in os.listdir(src_seq_root) if not isdir(join(src_seq_root, i))]
#     files.remove('sequence')
#     for f in files:
#         shutil.copy(join(src_seq_root, f), seq_save_root)
#
#     c = read_fun(join(src_seq_root, 'sequence'))
#     write_fun(join(seq_save_root, 'sequence'), c[1:])

