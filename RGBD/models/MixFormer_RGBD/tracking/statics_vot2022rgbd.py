import os
from glob import glob

dataset_dir = '/home/lz/Videos/VOT2022RGBD/sequences'
seq_dirs = sorted([d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))])

total_frame_num = 0
min_frame_num = 100000
max_frame_num = 0
average_frame_num = 0
seq_num = 0
for seq_dir in seq_dirs:
    seq = os.path.join(dataset_dir, seq_dir, 'color')
    seq_frame_num = len(glob(os.path.join(seq, '*.jpg')))
    total_frame_num += seq_frame_num
    if seq_frame_num < min_frame_num:
        min_frame_num = seq_frame_num

    if seq_frame_num > max_frame_num:
        max_frame_num = seq_frame_num
    seq_num += 1

print(f'total_frame_num: {total_frame_num}')                # 80741
print(f'min_frame_num: {min_frame_num}')                    # 31
print(f'max_frame_num: {max_frame_num}')                    # 2496
print(f'average_frame_num: {total_frame_num/seq_num}')      # 635