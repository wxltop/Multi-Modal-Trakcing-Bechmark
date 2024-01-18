import os
import sys
import numpy as np
import cv2
import json

env_path = os.path.join(os.path.dirname(__file__), '..')
if env_path not in sys.path:
    sys.path.append(env_path)


def eval(out_res, label_res):
    measure_per_frame = []
    for _pred, _gt in zip(out_res, label_res):
        measure_per_frame.append(iou(_pred, _gt))
    return np.mean(measure_per_frame)

def not_exist(pred):
    return (pred[0] == 0 and pred[1] == 0 and pred[2] == 0 and pred[1] == 0)#(len(pred) == 1 and pred[0] == 0) or len(pred) == 0

def iou(bbox1, bbox2):
    """
    Calculates the intersection-over-union of two bounding boxes.
    Args:
        bbox1 (numpy.array, list of floats): bounding box in format x,y,w,h.
        bbox2 (numpy.array, list of floats): bounding box in format x,y,w,h.
    Returns:
        int: intersection-over-onion of bbox1, bbox2
    """
    bbox1 = [float(x) for x in bbox1]
    bbox2 = [float(x) for x in bbox2]

    (x0_1, y0_1, w1_1, h1_1) = bbox1
    (x0_2, y0_2, w1_2, h1_2) = bbox2
    x1_1 = x0_1 + w1_1
    x1_2 = x0_2 + w1_2
    y1_1 = y0_1 + h1_1
    y1_2 = y0_2 + h1_2
    # get the overlap rectangle
    overlap_x0 = max(x0_1, x0_2)
    overlap_y0 = max(y0_1, y0_2)
    overlap_x1 = min(x1_1, x1_2)
    overlap_y1 = min(y1_1, y1_2)

    # check if there is an overlap
    if overlap_x1 - overlap_x0 <= 0 or overlap_y1 - overlap_y0 <= 0:
        return 0

    # if yes, calculate the ratio of the overlap to each ROI size and the unified size
    size_1 = (x1_1 - x0_1) * (y1_1 - y0_1)
    size_2 = (x1_2 - x0_2) * (y1_2 - y0_2)
    size_intersection = (overlap_x1 - overlap_x0) * (overlap_y1 - overlap_y0)
    size_union = size_1 + size_2 - size_intersection

    return size_intersection / size_union

video_root = '/data/Disk_A/xuefeng_data/TEST/'
results_root = '/data/Disk_B/xuefeng/trackers/Stark/tracking_results/stark_s/rgbd/'
# results_root = '/data/Disk_C/xuefeng_space/Stark/tracking_results/stark_s/baseline/'

video_files = sorted(os.listdir(video_root))

vis = True
vid = 0
overall_performance = []

for video_name in video_files:

        video_path = os.path.join(video_root, video_name)
        label_file = os.path.join(video_path, 'groundtruth_rect.txt')
        res_file = os.path.join(results_root, video_name) + '.txt'

        # load ground-truth
        ground_truth_rect = np.loadtxt(label_file, delimiter=',', dtype=np.float32)
        results = np.loadtxt(res_file, delimiter='\t', dtype=np.float32)

        mixed_measure = eval(results, ground_truth_rect)

        overall_performance.append(mixed_measure)


        vid += 1
        print(' %d) %s  %.03f' % (vid, video_name, mixed_measure))

print('[Overall] Mixed Measure 1: %.03f\n' % (np.mean(overall_performance)))





