import torch
import os
import sys

env_path = os.path.join(os.path.dirname(__file__), '../..')
if env_path not in sys.path:
    sys.path.append(env_path)

from pytracking.evaluation.data import Sequence
from pytracking.evaluation.datasets import get_dataset
from pytracking.evaluation.environment import env_settings
from ltr.data.image_loader import imread_indexed
from pytracking.evaluation import Tracker, get_dataset, trackerlist
from pathlib import Path
from collections import OrderedDict as odict
import pytracking.analysis.vos_utils as utils
import glob


def find_missing_ytvos_sequences(tracker, param, run_id):
    dset = get_dataset('yt2019_jjval')

    env = env_settings()
    results_path = '{}/{}/{}_{:03d}'.format(env.segmentation_path, tracker, param, run_id)
    trk_results_path = '{}/{}/{}_{:03d}'.format(env.results_path, tracker, param, run_id)
    for j, sequence in enumerate(dset):
        # Load all frames
        frames = sequence.ground_truth_seg

        success = True
        for f in frames:
            if f is None:
                continue

            file = Path(f)
            try:
                _ = imread_indexed(os.path.join(results_path, sequence.name, file.name))
            except:
                success = False
                break

        if not success:
            print('{}/{}*.txt'.format(trk_results_path, sequence.name))
            for filename in glob.glob('{}/{}*.txt'.format(trk_results_path, sequence.name)):
                os.remove(filename)

if __name__ == '__main__':
    find_missing_ytvos_sequences('dimp_vos', 'dolf_test1_lr2_nfilt32_ytvosjj_3_3_sw_mem_lr01_thresh', 0)
    find_missing_ytvos_sequences('dimp_vos', 'dolf_test1_lr2_nfilt32_ytvosjj_3_3_sw_mem_lr01_sk1', 0)

