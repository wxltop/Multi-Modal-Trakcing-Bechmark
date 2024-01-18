import numpy as np
import os
import zipfile
import shutil
from pytracking.evaluation.environment import env_settings


def pack_tracking_results(base_path, output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    trackers = os.listdir(base_path)

    for t in trackers:
        runfiles = os.listdir(base_path + '/' + t)

        for r in runfiles:
            save_path = output_path + '/' + t
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            shutil.make_archive(save_path + '/' + r, 'zip', base_path + '/' + t + '/' + r)


def main():
    base_path = '/home/goutam/data/tracking_results/kys/rel_clean/'
    output_path = '/home/goutam/data/tracking_results/kys/rel_clean_packed'
    pack_tracking_results(base_path, output_path)


if __name__ == '__main__':
    main()
