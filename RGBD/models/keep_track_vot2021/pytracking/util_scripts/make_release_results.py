import os
import sys
from pathlib import Path

env_path = os.path.join(os.path.dirname(__file__), '..')
if env_path not in sys.path:
    sys.path.append(env_path)

from pytracking.evaluation import get_dataset
from shutil import copyfile


def main():
    datasets = [get_dataset('otb'), get_dataset('nfs')]

    base_path = '/home/goutam/data/tracking_results/kys/rel/res50_rel_motion_trdt05_occ_fix_win_confentr_tnf05_offset_winbu'
    output_path = '/home/goutam/data/tracking_results/kys/rel_clean/'
    new_tracker_param_name = 'default'
    num_runs = 5

    for d in datasets:
        for run_id in range(num_runs):
            os.makedirs(os.path.join(output_path, '{}_{:03d}'.format(new_tracker_param_name, run_id)), exist_ok=True)

            for seq in d:
                copyfile(os.path.join(base_path + '_{:03d}'.format(run_id), seq.name + '.txt'),
                         os.path.join(output_path, '{}_{:03d}'.format(new_tracker_param_name, run_id), seq.name + '.txt'))

    print('Done')


if __name__ == '__main__':
    main()
