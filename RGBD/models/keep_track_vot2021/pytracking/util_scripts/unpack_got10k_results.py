import numpy as np
import os
import zipfile
import shutil
from pytracking.evaluation.environment import env_settings


def unpack_got10k_results(packed_results_path, tracker_name, param_name, output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for i in range(1,181):
        seq_name = 'GOT-10k_Test_{:06d}'.format(i)

        seq_packed_path = '{}/{}'.format(packed_results_path, seq_name)

        for run_id in range(3):
            res = np.loadtxt('{}/{}_{:03d}.txt'.format(seq_packed_path, seq_name, run_id+1), dtype=np.float64, delimiter=',')
            times = np.loadtxt('{}/{}_time.txt'.format(seq_packed_path, seq_name), dtype=np.float64)

            if not os.path.exists('{}/{}/{}_{:03d}'.format(output_path, tracker_name, param_name, run_id)):
                os.makedirs('{}/{}/{}_{:03d}'.format(output_path, tracker_name, param_name, run_id))
            np.savetxt('{}/{}/{}_{:03d}/{}.txt'.format(output_path, tracker_name, param_name, run_id, seq_name), res, delimiter='\t', fmt='%d')
            np.savetxt('{}/{}/{}_{:03d}/{}_time.txt'.format(output_path, tracker_name, param_name, run_id, seq_name), times, fmt='%f')

def main():
    packed_results_path = '/home/goutam/Desktop/GOT-10k-20200419T163824Z-001/GOT-10k'
    tracker_name = 'atom'
    param_name = 'default'
    output_path = '/home/goutam/Desktop/DiMP-18-20200419T141649Z-001/DiMP-18/got_unpacked'

    unpack_got10k_results(packed_results_path, tracker_name, param_name, output_path)


if __name__ == '__main__':
    main()
