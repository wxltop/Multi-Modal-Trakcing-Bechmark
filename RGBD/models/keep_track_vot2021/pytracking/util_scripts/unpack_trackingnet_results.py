import numpy as np
import os
import zipfile
import shutil
from pytracking.evaluation.environment import env_settings
from pytracking.evaluation.datasets import get_dataset


def unpack_trackingnet_results(packed_results_path, tracker_name, param_name, run_id, output_path):
    """ Packs trackingnet results into a zip folder which can be directly uploaded to the evaluation server. The packed
    file is saved in the folder env_settings().tn_packed_results_path

    args:
        tracker_name - name of the tracker
        param_name - name of the parameter file
        run_id - run id for the tracker
        output_name - name of the packed zip file
    """

    tn_dataset = get_dataset('trackingnet')

    for seq in tn_dataset:
        seq_name = seq.name

        if run_id is None:
            seq_results_path = '{}/{}/{}/{}.txt'.format(output_path, tracker_name, param_name, seq_name)
        else:
            seq_results_path = '{}/{}/{}_{:03d}/{}.txt'.format(output_path, tracker_name, param_name, run_id, seq_name)

        results = np.loadtxt('{}/{}.txt'.format(packed_results_path, seq_name), delimiter=',', dtype=np.float64)

        if not os.path.exists('{}/{}/{}_{:03d}'.format(output_path, tracker_name, param_name, run_id)):
            os.makedirs('{}/{}/{}_{:03d}'.format(output_path, tracker_name, param_name, run_id))

        np.savetxt(seq_results_path, results, delimiter='\t', fmt='%d')


def main():
    tracker_name = 'atom'
    param_name = 'default'
    run_id = 0
    output_path = '/home/goutam/Desktop/DiMP-18-20200419T141649Z-001/DiMP-18/tn_packed'
    packed_results_path = '/home/goutam/Desktop/TrackingNet-20200419T170746Z-001/TrackingNet'
    unpack_trackingnet_results(packed_results_path, tracker_name, param_name, run_id=run_id, output_path=output_path)


if __name__ == '__main__':
    main()
