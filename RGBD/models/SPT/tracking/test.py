import os
import sys
import argparse

prj_path = os.path.join(os.path.dirname(__file__), '..')
if prj_path not in sys.path:
    sys.path.append(prj_path)

from lib.test.evaluation import get_dataset
from lib.test.evaluation.running import run_dataset
from lib.test.evaluation.tracker import Tracker

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
def run_tracker(tracker_name, tracker_param, run_id=None, dataset_name='otb', sequence=None, debug=0, threads=0,
                num_gpus=8):
    """Run tracker on sequence or dataset.
    args:
        tracker_name: Name of tracking method.
        tracker_param: Name of parameter file.
        run_id: The run id.
        dataset_name: Name of dataset (otb, nfs, uav, tpl, vot, tn, gott, gotv, lasot).
        sequence: Sequence number or name.
        debug: Debug level.
        threads: Number of threads.
    """

    dataset = get_dataset(dataset_name)

    if sequence is not None:
        dataset = [dataset[sequence]]

    trackers = [Tracker(tracker_name, tracker_param, dataset_name, run_id)]

    run_dataset(dataset, trackers, debug, threads, num_gpus=num_gpus)


def main():
    # parser = argparse.ArgumentParser(description='Run tracker on sequence or dataset.')
    # parser.add_argument('tracker_name', type=str, help='Name of tracking method.')
    # parser.add_argument('tracker_param', type=str, help='Name of config file.')
    # parser.add_argument('--runid', type=int, default=None, help='The run id.')
    # parser.add_argument('--dataset_name', type=str, default='otb', help='Name of dataset (otb, nfs, uav, tpl, vot, tn, gott, gotv, lasot).')
    # parser.add_argument('--sequence', type=str, default=None, help='Sequence number or name.')
    # parser.add_argument('--debug', type=int, default=0, help='Debug level.')
    # parser.add_argument('--threads', type=int, default=0, help='Number of threads.')
    # parser.add_argument('--num_gpus', type=int, default=8)
    # args = parser.parse_args()

    tracker_name = 'stark_s'
    tracker_param = 'rgbd'
    runid = 1
    dataset_name = 'cdtb'
    sequence = None
    debug = 0
    threads = 0
    num_gpus = 1

    try:
        seq_name = int(sequence)
    except:
        seq_name = sequence

    # run_tracker(args.tracker_name, args.tracker_param, args.runid, args.dataset_name, seq_name, args.debug,
    #             args.threads, num_gpus=args.num_gpus)
    run_tracker(tracker_name, tracker_param, runid, dataset_name, seq_name, debug, threads, num_gpus=num_gpus)


if __name__ == '__main__':
    main()
