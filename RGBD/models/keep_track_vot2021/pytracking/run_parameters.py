import os
import sys
import argparse
import importlib

env_path = os.path.join(os.path.dirname(__file__), '..')
if env_path not in sys.path:
    sys.path.append(env_path)

from pytracking.evaluation.running import run_dataset
from pytracking.evaluation import get_dataset, trackerlist


def run_experiment(tracker_name: str, parameter_file_name: str, dataset_name: str, seed_start: int, seed_stop: int,
                   debug=0, threads=0, save_time=True):
    """Run experiment.
    args:
        tracker_name: Tracker name in the tracker folder.
        parameter_file_name: parameter file name in the parameter folder.
        dataset_name: name of dataset to run
        seed_start: random seed from
        seed_stop: random seed to
        debug: Debug level.
        threads: Number of threads.
    """
    trackers = trackerlist(tracker_name, parameter_file_name, range(seed_start, seed_stop))
    dataset = get_dataset(dataset_name)

    print('Running:  {}  {} on {} from range({}, {})'.format(tracker_name, parameter_file_name,
                                                             dataset_name, seed_start, seed_stop))
    run_dataset(dataset, trackers, debug, threads, save_time=save_time, is_oxuva=('oxuva' in dataset_name))


def main():
    parser = argparse.ArgumentParser(description='Run tracker.')
    parser.add_argument('tracker_name', type=str, help='Tracker name.')
    parser.add_argument('parameter_file_name', type=str, help='Name of parameter file (not path).')
    parser.add_argument('dataset_name', type=str, help='Dataset name.')
    parser.add_argument('seed', type=int, help='Random seed (Start) if --seed-stop not specified it will be +1.')
    parser.add_argument('--seed-stop', type=int, default=-1, help='Random seed (Stop).')
    parser.add_argument('--save-time', action='store_true', help='Save timings.')
    parser.add_argument('--debug', type=int, default=0, help='Debug level.')
    parser.add_argument('--threads', type=int, default=0, help='Number of threads.')

    args = parser.parse_args()
    if args.seed_stop < args.seed:
        args.seed_stop = args.seed + 1

    run_experiment(args.tracker_name, args.parameter_file_name, args.dataset_name,
                   args.seed, args.seed_stop, args.debug, args.threads, args.save_time)


if __name__ == '__main__':
    main()
