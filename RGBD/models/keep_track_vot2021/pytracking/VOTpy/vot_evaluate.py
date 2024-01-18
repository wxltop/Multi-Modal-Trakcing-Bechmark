import importlib
import logging
import os
import sys
import argparse

env_path = os.path.join(os.path.dirname(__file__), '../..')
if env_path not in sys.path:
    sys.path.append(env_path)

from pytracking.evaluation.environment import env_settings
from pytracking.VOTpy.utils.generate_tracker_file import generate_tracker_file
# from vot.tracker import load_trackers, TrackerException
# sys.path.append('../../../toolkit')
sys.path.append('/home/chmayer/github/toolkit')

import vot.utilities.cli as vot_cli


class Config:
    def __init__(self, trackers, workspace, registry, debug=False):
        self.trackers = trackers
        self.workspace = workspace
        self.debug = debug
        self.registry = registry
        self.force = False
        self.persist = False


def vot_evaluate(experiment_module, experiment_name, workspace_path=None, debug=False):

    expr_module = importlib.import_module('pytracking.VOTpy.experiments.{}'.format(experiment_module))
    expr_func = getattr(expr_module, experiment_name)
    trackers = expr_func()

    # Generate tracker file
    tracker_labels, tracker_file_path = generate_tracker_file(trackers, debug)

    logger = logging.getLogger("vot")
    logger.addHandler(logging.StreamHandler())

    if workspace_path is None:
        env = env_settings()
        vot_ws_path = env.vot2020_ws_path
    else:
        vot_ws_path = workspace_path

    registry = [tracker_file_path, ]
    config = Config(tracker_labels, vot_ws_path, registry, debug)

    vot_cli.do_evaluate(config)


def main():
    parser = argparse.ArgumentParser(description='VOT Evaluate')

    parser.add_argument('experiment_module', type=str, help='Name of experiment module in the experiments/ folder.')
    parser.add_argument('experiment_name', type=str, help='Name of the experiment function.')

    parser.add_argument('--workspace_path', type=str, default=None, help='Path to workspace dir.')
    parser.add_argument('--debug', dest='debug', action='store_true', help='Debug')
    parser.set_defaults(debug=False)

    args = parser.parse_args()

    vot_evaluate(args.experiment_module, args.experiment_name, args.workspace_path, args.debug)


if __name__ == '__main__':
    main()
