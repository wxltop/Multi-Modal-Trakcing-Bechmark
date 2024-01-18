import os
import hashlib
from pytracking.evaluation.environment import env_settings
from pytracking.evaluation import trackerlist


def get_tracker_name(tracker):
    if tracker.display_name is None:
        if tracker.run_id is not None:
            return '{}_{}_{:03d}'.format(tracker.name, tracker.parameter_name, tracker.run_id)
        else:
            return '{}_{}'.format(tracker.name, tracker.parameter_name)
    else:
        return tracker.display_name


def generate_file_id(tracker_file):
    return int(hashlib.sha256(tracker_file.encode('utf-8')).hexdigest(), 16) % (10 ** 8)


def generate_tracker_file(trackers, debug=False):
    pytracking_path = os.path.join(os.path.dirname(__file__), '../../..')
    pytracking_path = os.path.abspath(pytracking_path)
    tracker_file = ''

    tracker_labels = []
    for trk in trackers:
        tracker_name = get_tracker_name(trk)
        tracker_labels.append(tracker_name)
        tracker_file = '{}[{}]\n' \
                       'label = {}\n' \
                       'protocol = traxpython\n\n' \
                       "command = import pytracking.run_vot as run_vot; run_vot.run_vot2020('{}', '{}', {}, {})\n\n" \
                       'paths = {}\n\n'.format(tracker_file, tracker_name,
                                               tracker_name, trk.name,
                                               trk.parameter_name,
                                               trk.run_id, int(debug), pytracking_path)

    env = env_settings()
    ws_path = env.vot2020_ws_path
    tracker_file_dir = os.path.join(ws_path, 'tracker_files')
    if not os.path.exists(tracker_file_dir):
        os.mkdir(tracker_file_dir)

    filename = 'trackers_{:d}.ini'.format(generate_file_id(tracker_file))
    filepath = os.path.join(tracker_file_dir, filename)
    with open(filepath, "w") as text_file:
        print(tracker_file, file=text_file)

    return tracker_labels, filepath


if __name__ == '__main__':
    trackers = trackerlist('atom_cvpr19', 'default', range(0, 5))
    generate_tracker_file(trackers)