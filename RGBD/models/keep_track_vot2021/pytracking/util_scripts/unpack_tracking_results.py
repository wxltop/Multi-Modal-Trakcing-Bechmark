import os
import shutil
from pytracking.evaluation.environment import env_settings


def unpack_tracking_results(packed_results_path, output_path=None):
    """
    Unpacks zipped benchmark results. The directory 'packed_results_path' should have the following structure
    - root
        - tracker1
            - param1.zip
            - param2.zip
            .
            .
        - tracker2
            - param1.zip
            - param2.zip
        .
        .

    args:
        packed_results_path - Path to the directory where the zipped results are stored
        output_path - Path to the directory where the results will be unpacked. Set to env_settings().results_path
                      by default
    """

    if output_path is None:
        output_path = env_settings().results_path

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    trackers = os.listdir(packed_results_path)

    for t in trackers:
        runfiles = os.listdir(os.path.join(packed_results_path, t))

        for r in runfiles:
            save_path = os.path.join(output_path, t)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            shutil.unpack_archive(os.path.join(packed_results_path, t, r), os.path.join(save_path, r[:-4]), 'zip')


def main():
    packed_results_path = '/home/goutam/Desktop/tracker_results_packed'
    unpack_tracking_results(packed_results_path)


if __name__ == '__main__':
    main()
