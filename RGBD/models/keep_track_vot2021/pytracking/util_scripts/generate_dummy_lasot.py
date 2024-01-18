import os
import sys
from shutil import copyfile

env_path = os.path.join(os.path.dirname(__file__), '..')
if env_path not in sys.path:
    sys.path.append(env_path)

from pytracking.evaluation.environment import env_settings


def main():
    out_dir = ''

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    settings = env_settings()
    lasot_path = settings.lasot_path

    class_list = os.listdir(lasot_path)

    for c in class_list:
        if not os.path.isdir(os.path.join(lasot_path, c)):
            continue

        if not os.path.exists(os.path.join(out_dir, c)):
            os.makedirs(os.path.join(out_dir, c))

        seq_list = os.listdir(os.path.join(lasot_path, c))

        for s in seq_list:
            if not os.path.isdir(os.path.join(lasot_path, c, s)):
                continue

            if not os.path.exists(os.path.join(out_dir, c, s)):
                os.makedirs(os.path.join(out_dir, c, s))

            copyfile(os.path.join(lasot_path, c, s, 'full_occlusion.txt'), os.path.join(out_dir, c, s, 'full_occlusion.txt'))
            copyfile(os.path.join(lasot_path, c, s, 'groundtruth.txt'), os.path.join(out_dir, c, s, 'groundtruth.txt'))
            copyfile(os.path.join(lasot_path, c, s, 'nlp.txt'), os.path.join(out_dir, c, s, 'nlp.txt'))
            copyfile(os.path.join(lasot_path, c, s, 'out_of_view.txt'), os.path.join(out_dir, c, s, 'out_of_view.txt'))

    print('Done')


if __name__ == '__main__':
    main()
