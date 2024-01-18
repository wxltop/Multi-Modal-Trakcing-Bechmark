import os
import sys
from pathlib import Path

env_path = os.path.join(os.path.dirname(__file__), '..')
if env_path not in sys.path:
    sys.path.append(env_path)

from pytracking.evaluation import get_dataset

def main():
    otb = get_dataset('otb')
    base_path = Path('/home/goutam/Downloads/DaSiamRPN_OTB100/default/')
    files = base_path.glob('*.txt')
    for f in files:
        name = f.parts[-1]
        seq_name = name[:-4]
        new_name = None
        for seq in otb:
            if seq_name.lower() == seq.name.lower()[:]:
                # new_name = 'OPE_' + seq.name + '.mat'
                new_name = seq.name + '.txt'
                break
        if new_name is None:
            continue
        # new_name = 'OPE_' + name[:name.find('_SIAMRPN')] + '.mat'
        # new_name = name.replace('-','_')
        f_new = f.parent / new_name
        os.rename(f, f_new)




if __name__ == '__main__':
    main()
