import _init_paths
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [8, 8]

from lib.test.analysis.plot_results import plot_results, print_results, print_per_sequence_results, \
    print_results_per_video
from lib.test.evaluation import get_dataset, trackerlist

trackers = []

trackers.extend(trackerlist('ostrack', 'baseline_roi_ep300', 'LaSOT', None, 'baseline_roi_ep300'))

dataset = get_dataset('LaSOT')
# plot_results(trackers, dataset, 'got10k_val', merge_results=True, plot_types=('success', 'prec'),
#              skip_missing_seq=False, force_evaluation=True, plot_bin_gap=0.05, exclude_invalid_frames=False)
print_results_per_video(trackers, dataset, 'LaSOT', merge_results=True, plot_types=('success', 'prec', 'norm_prec'), per_video=True)
# print_per_sequence_results(trackers, dataset, report_name="debug")
