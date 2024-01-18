import tikzplotlib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

env_path = os.path.join(os.path.dirname(__file__), '../../..')
if env_path not in sys.path:
    sys.path.append(env_path)

from pytracking.evaluation.environment import env_settings


def plot_draw_save():
    num_filters = [1, 2, 4, 8, 16, 32, 64]
    j_score = [75.6, 78.8, 80.1, 80.6, 81.1, 81.2, 81.1]

    # Plot settings
    font_size = 12
    font_size_axis = 14
    line_width = 2
    font_size_legend = 13

    matplotlib.rcParams.update({'font.size': font_size})
    matplotlib.rcParams.update({'axes.titlesize': font_size_axis})
    matplotlib.rcParams.update({'axes.titleweight': 'black'})
    matplotlib.rcParams.update({'axes.labelsize': font_size_axis})

    fig, ax = plt.subplots()

    line = ax.plot(num_filters, j_score,
                   linewidth=line_width,
                   color=(1.0, 0, 0),
                   linestyle='-',
                   marker="o")

    ax.set(xlabel='Maximum training set size $K_\mathrm{max}$',
           ylabel='$\mathcal{J}$-Score',
           title='Impact of maximum training set size $K_\mathrm{max}$')
    plt.xscale('log', basex=2)
    plt.yticks(np.arange(75.0, 82.0, step=0.5))  # Set label locations.

    ax.grid(True, linestyle='-.')
    fig.tight_layout()

    settings = env_settings()

    plot_name = 'plot_max_memsz'
    result_plot_path = os.path.join(settings.result_plot_path, 'papers', 'eccv20')

    if not os.path.exists(result_plot_path):
        os.makedirs(result_plot_path)

    plt.draw()
    tikzplotlib.save('{}/{}.tex'.format(result_plot_path, plot_name))
    fig.savefig('{}/{}_plot.pdf'.format(result_plot_path, plot_name), dpi=300, format='pdf', transparent=True)

    plt.show()
    a = 1

if __name__ == '__main__':
    plot_draw_save()
