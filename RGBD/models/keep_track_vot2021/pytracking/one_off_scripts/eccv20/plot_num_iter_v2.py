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
    num_filters = [1/32, 1/16, 1/8, 1/4, 1/2, 1, 3, 5, 7]
    j_score = [77.2, 78.1, 79.1, 79.9, 80.7, 81.0, 81.20, 80.90, 80.7]
    j_score_noup = [75.6] * len(num_filters)
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

    line1 = ax.plot(num_filters, j_score,
                   linewidth=line_width,
                   color=(1.0, 0, 0),
                   linestyle='-',
                   marker="o")

    line2 = ax.plot(num_filters, j_score_noup,
                    linewidth=line_width,
                    color=(0.0, 0, 1.0),
                    linestyle='-',
                    )

    ax.legend([line1[0], line2[0]], ['Update', 'No Update'], fancybox=False, edgecolor='black',
              fontsize=font_size_legend, framealpha=1.0)

    ax.set(xlabel='Number of SD iterations $N_\mathrm{update}^\mathrm{inf}$',
           ylabel='$\mathcal{J}$-Score')
    #ax.set_xscale('symlog')
    plt.xscale('log', basex=2)
    plt.yticks(np.arange(75.0, 81.5, step=0.5))  # Set label locations.
    plt.xticks(num_filters)
    ax.set_xticklabels(['$\\frac{1}{32}$', '$\\frac{1}{16}$', '$\\frac{1}{8}$', '$\\frac{1}{4}$', '$\\frac{1}{2}$', '$1$', '$3$', '$5$', '$7$'])
    ax.grid(True, linestyle='-.')
    fig.tight_layout()

    settings = env_settings()

    plot_name = 'plot_num_iter_v2'
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
