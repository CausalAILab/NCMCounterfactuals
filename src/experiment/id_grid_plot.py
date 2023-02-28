import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from src.metric.queries import is_q_id_in_G

pd.set_option('display.max_columns', 10)
pd.set_option('display.max_rows', 500)
pd.set_option('display.width', 1000)


def get_avg_gaps(data, epoch_steps):
    avg_gaps = data.groupby(["graph", "t"])[epoch_steps].mean().reset_index()
    return avg_gaps


def get_acc_lines(data, epoch_steps, query, tau, rolling_window=1):
    avg_gaps = get_avg_gaps(data, epoch_steps)
    stderr_gaps = data.groupby(['graph', 't']).apply(lambda x: x[epoch_steps].std() / np.sqrt(len(x))).reset_index()
    classification = avg_gaps
    classification[epoch_steps] = (avg_gaps[epoch_steps] + 1.65 * stderr_gaps[epoch_steps]) < tau
    classification['truth'] = classification['graph'].apply(lambda x: is_q_id_in_G(x, query))
    for step in epoch_steps:
        classification[step] = (classification[step] == classification['truth']).astype(float)
    classification = classification.drop(columns=['truth'])
    classification = classification.groupby(["graph"])[epoch_steps].mean().reset_index()
    classification[epoch_steps] = classification[epoch_steps].rolling(rolling_window, min_periods=1, axis=1).mean()
    return classification


def get_grid_acc_plot(fig_name, acc_data, queries, graph_pairs, epoch_steps, colors=None, dashes=None):
    int_epochs = [int(ep) for ep in epoch_steps]

    num_rows = len(graph_pairs)
    num_double_cols = len(queries)
    #fig, axes = plt.subplots(len(graph_pairs), 2 * len(queries), sharex=True, sharey='row', figsize=(16, 6))

    left_space = 0.05
    right_space = 0.95
    minor_space = 0.1
    major_space = 0.3
    offset = (1 + minor_space) * (right_space - left_space) / \
             (2 * num_double_cols + num_double_cols * minor_space + (num_double_cols - 1) * major_space)
    wspace = major_space + minor_space + 1

    pv_plots = GridSpec(num_rows, num_double_cols, left=left_space, right=right_space - offset,
                        top=0.95, bottom=0.1, wspace=wspace)
    pvdox_plots = GridSpec(num_rows, num_double_cols, left=left_space + offset, right=right_space,
                           top=0.95, bottom=0.1, wspace=wspace)

    #fig.gca().set_prop_cycle(plt.cycler('color', plt.cm.jet(np.linspace(0, 1, 3))))

    fig = plt.figure(figsize=(16, 6))
    axes = []
    id_counter = 0
    for row in range(len(graph_pairs)):
        g1, g2 = graph_pairs[row]
        #axes[row][0].set_ylim([0.0, 1.01])

        for col in range(len(queries)):
            #ax1 = axes[row][2 * col]
            #ax2 = axes[row][2 * col + 1]
            axes.append(fig.add_subplot(pv_plots[row * len(queries) + col]))
            axes.append(fig.add_subplot(pvdox_plots[row * len(queries) + col]))
            ax1 = axes[-2]
            ax2 = axes[-1]

            q = queries[col]
            if colors is None:
                ax1.set_prop_cycle(plt.cycler('color', plt.cm.jet(np.linspace(0, 1, len(acc_data[q])))))
                ax2.set_prop_cycle(plt.cycler('color', plt.cm.jet(np.linspace(0, 1, len(acc_data[q])))))
                for tau in acc_data[q]:
                    ax1.plot(int_epochs, acc_data[q][tau].loc[g1])
                    ax2.plot(int_epochs, acc_data[q][tau].loc[g2])
            else:
                for method in acc_data[q]:
                    ax1.plot(int_epochs, acc_data[q][method].loc[g1], color=colors[method], dashes=dashes[method])
                    ax2.plot(int_epochs, acc_data[q][method].loc[g2], color=colors[method], dashes=dashes[method])

            if is_q_id_in_G(g1, q):
                id_counter += 1
                ax1.set_facecolor((0.95, 1.0, 0.95))
                plt.text(0.08, 0.8, str(id_counter),
                        bbox=dict(boxstyle="circle", facecolor='white', edgecolor='black'), transform=ax1.transAxes)
            else:
                ax1.set_facecolor((1.0, 1.0, 0.9))
            if is_q_id_in_G(g2, q):
                id_counter += 1
                ax2.set_facecolor((0.95, 1.0, 0.95))
                plt.text(0.08, 0.8, str(id_counter),
                         bbox=dict(boxstyle="circle", facecolor='white', edgecolor='black'), transform=ax2.transAxes)
            else:
                ax2.set_facecolor((1.0, 1.0, 0.9))

            ax1.set_ylim([0.0, 1.01])
            ax2.set_ylim([0.0, 1.01])
            if row != len(graph_pairs) - 1:
                plt.setp(ax1.get_xticklabels(), visible=False)
                plt.setp(ax2.get_xticklabels(), visible=False)
            if col != 0:
                plt.setp(ax1.get_yticklabels(), visible=False)
            plt.setp(ax2.get_yticklabels(), visible=False)

    #fig.tight_layout(rect=(0.025, 0.025, 1, 1))
    #fig.subplots_adjust(wspace=0.1, hspace=0.1)
    fig.supylabel('ID Accuracy')
    fig.supxlabel('Training Iteration')
    fig.savefig(fig_name, dpi=300, bbox_inches='tight')
    fig.clf()


def get_grid_gaps_plot(fig_name, gap_data, queries, graph_pairs, epoch_steps):
    int_epochs = [int(ep) for ep in epoch_steps]
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]

    num_rows = len(graph_pairs)
    num_double_cols = len(queries)

    left_space = 0.05
    right_space = 0.95
    minor_space = 0.1
    major_space = 0.3
    offset = (1 + minor_space) * (right_space - left_space) / \
             (2 * num_double_cols + num_double_cols * minor_space + (num_double_cols - 1) * major_space)
    wspace = major_space + minor_space + 1

    pv_plots = GridSpec(num_rows, num_double_cols, left=left_space, right=right_space - offset,
                        top=0.95, bottom=0.1, wspace=wspace)
    pvdox_plots = GridSpec(num_rows, num_double_cols, left=left_space + offset, right=right_space,
                           top=0.95, bottom=0.1, wspace=wspace)

    fig = plt.figure(figsize=(16, 6))
    axes = []
    id_counter = 0
    for row in range(len(graph_pairs)):
        g1, g2 = graph_pairs[row]

        for col in range(len(queries)):
            axes.append(fig.add_subplot(pv_plots[row * len(queries) + col]))
            axes.append(fig.add_subplot(pvdox_plots[row * len(queries) + col]))
            ax1 = axes[-2]
            ax2 = axes[-1]

            q = queries[col]
            avg_gap_data = get_avg_gaps(gap_data[q], epoch_steps)
            raw_graph_gaps = avg_gap_data.loc[avg_gap_data["graph"] == g1][epoch_steps].astype(float)
            gap_percentiles = np.percentile(raw_graph_gaps, percentiles, axis=0)
            ax1.set_prop_cycle(plt.cycler('color', plt.cm.jet(np.linspace(0, 1, len(percentiles)))))
            ax1.plot(int_epochs, gap_percentiles.T)
            if is_q_id_in_G(g1, q):
                id_counter += 1
                ax1.set_facecolor((0.95, 1.0, 0.95))
                plt.text(0.08, 0.8, str(id_counter),
                         bbox=dict(boxstyle="circle", facecolor='white', edgecolor='black'), transform=ax1.transAxes)
            else:
                ax1.set_facecolor((1.0, 1.0, 0.9))
            raw_graph_gaps = avg_gap_data.loc[avg_gap_data["graph"] == g2][epoch_steps].astype(float)
            gap_percentiles = np.percentile(raw_graph_gaps, percentiles, axis=0)
            ax2.set_prop_cycle(plt.cycler('color', plt.cm.jet(np.linspace(0, 1, len(percentiles)))))
            ax2.plot(int_epochs, gap_percentiles.T)
            if is_q_id_in_G(g2, q):
                id_counter += 1
                ax2.set_facecolor((0.95, 1.0, 0.95))
                plt.text(0.08, 0.8, str(id_counter),
                         bbox=dict(boxstyle="circle", facecolor='white', edgecolor='black'), transform=ax2.transAxes)
            else:
                ax2.set_facecolor((1.0, 1.0, 0.9))

            ax1.set_ylim([0.0, 1.2])
            ax2.set_ylim([0.0, 1.2])
            if row != len(graph_pairs) - 1:
                plt.setp(ax1.get_xticklabels(), visible=False)
                plt.setp(ax2.get_xticklabels(), visible=False)
            if col != 0:
                plt.setp(ax1.get_yticklabels(), visible=False)
            plt.setp(ax2.get_yticklabels(), visible=False)

    fig.supylabel('Max - Min Gap')
    fig.supxlabel('Training Iteration')
    fig.savefig(fig_name, dpi=300, bbox_inches='tight')
    fig.clf()


def get_r80_grid_plot(fig_name, data, graphs, epoch_steps, tau, mode="grid"):
    int_epochs = [int(ep) for ep in epoch_steps]
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    acc_data_gan = get_acc_lines(data['gan'], epoch_steps, 'ate', tau, rolling_window=10)
    acc_data_mle = get_acc_lines(data['mle'], epoch_steps, 'ate', tau, rolling_window=10)
    acc_data_gan = acc_data_gan.set_index('graph')
    acc_data_mle = acc_data_mle.set_index('graph')
    avg_gap_gan_data = get_avg_gaps(data['gan'], epoch_steps)
    avg_gap_mle_data = get_avg_gaps(data['mle'], epoch_steps)

    if mode == "grid":
        fig, axes = plt.subplots(3, len(graphs), sharex=True, sharey='row', figsize=(16, 4.5),
                                 gridspec_kw=dict(width_ratios=[8, 8, 8, 8, 1, 8, 8, 8, 8]))
        for col, graph in enumerate(graphs):
            if col == 4:
                continue

            ax_acc = axes[0][col]
            ax_gap1 = axes[1][col]
            ax_gap2 = axes[2][col]

            ax_acc.plot(int_epochs, acc_data_gan.loc[graph])
            ax_acc.plot(int_epochs, acc_data_mle.loc[graph])
            ax_acc.set_ylim([0.0, 1.01])

            raw_graph_gaps_gan = avg_gap_gan_data.loc[avg_gap_gan_data["graph"] == graph][epoch_steps].astype(float)
            raw_graph_gaps_mle = avg_gap_mle_data.loc[avg_gap_mle_data["graph"] == graph][epoch_steps].astype(float)
            gap_percentiles_gan = np.percentile(raw_graph_gaps_gan, percentiles, axis=0)
            gap_percentiles_mle = np.percentile(raw_graph_gaps_mle, percentiles, axis=0)
            ax_gap1.set_prop_cycle(plt.cycler('color', plt.cm.jet(np.linspace(0, 1, len(percentiles)))))
            ax_gap2.set_prop_cycle(plt.cycler('color', plt.cm.jet(np.linspace(0, 1, len(percentiles)))))
            ax_gap1.plot(int_epochs, gap_percentiles_gan.T)
            ax_gap2.plot(int_epochs, gap_percentiles_mle.T)
            ax_gap1.set_ylim([-0.1, 1.2])
            ax_gap2.set_ylim([-0.1, 1.2])

        axes[0][4].remove()
        axes[1][4].remove()
        axes[2][4].remove()

        axes[0][0].set_ylabel("Correct ID %")
        axes[1][0].set_ylabel("GAN Gaps")
        axes[2][0].set_ylabel("MLE Gaps")

        fig.tight_layout(rect=(0.025, 0.025, 1, 1))
        fig.subplots_adjust(wspace=0.1, hspace=0.1)
        fig.supxlabel('Training Iteration')
        fig.savefig(fig_name, dpi=300, bbox_inches='tight')
        fig.clf()
    else:
        for graph in graphs:
            plt.figure(figsize=(4, 3))
            plt.plot(int_epochs, acc_data_gan.loc[graph])
            plt.plot(int_epochs, acc_data_mle.loc[graph])
            plt.ylim([0.0, 1.01])
            plt.xlabel("Training Iteration")
            plt.ylabel("Correct ID %")
            plt.tight_layout()
            plt.savefig("{}_{}_acc.png".format(fig_name, graph))
            plt.clf()

            raw_graph_gaps_gan = avg_gap_gan_data.loc[avg_gap_gan_data["graph"] == graph][epoch_steps].astype(float)
            raw_graph_gaps_mle = avg_gap_mle_data.loc[avg_gap_mle_data["graph"] == graph][epoch_steps].astype(float)
            gap_percentiles_gan = np.percentile(raw_graph_gaps_gan, percentiles, axis=0)
            gap_percentiles_mle = np.percentile(raw_graph_gaps_mle, percentiles, axis=0)

            plt.figure(figsize=(4, 3))
            plt.gca().set_prop_cycle(plt.cycler('color', plt.cm.jet(np.linspace(0, 1, len(percentiles)))))
            plt.plot(int_epochs, gap_percentiles_gan.T)
            plt.ylim([-0.1, 1.2])
            plt.xlabel("Training Iteration")
            plt.ylabel("Max - Min Gaps")
            plt.tight_layout()
            plt.savefig("{}_{}_gan_gaps.png".format(fig_name, graph))
            plt.clf()

            plt.figure(figsize=(4, 3))
            plt.gca().set_prop_cycle(plt.cycler('color', plt.cm.jet(np.linspace(0, 1, len(percentiles)))))
            plt.plot(int_epochs, gap_percentiles_mle.T)
            plt.ylim([-0.1, 1.2])
            plt.xlabel("Training Iteration")
            plt.ylabel("Max - Min Gaps")
            plt.tight_layout()
            plt.savefig("{}_{}_mle_gaps.png".format(fig_name, graph))
            plt.clf()


parser = argparse.ArgumentParser(description="ID Experiment Grid Plot")
parser.add_argument('dir', help="directory of the experiment")
parser.add_argument('mode', help="which plot to make (expl, r80)")
parser.add_argument('--method', default="all", help="which method to plot (gan_1d, gan_16d, mle_1d)")
parser.add_argument('--rolling', type=int, default=10, help="rolling average window size")
parser.add_argument('--tau', type=float, default=0.03, help="choice of tau for hypothesis testing")

args = parser.parse_args()
d = args.dir
mode = args.mode
roll = args.rolling
tau_choice = args.tau

mode_options = ['expl', 'r80']
valid_methods = ["gan_1d", "gan_16d", "mle_1d"]
assert mode in mode_options
assert args.method in valid_methods or args.method == "all"

method_choice = args.method
if method_choice == "all":
    method_choice = valid_methods

queries = ['ate', 'ett', 'nde', 'ctfde']
graph_pairings = [["expl", "expl_dox"], ["expl_xm", "expl_xm_dox"],
                  ["expl_xy", "expl_xy_dox"], ["expl_my", "expl_my_dox"]]
r80_graphs = ["backdoor", "frontdoor", "m", "napkin", "placeholder", "bow", "extended_bow", "iv", "bad_m"]
tau_choices = [0.01, 0.03, 0.05]
colors = {
    "gan_1d": "blue",
    "gan_16d": "blue",
    "mle_1d": "orange"
}
dashes = {
    "gan_1d": '',
    "gan_16d": (3, 1),
    "mle_1d": ''
}

if mode == 'expl':
    if args.method == "all":
        # Get data
        acc_data = dict()
        epoch_steps = None
        for q in queries:
            acc_data[q] = dict()
            for method in method_choice:
                gap_data = pd.read_csv("{}/{}_{}_gap_results.csv".format(d, method, q))
                kl_data = pd.read_csv("{}/{}_{}_kl_results.csv".format(d, method, q))
                if epoch_steps is None:
                    epoch_steps = gap_data.columns.values[3:].tolist()

                acc_data[q][method] = get_acc_lines(gap_data, epoch_steps, q, tau_choice, rolling_window=roll)
                acc_data[q][method] = acc_data[q][method].set_index('graph')

        get_grid_acc_plot("{}/id_acc_plots.png".format(d), acc_data, queries, graph_pairings, epoch_steps,
                          colors=colors, dashes=dashes)
    else:
        # Get data
        gap_data = dict()
        kl_data = dict()
        acc_data = dict()
        epoch_steps = None
        for q in queries:
            gap_data[q] = pd.read_csv("{}/{}_{}_gap_results.csv".format(d, method_choice, q))
            kl_data[q] = pd.read_csv("{}/{}_{}_kl_results.csv".format(d, method_choice, q))
            if epoch_steps is None:
                epoch_steps = gap_data[q].columns.values[3:].tolist()

            acc_data[q] = dict()
            for tau in tau_choices:
                acc_data[q][tau] = get_acc_lines(gap_data[q], epoch_steps, q, tau, rolling_window=roll)
                acc_data[q][tau] = acc_data[q][tau].set_index('graph')

        get_grid_acc_plot("{}/id_acc_plots.png".format(d), acc_data, queries, graph_pairings, epoch_steps)
        get_grid_gaps_plot("{}/id_gaps_plots.png".format(d), gap_data, queries, graph_pairings, epoch_steps)

else:
    # Get data
    gap_data = dict()
    gap_data['gan'] = pd.read_csv("{}/gan_gap_results.csv".format(d))
    gap_data['mle'] = pd.read_csv("{}/mle_gap_results.csv".format(d))
    epoch_steps = gap_data['gan'].columns.values[3:].tolist()

    get_r80_grid_plot("{}/r80_id_plots.png".format(d), gap_data, r80_graphs, epoch_steps, tau_choice)
    get_r80_grid_plot("{}/single".format(d), gap_data, ["bdm"], epoch_steps, tau_choice, mode="single")
