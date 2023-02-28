import os
import sys
import shutil
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as mtrans
from matplotlib.ticker import FormatStrFormatter, FuncFormatter, SymmetricalLogLocator
from tensorboard.backend.event_processing import event_accumulator
import json

from src.metric.queries import is_q_id_in_G


def running_average(nums, horizon=10):
    new_nums = [0] * len(nums)
    new_nums[0] = nums[0]
    for i in range(1, len(nums)):
        new_nums[i] = new_nums[i - 1] + nums[i]
        if i >= horizon:
            new_nums[i] -= nums[i - horizon]

    for i in range(len(nums)):
        new_nums[i] = new_nums[i] / min(i + 1, horizon)

    return new_nums


def extract_iters(d):
    if os.path.exists("{}/hyperparams.json".format(d)):
        with open("{}/hyperparams.json".format(d), 'r') as f:
            hyperparams = json.load(f)
            return np.arange(0, int(hyperparams['max-query-iters']), 10)
    return None


def extract_data(d, clean=False):
    temp_gap_list = []
    temp_kl_list = []

    for r in os.listdir(d):
        if os.path.isdir("{}/{}".format(d, r)):
            if os.path.isdir("{}/{}/logs".format(d, r)):
                if not os.path.exists("{}/{}/results.json".format(d, r)):
                    if clean:
                        print("Run {} is incomplete. Deleting contents...".format(r))
                        shutil.rmtree("{}/{}".format(d, r))
                        if os.path.exists("{}/lock".format(d)):
                            os.remove("{}/lock".format(d))
                        if os.path.exists("{}/best.th".format(d)):
                            os.remove("{}/best.th".format(d))
                    else:
                        print("Trial {}, run {} is incomplete.".format(t, r))
                else:
                    max_dir = "{}/{}/logs/default/version_0".format(d, r)
                    min_dir = "{}/{}/logs/default/version_1".format(d, r)

                    if os.path.isdir(min_dir) and os.path.isdir(max_dir):
                        min_event = None
                        max_event = None
                        for item in os.listdir(min_dir):
                            if min_event is None and "events" in item:
                                min_event = item
                        for item in os.listdir(max_dir):
                            if max_event is None and "events" in item:
                                max_event = item
                        ea_min = event_accumulator.EventAccumulator("{}/{}".format(min_dir, min_event))
                        ea_max = event_accumulator.EventAccumulator("{}/{}".format(max_dir, max_event))
                        ea_min.Reload()
                        ea_max.Reload()
                        min_q_events = ea_min.Scalars('ncm_{}'.format(query.upper()))
                        min_kl_events = ea_min.Scalars('total_dat_KL')
                        max_q_events = ea_max.Scalars('ncm_{}'.format(query.upper()))
                        max_kl_events = ea_max.Scalars('total_dat_KL')

                        try:
                            min_max_gaps = []
                            total_kl = []
                            for i in range(len(min_q_events)):
                                min_q = np.nan_to_num(min_q_events[i].value, nan=-1.0)
                                max_q = np.nan_to_num(max_q_events[i].value, nan=1.0)
                                min_kl = np.nan_to_num(abs(min_kl_events[i].value), nan=2.0)
                                max_kl = np.nan_to_num(abs(max_kl_events[i].value), nan=2.0)

                                min_max_gaps.append(max_q - min_q)
                                total_kl.append(min_kl + max_kl)

                            temp_gap_list.append(min_max_gaps)
                            temp_kl_list.append(total_kl)
                        except Exception as e:
                            print("Error in run {}.".format(r))
                            print(e)

    gaps_means = None
    kl_means = None
    gaps_ucb = None
    if len(temp_gap_list) > 0:
        temp_gap_list = np.array(temp_gap_list)
        temp_kl_list = np.array(temp_kl_list)
        gaps_means = np.mean(temp_gap_list, axis=0)
        kl_means = np.mean(temp_kl_list, axis=0)

        if len(temp_gap_list) > 1:
            gaps_stderr = np.std(temp_gap_list, axis=0) / np.sqrt(len(temp_gap_list))
            gaps_ucb = gaps_means + 1.65 * gaps_stderr

    return gaps_means, kl_means, gaps_ucb


def fit_len(source, target):
    new_source = []
    skip = len(source) // len(target)
    for i in range(len(target)):
        new_source.append(source[skip * i])
    return np.array(new_source)


def error_plot(fig_name, iter_list, kl_list):
    plt.plot(iter_list, np.mean(kl_list, axis=0), color='red')
    plt.xlabel("Training Iteration")
    plt.ylabel("Average Total KL")
    plt.tight_layout()
    plt.savefig(fig_name)
    plt.clf()


def gaps_plot(fig_name, query, iter_list, q_gaps, percentiles, zoom_bounds=None, sep_bounds=None, sep_colors=None):
    q_gap_percentiles = np.percentile(q_gaps, percentiles, axis=0)

    plt.gca().set_prop_cycle(plt.cycler('color', plt.cm.jet(np.linspace(0, 1, len(percentiles)))))
    if zoom_bounds is not None:
        plt.gca().set_ylim(zoom_bounds)
    plt.plot(iter_list, q_gap_percentiles.T)
    plt.axhline(y=0.0, color='k', linestyle='-')
    if sep_bounds is not None:
        for i, b in enumerate(sep_bounds):
            plt.axhline(y=b, color=sep_colors[i], linestyle='--')
    plt.xlabel("Training Iteration")
    plt.ylabel("Max {} - Min {}".format(query.upper(), query.upper()))
    plt.legend(percentiles)
    plt.tight_layout()
    plt.savefig(fig_name)
    plt.clf()


def id_acc_plot(fig_name, iter_list, gaps_ucb_list, is_id, boundaries, run_avg=None):
    plt.gca().set_prop_cycle(plt.cycler('color', plt.cm.jet(np.linspace(0, 1, len(boundaries)))))
    plt.gca().set_ylim([0.0, 1.01])
    for b in boundaries:
        gaps_ucb_sep = []
        if isinstance(gaps_ucb_list, dict):
            for exp_type in gaps_ucb_list:
                gaps = gaps_ucb_list[exp_type]
                if is_q_id_in_G(exp_type, query):
                    result = (gaps <= b).astype(int)
                else:
                    result = (gaps > b).astype(int)
                gaps_ucb_sep.append(result)
            gaps_ucb_sep = np.concatenate(gaps_ucb_sep, axis=0)
        else:
            if is_id:
                gaps_ucb_sep = (gaps_ucb_list <= b).astype(int)
            else:
                gaps_ucb_sep = (gaps_ucb_list > b).astype(int)
        acc_list = np.mean(gaps_ucb_sep, axis=0)

        if run_avg is not None:
            acc_list = running_average(acc_list, horizon=run_avg)

        plt.plot(iter_list, acc_list)
    plt.xlabel("Training Iteration")
    plt.ylabel("Correct ID %")
    plt.legend(boundaries)
    plt.tight_layout()
    plt.savefig(fig_name)
    plt.clf()


parser = argparse.ArgumentParser(description="ID Experiment Results Parser")
parser.add_argument('dir', help="directory of the experiment")
parser.add_argument('query', help="identified query")
parser.add_argument('--clean', action="store_true",
                    help="delete unfinished experiments")
args = parser.parse_args()

d = args.dir

valid_graph_set = {"backdoor", "bow", "frontdoor", "napkin", "m", "bad_m", "iv", "extended_bow",
                   "med", "expl", "bdm", "simple",
                   "zid_a", "zid_b", "zid_c", "gid_a", "gid_b", "gid_c", "gid_d",
                   "expl_dox", "expl_xm", "expl_xm_dox", "expl_xy", "expl_xy_dox", "expl_my", "expl_my_dox"}
valid_queries = {"ate", "ett", "nde", "ctfde"}

query = args.query.lower()
assert query in valid_queries

boundaries = [0.01, 0.03, 0.05]
b_colors = ['r', 'g', 'b']
percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]

iter_list = None
id_q_gaps = []
nonid_q_gaps = []
all_q_gaps = dict()
all_gap_ucbs = dict()

os.makedirs("{}/figs".format(d), exist_ok=True)
for exp_type in os.listdir(d):
    if exp_type not in valid_graph_set:
        print("\nSkipping {} directory.".format(exp_type))
        continue
    print("\nScanning {} experiments...".format(exp_type))

    id_truth = is_q_id_in_G(exp_type, query)

    kl_list = []
    gaps_ucb_list = []
    graph_q_gaps = []
    for t in os.listdir("{}/{}".format(d, exp_type)):
        if iter_list is None:
            iter_list = extract_iters("{}/{}/{}".format(d, exp_type, t))

        if os.path.isdir("{}/{}/{}".format(d, exp_type, t)):
            gaps_means, kl_means, gaps_ucb = extract_data("{}/{}/{}".format(d, exp_type, t), args.clean)
            gaps_means = fit_len(gaps_means, iter_list)
            kl_means = fit_len(kl_means, iter_list)
            gaps_ucb = fit_len(gaps_ucb, iter_list)
            if gaps_means is not None:
                graph_q_gaps.append(gaps_means)
                kl_list.append(kl_means)
                if id_truth:
                    id_q_gaps.append(gaps_means)
                else:
                    nonid_q_gaps.append(gaps_means)

            if gaps_ucb is not None:
                gaps_ucb_list.append(gaps_ucb)
                if exp_type not in all_gap_ucbs:
                    all_gap_ucbs[exp_type] = []
                all_gap_ucbs[exp_type].append(gaps_ucb)

    kl_list = np.array(kl_list)
    all_q_gaps[exp_type] = graph_q_gaps

    # Plot KL error per graph
    error_plot("{}/figs/{}_errors.png".format(d, exp_type), iter_list, kl_list)

    # Plot gaps per graph
    gaps_plot("{}/figs/{}_gap_percentiles.png".format(d, exp_type), query, iter_list, graph_q_gaps, percentiles)

    # Plot accuracy per graph
    if len(gaps_ucb_list) > 0:
        gaps_ucb_list = np.array(gaps_ucb_list)
        id_acc_plot("{}/figs/{}_ID_classification.png".format(d, exp_type), iter_list, gaps_ucb_list, id_truth,
                    boundaries, run_avg=None)
        id_acc_plot("{}/figs/{}_ID_classification_10runavg.png".format(d, exp_type), iter_list, gaps_ucb_list, id_truth,
                    boundaries, run_avg=10)
        all_gap_ucbs[exp_type] = np.array(all_gap_ucbs[exp_type])

id_q_gaps = np.array(id_q_gaps)
nonid_q_gaps = np.array(nonid_q_gaps)

# Plot gaps overall
gaps_plot("{}/figs/ID_gap_percentiles.png".format(d), query, iter_list, id_q_gaps, percentiles,
          sep_bounds=boundaries, sep_colors=b_colors)
gaps_plot("{}/figs/nonID_gap_percentiles.png".format(d), query, iter_list, nonid_q_gaps, percentiles,
          sep_bounds=boundaries, sep_colors=b_colors)
gaps_plot("{}/figs/ID_gap_percentiles_zoomed.png".format(d), query, iter_list, id_q_gaps, percentiles,
          zoom_bounds=[-0.01, 0.04], sep_bounds=boundaries, sep_colors=b_colors)
gaps_plot("{}/figs/nonID_gap_percentiles_zoomed.png".format(d), query, iter_list, nonid_q_gaps, percentiles,
          zoom_bounds=[-0.01, 0.04], sep_bounds=boundaries, sep_colors=b_colors)

# Plot accuracy overall
id_acc_plot("{}/figs/overall_ID_classification.png".format(d), iter_list, all_gap_ucbs,
            False, boundaries, run_avg=None)
id_acc_plot("{}/figs/overall_ID_classification_10runavg.png".format(d), iter_list, all_gap_ucbs,
            False, boundaries, run_avg=10)
