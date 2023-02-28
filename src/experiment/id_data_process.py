import argparse
import csv
import json
import os
import numpy as np

from tensorboard.backend.event_processing import event_accumulator

parser = argparse.ArgumentParser(description="ID Experiment Data Processor")
parser.add_argument('dir', help="directory of the experiment")
parser.add_argument('query', help="identified query")

args = parser.parse_args()

d = args.dir

valid_graph_set = {"backdoor", "bow", "frontdoor", "napkin", "m", "bad_m", "iv", "extended_bow",
                   "med", "expl", "bdm", "simple",
                   "zid_a", "zid_b", "zid_c", "gid_a", "gid_b", "gid_c", "gid_d",
                   "expl_dox", "expl_xm", "expl_xm_dox", "expl_xy", "expl_xy_dox", "expl_my", "expl_my_dox"}
valid_queries = {"ate", "ett", "nde", "ctfde"}

query = args.query.lower()
assert query in valid_queries


def extract_iters(d):
    if os.path.exists("{}/hyperparams.json".format(d)):
        with open("{}/hyperparams.json".format(d), 'r') as f:
            hyperparams = json.load(f)
            return np.arange(0, int(hyperparams['max-query-iters']), 10)
    return None


def extract_data(d):
    r_label = []
    gap_lists = []
    kl_lists = []

    for r in os.listdir(d):
        if os.path.isdir("{}/{}".format(d, r)):
            if os.path.isdir("{}/{}/logs".format(d, r)):
                if not os.path.exists("{}/{}/results.json".format(d, r)):
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

                            r_label.append(r)
                            gap_lists.append(min_max_gaps)
                            kl_lists.append(total_kl)
                        except Exception as e:
                            print("Error in run {}.".format(r))
                            print(e)

    return r_label, gap_lists, kl_lists


def fit_len(source, target):
    new_source = []
    skip = len(source) // len(target)
    for i in range(len(target)):
        new_source.append(source[skip * i])
    return np.array(new_source)


def write_to_csv(filename, data, header=None):
    with open(filename, "w", newline='') as f:
        writer = csv.writer(f)
        if header is not None:
            writer.writerow(header)
        writer.writerows(data)


iter_list = None
header = ["graph", "t", "r"]
kl_rows = []
gap_rows = []
for exp_type in os.listdir(d):
    if exp_type not in valid_graph_set:
        print("\nSkipping {} directory.".format(exp_type))
        continue
    print("\nScanning {} experiments...".format(exp_type))

    for t in os.listdir("{}/{}".format(d, exp_type)):
        if iter_list is None:
            iter_list = extract_iters("{}/{}/{}".format(d, exp_type, t))
            header.extend(iter_list)

        if os.path.isdir("{}/{}/{}".format(d, exp_type, t)):
            t_r_labels, t_gap_lists, t_kl_lists = extract_data("{}/{}/{}".format(d, exp_type, t))
            for i in range(len(t_r_labels)):
                t_gap_list = [exp_type, t, t_r_labels[i]]
                t_kl_list = [exp_type, t, t_r_labels[i]]
                t_gap_list.extend(fit_len(t_gap_lists[i], iter_list))
                t_kl_list.extend(fit_len(t_kl_lists[i], iter_list))
                gap_rows.append(t_gap_list)
                kl_rows.append(t_kl_list)

os.makedirs('{}/results'.format(d), exist_ok=True)
write_to_csv("{}/results/gap_results.csv".format(d), gap_rows, header)
write_to_csv("{}/results/kl_results.csv".format(d), kl_rows, header)
