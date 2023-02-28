import glob
import json
import os
import sys

import matplotlib.pyplot as plt
import matplotlib.transforms as mtrans
import numpy as np
import pandas as pd
import argparse
import seaborn as sns
from tqdm.auto import tqdm

graph_Q = {
    "gid_a": "P(Y = 1 | do(X1 = 0, X2 = 0))",
    "gid_b": "P(Y = 1 | do(X1 = 0, X2 = 0))"
}

parser = argparse.ArgumentParser(description="Estimation Experiment Results Parser")
parser.add_argument('dir', help="directory of the experiment")
parser.add_argument('--legend', action="store_true", help="add legends to plots")
args = parser.parse_args()

d = args.dir

sns.set_style('darkgrid')
plt.rcParams.update({'font.size': 24})

records = []
dims = set()
for d in tqdm(glob.glob('{}/*'.format(d))):
    try:
        record = {}
        record['key'] = os.path.basename(d)
        record.update(map(lambda t: tuple(t.split('=')), record['key'].split('-')))
        q_name = "ate"
        if record['graph'] in graph_Q:
            q_name = graph_Q[record['graph']]
        if record['dim'] not in dims:
            dims.add(record['dim'])
        with open(f'{d}/results.json') as file:
            old_results = json.load(file)
            record['total_true_KL_dim_{}'.format(record['dim'])] = old_results['total_true_KL']
            record['err_ncm_Q_dim_{}'.format(record['dim'])] = old_results['err_ncm_{}'.format(q_name)]
        records.append(record)
    except Exception as e:
        print(d, e)

dims = sorted(list(dims))

df = pd.DataFrame.from_records(records)
df['n_samples'] = df['n_samples'].astype(int)

cols = ['err_ncm_Q_dim_{}'.format(d) for d in dims]
for col in cols:
    df[col] = df[col].astype(float).abs()


order = ["zid_a", "gid_a", "gid_b"]
fig, axes = plt.subplots(2, len(order), sharex=True, sharey='row')
for g_ind, graph in enumerate(order):
    ax_tv = axes[0][g_ind]
    ax_ate = axes[1][g_ind]

    df2 = (df.query(f'graph == "{graph}"')
           .rename(lambda s: str(s).replace('err_', 'mae_'), axis=1))
    melt_ate = df2.melt('n_samples', [col.replace('err_', 'mae_') for col in cols],
                    var_name='estimator', value_name='mae')
    sns_ate_ax = sns.lineplot(data=melt_ate, x='n_samples', y='mae', hue='estimator', marker='o', ax=ax_ate)

    melt_tv = df2.melt('n_samples', ['total_true_KL_dim_{}'.format(d) for d in dims],
                           var_name='estimator', value_name='kl_val')
    sns_tv_ax = sns.lineplot(data=melt_tv, x='n_samples', y='kl_val', hue='estimator', marker='o', ax=ax_tv)

    ax_ate.set_xlabel("")
    if not args.legend:
        ax_ate.get_legend().remove()
        ax_tv.get_legend().remove()

axes[0][0].set_xscale("log")
axes[0][0].set_yscale("log")
axes[1][0].set_yscale("log")

axes[0][0].set_ylabel("Total KL of Available Distributions")
axes[1][0].set_ylabel("MAE of Query")

trans = mtrans.blended_transform_factory(fig.transFigure,
                                             mtrans.IdentityTransform())
xlab = fig.text(.5, 20, "Number of Training Samples (n)", ha='center', fontsize=24)
xlab.set_transform(trans)

os.makedirs('img', exist_ok=True)
fig.set_figheight(10)
fig.set_figwidth(21)
fig.tight_layout()
fig.subplots_adjust(wspace=0.1, hspace=0.1, bottom=0.1)
fig.savefig(f'img/est_results.png', dpi=300, bbox_inches='tight')
