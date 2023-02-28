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


parser = argparse.ArgumentParser(description="Estimation Experiment Results Parser")
parser.add_argument('dir', help="directory of the experiment", default='out/EXPMLE')
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
        if record['dim'] not in dims:
            if int(record['dim']):
                dims.add(record['dim'])
        with open(f'{d}/time_results.json') as file:
            old_results = json.load(file)
            # record_1 = record.copy()
            # record['pipeline'] = 'MLE'
            record['MLE'] = old_results['mle']
        with open(f'{d}/time_results.json') as file:
            old_results = json.load(file)
            # record_2 = record.copy()
            # record['pipeline'] = 'GAN'
            record['GAN'] = old_results['gan']
        records.append(record)

    except Exception as e:
        print(d, e)

df = pd.DataFrame.from_records(records)
df['dim'] = df['dim'].astype(int)

cols = ['GAN', 'MLE']
for col in cols:
    df[col] = df[col].astype(float).abs()


fig, axes = plt.subplots(1, sharex=True, sharey='row')


ax_ate = axes
graph = 'expl'
df2 = df.query(f'graph == "{graph}"')
melt_runtime = df2.melt('dim', [col for col in cols],
                var_name='estimator', value_name='Runtime').sort_values(by=['dim', 'estimator'])


# print(melt_runtime)
sns_ate_ax = sns.lineplot(data=melt_runtime, x='dim', y='Runtime', hue='estimator', linewidth=2.5,
                          marker='o', err_style="band", ax=ax_ate, markersize=8,  palette=['green', 'orange'])


ax_ate.set_xlabel("")
ax_ate.set_xticks([1, 2, 3, 4, 5, 6, 7])
ax_ate.set_yticks([0, 200, 400, 600, 800])
ax_ate.get_legend().remove()
# axes.set_yscale("log")
axes.set_ylabel("Time (in seconds)")
trans = mtrans.blended_transform_factory(fig.transFigure,
                                             mtrans.IdentityTransform())
xlab = fig.text(.5, 30, "Dimension (d)", ha='center', fontsize=24)
xlab.set_transform(trans)

os.makedirs('img', exist_ok=True)
fig.set_figheight(5)
fig.set_figwidth(9)
fig.tight_layout()
fig.subplots_adjust(wspace=0.1, hspace=0.1, bottom=0.13)
fig.savefig(f'img/runtime_result.png', dpi=300, bbox_inches='tight')
