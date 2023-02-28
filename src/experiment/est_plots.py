import pandas as pd
import argparse
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as mtrans


parser = argparse.ArgumentParser(description="Estimation Experiment Results Parser")
parser.add_argument('dir', help="directory of the experiment")
parser.add_argument('query', help="identified query")
parser.add_argument('--legend', action="store_true", help="add legends to plots")
args = parser.parse_args()

valid_queries = {"ATE", "ETT", "NDE", "CTFDE"}

d = args.dir
query = args.query.upper()
assert query in valid_queries

df = pd.read_csv("{}/est_data.csv".format(d))

df['n_samples'] = df['n_samples'].astype(int)
dims = df['dim'].unique()
graphs = df['graph'].unique()
cols_Q = ['err_ncm_Q_dim_{}'.format(dim) for dim in dims]
cols_KL = ['total_true_KL_dim_{}'.format(dim) for dim in dims]
for dim in dims:
    df["total_true_KL_dim_{}".format(dim)] = ""
    df["err_ncm_Q_dim_{}".format(dim)] = ""
    df.loc[df['dim'] == dim, 'err_ncm_Q_dim_{}'.format(dim)] = df.loc[
        df['dim'] == dim, 'err_ncm_{}'.format(query)].values
    df.loc[df['dim'] == dim, 'total_true_KL_dim_{}'.format(dim)] = df.loc[
        df['dim'] == dim, 'total_true_KL'].values

for col in cols_Q:
    df[col] = pd.to_numeric(df[col], errors='coerce').abs()
for col in cols_KL:
    df[col] = pd.to_numeric(df[col])

fig, axes = plt.subplots(2, len(graphs), sharex=True, sharey='row')
for g_ind, graph in enumerate(graphs):
    ax_tv = axes[0][g_ind]
    ax_ate = axes[1][g_ind]

    df2 = (df.query(f'graph == "{graph}"')
           .rename(lambda s: str(s).replace('err_', 'mae_'), axis=1))
    melt_ate = df2.melt('n_samples', [col.replace('err_', 'mae_') for col in cols_Q],
                    var_name='estimator', value_name='mae')
    sns_ate_ax = sns.lineplot(data=melt_ate, x='n_samples', y='mae', hue='estimator', marker='o', ax=ax_ate)

    melt_tv = df2.melt('n_samples', cols_KL,
                           var_name='estimator', value_name='kl_val')
    sns_tv_ax = sns.lineplot(data=melt_tv, x='n_samples', y='kl_val', hue='estimator', marker='o', ax=ax_tv)

    ax_ate.set_xlabel("")
    if not args.legend:
        ax_ate.get_legend().remove()
        ax_tv.get_legend().remove()

    axes[0][g_ind].set_title(graph)

axes[0][0].set_xscale("log")
axes[0][0].set_yscale("log")
axes[1][0].set_yscale("log")

axes[0][0].set_ylabel("Total KL of Available Distributions")
axes[1][0].set_ylabel("MAE of Query")

trans = mtrans.blended_transform_factory(fig.transFigure,
                                             mtrans.IdentityTransform())
xlab = fig.text(.5, 20, "Number of Training Samples (n)", ha='center', fontsize=24)
xlab.set_transform(trans)

fig.set_figheight(10)
fig.set_figwidth(21)
fig.tight_layout()
fig.subplots_adjust(wspace=0.1, hspace=0.1, bottom=0.1)
fig.savefig('{}/est_results.png'.format(d), dpi=300, bbox_inches='tight')