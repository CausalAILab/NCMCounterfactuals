import pandas as pd
import argparse
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as mtrans


def process_dims(data, dims, model, query):
    df = data
    for dim in dims:
        df["err_{}_Q_dim_{}".format(model, dim)] = ""
        df.loc[df['dim'] == dim, 'err_{}_Q_dim_{}'.format(model, dim)] = df.loc[
            df['dim'] == dim, 'err_{}_{}'.format(model, query.upper())].values
    df = df.loc[df['dim'].isin(dims)]

    for col in ["err_{}_Q_dim_{}".format(model, dim) for dim in dims]:
        df[col] = pd.to_numeric(df[col], errors='coerce').abs()

    df['n_samples'] = df['n_samples'].astype(int)
    return df


def get_grid_mae_plot(fig_name, data, id_pairings, column_names, num_rows, num_cols, colors, styles):
    fig, axes = plt.subplots(num_rows, num_cols, sharex=True, sharey=True)
    i = 0
    for row in range(num_rows):
        for col in range(num_cols):
            ax = axes[row][col]
            q = id_pairings[i][0]
            graph = id_pairings[i][1]
            i += 1

            df = data[q].loc[data[q]["graph"] == graph]
            melt_q = df.melt('n_samples', column_names, var_name='estimator', value_name='mae')
            sns_q_ax = sns.lineplot(data=melt_q, x='n_samples', y='mae', hue='estimator', style='estimator',
                                    marker='.', ax=ax, palette=colors, dashes=styles, linewidth=0.5, ci=83)

            plt.text(0.78, 0.78, str(i),
                     bbox=dict(boxstyle="circle", facecolor='white', edgecolor='black'), transform=ax.transAxes)

            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.get_legend().remove()

    axes[0][0].set_yscale("log")
    axes[0][0].set_xscale("log")

    fig.supylabel('MAE of Query Estimation')
    fig.supxlabel('Number of Samples (n)')

    fig.tight_layout()
    fig.subplots_adjust(wspace=0.1, hspace=0.1, bottom=0.1)
    fig.savefig(fig_name, dpi=300, bbox_inches='tight')


parser = argparse.ArgumentParser(description="Estimation Experiment Grid Plots")
parser.add_argument('dir', help="directory of the experiment")
args = parser.parse_args()

d = args.dir
queries = ['ate', 'ett', 'nde', 'ctfde']
id_pairings = [["ate", "expl"],
               ["ate", "expl_dox"],
               ["ett", "expl"],
               ["ett", "expl_dox"],
               ["nde", "expl"],
               ["nde", "expl_dox"],
               ["ctfde", "expl"],
               ["ctfde", "expl_dox"],
               ["ate", "expl_xm_dox"],
               ["nde", "expl_xm_dox"],
               ["ate", "expl_xy_dox"],
               ["nde", "expl_xy_dox"],
               ["ate", "expl_my"],
               ["ate", "expl_my_dox"],
               ["ett", "expl_my"],
               ["ett", "expl_my_dox"]
               ]
dims = [1, 16]
models = ["gan_ncm", "mle_ncm"]
colors = dict()
for dim in dims:
    colors["err_gan_ncm_Q_dim_{}".format(dim)] = "blue"
    colors["err_mle_ncm_Q_dim_{}".format(dim)] = "orange"
styles = dict()
for model in models:
    styles["err_{}_Q_dim_1".format(model)] = ''
    styles["err_{}_Q_dim_16".format(model)] = (3, 1)
column_names = []
for model in models:
    for dim in dims:
        if model == "mle_ncm" and dim != 1:
            continue
        if model != "gan_ncm" and dim != 1:
            continue
        column_names.append('err_{}_Q_dim_{}'.format(model, dim))

# Get data
est_data = dict()
for q in queries:
    gan_est_data = pd.read_csv("{}/gan_{}_est_data.csv".format(d, q))
    gan_est_data = gan_est_data.rename(columns={'err_ncm_{}'.format(q.upper()): 'err_gan_ncm_{}'.format(q.upper())})
    gan_est_data = process_dims(gan_est_data, dims, "gan_ncm", q)

    mle_est_data = pd.read_csv("{}/mle_{}_est_data.csv".format(d, q))
    mle_est_data = mle_est_data.rename(columns={'err_ncm_{}'.format(q.upper()): 'err_mle_ncm_{}'.format(q.upper())})
    mle_est_data = process_dims(mle_est_data, dims, "mle_ncm", q)

    est_data[q] = pd.concat([gan_est_data, mle_est_data], axis=0, ignore_index=True)

get_grid_mae_plot("{}/est_mae_plots.png".format(d), est_data, id_pairings, column_names, 4, 4, colors, styles)
