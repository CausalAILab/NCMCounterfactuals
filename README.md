# Neural Causal Models for Counterfactuals

This repository contains the code for the paper ["Neural Causal Models for Counterfactual Identification and Estimation"](https://arxiv.org/abs/2210.00035) by Kevin Xia, Yushu Pan, and Elias Bareinboim.

Disclaimer: This code is offered publicly with the MIT License, meaning that others can use it for any purpose, but we are not liable for any issues arising from the use of this code. Please cite our work if you found this code useful.

## Setup

```
python -m pip install -r requirements.txt
```

## Running the code

Both identification and estimation procedures can be run using the `main.py` file with the desired arguments entered.

To run the identification experiments using the GAN-NCM, navigate to the base directory of the repository and run

```
python -m src.main <NAME> gan --lr 2e-5 --data-bs 256 --ncm-bs 256 --h-size 64 --u-size 2 --layer-norm --gan-mode wgangp --d-iters 1 --id-query <QUERY> -r 4 --max-lambda 1e-4 --min-lambda 1e-5 --max-query-iters 1000 --single-disc --gen-sigmoid --mc-sample-size 256 -G expl_set -t 20 -n 10000 -d <DIM> --gpu 0
```
where `<NAME>` is replaced with the name of the folder in which the results will be saved, `<QUERY>` is one of `ate`,
`ett`, `nde`, `ctfde`, depending on the query being identified, and `<DIM>` is the dimensionality of the Z variable
(1 or 16 in the paper).

The MLE-NCM can be run similarly using the command
```
python -m src.main <NAME> mle --full-batch --h-size 64 --id-query <QUERY> -r 4 --max-query-iters 1000 --mc-sample-size 10000 -G expl_set -t 20 -n 10000 -d <DIM> --gpu 0
```

Once completed, the results of the ID experiments can be aggregated into data files using the command
```
python -m src.experiment.id_data_process <DIR> <QUERY>
```
where `<DIR>` is the directory of the experiment folder, and `<QUERY>` is the query that is being identified in that
experiment. This will create a folder in that directory named `results` and add two files named `gap_results.csv`
and `kl_results.csv`.

When all queries are run for both the GAN and MLE method, move all results files to the same folder and rename them as
`<MODEL>_<DIM>_<QUERY>_gap_results.csv` and `<MODEL>_<DIM>_<QUERY>_kl_results.csv`, where `<MODEL>` refers to `gan` or
`mle`, and `<DIM>` refers to `1d` or `16d`. Finally, run
```
python -m src.experiment.id_grid_plot <DIR> expl --tau 0.05
```
where `<DIR>` is the folder will all results files. Plots will appear in a new `results` folder inside that directory.

To run the estimation experiments using the GAN-NCM, navigate to the base directory of the repository and run
```
python -m src.main <NAME> gan --lr 0.00002 --data-bs 256 --ncm-bs 256 --h-size 64 --u-size 2 --layer-norm --gan-mode wgangp --d-iters 1 --query-track <QUERY> --single-disc --gen-sigmoid --mc-sample-size 1000 -G expl_set -t 20 -n -1 -d -1 --gpu 0
```

Similarly for the MLE-NCM:
```
python -m src.main <NAME> mle --full-batch --h-size 64 --query-track <QUERY> --max-query-iters 1000 --mc-sample-size 10000 -G expl_set -t 20 -n -1 -d 1 --gpu 0
```

Once completed, the results of the estimation experiments can be aggregated into data files using the command
```
python -m src.experiment.est_data_process <DIR>
```
where `<DIR>` is the directory of the experiment folder. This will create a folder in that directory named `results`
and add a file named `est_data.csv`.

When all queries are run for both the GAN and MLE method, move all results files to the same folder and rename them as
`<MODEL>_<QUERY>_est_data.csv`, where `<MODEL>` refers to `gan` or `mle`. Finally, run
```
python -m src.experiment.est_grid_plot <DIR>
```
where `<DIR>` is the folder will all results files. Plots will appear inside that directory.

Finally, to run the runtime experiments, navigate to the base directory and run the command
```
python -m runtime_main <NAME> -G expl --gen xor -n 10000 --gpu 0
```

and the plots can be generated with
```
python -m src.experiment.runtime_results <DIR>
```
