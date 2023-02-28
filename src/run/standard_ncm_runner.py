import os
import glob
import shutil
import hashlib
import json

import numpy as np
import torch as T
import pytorch_lightning as pl

from src.metric import evaluation
from src.ds.causal_graph import CausalGraph
from src.scm.ctm import CTM
from src.scm.scm import expand_do
from .base_runner import BaseRunner


class NCMRunner(BaseRunner):
    def __init__(self, pipeline, dat_model, ncm_model):
        super().__init__(pipeline, dat_model, ncm_model)

    def create_trainer(self, directory, gpu=None):
        checkpoint = pl.callbacks.ModelCheckpoint(dirpath=f'{directory}/checkpoints/', monitor="train_loss")
        return pl.Trainer(
            callbacks=[
                checkpoint,
                pl.callbacks.EarlyStopping(monitor='train_loss',
                                           patience=self.pipeline.patience,
                                           min_delta=self.pipeline.min_delta,
                                           check_on_train_epoch_end=True)
            ],
            max_epochs=self.pipeline.max_epochs,
            accumulate_grad_batches=1,
            logger=pl.loggers.TensorBoardLogger(f'{directory}/logs/'),
            log_every_n_steps=1,
            terminate_on_nan=True,
            gpus=gpu
        ), checkpoint

    def run(self, exp_name, cg_file, n, dim, trial_index, hyperparams=None, gpu=None,
            lockinfo=os.environ.get('SLURM_JOB_ID', ''), verbose=False):
        key = self.get_key(cg_file, n, dim, trial_index)
        d = 'out/%s/%s' % (exp_name, key)  # name of the output directory

        if hyperparams is None:
            hyperparams = dict()

        with self.lock(f'{d}/lock', lockinfo) as acquired_lock:
            if not acquired_lock:
                print('[locked]', d)
                return

            try:
                # return if best.th is generated (i.e. training is already complete)
                if os.path.isfile(f'{d}/best.th'):
                    print('[done]', d)
                    return

                # since training is not complete, delete all directory files except for the lock
                print('[running]', d)
                for file in glob.glob(f'{d}/*'):
                    if os.path.basename(file) != 'lock':
                        if os.path.isdir(file):
                            shutil.rmtree(file)
                        else:
                            try:
                                os.remove(file)
                            except FileNotFoundError:
                                pass

                # set random seed to a hash of the parameter settings for reproducibility
                seed = int(hashlib.sha512(key.encode()).hexdigest(), 16) & 0xffffffff
                T.manual_seed(seed)
                np.random.seed(seed)
                print('Key:', key)
                print('Seed:', seed)

                # generate data-generating model, data, and model
                print('Generating data')
                cg = CausalGraph.read(cg_file)
                if self.dat_model is CTM:
                    v_sizes = {k: 1 if k in {'X', 'Y', 'M', 'W'} else dim for k in cg}
                    dat_m = self.dat_model(cg, v_size=v_sizes, regions=hyperparams.get('regions', 20),
                                           c2_scale=hyperparams.get('c2-scale', 1.0),
                                           batch_size=hyperparams.get('gen-bs', 10000),
                                           seed=seed)
                else:
                    dat_m = self.dat_model(cg, dim=dim, seed=seed)

                dat_sets = []
                for dat_do_set in hyperparams["do-var-list"]:
                    expand_do_set = {k: expand_do(v, n=n) for (k, v) in dat_do_set.items()}
                    dat_sets.append(dat_m(n=n, do=expand_do_set))
                m = self.pipeline(dat_m, hyperparams["do-var-list"], dat_sets, cg, dim, hyperparams=hyperparams,
                                  ncm_model=self.ncm_model)

                # print info
                print("Calculating metrics")
                stored_metrics = dict()
                for i, dat_do_set in enumerate(hyperparams["do-var-list"]):
                    name = evaluation.serialize_do(dat_do_set)
                    stored_metrics["true_{}".format(name)] = evaluation.probability_table(
                        dat_m, n=1000000, do={k: expand_do(v, n=1000000) for (k, v) in dat_do_set.items()})
                    stored_metrics["dat_{}".format(name)] = evaluation.probability_table(
                        dat_m, n=1000000, do={k: expand_do(v, n=1000000) for (k, v) in dat_do_set.items()},
                        dat=dat_sets[i])
                start_metrics = evaluation.all_metrics(m.generator, m.ncm, hyperparams["do-var-list"], dat_sets,
                                                       n=1000000, stored=stored_metrics,
                                                       query_track=hyperparams['eval-query'])
                if hyperparams['query-track'] is not None:
                    true_q = 'true_{}'.format(evaluation.serialize_query(hyperparams['eval-query']))
                    stored_metrics[true_q] = start_metrics[true_q]
                m.update_metrics(stored_metrics)

                # train model
                if gpu is None:
                    gpu = int(T.cuda.is_available())
                trainer, checkpoint = self.create_trainer(d, gpu)
                trainer.fit(m)
                ckpt = T.load(checkpoint.best_model_path)
                m.load_state_dict(ckpt['state_dict'])
                results = evaluation.all_metrics(m.generator, m.ncm, hyperparams["do-var-list"], dat_sets,
                                                 n=1000000, query_track=hyperparams['eval-query'])
                print(results)

                # save results
                with open(f'{d}/results.json', 'w') as file:
                    json.dump(results, file)
                with open(f'{d}/hyperparams.json', 'w') as file:
                    new_hp = {k: str(v) for (k, v) in hyperparams.items()}
                    json.dump(new_hp, file)
                T.save(dat_sets, f'{d}/dat.th')
                T.save(m.state_dict(), f'{d}/best.th')

                return m, results
            except Exception:
                # move out/*/* to err/*/*/#
                e = d.replace("out/", "err/").rsplit('-', 1)[0]
                e_index = len(glob.glob(e + '/*'))
                e += '/%s' % e_index
                os.makedirs(e.rsplit('/', 1)[0], exist_ok=True)
                shutil.move(d, e)
                print(f'moved {d} to {e}')
                raise
