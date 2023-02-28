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


class NCMMinMaxRunner(BaseRunner):
    def __init__(self, pipeline, dat_model, ncm_model):
        super().__init__(pipeline, dat_model, ncm_model)

    def create_trainer(self, directory, max_epochs, r, gpu=None):
        checkpoint = pl.callbacks.ModelCheckpoint(dirpath=f'{directory}/{r}/checkpoints/', monitor="train_loss")
        return pl.Trainer(
            callbacks=[
                checkpoint
            ],
            max_epochs=max_epochs,
            accumulate_grad_batches=1,
            logger=pl.loggers.TensorBoardLogger(f'{directory}/{r}/logs/'),
            log_every_n_steps=1,
            terminate_on_nan=True,
            gpus=gpu
        ), checkpoint

    def print_metrics(self, pl_model, do_var_list, dat_sets, verbose=False, stored_metrics=None, query_track=None):
        if stored_metrics is None:
            stored_metrics = dict()

        print("Calculating metrics")
        for i, dat_do_set in enumerate(do_var_list):
            name = evaluation.serialize_do(dat_do_set)
            if "true_{}".format(name) not in stored_metrics:
                stored_metrics["true_{}".format(name)] = evaluation.probability_table(
                    pl_model.generator, n=1000000, do={k: expand_do(v, n=1000000) for (k, v) in dat_do_set.items()})
            if "dat_{}".format(name) not in stored_metrics:
                stored_metrics["dat_{}".format(name)] = evaluation.probability_table(
                    pl_model.generator, n=1000000, do={k: expand_do(v, n=1000000) for (k, v) in dat_do_set.items()},
                    dat=dat_sets[i])
        start_metrics = evaluation.all_metrics(pl_model.generator, pl_model.ncm, do_var_list, dat_sets,
                                               n=1000000, stored=stored_metrics,
                                               query_track=query_track)
        if verbose:
            print(start_metrics)

        if query_track is not None:
            true_q = 'true_{}'.format(evaluation.serialize_query(query_track))
            if true_q not in stored_metrics:
                stored_metrics[true_q] = start_metrics[true_q]

        pl_model.update_metrics(stored_metrics)

        return stored_metrics

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

                # set random seed to a hash of the parameter settings for reproducibility
                seed = int(hashlib.sha512(key.encode()).hexdigest(), 16) & 0xffffffff

                # ensure the positivity assumption holds
                positivity = False
                while not positivity:
                    seed += 1
                    T.manual_seed(seed)
                    np.random.seed(seed)
                    print('Key:', key)
                    print('Seed:', seed)

                    # generate data-generating model, data, and model
                    print('Generating data')
                    cg = CausalGraph.read(cg_file)
                    v_sizes = {k: 1 if k in {'X', 'Y', 'M', 'W'} else dim for k in cg}
                    if self.dat_model is CTM:
                        dat_m = self.dat_model(cg, v_size=v_sizes, regions=hyperparams.get('regions', 20),
                                               c2_scale=hyperparams.get('c2-scale', 1.0),
                                               batch_size=hyperparams.get('gen-bs', 10000),
                                               seed=seed)
                    else:
                        dat_m = self.dat_model(cg, dim=dim, seed=seed)

                    # checks if the data has already been generated
                    if os.path.isfile(f'{d}/dat.th'):
                        dat_sets = T.load(f'{d}/dat.th')
                        positivity = True
                    else:
                        dat_sets = []
                        all_positive = True
                        for dat_do_set in hyperparams["do-var-list"]:
                            var_dims = 0
                            for k in v_sizes:
                                if k not in dat_do_set:
                                    var_dims += v_sizes[k]

                            expand_do_set = {k: expand_do(v, n=n) for (k, v) in dat_do_set.items()}
                            dat_set = dat_m(n=n, do=expand_do_set)
                            if hyperparams["positivity"]:
                                prob_table = evaluation.probability_table(dat=dat_set)
                                if len(prob_table) != (2 ** var_dims):
                                    all_positive = False
                                    print(prob_table)

                            dat_sets.append(dat_m(n=n, do=expand_do_set))

                        positivity = all_positive

                        if positivity:
                            T.save(dat_sets, f'{d}/dat.th')

                with open(f'{d}/hyperparams.json', 'w') as file:
                    new_hp = {k: str(v) for (k, v) in hyperparams.items()}
                    json.dump(new_hp, file)

                if gpu is None:
                    gpu = int(T.cuda.is_available())
                stored_metrics = dict()
                for r in range(hyperparams.get("id-reruns", 1)):
                    os.makedirs(f'{d}/{r}/', exist_ok=True)
                    if not os.path.isfile(f'{d}/{r}/best_max.th'):
                        # remove all files
                        for file in glob.glob(f'{d}/{r}/*'):
                            if os.path.isdir(file):
                                shutil.rmtree(file)
                            else:
                                try:
                                    os.remove(file)
                                except FileNotFoundError:
                                    pass

                        # reset seed
                        new_key = "{}-run={}".format(key, r)
                        seed = int(hashlib.sha512(new_key.encode()).hexdigest(), 16) & 0xffffffff
                        T.manual_seed(seed)
                        np.random.seed(seed)
                        print("Run {} seed: {}".format(r, seed))

                        m_max = self.pipeline(dat_m, hyperparams["do-var-list"], dat_sets, cg, dim,
                                              hyperparams=hyperparams, ncm_model=self.ncm_model,
                                              max_query=hyperparams.get('max-query-1', None))
                        m_min = self.pipeline(dat_m, hyperparams["do-var-list"], dat_sets, cg, dim,
                                              hyperparams=hyperparams, ncm_model=self.ncm_model,
                                              max_query=hyperparams.get('max-query-2', None))

                        # train models
                        trainer_max, checkpoint_max = self.create_trainer(d, hyperparams.get('max-query-iters', 3000),
                                                                          r, gpu)
                        trainer_min, checkpoint_min = self.create_trainer(d, hyperparams.get('max-query-iters', 3000),
                                                                          r, gpu)

                        print("\nTraining max model...")
                        stored_metrics = self.print_metrics(m_max, hyperparams['do-var-list'], dat_sets,
                                                            verbose=verbose, stored_metrics=stored_metrics,
                                                            query_track=hyperparams['eval-query'])
                        trainer_max.fit(m_max)
                        ckpt_max = T.load(checkpoint_max.best_model_path)
                        m_max.load_state_dict(ckpt_max['state_dict'])

                        print("\nTraining min model...")
                        stored_metrics = self.print_metrics(m_min, hyperparams['do-var-list'], dat_sets,
                                                            verbose=verbose, stored_metrics=stored_metrics,
                                                            query_track=hyperparams['eval-query'])
                        trainer_min.fit(m_min)
                        ckpt_min = T.load(checkpoint_min.best_model_path)
                        m_min.load_state_dict(ckpt_min['state_dict'])

                        results = evaluation.all_metrics_minmax(
                            m_max.generator, m_min.ncm, m_max.ncm, hyperparams['do-var-list'], dat_sets,
                            n=100000, query_track=hyperparams['eval-query'])
                        print(results)

                        # save results
                        with open(f'{d}/{r}/results.json', 'w') as file:
                            json.dump(results, file)
                        T.save(m_min.state_dict(), f'{d}/{r}/best_min.th')
                        T.save(m_max.state_dict(), f'{d}/{r}/best_max.th')

                    else:
                        print("Done with run {}".format(r))

                T.save(dict(), f'{d}/best.th')  # breadcrumb file
                return True
            except Exception:
                # move out/*/* to err/*/*/#
                e = d.replace("out/", "err/").rsplit('-', 1)[0]
                e_index = len(glob.glob(e + '/*'))
                e += '/%s' % e_index
                os.makedirs(e.rsplit('/', 1)[0], exist_ok=True)
                shutil.move(d, e)
                print(f'moved {d} to {e}')
                raise
