import numpy as np
import pandas as pd
import torch as T
from torch.autograd import grad

from src.metric.evaluation import all_metrics, probability_table
from src.ds.causal_graph import CausalGraph
from src.scm.ncm.mle_ncm import MLE_NCM
from src.scm.scm import expand_do
from src.metric import evaluation
from src.ds.counterfactual import CTF

from .base_pipeline import BasePipeline


class MLEPipeline(BasePipeline):
    patience = 400

    def __init__(self, generator, do_var_list, dat_sets, cg, dim, hyperparams=None, ncm_model=MLE_NCM, max_query=None):
        if hyperparams is None:
            hyperparams = dict()
        v_size = {k: 1 if k in {'X', 'Y', 'M', 'W'} else dim for k in cg}
        ncm = ncm_model(cg, v_size=v_size, default_u_size=hyperparams.get('u-size', 1), hyperparams=hyperparams,)

        super().__init__(generator, do_var_list, dat_sets, cg, dim, ncm, batch_size=hyperparams.get('data-bs', 1000))

        if isinstance(max_query, CTF):
            self.max_query = [max_query]
        else:
            self.max_query = max_query
        self.max_query_iters = hyperparams.get("max-query-iters", 3000)
        self.query_track = hyperparams["eval-query"]

        self.do_var_list = hyperparams["do-var-list"]

        self.ncm_batch_size = hyperparams.get('ncm-bs', 1000)
        self.lr = hyperparams.get('lr', 4e-3)
        self.mc_sample_size = hyperparams.get('mc-sample-size', 10000)
        self.min_lambda = hyperparams.get('min-lambda', 0.001)
        self.max_lambda = hyperparams.get('max-lambda', 1.0)
        self.ordered_v = cg.v

        self.data_counts = None
        self.full_batch = hyperparams["full-batch"]
        if self.full_batch:
            self.data_counts = []
            for i, do_set in enumerate(self.do_var_list):
                n = dat_sets[i][next(iter(dat_sets[i]))].shape[0]
                self.data_counts.append(self._get_data_counts(data=dat_sets[i], n=n))

        self.logged = False
        self.stored_kl = 1e8
        self.automatic_optimization = False

    def forward(self, n=1000, u=None, do={}):
        return self.ncm(n, u, do)

    def configure_optimizers(self):
        optim = T.optim.AdamW(self.ncm.parameters(), lr=self.lr)
        return {
            'optimizer': optim,
            'lr_scheduler': T.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optim, 50, 1, eta_min=1e-4)
        }

    '''
    def _get_data_counts(self, data, n, do_set=None):
        counts = dict()
        marg_set = set(self.ncm.v).difference(do_set)
        space_z0 = self.ncm.space(self.ncm.v_size, select=marg_set)
        for v in space_z0:
            v_joined = dict()
            v_joined.update(v)
            v_joined.update(do_set)
            data_point = {k: tuple(v.cpu().tolist()) for (k, v) in v_joined.items()}
            point_key = frozenset(data_point.items())
            counts[point_key] = 0
        for i in range(n):
            data_point = {k: tuple(v[i].cpu().tolist()) for (k, v) in data.items()}
            point_key = frozenset(data_point.items())
            counts[point_key] = counts[point_key] + 1
        return counts
    '''

    def _get_data_counts(self, data, n, do_set=None):
        counts = dict()
        for i in range(n):
            data_point = {k: tuple(v[i].cpu().tolist()) for (k, v) in data.items()}
            point_key = frozenset(data_point.items())
            counts[point_key] = counts.get(point_key, 0) + 1
        return counts

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        data_n = self.ncm_batch_size

        reg_ratio = min(self.current_epoch, self.max_query_iters) / self.max_query_iters
        reg_up = np.log(self.max_lambda)
        reg_low = np.log(self.min_lambda)
        max_reg = np.exp(reg_up - reg_ratio * (reg_up - reg_low))

        opt.zero_grad()
        loss = 0
        for i, do_set in enumerate(self.do_var_list):
            data_n = batch[i][next(iter(batch[i]))].shape[0]
            do_set_vars = set(do_set.keys())
            if self.full_batch:
                data_counts = self.data_counts[i]
            else:
                data_counts = self._get_data_counts(batch[i], data_n, do_set)

            for point, count in data_counts.items():
                data_point = {k: T.ByteTensor(v).to(device=self.device) for (k, v) in point}
                log_pv = self.ncm.likelihood(data_point, skip=do_set_vars, mc_size=self.mc_sample_size)
                loss -= count * log_pv
        loss = loss / data_n
        loss_record = loss.item()
        self.manual_backward(loss)

        q_loss_record = 0
        if self.max_query is not None:
            q_loss = 0
            for query in self.max_query:
                q_loss -= self.ncm.compute_ctf(query, n=self.mc_sample_size)
            q_loss = max_reg * q_loss
            q_loss_record = q_loss.item()
            self.manual_backward(q_loss)

        opt.step()

        # logging
        if (self.current_epoch + 1) % 10 == 0:
            if not self.logged:
                results = all_metrics(self.generator, self.ncm, self.do_var_list, self.dat_sets,
                                      n=100000, stored=self.stored_metrics, query_track=self.query_track)
                for k, v in results.items():
                    self.log(k, v)

                print(pd.Series(results))
                print("\nLambda: {}".format(max_reg))

                self.logged = True
                self.stored_kl = results["total_dat_KL"]
        else:
            self.logged = False

        self.log('train_loss', self.stored_kl, prog_bar=True)
        self.log('P_loss', loss_record, prog_bar=True)
        self.log('Q_loss', q_loss_record, prog_bar=True)

