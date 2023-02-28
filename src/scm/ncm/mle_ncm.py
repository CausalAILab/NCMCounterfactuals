import itertools
import numpy as np
import torch as T
import torch.nn as nn

from src.scm.distribution.continuous_distribution import UniformDistribution
from src.scm.nn.gumbel_mlp import GumbelMLP
from src.scm.scm import SCM, expand_do
from src.ds.counterfactual import CTF


class MLE_NCM(SCM):
    def __init__(self, cg, v_size={}, default_v_size=1, u_size={},
                 default_u_size=1, f={}, hyperparams=None, default_module=GumbelMLP):

        if hyperparams is None:
            hyperparams = dict()

        self.cg = cg
        self.u_size = {k: u_size.get(k, default_u_size) for k in self.cg.c2}
        self.v_size = {k: v_size.get(k, default_v_size) for k in self.cg}
        super().__init__(
            v=list(cg),
            f=nn.ModuleDict({
                k: f[k] if k in f else default_module(
                    {k: self.v_size[k] for k in self.cg.pa[k]},
                    {k: self.u_size[k] for k in self.cg.v2c2[k]},
                    self.v_size[k],
                    h_layers=hyperparams.get('h-layers', 2),
                    h_size=hyperparams.get('h-size', 128)
                )
                for k in cg}),
            pu=UniformDistribution(self.cg.c2, self.u_size))

    def get_space(self, fixed):
        vals = []
        for k in self.v:
            if k in fixed:
                vals.append([fixed[k]])
            else:
                vals.append(list(range(2 ** self.v_size[k])))

        space = []
        for val_item in itertools.product(*vals):
            val_dict = dict()
            for i, k in enumerate(self.v):
                val_dict[k] = self.dec_to_bin(T.as_tensor(val_item[i]), self.v_size[k])
            space.append(val_dict)
        return space

    def likelihood(self, v_vals, u=None, skip=set(), mc_size=1):
        assert not skip.difference(self.v)

        if u is None:
            u = self.pu.sample(mc_size)
        else:
            mc_size = u[next(iter(u))].shape[0]

        expanded_vals = dict()
        for (k, v) in v_vals.items():
            if not T.is_tensor(v) or len(v.shape) == 1:
                expanded_vals[k] = expand_do(v, mc_size).float()
            else:
                expanded_vals[k] = v.to(self.device_param)
        log_pv = T.zeros(mc_size).to(self.device_param)
        for k in self.v:
            if k not in skip:
                log_pv += self.f[k](expanded_vals, u, expanded_vals[k])
        averaged_log_pv = T.logsumexp(log_pv, dim=0) - np.log(mc_size)
        return averaged_log_pv

    def sample(self, n=None, u=None, do={}, select=None):
        assert not set(do.keys()).difference(self.v)
        assert (n is None) != (u is None)

        for k in do:
            do[k] = do[k].to(self.device_param)

        if u is None:
            u = self.pu.sample(n)
        if select is None:
            select = self.v
        v = {}
        remaining = set(select)
        for k in self.v:
            v[k] = do[k] if k in do else self.f[k](v, u, n=n)
            remaining.discard(k)
            if not remaining:
                break
        return {k: v[k] for k in select}

    def compute_ctf(self, query: CTF, n=1000000, u=None, get_prob=True, evaluating=False):
        if evaluating:
            return super().compute_ctf(query, n=n, u=u, get_prob=get_prob, evaluating=evaluating)

        if len(query.cond_term_set) > 0:
            cond_ctf = CTF(query.cond_term_set)
            full_ctf = CTF(query.term_set.union(query.cond_term_set))
            log_p_full = self.compute_ctf(full_ctf, n=n, u=u, get_prob=get_prob, evaluating=evaluating)
            log_p_cond = self.compute_ctf(cond_ctf, n=n, u=u, get_prob=get_prob, evaluating=evaluating)
            return log_p_full - log_p_cond

        u = self.pu.sample(n)
        log_prob = 0
        for term in query.term_set:
            fixed_vars = dict()
            do_vars = set()
            nested_vars = dict()
            for (k, v) in term.do_vals.items():
                if k == "nested":
                    nested_vars.update(self.compute_ctf(v, u=u, get_prob=False, evaluating=True))
                else:
                    fixed_vars[k] = v
                    do_vars.add(k)

            fixed_vars.update(term.var_vals)

            space = self.get_space(fixed_vars)

            log_prob_vals = []
            for row in space:
                val = {k: v.byte().to(self.device_param) for (k, v) in row.items()}
                for (k, v) in nested_vars.items():
                    val[k] = v
                log_prob_val = self.likelihood(val, u=u, skip=do_vars)
                log_prob_vals.append(log_prob_val)
            log_prob += T.logsumexp(T.stack(log_prob_vals, dim=0), dim=0)

        return log_prob

    def dec_to_bin(self, x, bits):
        mask = 2 ** T.arange(bits - 1, -1, -1).to(x.device, x.dtype)
        return x.unsqueeze(-1).bitwise_and(mask).ne(0).float()
