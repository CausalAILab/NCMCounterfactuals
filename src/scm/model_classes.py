import numpy as np
import torch

from src.scm.scm import SCM
from src.ds.causal_graph import CausalGraph
from src.scm.distribution.discrete_distribution import BernoulliDistribution, SplitBernoulliDistribution


class XORModel(SCM):
    def __init__(self, cg, dim=1, p=0.02, seed=None):
        self.cg = cg
        self.dim = dim
        self.p = p

        sizes = {k: 1 if k in {'X', 'Y', 'M'} else dim for k in cg}

        self.confounders = {V: [] for V in self.cg.v}
        for V1, V2 in cg.bi:
            conf_name = "U_{}{}".format(V1, V2)
            self.confounders[V1].append(conf_name)
            self.confounders[V2].append(conf_name)
            sizes[conf_name] = 1

        super().__init__(
            v=list(cg),
            f={V: self.get_xor_func(V) for V in cg},
            pu=BernoulliDistribution(list(sizes.keys()), sizes, p=p, seed=seed))

    def get_xor_func(self, V):
        conf_list = self.confounders[V]
        par_list = self.cg.pa[V]

        def xor_func(v, u):
            values = u[V]

            for conf in conf_list:
                values = torch.bitwise_xor(values, u[conf])
            for par in par_list:
                par_samp = v[par].long()
                if values.shape[1] >= par_samp.shape[1]:
                    values = torch.bitwise_xor(values, par_samp)
                else:
                    par_samp = torch.unsqueeze(torch.remainder(torch.sum(par_samp, 1), 2), 1)
                    values = torch.bitwise_xor(values, par_samp)

            return values

        return xor_func


class RoundModel(SCM):
    def __init__(self, cg, dim=1, p1=0.5, p2=0.2, seed=None):
        self.cg = cg
        self.dim = dim
        self.p1 = p1
        self.p2 = p2

        u1_names = []
        u2_names = []
        sizes = {k: 1 if k in {'X', 'Y', 'M'} else dim for k in cg}
        for V in cg.v:
            if len(cg.pa[V]) == 0:
                u1_names.append(V)
            else:
                u2_names.append(V)

        self.confounders = {V: [] for V in self.cg.v}
        for V1, V2 in cg.bi:
            conf_name = "U_{}{}".format(V1, V2)
            self.confounders[V1].append(conf_name)
            self.confounders[V2].append(conf_name)
            sizes[conf_name] = 1
            u2_names.append(conf_name)

        super().__init__(
            v=list(cg),
            f={V: self.get_round_func(V) for V in cg},
            pu=SplitBernoulliDistribution(u1_names, u2_names, sizes, p1=p1, p2=p2, seed=seed))

    def get_round_func(self, V):
        conf_list = self.confounders[V]
        par_list = self.cg.pa[V]

        def round_func(v, u):
            values = u[V]

            for conf in conf_list:
                values = torch.bitwise_or(values, u[conf])

            cumul_par_val = 0
            for par in par_list:
                par_val = torch.round(torch.mean(v[par].float(), dim=1) + 0.0000001).long()
                cumul_par_val = torch.bitwise_or(torch.unsqueeze(par_val, dim=1), cumul_par_val)

            values = torch.bitwise_xor(values, cumul_par_val)

            return values

        return round_func


if __name__ == "__main__":
    cg = CausalGraph.read("../../dat/cg/bow.cg")
    m = XORModel(cg, dim=1, p=0.25)
    print(m.compute_query({'Y': 1}, {}, {'X': 1}, evaluating=True))
    print(m.compute_query({'Y': 1}, {'X': 1}, {}, evaluating=True))
