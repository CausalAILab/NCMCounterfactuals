import numpy as np
import torch as T
import torch.nn as nn

from src.scm.distribution.continuous_distribution import UniformDistribution
from src.scm.nn.mlp import MLP
from src.scm.scm import SCM


class FF_NCM(SCM):
    def __init__(self, cg, v_size={}, default_v_size=1, u_size={},
                 default_u_size=1, f={}, hyperparams=None, default_module=MLP):
        if hyperparams is None:
            hyperparams = dict()

        self.cg = cg
        self.u_size = {k: u_size.get(k, default_u_size) for k in self.cg.c2}
        self.v_size = {k: v_size.get(k, default_v_size) for k in self.cg}
        super().__init__(
            v=list(cg),
            f=nn.ModuleDict({
                v: f[v] if v in f else default_module(
                    {k: self.v_size[k] for k in self.cg.pa[v]},
                    {k: self.u_size[k] for k in self.cg.v2c2[v]},
                    self.v_size[v],
                    h_size=hyperparams.get('h-size', 128)
                )
                for v in cg}),
            pu=UniformDistribution(self.cg.c2, self.u_size))

    def convert_evaluation(self, samples):
        return {k: T.round(samples[k]) for k in samples}
