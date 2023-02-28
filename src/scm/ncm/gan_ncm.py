import numpy as np
import torch as T
import torch.nn as nn

from src.scm.distribution.continuous_distribution import UniformDistribution, NeuralDistribution
from src.scm.nn.mlp import MLP
from src.scm.scm import SCM, expand_do


class GAN_NCM(SCM):
    def __init__(self, cg, v_size={}, default_v_size=1, u_size={},
                 default_u_size=1, f={}, hyperparams=None,
                 default_gen_module=MLP, disc_module=MLP, gen_use_sigmoid=True, disc_use_sigmoid=True):
        if hyperparams is None:
            hyperparams = dict()

        self.cg = cg
        self.u_size = {k: u_size.get(k, default_u_size) for k in self.cg.c2}
        self.v_size = {k: v_size.get(k, default_v_size) for k in self.cg}

        self.gen_use_sigmoid = gen_use_sigmoid

        gens = nn.ModuleDict({
                v: f[v] if v in f else default_gen_module(
                    {k: self.v_size[k] for k in self.cg.pa[v]},
                    {k: self.u_size[k] for k in self.cg.v2c2[v]},
                    self.v_size[v],
                    h_layers=hyperparams.get('h-layers', 2),
                    h_size=hyperparams.get('h-size', 128),
                    use_layer_norm=hyperparams.get('layer-norm', False),
                    use_sigmoid=gen_use_sigmoid
                )
                for v in cg})

        if hyperparams['neural-pu']:
            pu_dist = NeuralDistribution(self.cg.c2, self.u_size, hyperparams)
        else:
            pu_dist = UniformDistribution(self.cg.c2, self.u_size)


        super().__init__(
            v=list(cg),
            f=gens,
            pu=pu_dist
        )

        self.single_disc = hyperparams['single-disc']
        self.do_set_count = len(hyperparams['do-var-list'])

        if self.single_disc:
            disc_sizes = {k: v for (k, v) in self.v_size.items()}
            disc_sizes['_do_choice'] = self.do_set_count
            self.f_disc = disc_module(
                disc_sizes,
                {},
                1,
                h_layers=hyperparams.get('h-layers', 2),
                h_size=len(self.v_size) * hyperparams.get('h-size', 128),
                use_sigmoid=disc_use_sigmoid,
                use_layer_norm=hyperparams.get('layer-norm', False)
            )
        else:
            self.f_disc = nn.ModuleList([
                disc_module(
                    self.v_size,
                    {},
                    1,
                    h_layers=hyperparams.get('h-layers', 2),
                    h_size=len(self.v_size) * hyperparams.get('h-size', 128),
                    use_sigmoid=disc_use_sigmoid,
                    use_layer_norm=hyperparams.get('layer-norm', False)
                )
                for _ in range(len(hyperparams["do-var-list"]))
            ])

    def convert_evaluation(self, samples):
        return {k: T.gt(samples[k], 0.5).float() for k in samples}

    def query_loss(self, input, val):
        if self.gen_use_sigmoid:
            return super().query_loss(input, val)
        else:
            if T.is_tensor(val):
                raise NotImplementedError()
            else:
                return T.sum(T.square(input - val))

    def get_disc_outputs(self, samples, index, include_inp=False):
        if self.single_disc:
            n = len(samples[next(iter(samples))])
            one_hot_do = [0 for _ in range(self.do_set_count)]
            one_hot_do[index] = 1
            one_hot_do = T.FloatTensor(one_hot_do).to(self.device_param)
            inp = {k: v for (k, v) in samples.items()}
            inp['_do_choice'] = expand_do(one_hot_do, n)
            return self.f_disc(inp, {}, include_inp=include_inp)
        else:
            return self.f_disc[index](samples, {}, include_inp=include_inp)
