import numpy as np
import torch as T
import torch.nn as nn

from src.scm.distribution.continuous_distribution import UniformDistribution
from src.scm.scm import SCM
from src.ds.causal_graph import CausalGraph


class CTM(SCM):
    def __init__(self, cg, v_size={}, default_v_size=1, regions=4, c2_scale=1.0, batch_size=None, seed=None):
        self.cg = cg
        self.u_size = {k: 1 for k in self.cg.c2}
        self.v_size = {k: v_size.get(k, default_v_size) for k in self.cg}
        self.region_count = {k: int(regions * (c2_scale ** len(self.cg.v2c2[k]))) for k in self.cg}
        self.batch_size = batch_size

        if seed is not None:
            self.rand_state = np.random.RandomState(seed=seed)
        else:
            self.rand_state = np.random.RandomState()

        super().__init__(
            v=list(cg),
            f={V: self.get_ctm_func(V) for V in cg},
            pu=UniformDistribution(self.cg.c2, self.u_size))

    def get_ctm_func(self, V):
        v_pa = sorted(self.cg.pa[V])
        u_pa = sorted(self.cg.v2c2[V])

        outcomes = 2 ** (sum([self.v_size[k] for k in v_pa]))
        output_size = self.v_size[V]
        c2_size = len(u_pa)
        regions = []
        region_outputs = []
        for r in range(self.region_count[V]):
            intervals = [sorted(self.rand_state.rand(2)) for _ in range(c2_size)]
            output = self.rand_state.binomial(1, 0.5, size=(outcomes, output_size))
            regions.append(intervals)
            region_outputs.append(output)
        default_output = self.rand_state.binomial(1, 0.5, size=(outcomes, output_size))
        region_outputs.append(default_output)
        region_outputs = T.LongTensor(region_outputs)

        def ctm_func(v_raw, u_raw):
            v = {k: v.cpu() for (k, v) in v_raw.items()}
            u = {k: v.cpu() for (k, v) in u_raw.items()}

            u_key = next(iter(u))
            n = len(u[u_key])

            region_found = T.ones((n, 1), dtype=T.long) * len(regions)
            for i, region in enumerate(regions):
                in_region = T.ones((n, 1), dtype=T.bool)
                for j, u_name in enumerate(u_pa):
                    in_region *= (region[j][0] <= u[u_name]) * (u[u_name] < region[j][1])
                region_found[in_region] = i

            region_found = T.squeeze(region_found)
            used_func = region_outputs[region_found]

            if len(v_pa) == 0:
                return T.squeeze(used_func, dim=1)
            else:
                v_arr = T.cat([v[k] for k in v_pa], dim=1).long()
                v_ind = T.zeros(n, dtype=T.long)
                for i in range(v_arr.shape[1]):
                    v_ind = 2 * v_ind + v_arr[:, i]
                return used_func[range(n), v_ind]

        return ctm_func

    def sample(self, n=None, u=None, do={}, select=None):
        if self.batch_size is None:
            return super().sample(n=n, u=u, do=do, select=select)

        assert not set(do.keys()).difference(self.v)
        assert (n is None) != (u is None)

        if select is None:
            samp = {k: [] for k in self.v}
        else:
            samp = {k: [] for k in select}
        if n is None:
            u_key = next(iter(u))
            remaining = len(u[u_key])
        else:
            remaining = n

        i = 0
        while remaining > 0:
            if remaining > self.batch_size:
                if n is None:
                    new_n = None
                    new_u = {k: u[k][self.batch_size * i:self.batch_size * (i + 1)] for k in u}
                else:
                    new_n = self.batch_size
                    new_u = None
                new_do = {k: do[k][self.batch_size * i:self.batch_size * (i + 1)] for k in do}
                remaining -= self.batch_size
            else:
                if n is None:
                    new_n = None
                    new_u = {k: u[k][self.batch_size * i:] for k in u}
                else:
                    new_n = remaining
                    new_u = None
                new_do = {k: do[k][self.batch_size * i:] for k in do}
                remaining = 0

            batch = super().sample(n=new_n, u=new_u, do=new_do, select=select)
            for v in batch:
                samp[v].append(batch[v])
            i += 1

        samp = {k: T.cat(samp[k], dim=0) for k in samp}
        return samp


if __name__ == "__main__":
    cg = CausalGraph.read("../../dat/cg/zid_a.cg")
    m = CTM(cg, v_size={}, regions=20, c2_scale=1.0, batch_size=100000)
    result = m(20)
    print(result)
    for k in result:
        print("{}: {}".format(k, result[k].shape))
    print(m(10, do={'X': T.ones((10, 1), dtype=T.long)}))
