import numpy as np
import torch as T

from .distribution import Distribution


class DiscreteDistribution(Distribution):
    def __init__(self, u):
        super().__init__(u)


class BernoulliDistribution(DiscreteDistribution):
    def __init__(self, u_names, sizes, p, seed=None):
        assert set(sizes.keys()).issubset(set(u_names))

        super().__init__(list(u_names))
        self.sizes = {U: sizes[U] if U in sizes else 1 for U in u_names}
        self.p = p
        if seed is not None:
            self.rand_state = np.random.RandomState(seed=seed)
        else:
            self.rand_state = np.random.RandomState()

    def sample(self, n=1, device=None):
        if device is None:
            device = self.device_param.device

        u_vals = dict()
        for U in self.sizes:
            u_vals[U] = T.from_numpy(self.rand_state.binomial(1, self.p, size=(n, self.sizes[U]))).long().to(device)

        return u_vals


class SplitBernoulliDistribution(DiscreteDistribution):
    def __init__(self, u1_names, u2_names, sizes, p1, p2, seed=None):
        all_u_names = set(u1_names + u2_names)

        assert set(sizes.keys()).issubset(all_u_names)

        super().__init__(list(all_u_names))
        self.u1_names = u1_names
        self.u2_names = u2_names
        self.sizes = {U: sizes[U] if U in sizes else 1 for U in all_u_names}
        self.p1 = p1
        self.p2 = p2
        if seed is not None:
            self.rand_state = np.random.RandomState(seed=seed)
        else:
            self.rand_state = np.random.RandomState()

    def sample(self, n=1, device=None):
        if device is None:
            device = self.device_param.device

        u_vals = dict()
        for U in self.u1_names:
            u_vals[U] = T.from_numpy(self.rand_state.binomial(1, self.p1, size=(n, self.sizes[U]))).long().to(device)
        for U in self.u2_names:
            u_vals[U] = T.from_numpy(self.rand_state.binomial(1, self.p2, size=(n, self.sizes[U]))).long().to(device)

        return u_vals
