import itertools

import numpy as np
import torch as T
import torch.nn as nn

from .distribution.distribution import Distribution
from src.ds.counterfactual import CTF


def log(x):
    return T.log(x + 1e-8)


def expand_do(val, n):
    if T.is_tensor(val):
        return T.tile(val, (n, 1))
    else:
        return T.unsqueeze(T.ones(n, dtype=float) * val, 1)


def check_equal(input, val):
    if T.is_tensor(val):
        return T.all(T.eq(input, T.tile(val, (input.shape[0], 1))), dim=1).bool()
    else:
        return T.squeeze(input == val)

def soft_equals(input, val):
    if T.is_tensor(val):
        return T.sum(T.abs(T.tile(val, (input.shape[0], 1)) - input), dim=1)
    else:
        return T.squeeze(T.abs(val - input))

def cross_entropy_compare(input, val):
    if T.is_tensor(val):
        raise NotImplementedError()
    else:
        if val == 1:
            return T.sum(-log(input))
        elif val == 0:
            return T.sum(-log(1 - input))
        else:
            raise ValueError("Comparison to {} of type {} is not allowed.".format(val, type(val)))


class SCM(nn.Module):
    def __init__(self, v, f, pu: Distribution):
        super().__init__()
        self.v = v
        self.u = list(pu)
        self.f = f
        self.pu = pu
        self.device_param = nn.Parameter(T.empty(0))

    def space(self, v_size, select=None, tensor=True):
        if select is None:
            select = self.v
        for pairs in itertools.product(*([
            (vi, T.LongTensor(value).to(self.device_param.device) if tensor else value)
            for value in itertools.product(*([0, 1] for j in range(v_size[vi])))]
                for vi in select)):
            yield dict(pairs)

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
            v[k] = do[k] if k in do else self.f[k](v, u)
            remaining.discard(k)
            if not remaining:
                break
        return {k: v[k] for k in select}

    def convert_evaluation(self, samples):
        return samples

    def forward(self, n=None, u=None, do={}, select=None, evaluating=False):
        if evaluating:
            with T.no_grad():
                result = self.sample(n, u, do, select)
                result = self.convert_evaluation(result)
                return {k: result[k].cpu() for k in result}

        return self.sample(n, u, do, select)

    def query_loss(self, input, val):
        if T.is_tensor(val):
            raise NotImplementedError()
        else:
            if val == 1:
                return T.sum(-log(input))
            elif val == 0:
                return T.sum(-log(1 - input))
            else:
                raise ValueError("Comparison to {} of type {} is not allowed.".format(val, type(val)))

    def compute_ctf(self, query: CTF, n=1000000, u=None, get_prob=True, evaluating=False):
        if u is None:
            u = self.pu.sample(n)
            n_new = n
        else:
            n_new = len(u[next(iter(u))])

        for term in query.cond_term_set:
            samples = self(n=None, u=u, do={
                k: expand_do(v, n_new) for (k, v) in term.do_vals.items()
            }, select=term.vars, evaluating=True)

            cond_match = T.ones(n_new, dtype=T.bool)
            for (k, v) in term.var_vals.items():
                cond_match *= check_equal(samples[k], v)

            u = {k: v[cond_match] for (k, v) in u.items()}
            n_new = T.sum(cond_match.long()).item()

        if n_new <= 0:
            if evaluating:
                return float('nan')
            else:
                return T.tensor([float('nan')]).to(self.device_param)

        if evaluating:
            match = T.ones(n_new, dtype=T.bool, requires_grad=False)
            out_samples = dict()
            for term in query.term_set:
                expanded_do_terms = dict()
                for (k, v) in term.do_vals.items():
                    if k == "nested":
                        expanded_do_terms.update(self.compute_ctf(v, u=u, get_prob=False, evaluating=evaluating))
                    else:
                        expanded_do_terms[k] = expand_do(v, n_new)
                q_samples = self(n=None, u=u, do=expanded_do_terms, select=term.vars, evaluating=evaluating)

                if get_prob:
                    for (k, v) in term.var_vals.items():
                        match *= check_equal(q_samples[k], v)
                else:
                    out_samples.update(q_samples)

            if get_prob:
                return (T.sum(match.long()) / match.shape[0]).item()
            else:
                return out_samples
        
        else:
            loss = 0
            loss_count = 0
            out_samples = dict()
            for term in query.term_set:
                expanded_do_terms = dict()
                for (k, v) in term.do_vals.items():
                    if k == "nested":
                        expanded_do_terms.update(self.compute_ctf(v, u=u, get_prob=False, evaluating=evaluating))
                    else:
                        expanded_do_terms[k] = expand_do(v, n_new)

                q_samples = self(n=None, u=u, do=expanded_do_terms, select=term.vars, evaluating=evaluating)

                if get_prob:
                    for (k, v) in term.var_vals.items():
                        loss += self.query_loss(q_samples[k], v)
                        loss_count += 1
                else:
                    out_samples.update(q_samples)

            if get_prob:
                return loss / (n_new * loss_count)
            else:
                return out_samples
