import torch as T
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, pa_size, u_size, o_size, h_size=128, h_layers=2, use_sigmoid=True, use_layer_norm=True):
        super().__init__()
        self.pa = sorted(pa_size)
        self.set_pa = set(self.pa)
        self.u = sorted(u_size)
        self.pa_size = pa_size
        self.u_size = u_size
        self.o_size = o_size
        self.h_size = h_size

        self.i_size = sum(self.pa_size[k] for k in self.pa_size) + sum(self.u_size[k] for k in self.u_size)

        layers = [nn.Linear(self.i_size, self.h_size)]
        if use_layer_norm:
            layers.append(nn.LayerNorm(self.h_size))
        layers.append(nn.ReLU())
        for l in range(h_layers - 1):
            layers.append(nn.Linear(self.h_size, self.h_size))
            if use_layer_norm:
                layers.append(nn.LayerNorm(self.h_size))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(self.h_size, self.o_size))
        if use_sigmoid:
            layers.append(nn.Sigmoid())

        self.nn = nn.Sequential(*layers)

        self.device_param = nn.Parameter(T.empty(0))

        self.nn.apply(self.init_weights)

    def init_weights(self, m):
        if type(m) == nn.Linear:
            T.nn.init.xavier_normal_(m.weight,
                                     gain=T.nn.init.calculate_gain('relu'))

    def forward(self, pa, u, include_inp=False):
        if len(u.keys()) == 0:
            inp = T.cat([pa[k] for k in self.pa], dim=1)
        elif len(pa.keys()) == 0 or len(set(pa.keys()).intersection(self.set_pa)) == 0:
            inp = T.cat([u[k] for k in self.u], dim=1)
        else:
            inp_u = T.cat([u[k] for k in self.u], dim=1)
            inp_pa = T.cat([pa[k] for k in self.pa], dim=1)
            inp = T.cat((inp_pa, inp_u), dim=1)

        if include_inp:
            return self.nn(inp), inp

        return self.nn(inp)


if __name__ == '__main__':
    s = MLP(dict(v1=2, v2=1), dict(u1=1, u2=2), 3)
    print(s)
    pa = {
        'v1': T.tensor([[1, 2], [3, 4.]]),
        'v2': T.tensor([[5], [6.]])
    }
    u = {
        'u1': T.tensor([[7.], [8]]),
        'u2': T.tensor([[9, 10], [11, 12.]])
    }
    print(s(pa, u))
