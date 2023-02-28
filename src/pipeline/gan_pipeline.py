import numpy as np
import pandas as pd
import torch as T
from torch.autograd import grad

from src.metric.evaluation import all_metrics, probability_table
from src.ds.causal_graph import CausalGraph
from src.ds.counterfactual import CTF
from src.scm.ncm.gan_ncm import GAN_NCM
from src.scm.scm import expand_do

from .base_pipeline import BasePipeline


def log(x):
    return T.log(x + 1e-8)


class GANPipeline(BasePipeline):
    patience = 500
    max_epochs = 3000

    def __init__(self, generator, do_var_list, dat_sets, cg, dim, hyperparams=None, ncm_model=GAN_NCM, max_query=None):
        """
        gan-mode options: vanilla, bgan, wgan
        """
        if hyperparams is None:
            hyperparams = dict()

        v_size = {k: 1 if k in {'X', 'Y', 'M', 'W'} else dim for k in cg}
        ncm = ncm_model(cg, v_size=v_size, default_u_size=hyperparams.get('u-size', 1), hyperparams=hyperparams,
                        gen_use_sigmoid=hyperparams['gen-sigmoid'],
                        disc_use_sigmoid=(hyperparams.get("gan-mode", "NA") != "wgan"))
        super().__init__(generator, do_var_list, dat_sets, cg, dim, ncm, batch_size=hyperparams.get('data-bs', 1000))

        self.max_query = max_query
        self.max_query_iters = hyperparams.get("max-query-iters", 3000)
        self.query_track = hyperparams["eval-query"]
        self.gan_mode = hyperparams.get("gan-mode", "vanilla")
        self.gen_sigmoid = hyperparams["gen-sigmoid"]
        self.perturb_sd = hyperparams["perturb-sd"]

        self.do_var_list = hyperparams["do-var-list"]

        self.ncm_batch_size = hyperparams.get('ncm-bs', 1000)
        self.d_iters = hyperparams.get('d-iters', 1)
        self.cut_batch_size = hyperparams.get('data-bs', 1000) // self.d_iters
        self.grad_clamp = hyperparams.get('grad-clamp', 0.01)
        self.gp_weight = hyperparams.get('gp-weight', 10.0)
        self.lr = hyperparams.get('lr', 0.001)
        self.mc_sample_size = hyperparams.get('mc-sample-size', 10000)
        self.min_lambda = hyperparams.get('min-lambda', 0.001)
        self.max_lambda = hyperparams.get('max-lambda', 1.0)
        self.ordered_v = cg.v

        self.logged = False
        self.stored_kl = 1e8

        self.automatic_optimization = False

    def forward(self, n=1000, u=None, do={}):
        return self.ncm(n, u, do)

    def configure_optimizers(self):
        if self.gan_mode == "wgan":
            opt_gen = T.optim.RMSprop(self.ncm.f.parameters(), lr=self.lr)
            opt_disc = T.optim.RMSprop(self.ncm.f_disc.parameters(), lr=self.lr)
            opt_pu = T.optim.RMSprop(self.ncm.pu.parameters(), lr=self.lr)
        else:
            opt_gen = T.optim.Adam(self.ncm.f.parameters(), lr=self.lr)
            opt_disc = T.optim.Adam(self.ncm.f_disc.parameters(), lr=self.lr)
            opt_pu = T.optim.Adam(self.ncm.pu.parameters(), lr=self.lr)
        return opt_gen, opt_disc, opt_pu

    def _get_D_loss(self, real_out, fake_out):
        if self.gan_mode == "wgan" or self.gan_mode == "wgangp":
            return -(T.mean(real_out) - T.mean(fake_out))
        else:
            return -T.mean(log(real_out) + log(1 - fake_out))

    def _get_G_loss(self, fake_out):
        if self.gan_mode == "bgan":
            return 0.5 * T.mean((log(fake_out) - log(1 - fake_out)) ** 2)
        elif self.gan_mode == "wgan" or self.gan_mode == "wgangp":
            return -T.mean(fake_out)
        else:
            return -T.mean(log(fake_out))

    def _get_gradient_penalty(self, real_data, fake_data, disc_index):
        interpolated_data = dict()
        alpha = T.rand(self.ncm_batch_size, 1, device=self.device, requires_grad=True)
        for V in real_data:
            v_alpha = alpha.expand_as(real_data[V])
            interpolated_data[V] = v_alpha * real_data[V].detach() + (1 - v_alpha) * fake_data[V].detach()

        interpolated_out, inp = self.ncm.get_disc_outputs(interpolated_data, disc_index, include_inp=True)
        gradients = grad(outputs=interpolated_out, inputs=inp,
                         grad_outputs=T.ones(interpolated_out.size(), device=self.device),
                         create_graph=True, retain_graph=True)[0]
        gradients = gradients.view(self.ncm_batch_size, -1)
        gradients_norm = T.sqrt(T.sum(gradients ** 2, dim=1) + 1e-12)
        return self.gp_weight * (T.relu(gradients_norm - self.grad_clamp) ** 2).mean()

    def _get_q_loss(self):
        if isinstance(self.max_query, CTF):
            ctf_query = self.max_query
            query_loss = self.ncm.compute_ctf(ctf_query, n=self.mc_sample_size)
        else:
            query_loss = 0
            for query in self.max_query:
                cur_loss = self.ncm.compute_ctf(query, n=self.mc_sample_size)
                if T.isnan(cur_loss):
                    return cur_loss
                query_loss += cur_loss
        return query_loss

    def training_step(self, batch, batch_idx):
        G_opt, D_opt, PU_opt = self.optimizers()
        ncm_n = self.ncm_batch_size

        reg_ratio = min(self.current_epoch, self.max_query_iters) / self.max_query_iters
        reg_up = np.log(self.max_lambda)
        reg_low = np.log(self.min_lambda)
        max_reg = np.exp(reg_up - reg_ratio * (reg_up - reg_low))

        G_opt.zero_grad()
        PU_opt.zero_grad()

        # Train Discriminator
        total_d_loss = 0
        for d_iter in range(self.d_iters):
            D_opt.zero_grad()
            for i, do_set in enumerate(self.do_var_list):
                ncm_batch = self.ncm(ncm_n, do={k: expand_do(v, ncm_n) for (k, v) in do_set.items()})
                real_batch = {k: v[d_iter * self.cut_batch_size:(d_iter + 1) * self.cut_batch_size].float()
                              for (k, v) in batch[i].items()}
                if not self.gen_sigmoid:
                    new_real_batch = dict()
                    for k in real_batch:
                        v = real_batch[k]
                        if k not in do_set:
                            new_real_batch[k] = T.normal(mean=v,
                                                         std=self.perturb_sd * T.ones(v.shape, device=self.device))
                        else:
                            new_real_batch[k] = v
                    real_batch = new_real_batch
                ncm_disc_real_out = self.ncm.get_disc_outputs(real_batch, i)
                ncm_disc_fake_out = self.ncm.get_disc_outputs(ncm_batch, i)
                D_loss = self._get_D_loss(ncm_disc_real_out, ncm_disc_fake_out)

                if self.gan_mode == "wgangp":
                    grad_penalty = self._get_gradient_penalty(real_batch, ncm_batch, i)
                    self.log('grad_penalty', grad_penalty, prog_bar=True)
                    D_loss += grad_penalty

                total_d_loss += D_loss.item()
                self.manual_backward(D_loss)

            D_opt.step()

            if self.gan_mode == "wgan":
                for p in self.ncm.f_disc.parameters():
                    p.data.clamp_(-self.grad_clamp, self.grad_clamp)

            self.ncm.f.zero_grad()
            self.ncm.f_disc.zero_grad()
            self.ncm.pu.zero_grad()

        # Train Generator
        g_loss_record = 0
        for i, do_set in enumerate(self.do_var_list):
            ncm_batch = self.ncm(ncm_n, do={k: expand_do(v, ncm_n) for (k, v) in do_set.items()})
            ncm_disc_fake_out = self.ncm.get_disc_outputs(ncm_batch, i)
            G_loss = self._get_G_loss(ncm_disc_fake_out) / len(self.do_var_list)
            g_loss_record += G_loss.item()
            self.manual_backward(G_loss)

        q_loss_record = 0
        if self.max_query is not None:
            q_loss = max_reg * (len(self.do_var_list) ** 2) * self._get_q_loss()
            q_loss_record = q_loss.item()
            if not T.isnan(q_loss):
                self.manual_backward(q_loss)

        G_opt.step()
        PU_opt.step()

        self.ncm.f.zero_grad()
        self.ncm.f_disc.zero_grad()
        self.ncm.pu.zero_grad()

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
        self.log('G_loss', g_loss_record, prog_bar=True)
        self.log('D_loss', total_d_loss, prog_bar=True)
        self.log('Q_loss', q_loss_record, prog_bar=True)
