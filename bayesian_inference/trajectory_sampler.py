import copy
import numpy as np
import logging
import torch
from torch import cuda
from torch.nn import functional as F
import random
from multiprocessing import Pool
from typing import List, Callable
from collections import defaultdict
import tqdm

#from sg_mcmc import BaseOptimizer

FORMAT = '%(asctime)-15s %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)


class BurnInScheduler(object):
    def __init__(self, optimizer, burn_in_epochs, burn_lr, rest_lr, **kwargs):
        self.optimizer = optimizer
        self.burn_in_epochs = burn_in_epochs
        self.burn_lr = burn_lr
        self.rest_lr = rest_lr
        self.passed = False

    def step(self, epoch):
        if self.passed is False and epoch >= self.burn_in_epochs:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.rest_lr
            self.passed = True
            

class SG_MCMC_Inference:
    def __init__(self, 
                bayesian_nn,
                potential, 
                sg_mcmc, 
                loss_fn,
                valloader, 
                **kwargs):
        self.bayesian_nn = bayesian_nn
        self.sg_mcmc = sg_mcmc
        self.potential= potential
        self.loss_fn = loss_fn
        self.valloader = valloader

        self.err_fn = kwargs.get('err_fn', None)
        self.scheduler = kwargs.get('scheduler', None)
        self.resample_prior_every = kwargs.get('resample_prior_every', 100)
        self.resample_momentum_every = kwargs.get('resample_momentum_every', 50)
        self.device = torch.device(kwargs.get('device', 'cuda:0' if cuda.is_available() else 'cpu'))
        self.verbosity = kwargs.get('verbosity', 1)
        self.report_every = kwargs.get('report_every', 10)
        self.save_freq = kwargs.get('save_freq', 2)
        self.save_stoch_grad = kwargs.get('save_stoch_grad', True)
        
    def _report(self, bayesian_nn):
        training =  bayesian_nn.training
        bayesian_nn.eval()
        val_loss = 0
        val_err = 0
        n_pts = 0
        with torch.no_grad():
            for x, y in self.valloader:
                x, y = x.to(self.device), y.to(self.device)
                out = bayesian_nn(x.float())
                val_loss += self.loss_fn(out, y).item()
                if self.err_fn is not None:
                    val_err += self.err_fn(out, y).item()
                n_pts += out.shape[0]
        potential = self.potential(bayesian_nn, stoch=False)
        potential_grad = self.potential.grad(bayesian_nn, potential=potential)
        logger.info(f'potential: {potential.item()}, potential_grad: {potential_grad[:10].tolist()}')
        logger.info(f'val loss: {val_loss / n_pts}, val error: {val_err / n_pts}')

        if training:
            bayesian_nn.train()

    def _sample_traj(self, bayesian_nn, sg_mcmc, n_burn, n_sample, resample_prior_until=None, save_freq=None):
        traj = []
        traj_grad = []

        bayesian_nn.train()
        for it in range(1, n_burn + 1):
            resample_prior = (it % self.resample_prior_every == 0) and \
                (it < resample_prior_until)
            if resample_prior:
                bayesian_nn.resample_prior()

            resample_momentum = (it % self.resample_momentum_every == 0) 
            #    and (epoch < resample_prior_until) and (epoch < burn_in_epochs)

            potential = self.potential(bayesian_nn, stoch=self.save_stoch_grad)
            sg_mcmc.zero_grad()
            potential.backward()
            sg_mcmc.step(resample_momentum=resample_momentum, burn_in=True)

            if it % self.report_every == 0:
                logger.info(f'Iteration: {it}')
                self._report(bayesian_nn)

        prior_dict = bayesian_nn.prior_dict()

        for it in range(n_burn + 1, n_burn + n_sample + 1):
            resample_momentum = (it % self.resample_momentum_every == 0) #

            potential = self.potential(bayesian_nn, stoch=self.save_stoch_grad)
            sg_mcmc.zero_grad()
            potential_grad = self.potential.grad(bayesian_nn, potential=potential)
            sg_mcmc.step(resample_momentum=resample_momentum, burn_in=False)

            if it % self.report_every == 0:
                logger.info(f'Iteration: {it}')
                self._report(bayesian_nn)
            if it % save_freq == 0:
                traj.append(copy.deepcopy(bayesian_nn.state_dict()))
                traj_grad.append(potential_grad.detach().numpy())

        return traj, traj_grad, prior_dict

    def sample_trajs(self, n_trajs, n_burn, n_sample, resample_prior_until=None, save_freq=None):
        if save_freq is None:
            save_freq = self.save_freq
        if resample_prior_until is None:
            resample_prior_until = n_burn

        traj_grads = []
        trajs = []
        priors = []
        for _ in tqdm.trange(n_trajs):
            bayesian_nn = self.bayesian_nn
            sg_mcmc = self.sg_mcmc
            bayesian_nn.init()
            sg_mcmc.init(bayesian_nn.parameters())
            traj, traj_grad, prior_dict = self._sample_traj(bayesian_nn, sg_mcmc, n_burn, n_sample, resample_prior_until, save_freq)
            trajs.append(traj)
            traj_grads.append(traj_grad)
            priors.append(prior_dict)
            yield trajs, np.array(traj_grads), priors

        # total_steps = n_epoch * len(self.trainloader)
        # burn_in_steps = burn_in_epochs * len(self.trainloader)
        # if self.max_weight_set_size is not float('inf'):
        # start_save_step = total_steps - self.max_weight_set_size * self.save_freq
        # else:
        # start_save_step = burn_in_steps
        # best_loss = 1e9
        # no_significant_improvement_step = 0

        # self.bayesian_nn.init()
        # self.optimizer.init(self.model)
    #     for epoch in range(n_epoch):
    #         train_loss, val_loss = 0, 0
    #         train_err, val_err = 0, 0
    #         n_ex = 0
    #         self.model.train()
    #         for x, y in self.trainloader:
    #             resample_prior = ((it_cnt + 1) % self.resample_prior_every == 0) and \
    #             (epoch < resample_prior_until) and (epoch < burn_in_epochs)
    #             resample_momentum = ((it_cnt + 1) % self.resample_momentum_every == 0) and \
    #             (epoch < resample_prior_until) and (epoch < burn_in_epochs)
    #             loss, err = self.do_train_step(x.to(self.device), y.to(self.device), 
    #                 resample_prior=resample_prior, resample_momentum=resample_momentum)
    #             self.scheduler.step(epoch)
    #             train_loss += loss
    #             train_err += err
    #             n_ex += x.shape[0]
    #             it_cnt += 1

    #             if epoch >= burn_in_epochs and it_cnt % self.save_freq == 0 and it_cnt >= start_save_step:
    #                 self.save_sampled_net()

    #         if epoch % self.report_every == 0:
    #             n_ex = 0
    #             self.model.eval()
    #             for x, y in self.valloader:
    #                 with torch.no_grad():
    #                     x, y = x.to(self.device), y.to(self.device)
    #                     y_hat = self.model(x)
    #                     val_err += self.err_func(y_hat, y).sum().item()
    #                     val_loss += self.nll_func(y_hat, y)
    #                     n_ex += x.shape[0]
    #             if it_cnt <= start_save_step + 1:
    #                 potential = self.compute_potential(stoch=self.save_potential_grad_type == 'stoch')
    #                 self.optimizer.zero_grad()
    #                 potential.backward()
    #                 grad = self.get_potential_grad(add_prior_grad=False)[:10]
    #             else:
    #                 potential = self.potential_sample[-1]
    #                 grad = self.potential_grad_sample[-1][:10]

    #             logger.info(f'Epoch {epoch} finished. Val loss {val_loss / n_ex}, Val error {val_err / n_ex}')
    #             logger.info(f'Potential: {potential}')
    #             logger.info(f'Potential grad: {grad}')
    #         # if self.early_stopping:
    #         #     if val_loss / n_ex - best_loss < -self.min_delta:
    #         #         best_loss = val_loss / n_ex
    #         #     else:
    #         #         no_significant_improvement_step += 1

    #         #     if no_significant_improvement_step >= self.wait:
    #         #         logger.info('Early Stopping triggered')
    #         #         break

    # def do_train_step(self, x, y, **kwargs):
    #     y_hat = self.model(x)
    #     nll = self.nll_func(y_hat, y)
    #     nll *= self.N_train / x.shape[0]
    #     self.optimizer.zero_grad() 
    #     nll.backward()
    #     if isinstance(self.optimizer, BaseOptimizer):
    #         out = self.optimizer.step(**kwargs)
    #         #grad = out[1].detach().numpy()
    #     else:
    #         self.optimizer.step()
    #         #grad = self.get_potential_grad().detach().numpy()
        
    #     err = self.err_func(y_hat, y).sum().item()

    #     self.loss_history.append(nll)
    #     self.err_history.append(err)
        
    #     return nll, err

    # def get_weight_sample(self, Nsample=0):
    #     """return weight sample from posterior in a single-column array"""
    #     #weight_vec = []

    #     if Nsample == 0 or Nsample > len(self.weight_set_sample):
    #         Nsample = len(self.weight_set_sample)

    #     return self.weight_set_sample

    # def save_sampled_net(self):
    #     if len(self.weight_set_sample) >= self.max_weight_set_size:
    #         self.weight_set_sample.pop(0)
    #         self.potential_sample.pop(0)
    #         self.potential_grad_sample.pop(0)
    #     self.weight_set_sample.append(copy.deepcopy(self.model.state_dict()))
    #     potential = self.compute_potential(stoch=self.save_potential_grad_type == 'stoch')
    #     self.optimizer.zero_grad()
    #     potential.backward()
    #     grad = self.get_potential_grad(add_prior_grad=False)
    #     self.potential_sample.append(potential)
    #     self.potential_grad_sample.append(grad.detach().numpy())

    # def compute_potential(self, stoch=False):
    #     potential = 0
    #     self.model.to(self.device)
        
    #     if stoch is False:
    #         for x, y in self.trainloader:
    #             x, y = x.to(self.device), y.to(self.device)
    #             y_hat = self.model(x)
    #             potential += self.nll_func(y_hat, y)
    #     else:
    #         x, y = next(iter(self.datasampler))
    #         x, y = x.to(self.device), y.to(self.device)
    #         y_hat = self.model(x)
    #         potential = self.N_train / x.shape[0] * self.nll_func(y_hat, y)

        
    #     for group in self.optimizer.param_groups:
    #         for p in group['params']:
    #             if p.grad is None:
    #                 continue
    #             try:
    #                 state = self.optimizer.state[p]
    #                 weight_decay = state['weight_decay']
    #             except:
    #                 weight_decay = group['weight_decay']
    #             potential += weight_decay / 2 * (p**2).sum()
    #     return potential
 
    # def get_potential_grad(self, add_prior_grad=False):
    #     flat_grad = []
    #     for group in self.optimizer.param_groups:
    #         for p in group['params']:
    #             if p.grad is None:
    #                 continue
    #             try:
    #                 state = self.optimizer.state[p]
    #                 weight_decay = state['weight_decay']
    #             except:
    #                 weight_decay = group['weight_decay']

    #             d_p = p.grad

    #             if weight_decay != 0 and add_prior_grad is True:
    #                 d_p.add_(p, alpha=weight_decay)

    #             flat_grad.append(d_p.flatten())
    #     return torch.cat(flat_grad, dim=0)
