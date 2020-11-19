import copy
import numpy as np
import logging
import torch
from torch import cuda
from torch.nn import functional as F
import random
from torch import multiprocessing
from torch.multiprocessing import Pool
from typing import List, Callable
from collections import defaultdict
import tqdm
from collections import OrderedDict
import time

from .sg_mcmc_methods import SVRG_LD, SVRG_HMC


FORMAT = '%(asctime)-15s %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)

#pool = multiprocessing.Pool(multiprocessing.cpu_count() - 1)


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
        self.potential = potential
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
        self.epoch_length = kwargs.get('epoch_length', None)
        if self.epoch_length is None:
            self.epoch_length = len(self.potential.burn_batchsampler)
        
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
        logger.info(f'potential: {potential.item()}, potential_grad: {potential_grad[:10].tolist()}, {potential_grad.norm()}')
        logger.info(f'val loss: {val_loss / n_pts}, val error: {val_err / n_pts}')

        if training:
            bayesian_nn.train()

    #@torch.no_grad()
    @staticmethod
    def copy_state_dict(state_dict):
        new_dict = OrderedDict()
        for n, p in state_dict.items():
            new_dict[n] = p.detach().clone()
        return new_dict

    def _sample_traj(self, bayesian_nn, sg_mcmc, n_burn, n_sample, resample_prior_until=None, save_freq=None):
        traj = []
        traj_grad = []

        if isinstance(sg_mcmc, (SVRG_LD, SVRG_HMC)):
            bayesian_nn_fixed = copy.deepcopy(bayesian_nn)
            sg_mcmc_fixed = type(sg_mcmc)(bayesian_nn_fixed.parameters(), lr=0.0)
            sg_mcmc_fixed.load_state_dict(sg_mcmc.state_dict())

        bayesian_nn.train()
        for it in range(1, n_burn + n_sample + 1):
            # update fixed point, collect deterministic grads
            if isinstance(sg_mcmc, (SVRG_LD, SVRG_HMC)) and (((it-1) % self.epoch_length == 0 and it > n_burn) or it == n_burn + 1):
                bayesian_nn_fixed.load_state_dict(bayesian_nn.state_dict())
                for p, p_f in zip(bayesian_nn.parameters(), bayesian_nn_fixed.parameters()):
                   p_f.sigma2 = p.sigma2
                potential = self.potential(bayesian_nn_fixed, stoch=False)
                _ = self.potential.grad(bayesian_nn_fixed, potential=potential)
                sg_mcmc.load_dataset_grads(sg_mcmc_fixed.param_groups)

            resample_prior = (it % self.resample_prior_every == 0) and \
                (it < resample_prior_until)
            if resample_prior:
                bayesian_nn.resample_prior()
            resample_momentum = (it % self.resample_momentum_every == 0)

            # collect batch grads for fixed and current point
            if isinstance(sg_mcmc, (SVRG_LD, SVRG_HMC)) and it > n_burn:
                potential, x, y = self.potential(bayesian_nn_fixed, stoch=self.save_stoch_grad, burn=it <= n_burn)
                sg_mcmc_fixed.zero_grad()
                potential.backward()
                sg_mcmc.load_batch_grads(sg_mcmc_fixed.param_groups)
            else:
                x, y = None, None

            potential, _, _ = self.potential(bayesian_nn, stoch=self.save_stoch_grad, burn=it <= n_burn, x=x, y=y)
            sg_mcmc.zero_grad()
            potential.backward()
            _, potential_grad = sg_mcmc.step(resample_momentum=resample_momentum, burn_in=it <= n_burn)

            if it % self.report_every == 0 or it == n_burn + n_sample:
                logger.info(f'Iteration: {it}')
                self._report(bayesian_nn)
            if it > n_burn and it % save_freq == 0:
                traj.append(SG_MCMC_Inference.copy_state_dict(bayesian_nn.state_dict()))
                #traj.append(copy.deepcopy(bayesian_nn.state_dict()))
                traj_grad.append(potential_grad.detach())
 
        prior_dict = bayesian_nn.prior_dict()

        return traj, traj_grad, prior_dict

    def sample_trajs(self, n_trajs, n_burn, n_sample, resample_prior_until=None, save_freq=None):
        #global func     # HACK
        
        if save_freq is None:
            save_freq = self.save_freq
        if resample_prior_until is None:
            resample_prior_until = n_burn

        traj_grads = []
        trajs = []
        priors = []
        def func(seed=None):
            #bayesian_nn, sg_mcmc, seed = args
            if seed is not None:
                torch.manual_seed(seed)
            bayesian_nn = self.bayesian_nn
            sg_mcmc = self.sg_mcmc
            #bayesian_nn.init()
            bayesian_nn.init_norm()
            sg_mcmc.init(bayesian_nn.parameters())
            start = time.time()
            traj, traj_grad, prior_dict = self._sample_traj(bayesian_nn, sg_mcmc, n_burn, n_sample, resample_prior_until, save_freq)
            end = time.time()
            logger.info(prior_dict)
            logger.info(f'time: {end - start}')
            return (traj, torch.stack(traj_grad, dim=0), prior_dict)

        for _ in range(n_trajs):
            traj, traj_grad, prior_dict = func()
            trajs.append(traj)
            traj_grads.append(traj_grad)
            priors.append(prior_dict)
            yield trajs, torch.stack(traj_grads, dim=0), priors


        # ncpu = multiprocessing.cpu_count()
        # with Pool(1) as p:
        #     results = p.map(func, [(self.bayesian_nn, self.sg_mcmc, 1)]) #list(np.arange(n_trajs)))

        # trajs = [x[0] for x in results]
        # traj_grads = torch.stack([x[1] for x in results], dim=0)
        # priors = [x[2] for x in results]

        # return trajs, traj_grads, priors
