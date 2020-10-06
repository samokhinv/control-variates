import copy
import numpy as np
import logging
import torch
from torch import cuda

from .cv import SteinCV
from .optim import BaseOptimizer

FORMAT = '%(asctime)-15s %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)

from typing import List, Callable
from .uncertainty_quantification import ClassificationUncertaintyMCMC


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
            

class BNNTrainer(object):
    def __init__(self, model, optimizer, nll_func, err_func, trainloader, valloader, **kwargs):
        self.model = model
        self.optimizer = optimizer
        self.nll_func = nll_func
        self.err_func = err_func
        self.trainloader = trainloader
        self.valloader = valloader

        self.scheduler = kwargs.get('scheduler', None)
        self.resample_prior_every = kwargs.get('resample_prior_every', 100)
        self.resample_momentum_every = kwargs.get('resample_momentum_every', 50)
        device = kwargs.get('device', 'cuda:0' if cuda.is_available() else 'cpu')
        self.max_weight_set_size = kwargs.get('max_weight_set_size', float('inf'))
        self.report_every = kwargs.get('report_every', 10)
        self.save_freq = kwargs.get('save_freq', 2)
        self.batch_size = kwargs.get('batch_size', 500)
        
        self.device = torch.device(device)
        self.N_train = len(trainloader.dataset)
        print(f'N_train {self.N_train}')
        self.weight_set_sample = []
        self.loss_history = []
        self.err_history = []
        self.potential_grad_sample = []
        self.potential_sample = []
        
        # self.early_stopping = kwargs.get('early_stopping', False)
        # if self.early_stopping:
        #     self.min_delta = kwargs.get('min_delta', 1e-3)
        #     self.wait = kwargs.get('wait', 10)

    def train(self, n_epoch, burn_in_epochs=0, resample_prior_until=1):
        total_steps = n_epoch * len(self.trainloader)
        burn_in_steps = burn_in_epochs * len(self.trainloader)
        if self.max_weight_set_size is not float('inf'):
            start_save_step = total_steps - self.max_weight_set_size * self.save_freq
        else:
            start_save_step = burn_in_steps
        #best_loss = 1e9
        #no_significant_improvement_step = 0
        self.model.to(self.device)
        it_cnt = 0
        for epoch in range(n_epoch):
            train_loss, val_loss = 0, 0
            train_err, val_err = 0, 0
            n_ex = 0
            self.model.train()
            for x, y in self.trainloader:
                resample_prior = (it_cnt % self.resample_prior_every == 0) and \
                (epoch < resample_prior_until) and (epoch < burn_in_epochs)
                resample_momentum = (it_cnt % self.resample_momentum_every == 0) and \
                (epoch < resample_prior_until) and (epoch < burn_in_epochs)
                loss, err = self.do_train_step(x.to(self.device), y.to(self.device), 
                    resample_prior=resample_prior, resample_momentum=resample_momentum)
                self.scheduler.step(epoch)
                train_loss += loss
                train_err += err
                n_ex += x.shape[0]
                it_cnt += 1

                if epoch > burn_in_epochs and it_cnt % self.save_freq == 0 and it_cnt >= start_save_step:
                    self.save_sampled_net()

            if epoch % self.report_every == 0:
                n_ex = 0
                self.model.eval()
                for x, y in self.valloader:
                    with torch.no_grad():
                        x, y = x.to(self.device), y.to(self.device)
                        y_hat = self.model(x)
                        val_err += self.err_func(y_hat, y).sum().item()
                        val_loss += self.nll_func(y_hat, y)
                        n_ex += x.shape[0]
                if it_cnt <= start_save_step + 1:
                    potential = self.compute_potential()
                    self.optimizer.zero_grad()
                    potential.backward()
                    grad = self.get_potential_grad(add_prior_grad=False)[:10]
                else:
                    potential = self.potential_sample[-1]
                    grad = self.potential_grad_sample[-1][:10]

                logger.info(f'Epoch {epoch} finished. Val loss {val_loss / n_ex}, Val error {val_err / n_ex}')
                logger.info(f'Potential: {potential}')
                logger.info(f'Potential grad: {grad}')
            # if self.early_stopping:
            #     if val_loss / n_ex - best_loss < -self.min_delta:
            #         best_loss = val_loss / n_ex
            #     else:
            #         no_significant_improvement_step += 1

            #     if no_significant_improvement_step >= self.wait:
            #         logger.info('Early Stopping triggered')
            #         break

    def do_train_step(self, x, y, **kwargs):
        y_hat = self.model(x)
        nll = self.nll_func(y_hat, y)
        nll *= self.N_train / x.shape[0]
        self.optimizer.zero_grad() 
        nll.backward()
        if isinstance(self.optimizer, BaseOptimizer):
            out = self.optimizer.step(**kwargs)
            #grad = out[1].detach().numpy()
        else:
            self.optimizer.step()
            #grad = self.get_potential_grad().detach().numpy()
        
        err = self.err_func(y_hat, y).sum().item()

        self.loss_history.append(nll)
        self.err_history.append(err)
        
        return nll, err

    def get_weight_sample(self, Nsample=0):
        """return weight sample from posterior in a single-column array"""
        #weight_vec = []

        if Nsample == 0 or Nsample > len(self.weight_set_sample):
            Nsample = len(self.weight_set_sample)

        return self.weight_set_sample

    def save_sampled_net(self):
        if len(self.weight_set_sample) >= self.max_weight_set_size:
            self.weight_set_sample.pop(0)
            self.potential_sample.pop(0)
            self.potential_grad_sample.pop(0)
        self.weight_set_sample.append(copy.deepcopy(self.model.state_dict()))
        potential = self.compute_potential()
        self.optimizer.zero_grad()
        potential.backward()
        grad = self.get_potential_grad(add_prior_grad=False)
        self.potential_sample.append(potential)
        self.potential_grad_sample.append(grad.detach().numpy())

    def compute_potential(self):
        potential = 0
        self.model.to(self.device)
        for x, y in self.trainloader:
            x, y = x.to(self.device), y.to(self.device)
            y_hat = self.model(x)
            potential += self.nll_func(y_hat, y)
        
        for group in self.optimizer.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                try:
                    state = self.optimizer.state[p]
                    weight_decay = state['weight_decay']
                except:
                    weight_decay = group['weight_decay']
                potential += weight_decay / 2 * p**2.sum()
        return potential
 
    def get_potential_grad(self, add_prior_grad=False):
        flat_grad = []
        for group in self.optimizer.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                try:
                    state = self.optimizer.state[p]
                    weight_decay = state['weight_decay']
                except:
                    weight_decay = group['weight_decay']

                d_p = p.grad

                if weight_decay != 0 and add_prior_grad is True:
                    d_p.add_(p, alpha=weight_decay)

                flat_grad.append(d_p.flatten())
        return torch.cat(flat_grad, dim=0)
