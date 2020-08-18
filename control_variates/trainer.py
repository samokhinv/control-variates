import copy
import numpy as np
import logging

import torch
from torch import cuda

from .cv import SteinCV

FORMAT = '%(asctime)-15s %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)

from typing import List, Callable
from .uncertainty_quantification import ClassificationUncertaintyMCMC


class BNNTrainer(object):
    def __init__(self, model, optimizer, nll_func, err_func, trainloader, valloader, **kwargs):
        self.model = model
        self.optimizer = optimizer
        self.nll_func = nll_func
        self.err_func = err_func
        self.trainloader = trainloader
        self.valloader = valloader

        self.resample_prior_every = kwargs.get('resample_prior_every', 100)
        self.resample_momentum_every = kwargs.get('resample_momentum_every', 50)
        device = kwargs.get('device', 'cuda:0' if cuda.is_available() else 'cpu')
        self.max_weight_set_size = kwargs.get('max_weight_set_size', 200)
        self.report_every = kwargs.get('report_every', 10)
        self.save_freq = kwargs.get('save_freq', 2)
        self.batch_size = kwargs.get('batch_size', 500)
        #self.resample_prior_until = kwargs.get('resample_prior_intil', 100)
        #self.burn_in_steps = kwargs.get('burn_in_steps', 200)
        
        self.device = torch.device(device)
        self.N_train = len(trainloader) * self.batch_size
        self.weight_set_samples = []
        self.early_stopping = kwargs.get('early_stopping', False)
        if self.early_stopping:
            self.min_delta = kwargs.get('min_delta', 1e-3)
            self.wait = kwargs.get('wait', 10)


    def train(self, n_epoch, burn_in_epochs=0, resample_prior_until=1e8):
        best_loss = 1e9
        no_significant_improvement_step = 0
        self.model.to(self.device)
        it_cnt = 0
        for epoch in range(n_epoch):
            train_loss, val_loss = 0, 0
            train_err, val_err = 0, 0
            n_ex = 0
            self.model.train()
            burn_in = epoch < burn_in_epochs
            for x, y in self.trainloader:
                resample_prior = it_cnt % self.resample_prior_every == 0 and epoch < resample_prior_until
                resample_momentum = it_cnt % self.resample_momentum_every == 0
                loss, err = self.do_train_step(x.to(self.device), y.to(self.device), 
                    resample_prior=resample_prior, resample_momentum=resample_momentum, burn_in=burn_in)
                #if resample_prior:
                #    self.weight_set_samples = []
                train_loss += loss
                train_err += err
                n_ex += x.shape[0]
                it_cnt += 1

            if epoch > burn_in_epochs and epoch % self.save_freq == 0:
                self.save_sampled_net()


                #loc_iter = n_iter % self.N_train
                #if (loc_iter + 1) % self.report_every == 0:
                #    logger.info(f'Iteration {n_iter}, Loss {train_loss /  n_ex}, Error {train_err / n_ex}')
            
            n_ex = 0
            self.model.eval()
            for x, y in self.valloader:
                with torch.no_grad():
                    x, y = x.to(self.device), y.to(self.device)
                    y_hat = self.model(x)
                    val_err += self.err_func(y_hat, y).sum().item()
                    val_loss += self.nll_func(y_hat, y)
                    n_ex += x.shape[0]

            if epoch % self.report_every == 0:
                logger.info(f'Epoch {epoch} finished. Val loss {val_loss / n_ex}, Val error {val_err / n_ex}')
            if self.early_stopping:
                if val_loss / n_ex - best_loss < -self.min_delta:
                    best_loss = val_loss / n_ex
                else:
                    no_significant_improvement_step += 1

                if no_significant_improvement_step >= self.wait:
                    logger.info('Early Stopping triggered')
                    break

    def do_train_step(self, x, y, **kwargs):
        y_hat = self.model(x)
        nll = self.nll_func(y_hat, y)
        nll *= self.N_train / x.shape[0]
        self.optimizer.zero_grad() 
        nll.backward()
        self.optimizer.step(**kwargs)

        err = self.err_func(y_hat, y).sum().item()

        return nll, err

    def get_weight_samples(self, Nsamples=0):
        """return weight samples from posterior in a single-column array"""
        weight_vec = []

        if Nsamples == 0 or Nsamples > len(self.weight_set_samples):
            Nsamples = len(self.weight_set_samples)

        # for idx, state_dict in enumerate(self.weight_set_samples):
        #     if idx == Nsamples:
        #         break

        #     for key in state_dict.keys():
                # if 'weight' in key:
                #     weight_mtx = state_dict[key].cpu().data
                #     for weight in weight_mtx.view(-1):
                #         weight_vec.append(weight)

        return self.weight_set_samples

    def save_sampled_net(self):

        max_samples = self.max_weight_set_size
        if len(self.weight_set_samples) >= max_samples:
            self.weight_set_samples.pop(0)

        self.weight_set_samples.append(copy.deepcopy(self.model.state_dict()))

        return None


class NCVTrainer(object):
    def __init__(self,
                 ncv: SteinCV,
                 models: List[Callable],
                 var_criterion: Callable[[torch.Tensor], torch.Tensor],
                 valid_criterion: Callable[[torch.Tensor], torch.Tensor],
                 x_batch,
                 optimizer,
                 trainloader,
                 valloader,
                 **kwargs):
        self.ncv = ncv
        self.models = models
        self.x_batch = x_batch
        self.optimizer = optimizer
        self.trainloader = trainloader
        self.valloader = valloader

        self.var_criterion = var_criterion
        self.valid_criterion = valid_criterion
        self.report_every = kwargs.get('report_every', 10)
        self.early_stopping = kwargs.get('early_stopping', False)
        if self.early_stopping:
            self.min_delta = kwargs.get('min_delta', 1e-3)
            self.wait = kwargs.get('wait', 10)

    def train(self, x, n_epochs):
        best_loss = 1e9
        no_significant_improvement_step = 0
        for epoch in range(n_epochs):
            val_loss = []
            self.ncv.psy_model.train()
            for x, _ in self.trainloader:
                x.to(self.device)
                self.optimizer.zero_grad()
                mc_variance = self.var_criterion(x)
                mc_variance.backward()
                self.optimizer.step()

            self.ncv.psy_model.eval()
            for x, _ in self.valloader:
                x.to(self.device)
                with torch.no_grad():
                    val_loss.append(self.valid_criterion(x).mean().item())

            val_loss = np.mean(val_loss)

            if epoch % self.report_every == 0:
                logger.info(f'Epoch {epoch} finished. Val loss {val_loss}')
            if self.early_stopping:
                if val_loss - best_loss < -self.min_delta:
                    best_loss = val_loss
                else:
                    no_significant_improvement_step += 1

                if no_significant_improvement_step >= self.wait:
                    logger.info('Early Stopping triggered')
                    break
