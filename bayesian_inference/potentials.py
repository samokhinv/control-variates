import torch
from torch.nn import functional as F
import numpy as np
from abc import ABC, abstractmethod


class Potential(ABC):
    @abstractmethod
    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def __call__(self, data=None, theta=None):
        pass

    @abstractmethod
    def grad(self, data=None, theta=None):
        pass


class GaussPotential(Potential):
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma
        self.sigma_inv = torch.inverse(sigma)

    def __call__(self, point):
        return -0.5*(self.sigma_inv @ (point-self.mu)) @ (point-self.mu)
        
    def grad(self, point):
        return -self.sigma_inv @ (point - self.mu)


class ClassificationPotential(Potential):
    def __init__(self, burn_batchsampler, sample_batchsampler, device):
        self.burn_batchsampler = burn_batchsampler
        self.sample_batchsampler = sample_batchsampler
        self.device = device
        self.N_pts = len(burn_batchsampler.dataset)
        self.burn_batchiter = iter(burn_batchsampler)
        self.sample_batchiter = iter(sample_batchsampler)

    def reload_iterator(self, burn=False):
        if burn:
            self.burn_batchiter = iter(self.burn_batchsampler)
            return self.burn_batchiter
        else:
            self.sample_batchiter = iter(self.sample_batchsampler)
            return self.sample_batchiter

    def __call__(self, bayesian_nn, stoch=True, seed=None, burn=False, x=None, y=None):
        if seed is not None:
            torch.manual_seed(seed)

        if stoch is True:
            if x is None or y is None:
                batchiter = self.burn_batchiter if burn else self.sample_batchiter
                try:
                    x, y = next(batchiter)
                except StopIteration:
                    batchiter = self.reload_iterator(burn)
                    x, y = next(batchiter)
                #x, y = x.to(self.device), y.to(self.device)
            #print(x * x)
            out = bayesian_nn(x)
            log_prob = -F.cross_entropy(out, y, reduction='mean')

            potential = -self.N_pts * log_prob - bayesian_nn.get_log_prior()
            return potential, x, y
        else:
            batchsampler = self.burn_batchsampler if burn else self.sample_batchsampler
            potential = 0
            for x, y in batchsampler:
                x, y = x.to(self.device), y.to(self.device)
                out = bayesian_nn(x)
                log_prob = -F.cross_entropy(out, y, reduction='sum')
                potential -= log_prob
            potential -= bayesian_nn.get_log_prior()
            return potential

    def grad(self, bayesian_nn, stoch=True, potential=None, burn=False):
        if potential is None:
            potential = self.__call__(bayesian_nn, stoch=stoch, burn=burn)

        bayesian_nn.zero_grad()
        potential.backward()
        flat_grad = []
        for p in bayesian_nn.parameters():
            if p.grad is None:
                continue
            else:
                flat_grad.append(p.grad.flatten())
        
        return torch.cat(flat_grad, dim=0)
