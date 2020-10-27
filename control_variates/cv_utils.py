import torch
import numpy as np
from torch import nn
from torch.nn import functional as F

from joblib import Parallel


def state_dict_to_vec(state_dict):
    return torch.cat([w_i.view(-1) for w_i in state_dict.values()])


def compute_tricky_divergence(model, priors=None):
    div = 0
    for n, p in model.named_parameters():
        if p.grad is None:
            continue
        d_p = p.grad
        if priors is not None:
            if n in priors:
                d_p.add_(p.data, alpha=-priors[n])
        div += d_p.sum()
    return div


def compute_concat_gradient(model, priors=None):
    whole_grad = torch.Tensor()
    for n, p in model.named_parameters():
        if p.grad is None:
            continue
        d_p = p.grad
        if priors is not None:
            if n in priors:
                d_p.add_(p.data, alpha=-priors[n])
        whole_grad = torch.cat([whole_grad, d_p.flatten()])
    return whole_grad


def compute_log_likelihood(x, y, model):
    y_hat = model(x)
    log_likelihood = -F.cross_entropy(y_hat, y, reduction='mean')
    return log_likelihood


def compute_potential_grad(models, train_x, train_y, N_train, priors=None):
    for model in models:
        model.zero_grad()
    log_likelihoods = [(compute_log_likelihood(train_x, train_y, model) * N_train).backward() for model in models]
    potential_grad = -1 * torch.stack([compute_concat_gradient(model, priors) for model in models])
    return potential_grad


def compute_mc_estimate(function: callable, models, x: torch.tensor):
    return function(models, x).sum(0) / len(models)


def compute_naive_variance(function: callable, control_variate: callable, models, x: torch.tensor, potential_grad=None):
    def diff(model_, x_):
        return function(model_, x_) - control_variate(model_, x_, potential_grad=potential_grad)

    sample_mean = compute_mc_estimate(diff, models, x)
    sample_mean_no_cv = compute_mc_estimate(function, models, x)
    v = ((diff(models, x) - sample_mean) ** 2).sum(0) / (len(models) - 1)
    v_no_cv = ((function(models, x) - sample_mean_no_cv)**2).sum(0) / (len(models) - 1)

    return v, v_no_cv


class SampleVarianceEstimator(object):
    def __init__(self, function, traj, potential_grads=None, cv=None):
        self.traj = traj
        self.function = function
        self.potential_grads = potential_grads
        self.cv = cv

    def estimate_variance(self, x, use_cv=True, all_values=None):
        if all_values is None:
            if use_cv is True:
                all_values = self.function(self.traj, x) - self.cv(self.traj, x, potential_grad=self.potential_grads)
            else:
                all_values = self.function(self.traj, x)

        var = ((all_values - all_values.mean(0)) ** 2).sum(0) / (len(self.traj) - 1)

        return var


class SpectralVarianceEstimator(object):
    def __init__(self, function, traj, potential_grads=None, cv=None):
        self.traj = traj
        self.function = function
        self.potential_grads = potential_grads
        self.cv = cv

    def tukey_hanning_window(self, sequence, k):
        n = sequence.shape[0]
        pi = sequence.mean(0)
        w = 0
        for i in range(n-k):
            w += (1. / n) * (sequence[i] - pi)*(sequence[i+k] - pi)
        
        return w

    def compute_spectral_variance(self, sequence):
        n = len(sequence)
        floor_bord = - int(np.floor(np.sqrt(n))) + 1
        up_bord = int(np.floor(np.sqrt(n))) - 1
        sigma2 = 0

        for k in range(floor_bord, up_bord + 1):
            cos = np.cos(np.pi*np.abs(k) / np.floor(np.sqrt(n)))
            sigma2 += 0.5 * (1 + cos) * self.tukey_hanning_window(sequence, np.abs(k))

        return sigma2

    def estimate_variance(self, x, use_cv=True, all_values=None):
        if all_values is None:
            if use_cv is True:
                all_values = self.function(self.traj, x) - self.cv(self.traj, x, potential_grad=self.potential_grads)
            else:
                all_values = self.function(self.traj, x)

        var = self.compute_spectral_variance(all_values)
        return var

        

# class SpectralVariance(object):
#     def __init__(self, function, sample, window_lag_f:callable, truncation_point):
#         self.function = function
#         self.sample = sample
#         self.window_lag_f = window_lag_f
#         self.truncation_point = truncation_point
#         self.centr_values = []
#         # if x is not None:
#         #     values = np.array([function(model, x) for model in sample])
#         #     self.cetr_values = values - np.mean(values)

#         self.autocovariances = None #torch.zeros()

#     def __call__(self, x):
#         values = torch.stack([self.function(model, x) for model in self.sample], dim=0)
#         self.centr_values = values - torch.mean(values, dim=0)
#         self.autocovariances = torch.zeros(len(self.sample), x.shape[0])

#         return self.get_variance()

#     def get_autocovariance(self, s):
#         if s == 0:
#             left_side = self.centr_values
#         else:
#             left_side = self.centr_values[:-s]
#         rho = (left_side * self.centr_values[s:]).mean(0)
#         return rho
    
#     def get_variance(self):
#         for s in range(self.truncation_point):
#             self.autocovariances[s] = self.get_autocovariance(s)
#         variance = 0
#         for s in range(-self.truncation_point+1, self.truncation_point):
#             variance += self.autocovariances[np.abs(s)] * self.window_lag_f(s / self.truncation_point)

#         return variance


def trapezoidal_kernel(s):
    assert -1 <= s <= 1
    if -1 <= s < -0.5:
        return 2 * s + 2
    elif -0.5 <= s < 0.5:
        return 1
    else:
        return -2 * s + 2
        