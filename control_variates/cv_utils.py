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


def compute_log_likelihood(x, y, model):
    y_hat = model(x)
    log_likelihood = -F.cross_entropy(y_hat, y, reduction='mean')
    return log_likelihood


def compute_mc_estimate(function: callable, models, x: torch.tensor):
    return sum(function(model, x) for model in models) / len(models)


def compute_naive_variance(function: callable, control_variate: callable, models, x: torch.tensor):
    def diff(model_, x_):
        return function(model_, x_) - control_variate(model_, x_)

    sample_mean = compute_mc_estimate(diff, models, x)
    sample_mean_no_cv = compute_mc_estimate(function, models, x)
    v = sum((diff(model, x) - sample_mean)**2 for model in models) / (len(models) - 1)
    v_no_cv = sum((function(model, x) - sample_mean_no_cv)**2 for model in models) / (len(models) - 1)

    return v, v_no_cv


class SpectralVariance(object):
    def __init__(self, function, sample, window_lag_f:callable, truncation_point):
        self.function = function
        self.sample = sample
        self.window_lag_f = window_lag_f
        self.truncation_point = truncation_point
        self.centr_values = []
        # if x is not None:
        #     values = np.array([function(model, x) for model in sample])
        #     self.cetr_values = values - np.mean(values)

        self.autocovariances = None #torch.zeros()

    def __call__(self, x):
        values = torch.stack([self.function(model, x) for model in self.sample], dim=0)
        self.centr_values = values - torch.mean(values, dim=0)
        self.autocovariances = torch.zeros(len(self.sample), x.shape[0])

        return self.get_variance()

    def get_autocovariance(self, s):
        if s == 0:
            left_side = self.centr_values
        else:
            left_side = self.centr_values[:-s]
        rho = (left_side * self.centr_values[s:]).mean(0)
        return rho
    
    def get_variance(self):
        for s in range(self.truncation_point):
            self.autocovariances[s] = self.get_autocovariance(s)
        variance = 0
        for s in range(-self.truncation_point+1, self.truncation_point):
            variance += self.autocovariances[np.abs(s)] * self.window_lag_f(s / self.truncation_point)

        return variance


def trapezoidal_kernel(s):
    assert -1 <= s <= 1
    if -1 <= s < -0.5:
        return 2 * s + 2
    elif -0.5 <= s < 0.5:
        return 1
    else:
        return -2 * s + 2

    


