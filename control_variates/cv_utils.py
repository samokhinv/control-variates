import torch
import numpy as np
from torch import nn
from torch.nn import functional as F


def state_dict_to_vec(state_dict):
    return torch.cat([w_i.view(-1) for w_i in state_dict.values()])


def compute_tricky_divergence(model, priors=None):
    div = 0
    for n, p in model.named_parameters():
        if p.grad is None:
            continue
        d_p = p.grad
        if priors is not None:
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


def SpectralVariance(object):
    def __init__(self, function, sample):
        self.function = function
        self.sample = sample
        values = np.array(list(map(function, sample)))
        self.cetr_values = values - np.mean(values)

        self.autocovariances = []

    def get_autocovariance(self, s):
        rho = np.dot(self.centr_values[:s], self.centr_values[s:]) / len(self.centr_values)
        return rho
    
    def get_variance(self, window_lag_f, truncation_point):
        for s in range(truncation_point):
            self.autocovariances.append(self.get_autocovariance(s))
        variance = 0
        for s in range(-truncation_point-1, truncation_point):
            variance.append(self.autocovariances[np.abs(s)] * window_lag_f(s / truncation_point))

        return variance


def trapezoidal_kernel(s):
    assert -1 <= s <= 1
    if -1 <= s < -0.5:
        return 2 * s + 2
    elif -0.5 <= s < 0.5:
        return 1
    else:
        return -2 * s + 2

    


