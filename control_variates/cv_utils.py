import torch
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



