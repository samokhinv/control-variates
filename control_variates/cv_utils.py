import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from tqdm import trange, tqdm

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


# def compute_concat_gradient(model, priors=None):
#     whole_grad = torch.Tensor()
#     for n, p in model.named_parameters():
#         if p.grad is None:
#             continue
#         d_p = p.grad
#         if priors is not None:
#             if n in priors:
#                 d_p.add_(p.data, alpha=-priors[n])
#         whole_grad = torch.cat([whole_grad, d_p.flatten()])
#     return whole_grad


# def compute_log_likelihood(x, y, model):
#     y_hat = model(x)
#     log_likelihood = -F.cross_entropy(y_hat, y, reduction='mean')
#     return log_likelihood


# def compute_potential_grad(models, train_x, train_y, N_train, priors=None):
#     for model in models:
#         model.zero_grad()
#     log_likelihoods = [(compute_log_likelihood(train_x, train_y, model) * N_train).backward() for model in models]
#     potential_grad = -1 * torch.stack([compute_concat_gradient(model, priors) for model in models])
#     return potential_grad


# def compute_mc_estimate(function: callable, models, x: torch.tensor):
#     return function(models, x).sum(0) / len(models)


# def compute_naive_variance(function: callable, control_variate: callable, models, x: torch.tensor, potential_grad=None):
#     def diff(model_, x_):
#         return function(model_, x_) - control_variate(model_, x_, potential_grad=potential_grad)

#     sample_mean = compute_mc_estimate(diff, models, x)
#     sample_mean_no_cv = compute_mc_estimate(function, models, x)
#     v = ((diff(models, x) - sample_mean) ** 2).sum(0) / (len(models) - 1)
#     v_no_cv = ((function(models, x) - sample_mean_no_cv)**2).sum(0) / (len(models) - 1)

#     return v, v_no_cv


class SampleVarianceEstimator(object):
    def __init__(self, function=None, traj=None, potential_grads=None, cv=None):
        self.traj = traj
        self.function = function
        self.potential_grads = potential_grads
        self.cv = cv

    def estimate_variance(self, trajs):
        var = trajs.std(-1)**2
        return var


class SpectralVarianceEstimator(object):
    def __init__(self, window):
        self.W = torch.FloatTensor(window).view(-1)

    # def tukey_hanning_window(self, sequence, k):
    #     n = sequence.shape[-1]
    #     pi = sequence.mean(-1)
    #     w = torch.zeros(sequence.shape[:-1])
    #     for i in range(n-k):
    #         w += (1. / n) * (sequence[..., i] - pi)*(sequence[..., i+k] - pi)
    #     return w

    # def estimate_variance(self, sequence):
    #     n = sequence.shape[-1]
    #     floor_bord = - 10 # int(np.floor(np.sqrt(n))) + 1
    #     up_bord = 10 #int(np.floor(np.sqrt(n))) - 1
    #     sigma2 = torch.zeros(sequence.shape[:-1])

    #     #for seq_id, seq in tqdm(enumerate(sequence), leave=False):
    #     for k in trange(floor_bord, up_bord + 1, leave=False):
    #         cos = np.cos(np.pi*np.abs(k) / np.floor(np.sqrt(n)))
    #         sigma2 += 0.5 * (1 + cos) * self.tukey_hanning_window(sequence, np.abs(k))
    #     return sigma2
    
    @staticmethod
    def mult_W(x,c):
        """
        performs multiplication (fast, by FFT) with W - toeplitz (bn-diagonal) matrix
        Args:
            c - vector of 
        returns:
            matvec product;
        """
        n = x.shape[-1]

        x_emb = torch.zeros(1, 2*n-1, 2)
        x_emb[:, :n, 0] = x
        c_emb = torch.zeros(1, 2*n-1, 2)
        c_emb[..., 0] = c
        return torch.ifft(torch.fft(c_emb, 1)*torch.fft(x_emb, 1), 1)[0, :n, 0]

    @staticmethod
    def PWP_fast(x, c):
        """
        Same PWP as above, but now with FFT
        """
        y = SpectralVarianceEstimator.mult_W(x - torch.mean(x), c)
        return y - torch.mean(y)

    @staticmethod
    def spectral_variance(Y, W):
        n = Y.shape[-1]
        return torch.dot(SpectralVarianceEstimator.PWP_fast(Y,W),Y)/n

    def estimate_variance(self, sequence):
        W = self.W
        var = torch.zeros(sequence.shape[0])
        for seq_id, seq in enumerate(sequence):
            var[seq_id] = self.spectral_variance(seq, W)
        return var


def trapezoidal_kernel(s):
    assert -1 <= s <= 1
    if -1 <= s < -0.5:
        return 2 * s + 2
    elif -0.5 <= s < 0.5:
        return 1
    else:
        return -2 * s + 2
        