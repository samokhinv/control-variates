import torch
import numpy as np
import dill as pickle


def tukey_hanning_window(sequence, k):
    n = len(sequence)
    pi = np.mean(sequence)
    w = 0
    for i in range(n-k):
        w += (1. / n) * (sequence[i] - pi)*(sequence[i+k] - pi)
    
    return w


def compute_spectral_variance(sequence):
    n = len(sequence)
    floor_bord = - int(np.floor(np.sqrt(n))) + 1
    up_bord = int(np.floor(np.sqrt(n))) - 1
    sigma2 = 0

    for k in range(floor_bord, up_bord + 1):
        cos = np.cos(np.pi*np.abs(k) / np.floor(np.sqrt(n)))
        sigma2 += 0.5 * (1 + cos) * tukey_hanning_window(sequence, np.abs(k))

    return sigma2