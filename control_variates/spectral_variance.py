import torch
import numpy as np
import dill as pickle
from numpy.fft import fft, ifft


# def tukey_hanning_window(sequence, k):
#     n = len(sequence)
#     pi = np.mean(sequence)
#     w = 0
#     for i in range(n-k):
#         w += (1. / n) * (sequence[i] - pi)*(sequence[i+k] - pi)
    
#     return w


# def compute_spectral_variance(sequence):
#     n = len(sequence)
#     floor_bord = - int(np.floor(np.sqrt(n))) + 1
#     up_bord = int(np.floor(np.sqrt(n))) - 1
#     sigma2 = 0

#     for k in range(floor_bord, up_bord + 1):
#         cos = np.cos(np.pi*np.abs(k) / np.floor(np.sqrt(n)))
#         sigma2 += 0.5 * (1 + cos) * tukey_hanning_window(sequence, np.abs(k))

#     return sigma2


# def PWP(x,W):
#     """
#     performs multiplication (slow) with P - projector, W - topelitz (bn-diagonal) matrix
#     Args:
#         W - bn-diagonal matrix os shap (n,n) in csr format;
#     returns:
#         np.array of shape (n,) - result of PWP multiplicaton;
#     """
#     y = W @ (x - np.mean(x))
#     return y - np.mean(y)

# def mult_W(x,c):
#     """
#     performs multiplication (fast, by FFT) with W - toeplitz (bn-diagonal) matrix
#     Args:
#         c - vector of 
#     returns:
#         matvec product;
#     """
#     n = len(x)
#     x_emb = np.zeros(2*n-1)
#     x_emb[:n] = x
#     return ifft(fft(c)*fft(x_emb)).real[:n]
    
# def PWP_fast(x,c):
#     """
#     Same PWP as above, but now with FFT
#     """
#     y = mult_W(x - np.mean(x),c)
#     return y - np.mean(y)

# def Spectral_var(Y,W):
#     """
#     Compute spectral variance estimate for asymptotic variance with given kernel W for given vector Y
#     """
#     n = len(Y)
#     return np.dot(PWP_fast(Y,W),Y)/n
                  
# def set_bn(n):
#     """
#     function that sets size of the window in BM,OBM,SV estimates;
#     please, make changes only here to change them simulteneously
#     """
#     #return np.round(2*np.power(n,0.33)).astype(int)
#     return 10
    
def construct_ESVM_kernel(n):
    """
    Same as before, but now returns only first row of embedding circulant matrix;
    Arguments:
        n - int,size of the matrix;
    Returns:
        c - np.array of size (2n-1);
    """
    bn = set_bn(n)
    trap_left = np.linspace(0,1,bn)
    trap_center = np.ones(2*bn+1,dtype = float)
    trap_right = np.linspace(1,0,bn)
    diag_elems = np.concatenate([trap_left,trap_center,trap_right])
    c = np.zeros(2*n-1,dtype = np.float64)
    c[0:bn+1] = 1.0
    c[bn+1:2*bn+1] = trap_right
    c[-bn:] = 1.0
    c[-2*bn:-bn] = trap_left
    return c

def construct_Tukey_Hanning(n,bn):
    """
    Same as before, but now returns only first row of embedding circulant matrix;
    Arguments:
        n - int,size of the matrix;
        bn - truncation point (lag-window size);
    Returns:
        c - np.array of size (2n-1);
    """
    c = np.zeros(2*n-1,dtype = np.float64)
    #c = np.zeros(n,dtype = np.float64)
    if bn == 0:#1-dioagonal matrix
        c[0] = 1.0
        return c
    elif bn == 1:#3-diagonal matrix
        c[0] = 1.0
        c[1] = 1.0
        c[-1] = 1.0
        return c
    diag_elems = 1./2 + 1./2*np.cos(np.pi/bn*np.arange(-bn,bn+1))
    c[0:(bn+1)] = diag_elems[bn:]
    c[-bn:] = diag_elems[:bn]
    return c