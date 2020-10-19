import torch
from pathlib import Path
import dill as pickle
import numpy as np
import copy
import numpy as np
import random


def standartize(X_train,X_test,intercept = True):
    """Whitens noise structure, covariates updated
    """
    X_train = copy.deepcopy(X_train)
    X_test = copy.deepcopy(X_test)
    if intercept:#adds intercept term
        X_train = np.concatenate((np.ones(X_train.shape[0]).reshape(X_train.shape[0],1),X_train),axis=1)
        X_test = np.concatenate((np.ones(X_test.shape[0]).reshape(X_test.shape[0],1),X_test),axis=1)
    d = X_train.shape[1]
    # Centering the covariates 
    means = np.mean(X_train,axis=0)
    if intercept:#do not subtract the mean from the bias term
        means[0] = 0.0
    # Normalizing the covariates
    X_train -= means
    Cov_matr = np.dot(X_train.T,X_train)
    U,S,V_T = np.linalg.svd(Cov_matr,compute_uv = True)
    Sigma_half = U @ np.diag(np.sqrt(S)) @ V_T
    Sigma_minus_half = U @ np.diag(1./np.sqrt(S)) @ V_T
    X_train = X_train @ Sigma_minus_half
    # The same for test sample
    X_test = (X_test - means) @ Sigma_minus_half
    return X_train,X_test  


class DatasetStandarsScaler():
    def __init__(self, **kwargs):
        self.use_std = kwargs.get('use_std', True)
        self.biased = kwargs.get('biased', True)
        self.mean = kwargs.get('mean', 0)
        self.std = kwargs.get('std', 1)
        self.eps = kwargs.get('eps', 1e-7)

    def fit(self, dataset):
        N = len(dataset)
        mean = 0
        var = 0
        for x, _ in dataset:
            mean += x
        self.mean = mean / N
        for x, _ in dataset:
            var += (x - self.mean)**2
        self.std = (var / N)**(0.5)
        
    def transform(self, dataset):
        new_dataset = []
        for x, y in dataset:
            new_dataset.append(((x - self.mean) / (self.std + self.eps), y))
        return new_dataset


def load_trajs(trajs_path, bayesian_nn_class, canvas, requires_grad=False):
    trajs, traj_grads, priors = pickle.load(Path(trajs_path).open('rb'))
    
    for traj in trajs:
        for i, state_dict in enumerate(traj):
            traj[i] = bayesian_nn_class(canvas)
            traj[i].load_state_dict(state_dict)
            for p in traj[i].parameters():
                p.requires_grad = False
    
    traj_grads = torch.FloatTensor(np.stack(traj_grads))

    return trajs, traj_grads, priors


def random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
