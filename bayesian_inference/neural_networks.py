import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from abc import ABC, abstractmethod
from numpy.random import gamma


def get_prediction(model, x):
    return F.softmax(model(x.float()), dim=-1)


def get_binary_prediction(models, x, classes):
    assert len(classes) == 2
    if isinstance(models, nn.Module):
        models = (models,)
    pred = []
    for model in models: 
        pred.append(model(x.float()))
    pred = torch.stack(pred, 0)
    pred = F.softmax(pred[..., classes], dim=-1)[..., -1]
    return pred


def classif_err_fn(out, y):
    return out.argmax(-1).ne(y).sum()


def classif_loss_fn(out, y):
    return F.cross_entropy(out, y, reduction='sum')


def define_nn(config, x_shape, prior=None):
    for k, v in config.items():
        if isinstance(v, str):
            config[k] = eval(v)
    if config['model'] is LogRegression:
        input_dim = np.prod(x_shape)
        config['input_dim'] = input_dim
        canvas = input_dim, config['bias']
    else:
        raise NotImplementedError(config['model'])

    bayesian_nn = config['model'](canvas, prior)

    return bayesian_nn, config 


class BayesianNN(nn.Module):
    @abstractmethod
    def __init__(self, canvas, prior=None):
        super().__init__()
        self.canvas = canvas
        self.prior = prior

    @abstractmethod
    def get_log_prior(self):
        return self.prior.get_log_prior(self)

    @abstractmethod
    def prior_dict(self):
        sigma2 = dict()
        for n, p in self.named_parameters():
            sigma2[n] = p.sigma2
        return sigma2

    @abstractmethod
    def load_prior_dict(self, prior_dict):
        for n, p in self.named_parameters():
            p.sigma2 = prior_dict[n]
    
    @abstractmethod
    def sample_prior(self):
        self.prior.sample_prior(self)

    @abstractmethod
    def resample_prior(self):
        self.prior.resample_prior(self)

    @abstractmethod
    def init(self):
        self.load_state_dict(self.init_state_dict)
        self.load_prior_dict(self.init_prior_dict)

    def init_norm(self):
        for p in self.parameters():
            nn.init.normal_(p, mean=0, std=1.)


class MLP(BayesianNN):
    def __init__(self, canvas, prior=None):
        super().__init__(canvas, prior)
        input_dim, width, depth, output_dim = canvas

        layers = [nn.Linear(input_dim, width), nn.ReLU()]
        for i in range(depth - 1):
            layers.append(nn.Linear(width, width))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(width, output_dim))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        x = torch.flatten(x, 1)
        return self.block(x)


class LogRegression(BayesianNN):
    def __init__(self, canvas, prior=None):
        super().__init__(canvas, prior)
        input_size, bias = canvas
        self.linear = nn.Linear(input_size, 2, bias=bias)

    def forward(self, x):
        x = self.linear(x.flatten(1))
        return x


class SiBNN(BayesianNN):
    """
    The Bayesian Neural Net used in [1] for experiments

    [1] Neural Control Variates for Monte Carlo Variance Reduction <https://arxiv.org/pdf/1806.00159.pdf>
    """
    def __init__(self, canvas, prior=None):
        super().__init__(canvas, prior)
        in_channels, output_size = canvas

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 2, 5),
            nn.MaxPool2d(2),
            nn.Conv2d(2, 3, 3),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(75, output_size)
        )

    def forward(self, image):
        return self.net(image)


class Prior(ABC):
    @abstractmethod
    def __init__(self):
        pass

    # @abstractmethod
    # def __call__(self, bayesian_nn):
    #     pass

    @abstractmethod
    def get_log_prior(self, bayesian_nn):
        pass

    @abstractmethod
    def resample_prior(self, bayesian_nn):
        pass

    @abstractmethod
    def sample_prior(self, bayesian_nn):
        pass


class GammaGaussPrior(Prior):
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta

    def sample_prior(self, bayesian_nn):
        for p in bayesian_nn.parameters():
            p.sigma2 = 1 / gamma(shape=self.alpha, scale=1 / self.beta) 

    def resample_prior(self, bayesian_nn):
        for p in bayesian_nn.parameters():
            alpha = self.alpha + p.data.nelement() / 2
            beta = self.beta + (p.data ** 2).sum().item() / 2
            p.sigma2 = 1 / gamma(shape=alpha, scale=1 / beta)

    def get_log_prior(self, bayesian_nn):
        log_prior = 0
        for p in bayesian_nn.parameters():
            log_prior += - p.norm(2) ** 2 / (2 * p.sigma2)

        return log_prior
