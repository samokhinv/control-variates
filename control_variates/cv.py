from functools import partial

import torch
from torch import nn
from torch.nn import functional as F
from .cv_utils import (
    compute_log_likelihood, 
    compute_tricky_divergence, 
    state_dict_to_vec, 
    compute_concat_gradient, 
    compute_potential_grad
)


def reshape_m_i(models_vec, image_vec):
    """
    Function to get vectors for product of models and images sets

    :param models_vec: (n_models, *)
    :param image_vec: (n_images, *)
    :return: repeated inputs, both of shape (n_models, n_images, *)
    """
    models_vec = torch.repeat_interleave(models_vec.unsqueeze(1), repeats=image_vec.shape[0], dim=1)
    image_vec = torch.repeat_interleave(image_vec.unsqueeze(0), repeats=models_vec.shape[0], dim=0)
    return models_vec, image_vec


class SteinCV:
    def __init__(self, psy_model, train_x, train_y, priors, N_train, potential_grad=None):
        self.psy_model = psy_model
        self.train_x = train_x
        self.train_y = train_y
        self.priors = priors
        self.N_train = N_train
        self.potential_grad = potential_grad

    # def update_potential(self, train_x, train_y):
    #     log_likelihoods = [(compute_log_likelihood(self.train_x, self.train_y, model) * self.N_train).backward() for model in models]
    #     #ll_grad = torch.stack([compute_tricky_divergence(model, self.priors) for model in models])  # ll_grad для каждой модели
    #     ll_grad = self.train_x.shape[0] * torch.stack([compute_concat_gradient(model, self.priors) for model in models])
    #     if self.ll_grad is None:
    #         self.ll_grad = ll_grad / self.train_x.shape[0]
    #     else:
    #         self.ll_grad = (self.ll_grad * self.n_batch + ll_grad) / (self.n_batch + self.train_x.shape[0])

    #     self.n_batch += self.train_x.shape[0]
    #     self.priors = None 

    def __call__(self, models, x_batch, potential_grad=None):
        if potential_grad is None:
            potential_grad = self.potential_grad
        if isinstance(models, nn.Module):
            models = (models, )
        for model in models:
            model.zero_grad()
        if potential_grad is None:
            potential_grad = compute_potential_grad(models, self.train_x, self.train_y, self.N_train, self.priors)
        models_weights = torch.stack([state_dict_to_vec(model.state_dict()) for model in models])  # батч моделей
        models_weights.requires_grad = True
        psy_value = self.psy_model(models_weights, x_batch)  # хотим тензор число моделей X число примеров
        
        if isinstance(self.psy_model, PsyConstVector):
            psy_div = 0.
        else:
            psy_func = partial(self.psy_model, x=x_batch)
            psy_jac = torch.autograd.functional.jacobian(psy_func, models_weights, create_graph=True)
            psy_div = torch.einsum('ijil->ij', psy_jac)  # я чет завис с размерностями: i - n_models, j - n_images, l - n_weights
        if psy_value.ndim == 2:
            if potential_grad.ndim == 2:
                psy_value = psy_value.unsqueeze(-1).repeat(1, 1, potential_grad.shape[-1])
        ncv_value = -1 * torch.einsum('ijk,ik->ij', psy_value, potential_grad) + psy_div
        return ncv_value


class BasePsy(nn.Module):
    def __init__(self):
        super().__init__()

    def init_zero(self):
        for n, p in self.named_parameters():
            nn.init.zeros_(p)


class PsyConstVector(BasePsy):
    def __init__(self, input_dim):
        super().__init__()
        self.param = nn.Parameter(torch.zeros(input_dim))

    def forward(self, weights, x):
        return self.param.repeat([weights.shape[0], x.shape[0], 1])


class PsyLinear(BasePsy):
    def __init__(self, input_dim):
        super().__init__()
        self.layer = nn.Linear(input_dim, 1)#, bias=False)

    def forward(self, weights, x):
        return reshape_m_i(self.layer(weights), x)[0].squeeze(-1)


class PsyMLP(BasePsy):
    def __init__(self, input_dim, width, depth):
        super().__init__()

        self.input_dim = input_dim
        self.width = width
        self.depth = depth

        layers = [nn.Linear(input_dim, width), nn.LeakyReLU()]
        for i in range(depth - 1):
            layers.append(nn.Linear(width, width))
            layers.append(nn.LeakyReLU())
        layers.append(nn.Linear(width, 1, bias=False))

        self.block = nn.Sequential(*layers)

    def forward(self, weights, x):
        return self.block(weights).repeat(1, x.shape[0])


class PsyDoubleMLP(BasePsy):
    def __init__(self, input_dim1, width, depth1, input_dim2, depth2):
        super().__init__()


        layers1 = [nn.Linear(input_dim1, width), nn.ReLU()]
        for i in range(depth1 - 1):
            layers1.append(nn.Linear(width, width))
            layers1.append(nn.ReLU())

        self.block1 = nn.Sequential(*layers1)

        layers2 = [nn.Linear(input_dim2, width), nn.ReLU()]
        for i in range(depth2 - 1):
            layers2.append(nn.Linear(width, width))
            layers2.append(nn.ReLU())

        self.block2 = nn.Sequential(*layers2)

        self.final = nn.Linear(width, 1, bias=False)

        for n, p in self.named_parameters():
            if p.ndim >= 2:
                torch.nn.init.xavier_uniform_(p)

    def forward(self, weights, x):
        x = x.view(x.shape[0], -1)
        weights_hid = self.block1(weights) # n * h
        x_hid = self.block2(x)  # m * h
        hid = weights_hid.repeat(x_hid.shape[0], 1).reshape(weights_hid.shape[0], x_hid.shape[0], -1)
        hid = hid + x_hid
        return self.final(hid).squeeze(-1)


class PsyConv(BasePsy):
    """
    The NCV that repeats example from [1]

    [1] Neural Control Variates for Monte Carlo Variance Reduction <https://arxiv.org/pdf/1806.00159.pdf>
    """
    def __init__(self, weights_size, hidden_size):
        super(PsyConv, self).__init__()
        self.weights_size = weights_size
        self.hidden_size = hidden_size
        self.weights_block = nn.Linear(weights_size, hidden_size)
        self.image_block = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=2, kernel_size=5),
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=2, out_channels=3, kernel_size=3),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Flatten()
        )

        self.alpha = nn.Linear(in_features=hidden_size, out_features=1, bias=False)

    def forward(self, weights, x):
        return self.alpha(F.sigmoid(sum(reshape_m_i(self.weights_block(weights), self.image_block(x))))).squeeze(-1)
