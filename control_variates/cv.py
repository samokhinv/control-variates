from functools import partial

import torch
from torch import nn
from torch.nn import functional as F
from .cv_utils import (
    #compute_log_likelihood, 
    compute_tricky_divergence, 
    state_dict_to_vec, 
    #compute_concat_gradient, 
    #compute_potential_grad
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
    def __init__(self, psy_model, **kwargs):
        self.psy_model = psy_model

    def __call__(self, models_weights, x_batch, potential_grad):
        psy_value = self.psy_model(models_weights, x_batch)
        if isinstance(self.psy_model, PsyConstVector):
            psy_div = 0.
        else:
            psy_div = self.psy_model.divergence(models_weights, x_batch)
            #print(psy_div.shape)
            # psy_func = partial(self.psy_model, x=x_batch)
            # psy_jac = torch.autograd.functional.jacobian(psy_func, models_weights, create_graph=True)
            # if models_weights.ndim == 2:
            #     psy_div = torch.einsum('ijil->ij', psy_jac)
            # elif models_weights.ndim == 3:
            #     psy_div = torch.einsum('kijil->kij', psy_jac)
            # else:
            #     raise NotImplementedError

        if potential_grad.ndim == 2:
            ncv_value = -1 * torch.einsum('ijk,ik->ij', psy_value, potential_grad) + psy_div
        elif potential_grad.ndim == 3:
            ncv_value = -1 * torch.einsum('injk,ink->inj', psy_value, potential_grad) + psy_div
        else:
            raise NotImplementedError

        return ncv_value
        

class BasePsy(nn.Module):
    def __init__(self):
        super().__init__()

    def init_zero(self):
        for p in self.parameters():
            nn.init.zeros_(p)


class PsyConstVector(BasePsy):
    def __init__(self, input_dim):
        super().__init__()
        self.param = nn.Parameter(torch.ones(input_dim))

    def forward(self, weights, x):
        return self.param.repeat(list(weights.shape[:-1]) + [x.shape[0], 1])


class PsyLinear(BasePsy):
    def __init__(self, input_dim):
        super().__init__()
        self.layer = nn.Linear(input_dim, 1)#, bias=False)

    def forward(self, weights, x):
        return self.layer(weights).unsqueeze(-1).repeat([1]*(weights.ndim - 1) + [x.shape[0], weights.shape[-1]])

    def divergence(self, weights, x):
        return self.layer.weight.sum() * torch.ones(list(weights.shape[:-1]) + [x.shape[0]])


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
