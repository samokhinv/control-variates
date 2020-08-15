import torch
from torch import nn
from torch.nn import functional as F
from .cv_utils import compute_log_likelihood, compute_tricky_divergence, state_dict_to_vec


class SteinCV:
    def __init__(self, psy_model, train_x, train_y, priors, N_train):
        self.psy_model = psy_model
        self.train_x = train_x
        self.train_y = train_y
        self.priors = priors
        self.N_train = N_train

    def __call__(self, model, x):
        model.zero_grad()
        log_likelihood = compute_log_likelihood(self.train_x, self.train_y, model) * self.N_train
        log_likelihood.backward()

        model_weights = state_dict_to_vec(model.state_dict())
        model_weights.requires_grad = True
        psy_value = self.psy_model(model_weights, x)

        #psy_div = compute_tricky_divergence(self.psy_model)
        psy_div = []
        for i in range(x.shape[0]):
            psy_div.append(torch.autograd.grad(psy_value[:, i], model_weights, retain_graph=True)[0].sum())
        psy_div = torch.stack(psy_div, dim=0)
        
        ll_div = compute_tricky_divergence(model, self.priors)

        ncv_value = psy_value*ll_div.sum() + psy_div  # зачем повторять тензор? Женя: psy_value имеет дополнительную размерность - размерность x

        return ncv_value


class BasePsy(nn.Module):
    def __init__(self):
        super().__init__()

    def init_zero(self):
        for n, p in self.named_parameters():
            nn.init.zeros_(p)
            

class PsyLinear(BasePsy):
    def __init__(self, input_dim):
        super().__init__()
        self.layer = nn.Linear(input_dim, 1)#, bias=False)

    def forward(self, weights, x):
        return self.layer(weights).squeeze(-1).repeat(1, x.shape[0])


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
        return self.block(weights)


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
        weights_hid  = self.block1(weights) # n * h
        x_hid = self.block2(x)  # m * h
        hid = weights_hid.repeat(x_hid.shape[0]).reshape(weights_hid.shape[0], x_hid.shape[0], -1)
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
            nn.ReLU()
        )

        self.alpha = nn.Linear(in_features=hidden_size, out_features=1)

    def forward(self, flat_weights, image):
        return self.alpha(self.weights_block(flat_weights) + self.image_block(image))
