import torch
from torch import nn
from torch.nn import functional as F


def get_prediction(model, x):
    return F.softmax(model(x), dim=-1)


def get_binary_prediction(model, x, classes):
    assert len(classes) == 2
    return F.softmax(model(x)[..., classes], dim=-1)[..., -1]  # не сумируются в единицу?


class MLP(nn.Module):
    def __init__(self, input_dim, width, depth, output_dim):
        super(MLP, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.width = width
        self.depth = depth

        layers = [nn.Linear(input_dim, width), nn.ReLU()]
        for i in range(depth - 1):
            layers.append(nn.Linear(width, width))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(width, output_dim))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        x = torch.flatten(x, 1)
        return self.block(x)


class LogRegression(nn.Module):
    def __init__(self, input_size):
        super(LogRegression, self).__init__()
        self.linear = nn.Linear(input_size, 2)

    def forward(self, x):
        return self.linear(x.flatten(1))  # logits to use in cross entropy


class SiBNN(nn.Module):
    """
    The Bayesian Neural Net used in [1] for experiments

    [1] Neural Control Variates for Monte Carlo Variance Reduction <https://arxiv.org/pdf/1806.00159.pdf>
    """
    def __init__(self, in_channels=1, output_size=10):
        super(SiBNN, self).__init__()
        self.in_channels = in_channels
        self.output_size = output_size

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 2, 5),
            nn.MaxPool2d(2),
            nn.Conv2d(2, 3, 3),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(147, output_size)
        )

    def forward(self, image):
        return self.net(image)
