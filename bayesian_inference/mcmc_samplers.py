import numpy as np
import torch
from .potentials import Potential


def RWM(potential:Potential, gamma, n_burn, n_sample):
    traj, potential_grad = [], []
    point = torch.randn_like(potential.mu)
    for _ in range(n_burn):
        grad = potential.grad(point)
        new_point = point + (2 * gamma)**0.5 * torch.randn_like(point)
        logratio = potential(new_point) - potential(point)
        if torch.rand(1).log().item() <= logratio.item():
            point = new_point
    for _ in range(n_sample):
        traj.append(point)
        potential_grad.append(potential.grad(point))
        new_point = point + (2*gamma)**0.5*torch.randn_like(point)
        logratio = potential(new_point) - potential(point)
        if torch.rand(1).log().item() <= logratio.item():
            point = new_point

    return torch.stack(traj, dim=0), torch.stack(potential_grad, dim=0)
