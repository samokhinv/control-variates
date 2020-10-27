import torch
import dill as pickle
import numpy as np
from pathlib import Path
import argparse
from tqdm import trange

from bayesian_inference.potentials import GaussPotential
from bayesian_inference.mcmc_samplers import RWM
from utils import random_seed


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--n_trajs', type=int, default=100)
    parser.add_argument('--n_burn', type=int, default=10**4)
    parser.add_argument('--n_sample', type=int, default=10**5)

    parser.add_argument('--dim', type=int, default=2)
    parser.add_argument('--distribution_type', type=str,
        choices=['gauss', 'gauss_mixture'], default='gauss')
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--sampler', type=str, default='rwm')
    parser.add_argument('--save_path', type=str, required=True)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    return args


def main(args):
    if args.distribution_type == 'gauss':
        mu = torch.ones(args.dim)
        sigma = torch.randn(args.dim, args.dim)
        sigma = torch.mm(sigma, sigma.t())
        sigma.add_(torch.eye(args.dim))

        potential = GaussPotential(mu, sigma)

    trajs, potential_grads = [], []
    for _ in trange(args.n_trajs):
        traj, pot_grad = RWM(potential, args.gamma, args.n_burn, args.n_sample)
        trajs.append(traj)
        potential_grads.append(pot_grad)
    trajs = torch.stack(trajs, dim=0)
    potential_grads = torch.stack(potential_grads, dim=0)

    Path(args.save_path).parent.mkdir(exist_ok=True, parents=True)
    pickle.dump([trajs, potential_grads], Path(args.save_path).open('wb'))


if __name__ == '__main__':
    args = parse_arguments()
    random_seed(args.seed)
    main(args)