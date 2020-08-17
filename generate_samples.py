from control_variates.model import MLP
from control_variates.optim import LangevinSGD as SGLD, ScaleAdaSGHMC as H_SA_SGHMC
from mnist_utils import load_mnist_dataset
from control_variates.trainer import BNNTrainer
import torch
from torch.nn import functional as F
from control_variates.model import LogRegression

import numpy as np
import dill as pickle
from pathlib import Path
from functools import partial
from easydict import EasyDict as edict

from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm
import argparse

INPUT_DIM = 784

mcmc_grdadients = {'sgld': SGLD, 'sghmc': H_SA_SGHMC}


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bnn_lr', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=500)
    parser.add_argument('--n_hidden_layers', type=int, default=0)
    parser.add_argument('--hidden_dim', type=int, default=0)
    parser.add_argument('--n_epoch', type=int, default=200)
    parser.add_argument('--classes', type=int, nargs='+', default=[3,5])
    parser.add_argument('--alpha0', type=float, default=10)
    parser.add_argument('--beta0', type=float, default=10)
    parser.add_argument('--resample_prior_every', type=int, default=15)
    parser.add_argument('--resample_momentum_every', type=int, default=50)
    parser.add_argument('--burn_in_epochs', type=int, default=20)
    parser.add_argument('--resample_prior_until', type=int, default=100)
    parser.add_argument('--save_freq', type=int, default=2)
    parser.add_argument('--report_every', type=int, default=10)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--mcmc_gradient', type=str, choices=['sgld', 'sghmc'], default='sghmc')
    parser.add_argument('--n_samples', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    #parser.add_argument()
    args = parser.parse_args()

    return args


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    Path('data', 'mnist').mkdir(exist_ok=True, parents=True)
    trainloader, valloader = load_mnist_dataset(Path('data', 'mnist'), args.batch_size, classes=args.classes)
   
    def nll_func(y_hat, y):
        nll = F.cross_entropy(y_hat, y, reduction='sum')
        return nll

    def err_func(y_hat, y):
        err = y_hat.argmax(-1).ne(y)
        return err

    mcmc_class = mcmc_grdadients[args.mcmc_gradient]
    all_weights_and_priors = []

    for _ in range(args.n_samples):
        if args.n_hidden_layers == 0:
            model = LogRegression(INPUT_DIM)
        else:
            model = MLP(input_dim=INPUT_DIM, width=args.hidden_dim, depth=args.n_hidden_layers, output_dim=len(args.classes))

        optimizer = mcmc_class(model.parameters(), lr=args.bnn_lr, alpha0=args.alpha0, beta0=args.beta0)

        trainer = BNNTrainer(model, 
            optimizer, 
            nll_func, 
            err_func, 
            trainloader, 
            valloader, 
            device=device, 
            resample_prior_every=args.resample_prior_every,
            resample_momentum_every=args.resample_momentum_every,
            save_freq=args.save_freq,
            batch_size=args.batch_size
            )
        trainer.train(n_epoch=args.n_epoch, burn_in_epochs=args.burn_in_epochs, resample_prior_until=args.resample_prior_until)
        weights_set = trainer.weight_set_samples[-(args.n_epoch - args.resample_prior_until) // args.save_freq:]

        opt_with_priors = trainer.optimizer
        priors = {}
        group_params = opt_with_priors.param_groups[0]['params']
        for (n, _), p in zip(weights_set, group_params):  
            state = opt_with_priors.state[p]  
            priors[n] = state['weight_decay']

        all_weights_and_priors.append((weights_set, priors))
    
    pickle.dump(all_weights_and_priors, Path('../saved_samples', 'mnist_weights', f'{args.n_samples}_samples.pkl').open('wb'))


if __name__ == '__main__':
    args = parse_arguments()
    main(args)
