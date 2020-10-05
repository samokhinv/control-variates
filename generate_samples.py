import torch
from torch.nn import functional as F
import numpy as np
import dill as pickle
from pathlib import Path
from functools import partial
from easydict import EasyDict as edict
import random
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm
import argparse

from control_variates.model import MLP
from control_variates.optim import  SGLD, ScaleAdaSGHMC as H_SA_SGHMC
from mnist_utils import load_mnist_dataset
from UCI_utils import load_uci_dataset
from control_variates.trainer import BNNTrainer, BurnInScheduler
from control_variates.cv_utils import compute_potential_grad
from control_variates.model import LogRegression


mcmc_grdadients = {'sgld': SGLD, 'sghmc': H_SA_SGHMC}


def random_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bnn_lr', type=float, default=1e-7)
    parser.add_argument('--batch_size', type=int, default=500)
    parser.add_argument('--n_hidden_layers', type=int, default=0)
    parser.add_argument('--hidden_dim', type=int, default=0)
    parser.add_argument('--n_epoch', type=int, default=400)
    parser.add_argument('--classes', type=int, nargs='+', default=[3,5])
    parser.add_argument('--alpha0', type=float, default=1)
    parser.add_argument('--beta0', type=float, default=1)
    parser.add_argument('--resample_prior_every', type=int, default=15)
    parser.add_argument('--resample_momentum_every', type=int, default=50)
    parser.add_argument('--burn_in_epochs', type=int, default=200)
    parser.add_argument('--resample_prior_until', type=int, default=100)
    parser.add_argument('--save_freq', type=int, default=2)
    parser.add_argument('--report_every', type=int, default=10)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--mcmc_gradient', type=str, choices=['sgld', 'sghmc'], default='sghmc')
    parser.add_argument('--n_samples', type=int, default=10)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--save_path', type=str)
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--dataset', type=str, choices=['mnist', 'uci'], default='mnist')
    parser.add_argument('--input_dim', type=int, default=784)
    parser.add_argument('--not_normalize', action='store_true')
    parser.add_argument('--burn_lr', type=float, default=1e-5)

    args = parser.parse_args()

    return args


def main(args):
    if args.seed is not None:
        random_seed(args.seed)
    else:
        args.seed = -1

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
 
    if args.dataset == 'mnist':
        Path(args.data_dir).mkdir(exist_ok=True, parents=True)
        trainloader, valloader = load_mnist_dataset(Path(args.data_dir), 
                args.batch_size, classes=args.classes, normalize=not args.not_normalize)
    elif args.dataset == 'uci':
        trainloader, valloader = load_uci_dataset(Path(args.data_dir), 
                args.batch_size, normalize=not args.not_normalize)
   
    def nll_func(y_hat, y):
        nll = F.cross_entropy(y_hat, y, reduction='sum')
        return nll

    def err_func(y_hat, y):
        err = y_hat.argmax(-1).ne(y)
        return err

    mcmc_class = mcmc_grdadients[args.mcmc_gradient]
    weights_grads_priors = []

    def init_model(state_dict=None):
        if args.n_hidden_layers == 0:
            model = LogRegression(args.input_dim)
        else:
            model = MLP(input_dim=args.input_dim, width=args.hidden_dim, depth=args.n_hidden_layers, output_dim=len(args.classes))
        if state_dict is not None:
            model.load_state_dict(state_dict)
        return model

    initial_state_dict = init_model().state_dict()

    for _ in tqdm(range(args.n_samples)):
        model = init_model(initial_state_dict)
        
        optimizer = mcmc_class(model.parameters(), lr=args.burn_lr, alpha0=args.alpha0, beta0=args.beta0)
        scheduler = BurnInScheduler(optimizer, args.burn_in_epochs, args.burn_lr, args.bnn_lr)

        trainer = BNNTrainer(model, 
            optimizer, 
            nll_func, 
            err_func, 
            trainloader, 
            valloader,
            scheduler=scheduler, 
            device=device, 
            resample_prior_every=args.resample_prior_every,
            resample_momentum_every=args.resample_momentum_every,
            save_freq=args.save_freq,
            batch_size=args.batch_size,
            report_every=args.report_every
            )
        trainer.train(n_epoch=args.n_epoch, burn_in_epochs=args.burn_in_epochs, resample_prior_until=args.resample_prior_until)
        weights_sample = trainer.weight_set_sample
        potential_grad_sample = trainer.potential_grad_sample

        opt_with_priors = trainer.optimizer
        priors = {}
        group_params = opt_with_priors.param_groups[0]['params']
        for (n, _), p in zip(weights_sample[0].items(), group_params):  
            state = opt_with_priors.state[p]
            priors[n] = state['weight_decay']
        print(priors)

        weights_grads_priors.append((weights_sample, potential_grad_sample, priors))

        pickle.dump(weights_grads_priors, Path(args.save_path).open('wb'))


if __name__ == '__main__':
    args = parse_arguments()
    main(args)
