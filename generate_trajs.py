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
import json

from bayesian_inference.neural_networks import define_nn, MLP, LogRegression, classif_err_fn, classif_loss_fn, GammaGaussPrior
from bayesian_inference.potentials import ClassificationPotential
from bayesian_inference.sg_mcmc_methods import  SGLD, SVRG_LD, ScaleAdaSGHMC as H_SA_SGHMC
from bayesian_inference.trajectory_sampler import SG_MCMC_Inference, BurnInScheduler

from mnist_utils import load_mnist_dataset
from UCI_utils import load_uci_dataset
from utils import random_seed


sg_mcmc_method = {'sgld': SGLD, 'svrg-ld': SVRG_LD, 'sghmc': H_SA_SGHMC}


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_trajs', type=int, default=105)
    parser.add_argument('--n_burn', type=int, default=1e3)
    parser.add_argument('--n_sample', type=int, default=1e4)
    
    parser.add_argument('--alpha', type=float, default=1)
    parser.add_argument('--beta', type=float, default=1)
    parser.add_argument('--resample_prior_every', type=int, default=100)
    parser.add_argument('--resample_prior_until', type=int, default=None)
    
    parser.add_argument('--lr', type=float, default=1e-7)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--resample_momentum_every', type=int, default=100)
    parser.add_argument('--sg_mcmc_method', type=str, choices=[
      'sgld', 'svrg-ld', 'sghmc'
      ], default='sghmc')
    parser.add_argument('--save_determ_grad', action='store_true')
    parser.add_argument('--epoch_length', type=int, default=None)

    parser.add_argument('--save_freq', type=int, default=1)
    parser.add_argument('--report_every', type=int, default=1000)

    parser.add_argument('--model_config_path', type=str, required=True)
    parser.add_argument('--save_path', type=str, required=True)
    parser.add_argument('--data_dir', type=str)
    
    parser.add_argument('--dataset', type=str, choices=['mnist', 'uci'], default='mnist')
    parser.add_argument('--classes', type=int, nargs='+', default=[3,5])
    parser.add_argument('--not_normalize', action='store_true')

    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--seed', type=int, default=None)

    args = parser.parse_args()

    return args


def generate(args, batchsampler, valloader, config, device):
    potential = ClassificationPotential(batchsampler, device)
    prior = GammaGaussPrior(alpha=args.alpha, beta=args.beta)
    
    x, _ = valloader.dataset[0]
    x_shape = x.shape
    
    bayesian_nn, config = define_nn(config, x_shape, prior)
    bayesian_nn.sample_prior()
    bayesian_nn.init_state_dict = bayesian_nn.state_dict()
    bayesian_nn.init_prior_dict = bayesian_nn.prior_dict()

    mcmc_class = sg_mcmc_method[args.sg_mcmc_method]
    sg_mcmc = mcmc_class(bayesian_nn.parameters(), lr=args.lr)

    sampler = SG_MCMC_Inference(
            bayesian_nn,
            potential, 
            sg_mcmc, 
            classif_loss_fn,
            valloader,
            err_fn=classif_err_fn, 
            device=device, 
            save_freq=args.save_freq,
            resample_prior_every=args.resample_prior_every,
            resample_momentum_every=args.resample_momentum_every,
            epoch_length=args.epoch_length,
            report_every=args.report_every,
            save_stoch_grad=not args.save_determ_grad,
            )

    for trajs, traj_grads, priors in sampler.sample_trajs(
                args.n_trajs, 
                args.n_burn, 
                args.n_sample, 
                resample_prior_until=args.resample_prior_until):
        yield trajs, traj_grads, priors


if __name__ == '__main__':
    args = parse_arguments()
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
        
    randsampler = torch.utils.data.RandomSampler(trainloader.dataset, replacement=False)
    batchsampler = torch.utils.data.DataLoader(trainloader.dataset, batch_size=args.batch_size, sampler=randsampler)
    config = json.load(Path(args.model_config_path).open('r'))

    Path(args.save_path).parent.mkdir(exist_ok=True, parents=True)
    json.dump(config, Path(args.save_path + '_config.json').open('w'))
    for  trajs, traj_grads, priors in generate(args, batchsampler, valloader, config, device):
        pickle.dump((trajs, traj_grads, priors), Path(args.save_path + '_traj.pkl').open('wb'))





    # trash
    #weights_grads_priors = []

    # def init_model(state_dict=None):
    #     if args.n_hidden_layers == 0:
    #         model = LogRegression(args.input_dim)
    #     else:
    #         model = MLP(input_dim=args.input_dim, width=args.hidden_dim, depth=args.n_hidden_layers, output_dim=len(args.classes))
    #     if state_dict is not None:
    #         model.load_state_dict(state_dict)
    #     return model

    # initial_state_dict = init_model().state_dict()

    # for _ in tqdm(range(args.n_samples)):
    #     model = init_model(initial_state_dict)
        
    #     if args.burn_lr is None:
    #         args.burn_lr = args.bnn_lr
    #     optimizer = mcmc_class(model.parameters(), lr=args.burn_lr, alpha0=args.alpha0, beta0=args.beta0, 
    #                             sample_prior=not args.not_sample_prior)
    #     scheduler = BurnInScheduler(optimizer, args.burn_in_epochs, args.burn_lr, args.bnn_lr)

    #     if args.not_sample_prior is True:
    #         args.resample_prior_every = float('inf')
    #         args.resample_prior_until = 0

        
    #     trainer.train(n_epoch=args.n_epoch, burn_in_epochs=args.burn_in_epochs, resample_prior_until=args.resample_prior_until)
    #     weights_sample = trainer.weight_set_sample
    #     potential_grad_sample = trainer.potential_grad_sample

    #     opt_with_priors = trainer.optimizer
    #     priors = {}
    #     group_params = opt_with_priors.param_groups[0]['params']
    #     for (n, _), p in zip(weights_sample[0].items(), group_params):  
    #         state = opt_with_priors.state[p]
    #         priors[n] = state['weight_decay']
    #     print(priors)

    #     weights_grads_priors.append((weights_sample, potential_grad_sample, priors))

    #     pickle.dump(weights_grads_priors, Path(args.save_path).open('wb'))
