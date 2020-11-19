import torch
import numpy as np
import dill as pickle
from pathlib import Path
import numpy as np
import argparse
import json
from tqdm import tqdm

from bayesian_inference.neural_networks import define_nn, MLP, LogRegression, classif_err_fn, classif_loss_fn, GammaGaussPrior
from bayesian_inference.potentials import ClassificationPotential
from bayesian_inference.sg_mcmc_methods import  SGLD, SVRG_LD, SVRG_HMC, ScaleAdaSGHMC as H_SA_SGHMC
from bayesian_inference.trajectory_sampler import SG_MCMC_Inference, BurnInScheduler

from mnist_utils import load_mnist_dataset
from UCI_utils import load_uci_dataset
from utils import random_seed


sg_mcmc_method = {'sgld': SGLD, 'svrg-ld': SVRG_LD, 'sghmc': H_SA_SGHMC, 'svrg-hmc': SVRG_HMC}


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_trajs', type=int, default=105)
    parser.add_argument('--n_burn', type=int, default=1e3)
    parser.add_argument('--n_sample', type=int, default=1e4)
    
    parser.add_argument('--alpha', type=float, default=1)
    parser.add_argument('--beta', type=float, default=1)
    parser.add_argument('--resample_prior_every', type=int, default=100)
    parser.add_argument('--resample_prior_until', type=int, default=None)
    
    parser.add_argument('--lr', type=float, default=1e-1)
    parser.add_argument('--sample_lr', type=float, default=None)
    parser.add_argument('--burn_batch_size', type=int, default=128)
    parser.add_argument('--sample_batch_size', type=int, default=128)
    parser.add_argument('--resample_momentum_every', type=int, default=100)
    parser.add_argument('--sg_mcmc_method', type=str, choices=sg_mcmc_method.keys(), default='sghmc')
    parser.add_argument('--save_determ_grad', action='store_true')
    parser.add_argument('--epoch_length', type=int, default=None)

    parser.add_argument('--save_freq', type=int, default=1)
    parser.add_argument('--report_every', type=int, default=1000)

    parser.add_argument('--model_config_path', type=str, required=True)
    parser.add_argument('--save_dir', type=str, required=True)
    parser.add_argument('--prefix_name', type=str, default='sgmcmc')
    parser.add_argument('--data_dir', type=str)
    
    parser.add_argument('--dataset', type=str, choices=['mnist', 'uci'], default='mnist')
    parser.add_argument('--classes', type=int, nargs='+', default=[3,5])
    parser.add_argument('--not_normalize', action='store_true')
    parser.add_argument('--standard_scale', action='store_true')

    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    return args


def generate(args, burn_batchsampler, sample_batchsampler, valloader, config, device):
    potential = ClassificationPotential(burn_batchsampler, sample_batchsampler, device)
    prior = GammaGaussPrior(alpha=args.alpha, beta=args.beta)
    
    x, _ = valloader.dataset[0]
    x_shape = x.shape

    bayesian_nn, config = define_nn(config, x_shape, prior)
    if args.resample_prior_until == 0 or args.resample_prior_every >= args.n_burn:
        for p in bayesian_nn.parameters():
            p.sigma2 = 100.
    else:
        bayesian_nn.sample_prior()
    bayesian_nn.init_norm()
    #bayesian_nn.init_state_dict = bayesian_nn.state_dict()
    #bayesian_nn.init_prior_dict = bayesian_nn.prior_dict()

    mcmc_class = sg_mcmc_method[args.sg_mcmc_method]
    sg_mcmc = mcmc_class(bayesian_nn.parameters(), lr=args.lr, sample_lr=args.sample_lr)

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
    random_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
 
    if args.dataset == 'mnist':
        Path(args.data_dir).mkdir(exist_ok=True, parents=True)
        trainloader, valloader = load_mnist_dataset(Path(args.data_dir), 
                args.burn_batch_size, classes=args.classes, normalize=not args.not_normalize, standard_scale=args.standard_scale)
    
    elif args.dataset == 'uci':
        trainloader, valloader = load_uci_dataset(Path(args.data_dir), 
                args.burn_batch_size, normalize=not args.not_normalize, standard_scale=args.standard_scale)
        
    randsampler = torch.utils.data.RandomSampler(trainloader.dataset, replacement=False)
    burn_batchsampler = torch.utils.data.DataLoader(trainloader.dataset, batch_size=args.burn_batch_size, sampler=randsampler)
    sample_batchsampler = torch.utils.data.DataLoader(trainloader.dataset, batch_size=args.sample_batch_size, sampler=randsampler)
    config = json.load(Path(args.model_config_path).open('r'))

    if args.sample_lr is None:
        args.sample_lr = args.lr

    Path(args.save_dir).mkdir(exist_ok=True, parents=True)
    json.dump(config, Path(args.save_dir, args.prefix_name + '_config.json').open('w'))
    #trajs, traj_grads, priors = generate(args, burn_batchsampler, sample_batchsampler, valloader, config, device)
    for  i, (trajs, traj_grads, priors) in tqdm(enumerate(generate(args, burn_batchsampler, sample_batchsampler, valloader, config, device))):
        pass
        # if len(trajs) % 25 == 0 or len(trajs) == 5:
        #     print('DUMP:', len(trajs))
        #     pickle.dump((trajs, traj_grads, priors), Path(args.save_dir, args.prefix_name + '_traj.pkl').open('wb'))
    #if not (len(trajs) % 25 == 0 or len(trajs) == 5):
    pickle.dump((trajs, traj_grads, priors), Path(args.save_dir, args.prefix_name + '_traj.pkl').open('wb'))
