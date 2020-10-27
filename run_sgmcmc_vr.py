from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import torch
from tqdm import tqdm, trange
import pickle
from copy import deepcopy
import argparse
from pathlib import Path
import json
import random
from sklearn.model_selection import train_test_split

from utils import load_trajs, random_seed
from mnist_utils import load_mnist_dataset
from UCI_utils import load_uci_dataset
from control_variates.cv_utils import (
        SampleVarianceEstimator, 
        SpectralVarianceEstimator, 
        state_dict_to_vec,
        compute_naive_variance,
    )
from bayesian_inference.neural_networks import LogRegression, get_binary_prediction
from control_variates.cv import PsyLinear, SteinCV, PsyConstVector, PsyMLP
from control_variates.uncertainty_quantification import ClassificationUncertaintyMCMC
from control_variates.train_test_cv import train_cv, test_cv


def get_cv(psy_input_dim, device, args, potential_grad=None, 
            train_x=None, train_y=None, N_train=None):
    if args.psy_type == 'const':
        psy_model = PsyConstVector(input_dim=psy_input_dim)
        psy_model.init_zero()
    elif args.psy_type == 'linear':
        psy_model = PsyLinear(input_dim=psy_input_dim)
    elif args.psy_type == 'mlp':
        psy_model = PsyMLP(input_dim=psy_input_dim, width=args.width, depth=args.depth)
    psy_model.init_zero()
    psy_model.to(device)

    ncv = SteinCV(psy_model, priors=None, N_train=N_train, 
                train_x=train_x, train_y=train_y, potential_grad=potential_grad)
    return ncv


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--cv_lr', default=1e-6, type=float)
    parser.add_argument('--n_cv_iter', default=100, type=int)
    parser.add_argument('--psy_type', type=str, choices=['const', 'mlp', 'linear'], default='const')
    parser.add_argument('--var_estimator', type=str, choices=['sample', 'spectral'], default='sample')
    parser.add_argument('--centr_reg_coef', type=float, default=0)

    parser.add_argument('--n_train_traj', type=int)
    parser.add_argument('--n_test_traj', type=int)
    
    parser.add_argument('--max_sample_size', type=float, default=1000)
    parser.add_argument('--keep_n_last', type=int)
    parser.add_argument('--cut_n_first', type=int, default=0)
    parser.add_argument('--n_points', type=int, default=100)
    parser.add_argument('--sample_points', action='store_true')
    parser.add_argument('--n_batches', type=int, default=1)
    parser.add_argument('--predictive_distribution', action='store_true')

    parser.add_argument('--data_dir', type=str, default='..data/mnist')
    parser.add_argument('--figs_dir', type=str)
    parser.add_argument('--metrics_dir', type=str)
    parser.add_argument('--prefix_name', type=str)
    parser.add_argument('--model_config_path', type=str, required=True)
    parser.add_argument('--trajs_path', type=str, required=True)

    parser.add_argument('--dataset', type=str, choices=['mnist', 'uci'], default='mnist')
    parser.add_argument('--not_normalize', action='store_true')
    parser.add_argument('--classes', type=int, nargs='+', default=[3, 5])

    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()
    return args
        

def main(trajs, traj_grads, x, device, args):
    if args.keep_n_last is not None:
        args.keep_n_last = min(args.keep_n_last, len(trajs[0]))
        args.max_sample_size = min(args.max_sample_size, args.keep_n_last)
        every = (args.keep_n_last) // args.max_sample_size
        trajs = [x[-args.keep_n_last:][::every][-args.max_sample_size:] for x in trajs]
        traj_grads = [x[-args.keep_n_last:][::every][-args.max_sample_size:] for x in traj_grads]
    else:
        every = (len(trajs[0]) - args.cut_n_first) // args.max_sample_size
        trajs = [x[args.cut_n_first:][::every][-args.max_sample_size:] for x in trajs]
        traj_grads = [x[args.cut_n_first:][::every][-args.max_sample_size:] for x in traj_grads]

    #nns_and_grads = list(zip(trajs, traj_grads))
    #train_nns_and_grads , test_nns_and_grads , _, _ = train_test_split(nns_and_grads, [None]*len(trajs), test_size=len(trajs)-args.n_train_traj)
    train_trajs = list(zip(trajs[:args.n_train_traj], traj_grads[:args.n_train_traj]))
    test_trajs = list(zip(trajs[args.n_train_traj:], traj_grads[args.n_train_traj:]))
    
    batches = []
    randperm = torch.randperm(x.shape[0])
    for idx in range(0, min(x.shape[0], args.n_batches*args.n_points), args.n_points):
        if args.sample_points:
            x = x[randperm[idx:idx + args.n_points]]
        else:
            x = x[idx:idx + args.n_points]
        batches.append(x)

    psy_input_dim = state_dict_to_vec(trajs[0][0].state_dict()).shape[0]
    ncv = get_cv(psy_input_dim, device, args)
    function_f = lambda bayesian_nns, x: get_binary_prediction(bayesian_nns, x, classes=[0, 1])

    if args.var_estimator == 'sample':
        var_estimator = SampleVarianceEstimator(function_f, None)
    else:
        var_estimator = SpectralVarianceEstimator(function_f, None)
    
    loss_history = train_cv(train_trajs, ncv, batches, function_f, var_estimator,
                cv_lr=args.cv_lr, 
                n_cv_iter=args.n_cv_iter, 
                centr_reg_coef=args.centr_reg_coef, 
                predictive_distribution=args.predictive_distribution)

    _, ax = plt.subplots()
    ax.plot(np.arange(len(loss_history)), loss_history)
    plt.savefig(f'sgmcmc_vr_cv_opt.png')

    n_traj = len(test_trajs)
    traj_size = len(test_trajs[0][0])
    n_batches = len(batches)
    n_pts = len(batches[0])
    Path(args.figs_dir).mkdir(exist_ok=True)
    fig_path = Path(args.figs_dir, f'{args.prefix_name}_{n_traj}traj_{args.psy_type}_psy_{traj_size}size_{n_batches}batch_{n_pts}pts.png')
    Path(args.metrics_dir).mkdir(exist_ok=True)
    metrics_path = Path(args.metrics_dir, f'{args.prefix_name}_{n_traj}traj_{args.psy_type}_psy_{traj_size}size_{n_batches}batch_{n_pts}pts.json')

    mean_avg_pred, mean_avg_pred_cv, metrics = test_cv(test_trajs, ncv, batches, function_f)

    _, _ = plt.subplots()
    sns.boxplot(data=[mean_avg_pred, mean_avg_pred_cv])
    plt.xticks([0,1], ['without CV', 'with CV'])
    plt.savefig(fig_path)
    json.dump(metrics, metrics_path.open('w'))


if __name__ == "__main__":
    args = parse_arguments()
    random_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    if args.dataset == 'mnist':
        Path.mkdir(Path(args.data_dir), exist_ok=True, parents=True)
        train_dl, valid_dl = load_mnist_dataset(args.data_dir, batch_size=-1, classes=args.classes, normalize=not args.not_normalize)
    elif args.dataset == 'uci':
        train_dl, valid_dl = load_uci_dataset(args.data_dir, batch_size=-1, normalize=not args.not_normalize)
    N_train = len(train_dl.dataset)
    print(f'N_train: {N_train}')

    x, _ = next(iter(valid_dl))
    x_shape = x[0].shape

    config = json.load(Path(args.model_config_path).open('r'))
    
    trajs, traj_grads, _ = load_trajs(
        args.trajs_path,
        config,
        x_shape)
    
    print(f'N trajs: {len(trajs)}, len of traj: {len(trajs[0])}')

    args.max_sample_size = int(args.max_sample_size)
    main(trajs, traj_grads, x, device, args)
