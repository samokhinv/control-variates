from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import torch
from torch.nn import functional as F
from tqdm import tqdm, trange
import pickle
from copy import deepcopy
import argparse
from pathlib import Path
import json
import random
from sklearn.model_selection import train_test_split
import copy
from statsmodels.tsa.stattools import acf, acovf

from utils import load_trajs, random_seed
from mnist_utils import load_mnist_dataset
from UCI_utils import load_uci_dataset
from control_variates.cv_utils import (
        SampleVarianceEstimator, 
        SpectralVarianceEstimator, 
        state_dict_to_vec,
        #compute_naive_variance,
    )
from bayesian_inference.neural_networks import LogRegression, get_binary_prediction
from control_variates.cv import PsyLinear, SteinCV, PsyConstVector, PsyMLP
from control_variates.uncertainty_quantification import ClassificationUncertaintyMCMC
from control_variates.train_test_cv import train_cv, test_cv
from control_variates.spectral_variance import construct_Tukey_Hanning


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
    parser.add_argument('--var_estimator', type=str, nargs='+', choices=['evm', 'esvm'], default=['evm'])
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
    parser.add_argument('--train_trajs_path', type=str)
    parser.add_argument('--test_trajs_path', type=str, required=True)
    parser.add_argument('--cv_path', type=str, nargs='+', default=[])
    parser.add_argument('--cv_dir', type=str)

    parser.add_argument('--dataset', type=str, choices=['mnist', 'uci'], default='mnist')
    parser.add_argument('--not_normalize', action='store_true')
    parser.add_argument('--standard_scale', action='store_true')
    parser.add_argument('--classes', type=int, nargs='+', default=[3, 5])

    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()
    return args
        

def main(test_trajs, test_traj_weights, test_traj_grads, ncvs, x, device, args, 
            train_trajs=None, train_traj_weights=None, train_traj_grads=None):
    if train_trajs is not None:
        train_trajs = (train_trajs, train_traj_weights, train_traj_grads)

    test_trajs = (test_trajs, test_traj_weights, test_traj_grads)
    
    batches = []
    randperm = torch.randperm(x.shape[0])
    for idx in range(0, min(x.shape[0], args.n_batches*args.n_points), args.n_points):
        if args.sample_points:
            x = x[randperm[idx:idx + args.n_points]]
        else:
            x = x[idx:idx + args.n_points]
        batches.append(x)

    function_f = lambda bayesian_nns, x: get_binary_prediction(bayesian_nns, x, classes=[0, 1])
    
    if train_trajs is not None:
        preds = None
        for var_estimator, ncv in zip(args.var_estimator, ncvs):
            if var_estimator == 'evm':
                var_estimator = SampleVarianceEstimator()
            elif var_estimator == 'esvm':
                n = train_trajs[-1].shape[-2]
                bn = int(np.sqrt(n))
                window = construct_Tukey_Hanning(n, bn)
                var_estimator = SpectralVarianceEstimator(window)
            else:
                raise KeyError(var_estimator)

            loss_history, preds = train_cv(train_trajs, ncv, batches, function_f, var_estimator,
                        preds=preds,
                        cv_lr=args.cv_lr, 
                        n_cv_iter=args.n_cv_iter, 
                        centr_reg_coef=args.centr_reg_coef, 
                        predictive_distribution=args.predictive_distribution)

            _, ax = plt.subplots()
            ax.plot(np.arange(len(loss_history)), loss_history)
            plt.savefig(f'sgmcmc_vr_cv_opt.png')

    mean_avg_pred, mean_avg_pred_cv, metrics = test_cv(test_trajs, ncvs, batches, function_f)

    return mean_avg_pred, mean_avg_pred_cv, metrics


if __name__ == "__main__":
    args = parse_arguments()
    args.max_sample_size = int(args.max_sample_size)
    assert len(args.cv_path) == 0 or len(args.cv_path) == len(args.var_estimator)
    random_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    if args.dataset == 'mnist':
        Path.mkdir(Path(args.data_dir), exist_ok=True, parents=True)
        train_dl, valid_dl = load_mnist_dataset(args.data_dir, 
            batch_size=-1, classes=args.classes, normalize=not args.not_normalize, standard_scale=args.standard_scale)
    elif args.dataset == 'uci':
        train_dl, valid_dl = load_uci_dataset(args.data_dir, 
            batch_size=-1, normalize=not args.not_normalize, standard_scale=args.standard_scale)
    N_train = len(train_dl.dataset)
    print(f'N_train: {N_train}')
    x, _ = next(iter(valid_dl))
    x_shape = x[0].shape

    config = json.load(Path(args.model_config_path).open('r'))
    if args.train_trajs_path is not None and len(args.cv_path) == 0:
        train_trajs, train_traj_weights, train_traj_grads, _ = load_trajs(
            args.test_trajs_path,
            config,
            x_shape,
            max_sample_size=args.max_sample_size)
        # train_trajs = train_trajs[:10]
        # train_traj_weights = train_traj_weights[:10]
        # train_traj_grads = train_traj_grads[:10]
    else:
        train_trajs, train_traj_weights, train_traj_grads = None, None, None
    
    test_trajs, test_traj_weights, test_traj_grads, _ = load_trajs(
        args.test_trajs_path,
        config,
        x_shape)
    # test_trajs = test_trajs[30:60]
    # test_traj_weights = test_traj_weights[30:60]
    # test_traj_grads = test_traj_grads[30:60]
    #test_trajs = train_trajs[30:60]
    #test_traj_weights = train_traj_weights[30:60]
    #test_traj_grads = train_traj_grads[30:60]
    #train_trajs = train_trajs[:30]
    #train_traj_weights = train_traj_weights[:30]
    #train_traj_grads = train_traj_grads[:30]
    print(f'N test trajs: {len(test_trajs)}, len of traj: {len(test_trajs[0])}')

    psy_input_dim = state_dict_to_vec(test_trajs[0][0].state_dict()).shape[0]
    ncvs = []
    for cv_id in range(len(args.var_estimator)):
        ncv = get_cv(psy_input_dim, device, args)
        if len(args.cv_path) > 0:
            ncv.psy_model.load_state_dict(torch.load(args.cv_path[cv_id]))
        ncvs.append(ncv)

    mean_avg_pred, mean_avg_pred_cv, metrics = main(test_trajs, test_traj_weights, test_traj_grads, ncvs, x, device, args, 
            train_trajs=train_trajs, train_traj_weights=train_traj_weights, train_traj_grads=train_traj_grads)

    n_traj = len(test_trajs)
    traj_size = len(test_trajs[0])
    n_batches = args.n_batches
    n_pts = args.n_points
    Path(args.figs_dir).mkdir(exist_ok=True)
    fig_path = Path(args.figs_dir, f'{args.prefix_name}_{n_traj}traj_{args.psy_type}_psy_{traj_size}size_{n_batches}batch_{n_pts}pts.png')
    Path(args.metrics_dir).mkdir(exist_ok=True)
    metrics_path = Path(args.metrics_dir, f'{args.prefix_name}_{n_traj}traj_{args.psy_type}_psy_{traj_size}size_{n_batches}batch_{n_pts}pts.json')

    _, _ = plt.subplots()
    plt.grid()
    sns.boxplot(data=[mean_avg_pred]+list(mean_avg_pred_cv))
    plt.xticks(np.arange(1 + len(args.var_estimator)), ['vanilla'] + args.var_estimator)
    
    plt.savefig(fig_path)
    plt.close()
    json.dump(metrics, metrics_path.open('w'))
    for cv_id, ncv in enumerate(ncvs):
        if len(args.cv_path) == 0:
            Path(args.cv_dir).mkdir(exist_ok=True)
            cv_path = Path(args.cv_dir, f'{args.prefix_name}_{args.var_estimator[cv_id]}_{args.psy_type}.pt')
            torch.save(ncv.psy_model.state_dict(), cv_path)


    # plot traj parts
    ind = random.randint(0, len(test_trajs)-1)
    traj, traj_weights, traj_grads = test_trajs[ind], test_traj_weights[ind:ind+1], test_traj_grads[ind:ind+1]
    x = x[:100].float()
    traj_part = np.array([F.softmax(m(x), dim=-1)[:, 1].mean().item() for m in traj])
    cv_vals = [ncv(traj_weights, x, potential_grad=traj_grads) for ncv in ncvs]
    traj_part_cv = [np.array([(F.softmax(m(x), dim=-1)[:, 1] - cv_v[0, i]).mean().item() for i, m in enumerate(traj)]) for cv_v in cv_vals]

    sample_acf_vanilla = acovf(traj_part, adjusted = True,fft = True)#,nlags=100)
    sample_acf_cvs = [acovf(traj_part_cv_, adjusted = True,fft = True) for traj_part_cv_ in traj_part_cv]#,nlags=100)
    
    _, axs = plt.subplots(nrows=1, ncols=3, figsize=(20, 5))
    N1 = len(traj_part) // 2 - 500
    N2 = len(traj_part) // 2
    axs[0].grid()
    axs[0].plot(np.arange(len(traj_part))[N1:N2], traj_part[N1:N2])
    axs[1].grid()
    axs[1].plot(np.arange(len(traj_part_cv[-1]))[N1:N2], traj_part_cv[-1][N1:N2])

    N_cov_1 = 0
    N_cov = 30
    axs[2].grid()
    axs[2].fill_between(N_cov_1+np.arange(N_cov-N_cov_1), sample_acf_vanilla[N_cov_1:N_cov],alpha=0.5,label='ACF Vanilla')
    for var_estimator, sample_acf_cv in zip(args.var_estimator, sample_acf_cvs):
        axs[2].fill_between(N_cov_1+np.arange(N_cov-N_cov_1), sample_acf_cv[N_cov_1:N_cov],alpha=0.5,label=f'ACF {var_estimator.upper()}')
    #axs[2].set_ylim(0,5e-6)
    axs[2].ticklabel_format(axis='y',style='sci',scilimits=(-2,2))
    axs[2].tick_params(axis='y',labelsize = 16)
    axs[2].tick_params(axis='x',labelsize=16)
    plt.legend()
    plt.savefig(Path(args.figs_dir, f'{args.prefix_name}_{args.psy_type}_traj_aconv.png'))
    plt.close()
