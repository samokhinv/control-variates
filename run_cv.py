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
    #    compute_log_likelihood, 
    #    compute_concat_gradient,
        state_dict_to_vec,
        compute_naive_variance,
    #    compute_potential_grad,
    )
from bayesian_inference.neural_networks import LogRegression, get_binary_prediction
from control_variates.cv import PsyLinear, SteinCV, PsyConstVector, PsyMLP
from control_variates.uncertainty_quantification import ClassificationUncertaintyMCMC


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


def test_cv(trajs, ncv, batches, function_f, args):
    n_traj = len(trajs)
    traj_size = len(trajs[0][0])
    n_batches = len(batches)
    n_pts = len(batches[0])

    Path(args.figs_dir).mkdir(exist_ok=True)
    fig_path = Path(args.figs_dir, f'{args.prefix_name}_{n_traj}traj_{args.psy_type}_psy_{traj_size}size_{n_batches}batch_{n_pts}pts.png')
    Path(args.metrics_dir).mkdir(exist_ok=True)
    #metrics_path = Path(args.metrics_dir, f'{args.prefix_name}_{n_traj}traj_{args.psy_type}_psy_{traj_size}size_{n_batches}batch_{n_pts}pts.json')
    
    mean_avg_pred = np.zeros(n_traj)
    mean_avg_pred_cv = np.zeros(n_traj)

    # metrics = {'sample_var': 0., 'sample_var_cv': 0, 'sample_var_reduction': 0., 
    # 'spectral_var': 0., 'spectral_var_cv': 0., 'spectral_var_reduction': 0.}

    for x in tqdm(batches):
        avg_predictions_cv = []
        avg_predictions_no_cv = []

        for (bayesian_nns, potential_grads) in trajs:
            #uncertainty_quant = ClassificationUncertaintyMCMC(models, ncv)
            #sample_var_estimator = SampleVarianceEstimator(function_f, bayesian_nns)
            #spectral_var_estimator = SpectralVarianceEstimator(function_f, models)

            pred = function_f(bayesian_nns, x)
            cv_vals = ncv(bayesian_nns, x, potential_grad=potential_grads)
            # if args.predictive_distribution is True:
            #     pred = pred.mean(-1)
            #     cv_vals = cv_vals.mean(-1)

            avg_predictions_cv.append((pred-cv_vals).mean().item())
            avg_predictions_no_cv.append(pred.mean().item())

            #sample_var_cv = sample_var_estimator.estimate_variance(x, use_cv=True, all_values=(pred - cv_vals))
            #sample_var = sample_var_estimator.estimate_variance(x, use_cv=False, all_values=pred)
            
            #spectral_var_cv = spectral_var_estimator.estimate_variance(x, use_cv=True, all_values=(pred - cv_vals))
            #spectral_var = spectral_var_estimator.estimate_variance(x, use_cv=True, all_values=pred)

            # metrics['sample_var'] += sample_var.mean().item() / (n_batches * n_traj)
            # metrics['sample_var_cv'] += sample_var_cv.mean().item() / (n_batches * n_traj)
            # metrics['sample_var_reduction'] += (sample_var / sample_var_cv).mean().item() / (n_batches * n_traj)
            # metrics['spectral_var'] += spectral_var.mean().item() / (n_batches * n_traj)
            # metrics['spectral_var_cv'] += spectral_var_cv.mean().item() / (n_batches * n_traj)
            # metrics['spectral_var_reduction'] += (spectral_var / spectral_var_cv).mean().item() / (n_batches * n_traj)

            # avg_predictions_cv.append(
            #   uncertainty_quant.estimate_emperical_mean(x=x, predictions=pred, cv_values=cv_vals, use_cv=True).mean().item())
            # avg_predictions_no_cv.append(
            #   uncertainty_quant.estimate_emperical_mean(x=x, predictions=pred, use_cv=False).mean().item())

        mean_avg_pred += np.array(avg_predictions_no_cv) / n_batches
        mean_avg_pred_cv += np.array(avg_predictions_cv) / n_batches
    _, _ = plt.subplots()
    sns.boxplot(data=[mean_avg_pred, mean_avg_pred_cv])
    plt.xticks([0,1], ['without CV', 'with CV'])
    plt.savefig(fig_path)

    # with Path(metrics_path).open('w') as f:
    #     json.dump(metrics, f)


def train_cv(trajs, ncv, batches, function_f, args):
    def centr_regularizer(ncv, bayesian_nns, x, potential_grads=None):
        return (ncv(bayesian_nns, x, potential_grads).mean(0))**2

    for x_id, x in tqdm(enumerate(batches)):
        ncv_optimizer = torch.optim.Adam(ncv.psy_model.parameters(), lr=args.cv_lr, weight_decay=0.0)

        loss_history = []
        preds = []
        for it in range(args.n_cv_iter):
            loss = 0
            ncv_optimizer.zero_grad()
            for traj_id, (bayesian_nns, potential_grads) in enumerate(trajs):
                if it == 0:
                    preds.append(function_f(bayesian_nns, x))

                pred = preds[traj_id]

                if args.var_estimator == 'sample':
                    var_estimator = SampleVarianceEstimator(function_f, bayesian_nns)
                else:
                    var_estimator = SpectralVarianceEstimator(function_f, bayesian_nns)

                cv_vals = ncv(bayesian_nns, x, potential_grad=potential_grads)
                if args.predictive_distribution is True:
                    pred = pred.mean(-1)
                    cv_vals = cv_vals.mean(-1)

                var_cv = var_estimator.estimate_variance(x, use_cv=True, all_values=(pred - cv_vals))
                loss += var_cv.mean()
                if args.centr_reg_coef != 0:
                    loss += args.centr_reg_coef * centr_regularizer(ncv, bayesian_nns, x).mean()

            loss_history.append(loss.mean().item())
            loss.backward()
            ncv_optimizer.step()

        if x_id == 0:
            _, ax = plt.subplots()
            ax.plot(np.arange(len(loss_history)), loss_history)
            plt.savefig(f'convergence.png')


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cv_lr', default=1e-6, type=float)
    parser.add_argument('--n_cv_iter', default=100, type=int)
    parser.add_argument('--batch_size', default=-1, type=int)
    parser.add_argument('--width', type=int, default=100)
    parser.add_argument('--depth', type=int, default=1)
    parser.add_argument('--output_dim', type=int, default=2)
    parser.add_argument('--samples_path', type=str, required=True)
    parser.add_argument('--psy_type', type=str, choices=['const', 'mlp', 'linear'], default='const')
    parser.add_argument('--classes', type=int, nargs='+', default=[3,5])
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--classes', type=int, nargs='+', default=[3, 5])
    parser.add_argument('--save_path', type=str, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--data_dir', type=str, default='..data/mnist')
    parser.add_argument('--dataset', type=str, choices=['mnist', 'uci'], default='mnist')
    parser.add_argument('--centr_reg_coef', type=float, default=0)
    parser.add_argument('--figs_dir', type=str)
    parser.add_argument('--metrics_dir', type=str)
    parser.add_argument('--prefix_name', type=str)
    parser.add_argument('--max_sample_size', type=int, default=100)
    parser.add_argument('--n_points', type=int, default=10)
    parser.add_argument('--not_normalize', action='store_true')
    parser.add_argument('--cut_n_first', type=int, default=0)
    parser.add_argument('--sample_points', action='store_true')
    parser.add_argument('--n_batches', type=int, default=1)
    parser.add_argument('--var_estimator', type=str, choices=['sample', 'spectral'], default='sample')
    parser.add_argument('--predictive_distribution', action='store_true')
    parser.add_argument('--keep_n_last', type=int)
    parser.add_argument('--n_train_traj', type=int)
    parser.add_argument('--n_test_traj', type=int)

    args = parser.parse_args()
    return args
        

def main(trajs, traj_grads, x, device, args):
    #if potential_grads is None:
    #    potential_grads = [compute_potential_grad(ms, train_x, train_y, N_train, priors=ps) for ms, ps in zip(trajectories, priors)]
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
    train_trajs = trajs[:args.n_train_traj], traj_grads[:args.n_train_traj]
    test_trajs = trajs[args.n_train_traj:], traj_grads[args.n_train_traj:]
    
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
    
    train_cv(train_trajs, ncv, batches, function_f, args)
    test_cv(test_trajs, ncv, batches, function_f, args)
    # psy_weight = deepcopy(ncv.psy_model.state_dict())
    # with Path(args.save_path).open('wb') as fp:
    #     pickle.dump(psy_weight, fp)


if __name__ == "__main__":
    args = parse_arguments()
    random_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    if args.dataset == 'mnist':
        Path.mkdir(Path(args.data_dir), exist_ok=True, parents=True)
        train_dl, valid_dl = load_mnist_dataset(args.data_dir, args.batch_size, classes=args.classes, normalize=not args.not_normalize)
    elif args.dataset == 'uci':
        train_dl, valid_dl = load_uci_dataset(args.data_dir, batch_size=args.batch_size, normalize=not args.not_normalize)
    N_train = len(train_dl.dataset)
    print(f'N_train: {N_train}')

    x, _ = train_dl.dataset[0]
    args.input_dim = 1
    for d in x.shape:
        args.input_dim *= d

    x, _ = next(iter(valid_dl))
    
    trajs, traj_grads, _ = load_trajs(
        args.trajs_path,
        bayesian_nn_class=LogRegression, 
        canvas=(args.input_dim,))
    
    print(f'N trajs: {len(trajs)}, len of traj: {len(trajs[0])}')

    main(trajs, traj_grads, x, device, args)