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

from utils import load_samples
from mnist_utils import load_mnist_dataset
from UCI_utils import load_uci_dataset
import control_variates
from control_variates.cv_utils import SampleVarianceEstimator, SpectralVarianceEstimator
from control_variates.model import LogRegression
from control_variates.cv import PsyLinear, SteinCV, PsyConstVector, PsyMLP
from control_variates.cv_utils import state_dict_to_vec, compute_naive_variance, compute_potential_grad
from control_variates.model import get_binary_prediction
from control_variates.uncertainty_quantification import ClassificationUncertaintyMCMC
from control_variates.cv_utils import compute_log_likelihood, compute_concat_gradient


def random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def get_cv(psy_input_dim, priors, potential_grads, args, 
            train_x=None, train_y=None, N_train=None):
    ncv_s = []
    for prior, potential_grad in zip(priors, potential_grads):
        if args.psy_type == 'const':
            psy_model = PsyConstVector(input_dim=psy_input_dim)
            psy_model.init_zero()
        elif args.psy_type == 'linear':
            psy_model = PsyLinear(input_dim=psy_input_dim)
        elif args.psy_type == 'mlp':
            psy_model = PsyMLP(input_dim=psy_input_dim, width=args.width, depth=args.depth)
        psy_model.init_zero()
        psy_model.to(args.device)

        neural_control_variate = SteinCV(psy_model, prior, 
                N_train=N_train, train_x=train_x, train_y=train_y, potential_grad=potential_grad)
        ncv_s.append(neural_control_variate)
    return ncv_s


def train_cv(trajectories, ncv_s, batches, args):
    def centr_regularizer(ncv, models, x, potential_grad=None):
        return (ncv(models, x, potential_grad).mean(0))**2

    n_traj = len(trajectories)
    traj_size = len(trajectories[0])
    n_batches = len(batches)
    n_pts = len(batches[0])

    Path(args.figs_dir).mkdir(exist_ok=True)
    fig_path = Path(args.figs_dir, f'{args.prefix_name}_{n_traj}traj_{args.psy_type}_psy_{traj_size}size_{n_batches}batch_{n_pts}pts.png')
    Path(args.metrics_dir).mkdir(exist_ok=True)
    metrics_path = Path(args.metrics_dir, f'{args.prefix_name}_{n_traj}traj_{args.psy_type}_psy_{traj_size}size_{n_batches}batch_{n_pts}pts.json')
    
    mean_avg_pred = np.zeros(n_traj)
    mean_avg_pred_cv = np.zeros(n_traj)

    metrics = {'sample_var': 0., 'sample_var_cv': 0, 'sample_var_reduction': 0., 
    'spectral_var': 0., 'spectral_var_cv': 0., 'spectral_var_reduction': 0.}

    for x_id, x in tqdm(enumerate(batches)):
        avg_predictions_cv = []
        avg_predictions_no_cv = []
        for tr_id, (models, ncv) in enumerate(zip(trajectories, ncv_s)):
            psy_model = ncv.psy_model
            ncv_optimizer = torch.optim.Adam(psy_model.parameters(), lr=args.cv_lr, weight_decay=0.0)
            uncertainty_quant = ClassificationUncertaintyMCMC(models, ncv)

            function_f = lambda model, x: get_binary_prediction(model, x, classes=[0, 1])

            sample_var_estimator = SampleVarianceEstimator(function_f, models)
            spectral_var_estimator = SpectralVarianceEstimator(function_f, models)

            if args.var_estimator == 'sample':
                var_estimator = sample_var_estimator
            else:
                var_estimator = spectral_var_estimator

            history = []
            for _ in range(args.n_cv_iter):
                ncv_optimizer.zero_grad()

                pred = function_f(models, x)
                cv_vals = ncv(models, x)
                if args.predictive_distribution is True:
                    pred = pred.mean(-1)
                    cv_vals = cv_vals.mean(-1)

                var_cv = var_estimator.estimate_variance(x, use_cv=True, all_values=(pred - cv_vals))

                history.append(var_cv.mean().item())
                loss = var_cv.mean()
                if args.centr_reg_coef != 0:
                    loss += args.centr_reg_coef * centr_regularizer(ncv, models, x).mean()
                loss.backward()
                ncv_optimizer.step()
            pred = function_f(models, x)
            cv_vals = ncv(models, x)
            if args.predictive_distribution is True:
                pred = pred.mean(-1)
                cv_vals = cv_vals.mean(-1)

            sample_var_cv = sample_var_estimator.estimate_variance(x, use_cv=True, all_values=(pred - cv_vals))
            sample_var = sample_var_estimator.estimate_variance(x, use_cv=False, all_values=pred)
            
            spectral_var_cv = spectral_var_estimator.estimate_variance(x, use_cv=True, all_values=(pred - cv_vals))
            spectral_var = spectral_var_estimator.estimate_variance(x, use_cv=True, all_values=pred)

            metrics['sample_var'] += sample_var.mean().item() / (n_batches * n_traj)
            metrics['sample_var_cv'] += sample_var_cv.mean().item() / (n_batches * n_traj)
            metrics['sample_var_reduction'] += (sample_var / sample_var_cv).mean().item() / (n_batches * n_traj)
            metrics['spectral_var'] += spectral_var.mean().item() / (n_batches * n_traj)
            metrics['spectral_var_cv'] += spectral_var_cv.mean().item() / (n_batches * n_traj)
            metrics['spectral_var_reduction'] += (spectral_var / spectral_var_cv).mean().item() / (n_batches * n_traj)

            avg_predictions_cv.append(uncertainty_quant.estimate_emperical_mean(x, use_cv=True).mean().item())
            avg_predictions_no_cv.append(uncertainty_quant.estimate_emperical_mean(x, use_cv=False).mean().item())

            if x_id == 0 and tr_id < 10:
                _, ax = plt.subplots()
                ax.plot(np.arange(len(history)), history)
                plt.savefig(f'{tr_id}.png')
        mean_avg_pred += np.array(avg_predictions_no_cv) / n_batches
        mean_avg_pred_cv += np.array(avg_predictions_cv) / n_batches
    _, ax = plt.subplots()
    sns.boxplot(data=[mean_avg_pred, mean_avg_pred_cv])
    plt.xticks([0,1], ['without CV', 'with CV'])
    plt.savefig(fig_path)

    with Path(metrics_path).open('w') as f:
        json.dump(metrics, f)


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

    args = parser.parse_args()
    return args
        

def main(args):
    random_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    args.device = device

    if args.dataset == 'mnist':
        Path.mkdir(Path(args.data_dir), exist_ok=True, parents=True)
        train_dl, valid_dl = load_mnist_dataset(args.data_dir, args.batch_size, classes=[3, 5], normalize=not args.not_normalize)
    elif args.dataset == 'uci':
        train_dl, valid_dl = load_uci_dataset(args.data_dir, batch_size=args.batch_size, normalize=not args.not_normalize)
    N_train = len(train_dl.dataset)
    print(f'Train dataset size: {N_train}')

    x, _ = train_dl.dataset[0]
    args.input_dim = 1
    for d in x.shape:
        args.input_dim *= d

    x_new, y_new = next(iter(valid_dl))
    train_x, train_y = next(iter(train_dl))
    
    with Path(args.samples_path).open('rb') as fp:
        samples = pickle.load(fp)

    print(f'N samples: {len(samples)}, volume of sample: {len(samples[0][0])}')

    trajectories, priors, potential_grads = load_samples(
        args.samples_path,
        model_class=LogRegression, 
        model_kwargs={'input_size': args.input_dim})
    
    if potential_grads is None:
        potential_grads = [compute_potential_grad(ms, train_x, train_y, N_train, priors=ps) for ms, ps in zip(trajectories, priors)]
    
    if args.keep_n_last is not None:
        args.keep_n_last = min(args.keep_n_last, len(trajectories[0]))
        args.max_sample_size = min(args.max_sample_size, args.keep_n_last)
        every = (args.keep_n_last) // args.max_sample_size
        trajectories = [x[-args.keep_n_last:][::every][-args.max_sample_size:] for x in trajectories]
        potential_grads = [x[-args.keep_n_last:][::every][-args.max_sample_size:] for x in potential_grads]
    else:
        every = (len(trajectories[0]) - args.cut_n_first) // args.max_sample_size
        trajectories = [x[args.cut_n_first:][::every][-args.max_sample_size:] for x in trajectories]
        potential_grads = [x[args.cut_n_first:][::every][-args.max_sample_size:] for x in potential_grads]

    #class_ = 1 if (y_new == 1).sum() > y_new.shape[0] / 2 else 0
    x1 = x_new #(x_new[y_new == class_])
    batches = []
    randperm = torch.randperm(x1.shape[0])
    for idx in range(0, min(x1.shape[0], args.n_batches*args.n_points), args.n_points):
        if args.sample_points:
            x = x1[randperm[idx:idx + args.n_points]]
        else:
            x = x1[idx:idx + args.n_points]
        batches.append(x)

    psy_input_dim = state_dict_to_vec(trajectories[0][0].state_dict()).shape[0]
    ncv_s = get_cv(psy_input_dim, priors, potential_grads, args)
    train_cv(trajectories, ncv_s, batches, args)

    psy_weights = [deepcopy(ncv.psy_model.state_dict()) for ncv in ncv_s]
    with Path(args.save_path).open('wb') as fp:
        pickle.dump(psy_weights, fp)


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
