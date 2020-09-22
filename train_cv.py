from matplotlib import pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
import pickle
from copy import deepcopy
import argparse
from pathlib import Path

from utils import load_samples
from mnist_utils import load_mnist_dataset
from UCI_utils import load_uci_dataset
import control_variates
from control_variates.spectral_variance import compute_spectral_variance
from control_variates.model import LogRegression
from control_variates.cv import PsyLinear, SteinCV, PsyConstVector, PsyMLP
from control_variates.cv_utils import state_dict_to_vec, compute_naive_variance, compute_potential_grad
from control_variates.model import get_binary_prediction
from control_variates.uncertainty_quantification import ClassificationUncertaintyMCMC
from control_variates.cv_utils import trapezoidal_kernel, SpectralVariance
from control_variates.cv_utils import compute_log_likelihood, compute_concat_gradient

import random


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


def train_cv(trajectories, ncv_s, x, args):
    def centr_regularizer(ncv, models, x, potential_grad=None):
        return (ncv(models, x, potential_grad).mean(0))**2

    predictions_cv = []
    predictions_no_cv = []
    for tr_id, (models, ncv) in enumerate(zip(trajectories, ncv_s)):
        psy_model = ncv.psy_model
        ncv_optimizer = torch.optim.Adam(psy_model.parameters(), lr=args.cv_lr, weight_decay=0.0)
        uncertainty_quant = ClassificationUncertaintyMCMC(models, ncv)

        function_f = lambda model, x: get_binary_prediction(model, x, classes=[0, 1])
        history = []
        for _ in tqdm(range(args.n_cv_iter)):
            ncv_optimizer.zero_grad()
            mc_variance, no_cv_variance = compute_naive_variance(function_f, ncv, models, x)
            history.append(mc_variance.mean().item())
            loss = mc_variance.mean()
            if args.centr_reg_coef != 0:
                loss += args.centr_reg_coef * centr_regularizer(ncv, models, x).mean()
            loss.backward()
            ncv_optimizer.step()
        print(f'Sample Var with CV: {mc_variance.mean().item()}, w/o CV: {no_cv_variance.mean().item()}')
        print(f'Mean value of CV: {ncv(models, x).mean()}')
        pred = np.array(uncertainty_quant.get_predictions(x).tolist())
        pred_cv = pred - np.array(uncertainty_quant.get_cv_values(x).tolist())
        print(f'Spectral Var with CV: {compute_spectral_variance(pred_cv)}, w/o CV {compute_spectral_variance(pred)}')

        avg_predictions_cv.append(uncertainty_quant.estimate_emperical_mean(x, use_cv=True).mean().item())
        avg_predictions_no_cv.append(uncertainty_quant.estimate_emperical_mean(x, use_cv=False).mean().item())

        fig, ax = plt.subplots()
        ax.plot(np.arange(len(history)), history)
        plt.savefig(f'{tr_id}.png')
    
    fig, ax = plt.subplots()
    ax.boxplot(x=['without CV', 'with CV'], y=[avg_predictions_no_cv, avg_predictions_cv])
    plt.savefig(args.figure_path)


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
    parser.add_argument('--figure_path', type=str)
    parser.add_argument('--max_sample_size', type=int, default=100)
    parser.add_argument('--n_points', type=int, default=10)
    parser.add_argument('--not_normalize', action='store_true')
    parser.add_argument('--cut_n_first', type=int, default=0)

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
    
    every = len(trajectories[0] - args.cut_n_first) // args.max_sample_size
    trajectories = [x[args.cut_n_first:][::every][-args.max_sample_size:] for x in trajectories]
    potential_grads = [x[args.cut_n_first:][::every][-args.max_sample_size:] for x in potential_grads]

    x = (x_new[y_new == 1])[:args.n_points]

    psy_input_dim = state_dict_to_vec(trajectories[0][0].state_dict()).shape[0]
    ncv_s = get_cv(psy_input_dim, priors, potential_grads, args)
    train_cv(trajectories, ncv_s, x, args)

    psy_weights = [deepcopy(ncv.psy_model.state_dict()) for ncv in ncv_s]
    with Path(args.save_path).open('wb') as fp:
        pickle.dump(psy_weights, fp)


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
