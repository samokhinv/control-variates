from matplotlib import pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
import pickle
from copy import deepcopy
import argparse
from pathlib import Path

from mnist_utils import load_mnist_dataset
import control_variates
from control_variates.model import LogRegression
from control_variates.cv import PsyLinear, SteinCV, PsyConstVector, PsyMLP
from control_variates.cv_utils import state_dict_to_vec, compute_naive_variance
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


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cv_lr', default=1e-6, type=float)
    parser.add_argument('--n_cv_iter', default=100, type=int)
    parser.add_argument('--batch_size', default=-1, type=int)
    parser.add_argument('--input_dim', default=784, type=int)
    parser.add_argument('--width', type=int, default=100)
    parser.add_argument('--depth', type=int, default=0)
    parser.add_argument('--output_dim', type=int, default=2)
    parser.add_argument('--samples_path', type=str, required=True)
    parser.add_argument('--cv_type', type=str, choices=['const', 'mlp'], default='const')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_arguemnt('--data_dir', type=str, default='../data/mnist')
    parser.add_arguemnt('--classes', type=int, nargs='+', default=[3, 5])
    parser.add_argument('--save_path', type=str, required=True)

    args = parser.parse_args()
    return args


def main(args):
    random_seed(seed)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    with Path(args.samples_path).open(rb) as fp:
        samples = pickle.load(fp)

    print(f'N samples: {len(samples)}, volume of sample: {len(samples[0])}')

    Path.mkdir(args.data_dir, exist_ok=True, parents=True)
    train_dl, valid_dl = load_mnist_dataset(args.data_dir, args.batch_size, classes=[3, 5])
    N_train = len(train_dl.dataset)

    trajectories = [[LogRegression(args.input_dim)
                 for j in range(len(samples[i][0]))]
                for i in range(len(samples))]

    for i in range(len(samples)):
        for j in range(len(samples[i][0])):
            trajectories[i][j].load_state_dict(samples[i][0][j])

    priors = [samples[i][1] for i in range(len(samples))]

    x_new, y_new = next(iter(valid_dl))
    train_x, train_y = next(iter(train_dl))

    def centr_regularizer(ncv, models, x, ll_div=None):
        return (ncv(models, x, ll_div).mean(0))**2

    def compute_ll_div(models, train_x, train_y, N_train, priors=None):
        for model in models:
            model.zero_grad()
        log_likelihoods = [(compute_log_likelihood(train_x, train_y, model) * N_train).backward() for model in models]
        ll_div = torch.stack([compute_concat_gradient(model, priors) for model in models])
        return ll_div

    x = (x_new[y_new == 1.0])[[43]]

    ncv_s = []
    psy_input_dim = 1570
    for models, pr in zip(trajectories, priors):
        psy_model = PsyConstVector(psy_input_dim)
        psy_model.init_zero()
        psy_model.to(device)

        neural_control_variate = SteinCV(psy_model, train_x, train_y, pr, N_train)
        ncv_optimizer = torch.optim.Adam(psy_model.parameters(), lr=args.cv_lr, weight_decay=0.0) #1e-4)
        uncertainty_quant = ClassificationUncertaintyMCMC(models, neural_control_variate)
        train_x, train_y = next(iter(train_dl))
        ll_div = compute_ll_div(models, train_x, train_y, N_train, priors=priors)
        function_f = lambda model, x: get_binary_prediction(model, x, classes=[0, 1])
        history = []

        for it in tqdm(range(args.n_cv_iter)):
            ncv_optimizer.zero_grad()
            mc_variance, no_cv_variance = compute_naive_variance(function_f, neural_control_variate, models, x, ll_div)
            history.append(mc_variance.mean().item())
            #(mc_variance + centr_regularizer(neural_control_variate, models, x, ll_div)).mean().backward()
            mc_variance.mean().backward()
            ncv_optimizer.step()
        ncv_s.append(neural_control_variate)
        print(f'Var with CV: {mc_variance.mean().item()}, Var w/o CV: {no_cv_variance.mean().item()}')
        print(f'Mean value of CV: {neural_control_variate(models, x, ll_div).mean()}')
    psy_weights = [deepcopy(ncv.psy_model.state_dict()) for ncv in ncv_s]
    with Path(args.save_path).open('wb') as fp:
        pickle.dump(psy_weights, fp)


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
