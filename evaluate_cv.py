import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import dill as pickle
import random
import torch
import argparse
from pathlib import Path

from utils import load_samples
from control_variates.uncertainty_quantification import ClassificationUncertaintyMCMC
from control_variates.model import LogRegression
from control_variates.cv import PsyLinear, SteinCV, PsyConstVector
from control_variates.cv_utils import state_dict_to_vec, compute_naive_variance, compute_potential_grad
from control_variates.model import get_binary_prediction
from mnist_utils import load_mnist_dataset
from UCI_utils import load_uci_dataset


psy_class = {'const': PsyConstVector, 'linear': PsyLinear}


def random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=-1, type=int)
    parser.add_argument('--width', type=int, default=100)
    parser.add_argument('--depth', type=int, default=1)
    parser.add_argument('--output_dim', type=int, default=2)
    parser.add_argument('--samples_path', type=str, required=True)
    parser.add_argument('--psy_type', type=str, choices=['const', 'mlp', 'linear'], default='const')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--classes', type=int, nargs='+', default=[3, 5])
    parser.add_argument('--psy_path', type=str, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--data_dir', type=str, default='..data/mnist')
    parser.add_argument('--dataset', type=str, choices=['mnist', 'uci'], default='mnist')
    parser.add_argument('--figure_path', type=str)

    args = parser.parse_args()
    return args


def main(args):
    random_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    if args.dataset == 'mnist':
        Path.mkdir(Path(args.data_dir), exist_ok=True, parents=True)
        if args.batch_size == -1:
            args.batch_size = 20000
        train_dl, valid_dl = load_mnist_dataset(args.data_dir, args.batch_size, classes=[3, 5])
    elif args.dataset == 'uci':
        train_dl, valid_dl = load_uci_dataset(args.data_dir, batch_size=args.batch_size)
    N_train = len(train_dl.dataset)

    x, _ = train_dl.dataset[0]
    args.input_dim = 1
    for d in x.shape:
        args.input_dim *= d

    x_new, y_new = next(iter(valid_dl))
    train_x, train_y = next(iter(train_dl))

    trajectories, priors, potential_grads = load_samples(
        args.samples_path,
        model_class=LogRegression, 
        model_kwargs={'input_size': args.input_dim})
    if potential_grads is None:
        potential_grads = [compute_potential_grad(ms, train_x, train_y, N_train, priors=ps) for ms, ps in zip(trajectories, priors)]
    
    with Path(args.psy_path).open('rb') as fp:
        psy_weights = pickle.load(fp)
    psy_input_dim = state_dict_to_vec(trajectories[0][0].state_dict()).shape[0]
    psy_model_class = psy_class[args.psy_type]
    psy_models = [psy_model_class(psy_input_dim) for _ in range(len(psy_weights))]
    for i in range(len(psy_weights)):
        psy_models[i].load_state_dict(psy_weights[i])

    ncv_s = [SteinCV(psy_model, train_x, train_y, prior, len(train_dl.dataset), potential_grad=potential_grad) \
         for psy_model, prior, potential_grad in zip(psy_models, priors, potential_grads)]

    x_batch = x_new[y_new == 1.0][[43]]

    for x in x_batch:
        predictions_cv = []
        predictions_no_cv = []
        for models, ncv in tqdm(zip(trajectories, ncv_s)):
            uq = ClassificationUncertaintyMCMC(models, ncv)
            predictions_cv.append(uq.estimate_emperical_mean(x.unsqueeze(0), use_cv=True).mean().item())
            predictions_no_cv.append(uq.estimate_emperical_mean(x.unsqueeze(0), use_cv=False).mean().item())
        fig7, ax7 = plt.subplots()
        ax7.boxplot([predictions_no_cv, predictions_cv])
        plt.savefig(args.figure_path)


if __name__ == "__main__":
    args = parse_arguments()
    main(args)