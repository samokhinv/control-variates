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
import copy

from utils import load_trajs, random_seed
from control_variates.cv_utils import state_dict_to_vec
from mnist_utils import load_mnist_dataset
from UCI_utils import load_uci_dataset

from control_variates.cv import PsyLinear, SteinCV, PsyConstVector, PsyMLP


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

    parser.add_argument('--psy_type', type=str, choices=['const', 'mlp', 'linear'], default='const')

    parser.add_argument('--n_points', type=int, default=100)
    parser.add_argument('--sample_points', action='store_true')
    parser.add_argument('--n_batches', type=int, default=1)
    parser.add_argument('--predictive_distribution', action='store_true')

    parser.add_argument('--data_dir', type=str, default='..data/mnist')
    parser.add_argument('--model_config_path', type=str, required=True)
    parser.add_argument('--trajs_path', type=str, required=True)
    parser.add_argument('--cv_path', type=str)

    parser.add_argument('--dataset', type=str, choices=['mnist', 'uci'], default='mnist')
    parser.add_argument('--not_normalize', action='store_true')
    parser.add_argument('--standard_scale', action='store_true')
    parser.add_argument('--classes', type=int, nargs='+', default=[3, 5])

    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_arguments()
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
    test_trajs, test_traj_weights, test_traj_grads, _ = load_trajs(
        args.trajs_path,
        config,
        x_shape)

    psy_input_dim = state_dict_to_vec(test_trajs[0][0].state_dict()).shape[0]
    ncv = get_cv(psy_input_dim, device, args)
    if args.cv_path is not None:
        ncv.psy_model.load_state_dict(torch.load(args.cv_path))
    
    print(f'N test trajs: {len(test_trajs)}, len of traj: {len(test_trajs[0])}')

    x = x[:100].float()
    ind = random.randint(0, len(test_trajs)-1)
    traj, traj_weights, traj_grads = test_trajs[ind], test_traj_weights[ind], test_traj_grads[ind]
    traj_part = np.array([F.softmax(m(x), dim=-1)[:, 1].mean().item() for m in traj])
    cv_vals = ncv(traj_weights, x, potential_grad=traj_grads)
    traj_part_cv = np.array([(F.softmax(m(x), dim=-1)[:, 1]-cv_vals[i]).mean().item() for i, m in enumerate(traj)])
    _, axs = plt.subplots(nrows=1, ncols=2, figsize=(13, 5))
    axs[0].plot(np.arange(len(traj_part)), traj_part)
    axs[1].plot(np.arange(len(traj_part_cv)), traj_part_cv)
    plt.savefig('traj_pred.png')
