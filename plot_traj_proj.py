#!/usr/bin/env
import numpy as np
from matplotlib import pyplot as plt
import torch
import dill as pickle
import argparse
from pathlib import Path
import random

from utils import random_seed


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--trajs_path', type=str, required=True)
    parser.add_argument('--save_fig_path', type=str, required=True)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    return args


def main(args):
    trajs, _, _ = pickle.load(Path(args.trajs_path).open('rb'))
    traj = random.sample(trajs, 1)[0]
    traj_proj = np.array([nn['linear.weight'][0, 2].item() for nn in traj])
    _, _ = plt.subplots()
    plt.plot(np.arange(len(traj_proj)), traj_proj)
    plt.title('Trajectory projection')
    plt.savefig(args.save_fig_path)
    plt.close()


if __name__ == '__main__':
    args = parse_arguments()
    random_seed(args.seed)
    main(args)
