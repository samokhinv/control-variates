import torch
from pathlib import Path
import dill as pickle


def load_samples(samples_path, model_class, model_kwargs=None):
    with Path(samples_path).open('rb') as fp:
        samples = pickle.load(fp)

    print(f'N samples: {len(samples)}, volume of sample: {len(samples[0][0])}')

    trajectories = [[model_class(**model_kwargs)
                 for j in range(len(samples[i][0]))]
                for i in range(len(samples))]

    for i in range(len(samples)):
        for j in range(len(samples[i][0])):
            trajectories[i][j].load_state_dict(samples[i][0][j])

    priors = [samples[i][1] for i in range(len(samples))]

    if len(samples[0]) == 3:
        potential_grads = [samples[i][2] for i in range(len(samples))]
    else:
        potential_grads = None

    return trajectories, priors, potential_grads
