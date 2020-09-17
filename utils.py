import torch
from pathlib import Path
import dill as pickle


class DatasetStandarsScaler():
    def __init__(self, **kwargs):
        self.use_std = kwargs.get('use_std', True)
        self.biased = kwargs.get('biased', True)
        self.mean = kwargs.get('mean', 0)
        self.std = kwargs.get('std', 1)
        self.eps = kwargs.get('eps', 1e-7)

    def fit(self, dataset):
        N = len(dataset)
        mean = 0
        var = 0
        for x, _ in dataset:
            mean += x
        self.mean = mean / N
        for x, _ in dataset:
            var += (x - self.mean)**2
        self.std = (var / N)**(0.5)
        
    def transform(self, dataset):
        new_dataset = []
        for x, y in dataset:
            new_dataset.append(((x - self.mean) / (self.std + self.eps), y))
        return new_dataset


def load_samples(samples_path, model_class, model_kwargs=None):
    with Path(samples_path).open('rb') as fp:
        samples = pickle.load(fp)

    trajectories = [[model_class(**model_kwargs)
                 for j in range(len(samples[i][0]))]
                for i in range(len(samples))]

    for i in range(len(samples)):
        for j in range(len(samples[i][0])):
            trajectories[i][j].load_state_dict(samples[i][0][j])

    priors = [samples[i][2] for i in range(len(samples))]

    if len(samples[0]) == 3:
        potential_grads = torch.tensor(
          [samples[i][1] for i in range(len(samples))], dtype=torch.float
          ) 
    else:
        potential_grads = None

    return trajectories, priors, potential_grads
