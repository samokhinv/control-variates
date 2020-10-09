import numpy as np
import torch
import torch.utils.data
from torchvision import transforms, datasets
from pathlib import Path
from torch.utils.data import Subset
import dill as pickle
from utils import DatasetStandarsScaler, standartize


def load_uci_dataset(data_dir, batch_size, classes=None, normalize=True):
    with Path(data_dir, 'tr_dataset.pkl').open('rb') as fp:
        trainset = pickle.load(fp)
    with Path(data_dir, 'test_dataset.pkl').open('rb') as fp:
        valset = pickle.load(fp)

    # if classes is not None:
    #     if len(classes) == 2:
    #         train_idx = torch.logical_or(trainset.targets == classes[0], trainset.targets == classes[1])
    #         val_idx = torch.logical_or(valset.targets == classes[0], valset.targets == classes[1])

    #         trainset.data, valset.data = trainset.data[train_idx], valset.data[val_idx]
    #         trainset.targets, valset.targets = trainset.targets[train_idx] == classes[1], valset.targets[val_idx] == classes[1]
    #     else:
    #         targets = trainset.targets
    #         target_indices = np.arange(len(targets))
    #         idx_to_keep = [i for i, x in enumerate(targets) if x in classes]
    #         train_idx = target_indices[idx_to_keep]
    #         trainset = Subset(trainset, train_idx)

    #         targets = valset.targets
    #         target_indices = np.arange(len(targets))
    #         idx_to_keep = [i for i, x in enumerate(targets) if x in classes]
    #         train_idx = target_indices[idx_to_keep]
    #         valset = Subset(valset, train_idx)

    if normalize is True:
        X_train = np.vstack([x.numpy() for x, _ in trainset])
        X_test = np.vstack([x.numpy() for x, _ in valset])
        X_train, X_test = standartize(X_train, X_test, intercept=True)
        # scaler = DatasetStandarsScaler()
        # scaler.fit(trainset)
        # trainset = scaler.transform(trainset)
        # valset = scaler.transform(valset)
        trainset = [(x, y) for x, (_, y) in zip(X_train, trainset)]
        valset = [(x, y) for x, (_, y) in zip(X_train, valset)]

    if batch_size == -1:
        train_batch_size = len(trainset)
        val_batch_size = len(valset)
    else:
        train_batch_size = val_batch_size = batch_size
        
    if torch.cuda.is_available():
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size, shuffle=True, pin_memory=True)
        valloader = torch.utils.data.DataLoader(valset, batch_size=val_batch_size, shuffle=False, pin_memory=True)
    else:
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size, shuffle=True, pin_memory=False)
        valloader = torch.utils.data.DataLoader(valset, batch_size=val_batch_size, shuffle=False, pin_memory=False)

    return trainloader, valloader