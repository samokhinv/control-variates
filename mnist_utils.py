import numpy as np
import torch
import torch.utils.data
from torchvision import transforms, datasets
from pathlib import Path
from torch.utils.data import Subset
from utils import DatasetStandarsScaler, standartize


def load_mnist_dataset(data_dir, batch_size, classes=None, normalize=True, standard_scale=False):
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize(mean=(0.1307,), std=(0.3081,))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize(mean=(0.1307,), std=(0.3081,))
    ])

    Path(data_dir).mkdir(exist_ok=True, parents=True)
    trainset = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform_train)
    valset = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform_test)

    if classes is not None:
        if len(classes) == 2:
            train_idx = torch.logical_or(trainset.targets == classes[0], trainset.targets == classes[1])
            val_idx = torch.logical_or(valset.targets == classes[0], valset.targets == classes[1])

            trainset.data, valset.data = trainset.data[train_idx], valset.data[val_idx]
            trainset.targets, valset.targets = trainset.targets[train_idx] == classes[1], valset.targets[val_idx] == classes[1]
        else:
            targets = trainset.targets
            target_indices = np.arange(len(targets))
            idx_to_keep = [i for i, x in enumerate(targets) if x in classes]
            train_idx = target_indices[idx_to_keep]
            trainset = Subset(trainset, train_idx)

            targets = valset.targets
            target_indices = np.arange(len(targets))
            idx_to_keep = [i for i, x in enumerate(targets) if x in classes]
            train_idx = target_indices[idx_to_keep]
            valset = Subset(valset, train_idx)

    if normalize is True:
        if standard_scale is True:
            scaler = DatasetStandarsScaler()
            scaler.fit(trainset)
            trainset = scaler.transform(trainset)
            valset = scaler.transform(valset)
        else:
            X_train = np.vstack([x.view(-1).numpy() for x, _ in trainset])
            X_test = np.vstack([x.view(-1).numpy() for x, _ in valset])
            X_train, X_test = standartize(X_train, X_test, intercept=True)
            trainset = [(torch.from_numpy(x).float(), y) for x, (_, y) in zip(X_train, trainset)]
            valset = [(torch.from_numpy(x).float(), y) for x, (_, y) in zip(X_test, valset)]

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
