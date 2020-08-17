import torch
import torch.utils.data
from torchvision import transforms, datasets
from pathlib import Path


def load_mnist_dataset(data_dir, batch_size, two_classes=None):
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.1307,), std=(0.3081,))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.1307,), std=(0.3081,))
    ])

    Path(data_dir).mkdir(exist_ok=True)
    trainset = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform_train)
    valset = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform_test)

    if two_classes is not None:
        train_idx = torch.logical_or(trainset.targets == two_classes[0], trainset.targets == two_classes[1])
        val_idx = torch.logical_or(valset.targets == two_classes[0], valset.targets == two_classes[1])

        trainset.data, valset.data = trainset.data[train_idx], valset.data[val_idx]
        trainset.targets, valset.targets = trainset.targets[train_idx] == two_classes[1], valset.targets[val_idx] == two_classes[1]

    if torch.cuda.is_available():
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, pin_memory=True)
        valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False, pin_memory=True)

    else:
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, pin_memory=False)
        valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False, pin_memory=False)

    return trainloader, valloader

