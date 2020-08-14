import torch
import torch.utils.data
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader
from pathlib import Path


def load_mnist_dataset(data_dir, batch_size, ):
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

    
    if torch.cuda.is_available():
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=3)
        valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=3)

    else:
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, pin_memory=False,
                                                num_workers=3)
        valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False, pin_memory=False,
                                                num_workers=3)

    return trainloader, valloader

