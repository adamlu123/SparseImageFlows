from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
import numpy as np
import h5py

transform = transforms.Compose([
    transforms.ToTensor()
])


def get_loader(directory='./datasets', batch_size=128, train=True, num_workers=1,
               pin_memory=True):
    # 32 x 32
    dataset = datasets.CIFAR10(directory,
                               train=train,
                               download=True,
                               transform=transform)
    shuffle = not train
    loader = DataLoader(dataset,
                        batch_size=batch_size,
                        num_workers=num_workers,
                        shuffle=shuffle,
                        pin_memory=False)
    return loader


def get_lagan_loader(subset, batch_size):
    """
    :param subset: signal/background
    :param batch_size:
    :return: torch data loader
    """
    img_dir = "/baldig/physicsprojects/lagan"
    with h5py.File(img_dir + '/discretized_lagan.h5', 'r') as f:
        dataset = np.asarray(f[subset])
    dataset = torch.tensor(dataset, dtype=torch.float).view((-1, 1, 25, 25))
    # print('finish creating {} data loader! shape: {}'.format(subset, dataset.shape()))
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader