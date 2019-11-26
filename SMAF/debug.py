import torch
from torchvision import datasets, transforms
from torch.distributions import Gamma

train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=128, shuffle=True)


Gamma(concentration=alpha[inputs>0], rate=beta).log_prob(inputs[inputs>0])