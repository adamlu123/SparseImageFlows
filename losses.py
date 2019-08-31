import torch
from torch import nn

from utils import safe_log


class FreeEnergyBound(nn.Module):

    def __init__(self, density):
        super().__init__()

        self.density = density

    def forward(self, zk, log_jacobians):

        sum_of_log_jacobians = sum(log_jacobians)
        return (-sum_of_log_jacobians - safe_log(self.density(zk))).mean()


class SparseCE(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pi, beta, x):
        zero = torch.zeros_like(x)
        one = torch.ones_like(x)
        z = torch.where(x < 1e-4, zero, one)
        ce_beta = torch.log(1 / torch.sqrt(2 * torch.pi) - torch.square(beta - x) / 2)
        ce_beta = torch.where(z > 0, ce_beta, zero)
        cross_entropy = (z * torch.log(pi) + (1 - z) * torch.log(1 - pi) + ce_beta).sum()
        return -cross_entropy

