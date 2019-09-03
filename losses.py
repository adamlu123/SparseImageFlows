import torch
from torch import nn

from utils import safe_log
import numpy as np

class FreeEnergyBound(nn.Module):

    def __init__(self, density):
        super().__init__()

        self.density = density

    def forward(self, zk, log_jacobians):

        sum_of_log_jacobians = sum(log_jacobians)
        return (-sum_of_log_jacobians - safe_log(self.density(zk))).mean()


class SparseCE(nn.Module):
    def __init__(self):
         super(SparseCE).__init__()

    def forward(self, pi, beta, std, x):
        zero = torch.zeros_like(x).cuda()
        one = torch.ones_like(x).cuda()
        z = torch.where(x > zero, one, zero)
        ce_beta = np.log(1 / np.sqrt(2 * np.pi)) - (beta - x)**2 / (2*1**2)
        ce_beta = torch.where(z > 0, ce_beta, zero)
        # if ce_beta.shape[0] != 128:
        #     print(ce_beta.shape)
        cross_entropy = (z * torch.log(pi) + (1 - z) * torch.log(1 - pi) + ce_beta).mean()
        return -cross_entropy


# class SparseCE_mixture_flow(nn.Module):
#     def __init__(self):
#         super(SparseCE_mixture_flow).__init__()
#
#     def forward(self, pi, beta, std, x):
#         zero = torch.zeros_like(x).cuda()
#         one = torch.ones_like(x).cuda()
#         z = torch.where(x > zero, one, zero)
#         loglike_beta = np.log(1 / np.sqrt(2 * np.pi)) - (beta - x)**2 / (2*std**2)
#         loglike_beta = torch.where(z > 0, loglike_beta, zero)
#         cross_entropy = (z * torch.log(pi) + (1 - z) * torch.log(1 - pi) + loglike_beta).mean()
#         return -cross_entropy

def SparseCE_mixture_flow(pi, beta, std, x):
    zero = torch.zeros_like(x).cuda()
    one = torch.ones_like(x).cuda()
    z = torch.where(x > zero, one, zero)
    loglike_beta = np.log(1 / np.sqrt(2 * np.pi)) - (beta - x) ** 2 / (2 * std ** 2)
    loglike_beta = torch.where(z > 0, loglike_beta, zero)
    cross_entropy = (z * torch.log(pi) + (1 - z) * torch.log(1 - pi) + loglike_beta).mean()
    return -cross_entropy