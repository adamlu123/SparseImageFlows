import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
import numpy as np
from utils import safe_log


class NormalizingFlow(nn.Module):

    def __init__(self, dim, flow_length):
        super().__init__()

        self.transforms = nn.Sequential(*(
            PlanarFlow(dim) for _ in range(flow_length)
        ))

        self.log_jacobians = nn.Sequential(*(
            PlanarFlowLogDetJacobian(t) for t in self.transforms
        ))

    def forward(self, z):

        log_jacobians = []

        for transform, log_jacobian in zip(self.transforms, self.log_jacobians):
            log_jacobians.append(log_jacobian(z))
            z = transform(z)

        zk = z

        return zk, log_jacobians


class PlanarFlow(nn.Module):

    def __init__(self, dim):
        super().__init__()

        self.weight = nn.Parameter(torch.Tensor(1, dim))
        self.bias = nn.Parameter(torch.Tensor(1))
        self.scale = nn.Parameter(torch.Tensor(1, dim))
        self.tanh = nn.Tanh()

        self.reset_parameters()

    def reset_parameters(self):

        self.weight.data.uniform_(-0.01, 0.01)
        self.scale.data.uniform_(-0.01, 0.01)
        self.bias.data.uniform_(-0.01, 0.01)

    def forward(self, z):

        activation = F.linear(z, self.weight, self.bias)
        return z + self.scale * self.tanh(activation)


class PlanarFlowLogDetJacobian(nn.Module):
    """A helper class to compute the determinant of the gradient of
    the planar flow transformation."""

    def __init__(self, affine):
        super().__init__()

        self.weight = affine.weight
        self.bias = affine.bias
        self.scale = affine.scale
        self.tanh = affine.tanh

    def forward(self, z):

        activation = F.linear(z, self.weight, self.bias)
        psi = (1 - self.tanh(activation) ** 2) * self.weight
        det_grad = 1 + torch.mm(psi, self.scale.t())
        return safe_log(det_grad.abs())




class Linear_layers(nn.Module):
    def __init__(self, base_dim, out_dim):
        super().__init__()
        self.linear1 = nn.Linear(base_dim, 32)
        self.linear2 = nn.Linear(32, 64)
        self.linear3 = nn.Linear(64, out_dim)
    def forward(self, x):
        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear2(x))
        x = torch.relu(self.linear3(x))
        return x

class PlainGenerator(nn.Module):
    def __init__(self, base_dim, img_dim):
        super().__init__()
        # self.linear1 = nn.Linear(base_dim, 32)
        # self.linear2 = nn.Linear(32, 64)
        # self.linear3 = nn.Linear(64, 128)
        self.Linear_layers1 = Linear_layers(base_dim, out_dim=128)
        self.Linear_layers2 = Linear_layers(base_dim, out_dim=128)
        self.linear_pi = nn.Linear(128, img_dim)
        self.linear_beta = nn.Linear(128, img_dim)
        self.linear_std = nn.Linear(128, img_dim)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x_pi,x_beta):
        # x = self.sigmoid(self.linear1(x))
        # x = self.sigmoid(self.linear2(x))
        # x = self.linear3(x)
        x_pi = self.Linear_layers1(x_pi)
        x_beta = self.Linear_layers2(x_beta)
        pi = torch.sigmoid(self.linear_pi(x_pi))
        beta = torch.relu(self.linear_beta(x_beta))
        std = torch.exp(self.linear_std(x_beta))  #np.sqrt(0.5) * torch.ones_like(beta)#
        return pi, beta, std




class DeConvLayers(nn.Module):
    def __init__(self, base_dim, out_dim):
        super().__init__()
        self.ConvTranspose2d_1 = nn.ConvTranspose2d(in_channels=1, out_channels=1,
                                                    kernel_size=5, stride=1, dilation=1)
        self.ConvTranspose2d_2 = nn.ConvTranspose2d(in_channels=1, out_channels=1,
                                                    kernel_size=5, stride=1, dilation=1)
        self.ConvTranspose2d_3 = nn.ConvTranspose2d(in_channels=1, out_channels=1,
                                                    kernel_size=5, stride=1, dilation=1)
        self.ConvTranspose2d_4 = nn.ConvTranspose2d(in_channels=1, out_channels=1,
                                                    kernel_size=5, stride=1, dilation=1)

    def forward(self, x):
        out = self.ConvTranspose2d_1(x)
        out = self.ConvTranspose2d_2(out)
        out = self.ConvTranspose2d_3(out)
        out = self.ConvTranspose2d_4(out)
        return out

class PlainDeconvGenerator(nn.Module):
    def __init__(self, base_dim, img_dim):
        super().__init__()
        self.DeConvLayers = DeConvLayers(base_dim, out_dim=32)
        self.DeConvLayers_beta = DeConvLayers(base_dim, out_dim=32)
        self.linear_pi = nn.Linear(32*32, img_dim)
        self.linear_beta = nn.Linear(32*32, img_dim)
        self.linear_std = nn.Linear(32*32, img_dim)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x_pi, x_beta):
        x_pi = self.DeConvLayers(x_pi)
        x_pi = x_pi.view(-1,32*32)
        x_beta = self.DeConvLayers(x_beta) #.view(-1,32*32)
        x_beta = x_beta.view(-1, 32 * 32)

        pi = torch.sigmoid(self.linear_pi(x_pi))
        beta = torch.relu(self.linear_beta(x_beta))
        std = torch.exp(self.linear_std(x_beta))
        return pi, beta, std


class PlainGenerator_pi(nn.Module):
    def __init__(self, base_dim, img_dim):
        super().__init__()
        self.Linear_layers1 = Linear_layers(base_dim, out_dim=128)
        self.linear_pi = nn.Linear(128, img_dim)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x_pi):
        x_pi = self.Linear_layers1(x_pi)
        pi = torch.sigmoid(self.linear_pi(x_pi))
        return pi

class FlowGenerator(nn.Module):
    def __init__(self, base_dim, img_dim):
        super().__init__()
        self.pi_generator = PlainGenerator_pi(base_dim, img_dim)
        # self.beta_generator = NormalizingFlow(dim=32*32, flow_length=8)
        ones = torch.ones(32*32)
        self.beta = nn.Parameter(0.01*ones).cuda()
        self.std = nn.Parameter(0.01*ones).cuda()
    def forward(self, x_pi, x_beta):
        pi = self.pi_generator(x_pi)
        # beta, std = self.beta_generator(x_beta)
        return pi, self.beta, self.std