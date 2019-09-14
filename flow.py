import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
import numpy as np
from utils import safe_log



class Linear_layersLeakyReLU(nn.Module):
    def __init__(self, base_dim, out_dim):
        super().__init__()
        self.linear1 = nn.Linear(base_dim, 32)
        self.linear2 = nn.Linear(32, 128)
        self.linear3 = nn.Linear(128, 256)
        self.linear4 = nn.Linear(256, out_dim)
    def forward(self, x):
        x = torch.nn.LeakyReLU()(self.linear1(x))
        x = torch.nn.LeakyReLU()(self.linear2(x))
        x = torch.nn.LeakyReLU()(self.linear3(x))
        x = torch.nn.LeakyReLU()(self.linear4(x))
        return x


class Linear_layers(nn.Module):
    def __init__(self, base_dim, out_dim):
        super().__init__()
        self.linear1 = nn.Linear(base_dim, 32)
        self.linear2 = nn.Linear(32, 128)
        self.linear3 = nn.Linear(128, 256)
        self.linear4 = nn.Linear(256, out_dim)
    def forward(self, x):
        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear2(x))
        x = torch.relu(self.linear3(x))
        x = torch.relu(self.linear4(x))
        return x

class PlainGenerator(nn.Module):
    def __init__(self, base_dim, img_dim):
        super().__init__()
        # self.linear1 = nn.Linear(base_dim, 32)
        # self.linear2 = nn.Linear(32, 64)
        # self.linear3 = nn.Linear(64, 128)
        self.Linear_layers1 = Linear_layersLeakyReLU(base_dim, out_dim=128)
        self.Linear_layers2 = Linear_layersLeakyReLU(base_dim, out_dim=128)
        self.linear_pi = nn.Linear(128, img_dim)
        self.linear_beta = nn.Linear(128, img_dim)
        self.linear_std = nn.Linear(128, img_dim)
        self.sigmoid = torch.nn.Sigmoid()
        self.BatchNorm = nn.BatchNorm1d(128)

    def forward(self, x_pi,x_beta):
        # x = self.sigmoid(self.linear1(x))
        # x = self.sigmoid(self.linear2(x))
        # x = self.linear3(x)
        x_pi = self.Linear_layers1(x_pi)
        x_beta = self.BatchNorm(self.Linear_layers2(x_beta))
        pi = torch.sigmoid(self.linear_pi(x_pi))
        # pi = torch.max(torch.ones_like(pi), pi)

        beta = self.linear_beta(x_beta)
        # beta = torch.max(2*torch.ones_like(pi), beta)
        std = 3*torch.sigmoid(self.linear_std(x_beta))  #np.sqrt(0.5) * torch.ones_like(beta)#
        # std = torch.min(1 * torch.ones_like(pi), std)

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



class SinglePixelLinear(nn.Module):
    def __init__(self, width=25):
        super().__init__()
        self.W = nn.Parameter(torch.ones(width,width))
        self.bias = nn.Parameter(torch.ones(width,width))
    def forward(self, x_beta):
        return x_beta * self.W + self.bias


class PixelwiseLinears(nn.Module):
    def __init__(self,num_layers):
        super().__init__()
        self.num_layers = num_layers
        modules = []
        for i in range(self.num_layers):
            modules.append(SinglePixelLinear())
            modules.append(nn.LeakyReLU())
        self.layers = nn.Sequential(*modules)

    def forward(self, x_beta):
        return self.layers(x_beta)


class PixelwiseGenerator(nn.Module):
    def __init__(self, base_dim, img_dim):
        super().__init__()
        self.Linear_layers = Linear_layersLeakyReLU(base_dim, out_dim=128)
        self.pixelwiselinears = PixelwiseLinears(num_layers=3)
        self.SinglePixelLinear = SinglePixelLinear()
        self.linear_pi = nn.Linear(128, img_dim)
        self.SinglePixelLinear_std = SinglePixelLinear()
        self.sigmoid = torch.nn.Sigmoid()
        self.BatchNorm = nn.BatchNorm2d(1)

    def forward(self, x_pi,x_beta):
        x_pi = self.Linear_layers(x_pi)
        x_beta = self.BatchNorm(self.pixelwiselinears(x_beta))
        pi = torch.sigmoid(self.linear_pi(x_pi))

        beta = torch.relu(self.SinglePixelLinear(x_beta))
        beta_scale = torch.ones_like(beta)
        beta_scale[:, 0, 12, 12] = 25
        beta = beta_scale * beta
        std = 1*torch.sigmoid(self.SinglePixelLinear_std(x_beta))  #np.sqrt(0.5) * torch.ones_like(beta)#
        scale = torch.ones_like(std)
        scale[:,0,11:14,11:14] = 60
        std = scale * std
        return pi, beta, std


def logit(x):
    return torch.log(x) - torch.log(1 - x)

class DSF(nn.Module):
    def __init__(self):
        super().__init__()
        self.W1 = nn.Parameter(torch.tensor(1.)).cuda()
        self.bias = nn.Parameter(torch.tensor(1.)).cuda()
        self.W2 = nn.Parameter(torch.tensor(1.)).cuda()

    def forward(self, x):
        x = torch.sigmoid(self.W2) * torch.sigmoid(self.W1 * x + self.bias)
        x = logit(x)
        return x


class JointConvGenerator(nn.Module):
    def __init__(self, base_dim, img_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 4, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(4, 8, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(16, 1, kernel_size=3, padding=1)
        modules = [self.conv1, self.conv2, self.conv3, self.conv4]
        self.layers = nn.Sequential(*modules)
        self.SinglePixelLinear_pi = SinglePixelLinear()
        self.linear_pi = nn.Linear(25**2, 25**2)
        self.DSF = DSF()

    def forward(self, x):
        x = x.view(-1, 1, 25, 25)
        x = self.layers(x)
        pi = torch.sigmoid(self.linear_pi(x.view(-1,25**2)))
        beta = torch.exp(self.DSF(x))
        std = torch.ones_like(beta)
        return pi, beta, std


class JointLinearGenerator(nn.Module):
    def __init__(self, base_dim, img_dim):
        super().__init__()
        self.Linear_layers1 = Linear_layersLeakyReLU(base_dim, out_dim=128)
        self.SinglePixelLinear_beta = SinglePixelLinear()
        self.linear_pi = nn.Linear(128, 25**2)
        self.linear_beta = nn.Linear(128, 25**2)
        # self.DSF = DSF()

    def forward(self, x):
        x = self.Linear_layers1(x)
        pi = torch.sigmoid(self.linear_pi(x))

        beta = self.linear_beta(x).view(-1, 25, 25)
        # beta = torch.exp(self.SinglePixelLinear_beta(beta))
        # beta = torch.exp(self.SinglePixelLinear_beta(beta))
        beta = torch.exp(self.SinglePixelLinear_beta(beta))
        std = torch.ones_like(beta)
        return pi, beta, std
