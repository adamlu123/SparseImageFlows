import math
import types

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Gamma, Bernoulli
import utils
import plot_utils

def get_mask(in_features, out_features, in_flow_features, mask_type=None):
    """
    mask_type: input | None | output

    See Figure 1 for a better illustration:
    https://arxiv.org/pdf/1502.03509.pdf
    """
    if mask_type == 'input':
        in_degrees = torch.arange(in_features) % in_flow_features
    else:
        # in_degrees = torch.arange(in_features) % (in_flow_features - 1)
        in_degrees = torch.arange(in_features) % in_flow_features

    if mask_type == 'output':
        out_degrees = torch.arange(out_features) % in_flow_features - 1
    else:
        # out_degrees = torch.arange(out_features) % (in_flow_features - 1)
        out_degrees = torch.arange(out_features) % in_flow_features
    return (out_degrees.unsqueeze(-1) >= in_degrees.unsqueeze(0)).float()

class MaskedLinear(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 mask,
                 cond_in_features=None,
                 bias=True):
        super(MaskedLinear, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        if cond_in_features is not None:
            self.cond_linear = nn.Linear(
                cond_in_features, out_features, bias=False)

        self.register_buffer('mask', mask)

    def forward(self, inputs, cond_inputs=None):
        output = F.linear(inputs, self.linear.weight * self.mask,
                          self.linear.bias)
        if cond_inputs is not None:
            output += self.cond_linear(cond_inputs)
        return output

nn.MaskedLinear = MaskedLinear


class ARBase(nn.Module):
    def __init__(self,
                 num_inputs,
                 num_hidden,
                 num_cond_inputs=None,
                 act='relu',
                 num_latent_layer=2):
        super(ARBase, self).__init__()
        activations = {'relu': nn.ReLU, 'sigmoid': nn.Sigmoid, 'tanh': nn.Tanh}
        act_func = activations[act]

        input_mask = get_mask(num_inputs, num_hidden, num_inputs, mask_type='input')
        hidden_mask = get_mask(num_hidden, num_hidden, num_inputs)
        output_mask = get_mask(num_hidden, num_inputs * 2, num_inputs, mask_type='output')

        self.joiner = nn.MaskedLinear(num_inputs, num_hidden, input_mask, num_cond_inputs)

        latent_modules = []
        for i in range(num_latent_layer):
            latent_modules.append(act_func())
            latent_modules.append(nn.MaskedLinear(num_hidden, num_hidden, hidden_mask))
        latent_modules.append(nn.MaskedLinear(num_hidden, num_inputs * 2, output_mask))
        self.trunk = nn.Sequential(*latent_modules)

        self.conv1d = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=90, stride=1, kernel_size=1), nn.BatchNorm1d(90), nn.ReLU(),
            nn.Conv1d(in_channels=90, out_channels=180, stride=1, kernel_size=1), nn.BatchNorm1d(180), nn.ReLU(),
            nn.Conv1d(in_channels=180, out_channels=277, stride=1, kernel_size=1)
        )
        self.conv1d_gamma = nn.Conv1d(in_channels=1, out_channels=1, stride=1, kernel_size=1)

    def forward(self, inputs, cond_inputs=None, mode='direct', epoch=0):
        input_dim = inputs.shape[1]
        if mode == 'direct':
            h = self.joiner(inputs, cond_inputs)
            gamma, theta = self.trunk(h).chunk(2, 1)
            gamma = self.conv1d_gamma(gamma.view(-1, 1, input_dim))
            gamma = torch.sigmoid(gamma)
            pred = self.conv1d(theta.view(-1,1,input_dim))  # shape=(batch, 276, 625)
            return gamma, pred

        else:
            x = torch.zeros_like(inputs)
            with torch.no_grad():
                for i in range(input_dim):
                    h = self.joiner(x.view(-1, input_dim), cond_inputs)
                    gamma, theta = self.trunk(h).chunk(2, 1)  #
                    gamma = self.conv1d_gamma(gamma.view(-1, 1, input_dim)).squeeze()
                    gamma = torch.sigmoid(gamma[:, i])
                    z = Bernoulli(probs=gamma).sample()

                    out = self.conv1d(theta.view(-1,1,input_dim))
                    probs = F.softmax(out[:, :, i], dim=1).data  # shape=(batchsize, 277)
                    nonzeros = torch.multinomial(probs, 1).float().view(-1)  # shape=(batchsize)
                    x[:, i] = torch.where(z > 0, nonzeros, torch.zeros_like(nonzeros))
            return x


class MultiscaleAR(nn.Module):
    """
    Implementation of 2-scale AR model consisting of ARinner and ARouter,
    window is assumed to be square with area 'window_area'.
    """
    def __init__(self, window_area,
                 num_inputs,
                 num_hidden,
                 act='relu',
                 num_latent_layer=2):
        super(MultiscaleAR, self).__init__()
        self.ARinner = ARBase(window_area, num_hidden[0], None, act, num_latent_layer)
        self.ARouter = ARBase(num_inputs-window_area, num_hidden[1], window_area, act, num_latent_layer)
        self.window_area = window_area

    def forward(self, inputs, mode='direct'):
        if mode == 'direct':
            gamma_i, pred_i = self.ARinner(inputs[:, :self.window_area], mode='direct')
            inner_mean = (gamma_i*pred_i).mean(dim=1)  # shape=(batch, 9)
            gamma_o, pred_o = self.ARouter(inputs[:, self.window_area:], cond_inputs=inner_mean, mode='direct')
            gamma, pred = torch.cat([gamma_i, gamma_o], -1), torch.cat([pred_i, pred_o], -1)
            self.gamma = gamma.detach().cpu().numpy()

            nll_positive = F.cross_entropy(pred.view(-1, 277, 625), inputs.long(), reduction="none")  # shape=(batchsize, 625)
            ll = torch.where(inputs > 0,
                             0.1 * (gamma + 1e-10).log() - nll_positive,
                             0.1 * (1 - gamma + 1e-10).log()).sum(dim=-1, keepdim=True)
            return ll

        else:
            x_i = self.ARinner(inputs[:, :self.window_area], mode=mode)
            x_o = self.ARouter(inputs[:, self.window_area:], cond_inputs=x_i, mode=mode)
            x = torch.cat([x_i, x_o], -1)
            return x



class FlowSequential(nn.Sequential):
    """ A sequential container for Mixture of Dirac delta mass and Gamma/Normal density flows. (one layer)
    In addition to a forward pass it implements a backward pass and
    computes log jacobians.
    """

    def forward(self, inputs, cond_inputs=None, mode='direct', logdets=None):
        """ Performs a forward or backward pass for flow modules.
        Args:
            inputs: a tuple of inputs and logdets
            mode: to run direct computation or inverse
        """
        assert mode in ['direct', 'inverse']
        if mode == 'direct':
            for module in self._modules.values():
                ll = module(inputs, mode)
            return ll
        else:
            for module in reversed(self._modules.values()):
                inputs = module(inputs, mode)
            return inputs

    def log_probs(self, inputs):
        ll = self(inputs)
        self.log_jacob = torch.tensor(0.)
        self.u = torch.tensor(0.)
        return ll

    def sample(self, num_samples=None, noise=None, cond_inputs=None, input_size=1024):
        if noise is None:
            noise = torch.Tensor(num_samples, input_size).normal_()
        device = next(self.parameters()).device
        noise = noise.to(device)
        if cond_inputs is not None:
            cond_inputs = cond_inputs.to(device)
        samples = self.forward(noise, cond_inputs, mode='inverse')
        return samples















