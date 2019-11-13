import math
import types

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Gamma, Bernoulli
import utils


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


class ElementWiseLinear(nn.Module):
    def __init__(self, dim):
        super(ElementWiseLinear, self).__init__()
        # init = torch.nn.init.uniform(torch.empty(25, 25)).view(-1)
        self.weight = nn.Parameter(torch.zeros(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.conv = nn.Conv1d(in_channels=2, out_channels=1, kernel_size=1)

    def forward(self,
                coordinate_map):  # TODO: determine whether to use polar coordinate or Cartisan coordinate for position
        coordinate_map = self.conv(coordinate_map).squeeze()
        return coordinate_map * self.weight + self.bias


class AffineTransform(nn.Module):
    """
    Affine transform where weight and bias are output by a function of pixel position (call ElementWiseLinear)
    TODO: think about whether we need to use two separate ElementWiseLinear to output weight and bias.
    """

    def __init__(self):
        super(AffineTransform, self).__init__()
        self.ElementWiseLinear_weight = ElementWiseLinear(dim=625)
        self.ElementWiseLinear_bias = ElementWiseLinear(dim=625)

    def forward(self, x, coordinate_map):
        weight = self.ElementWiseLinear_weight(coordinate_map)
        bias = self.ElementWiseLinear_weight(coordinate_map)
        return x + bias


class ParameterFilter(nn.Module):
    """
    A wrapper for AffineTransform to deal with gamma, mu, and log_std together.
    """

    def __init__(self):
        super(ParameterFilter, self).__init__()
        self.AffineTransform_gamma = AffineTransform()
        self.AffineTransform_mu = AffineTransform()
        self.AffineTransform_logstd = AffineTransform()
        self.coordinate_map = (torch.stack((torch.tensor(np.stack([np.arange(0, 25.)] * 25), dtype=torch.float).t(),
                                            torch.tensor(np.stack([np.arange(0, 25.)] * 25),
                                                         dtype=torch.float)))/12 - 1).view(1, 2, 625).cuda()

    def forward(self, gamma, mu, log_std):
        gamma = self.AffineTransform_gamma(gamma, self.coordinate_map)
        mu = self.AffineTransform_mu(mu, self.coordinate_map)
        log_std = self.AffineTransform_logstd(log_std, self.coordinate_map)
        return gamma, mu, log_std


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


class MADE(nn.Module):
    """ An implementation of MADE
    (https://arxiv.org/abs/1502.03509s).
    """

    def __init__(self,
                 num_inputs,
                 num_hidden,
                 num_cond_inputs=None,
                 act='relu',
                 pre_exp_tanh=False):
        super(MADE, self).__init__()

        activations = {'relu': nn.ReLU, 'sigmoid': nn.Sigmoid, 'tanh': nn.Tanh}
        act_func = activations[act]

        input_mask = get_mask(
            num_inputs, num_hidden, num_inputs, mask_type='input')
        hidden_mask = get_mask(num_hidden, num_hidden, num_inputs)
        output_mask = get_mask(
            num_hidden, num_inputs * 2, num_inputs, mask_type='output')

        self.joiner = nn.MaskedLinear(num_inputs, num_hidden, input_mask,
                                      num_cond_inputs)

        self.trunk = nn.Sequential(act_func(),
                                   nn.MaskedLinear(num_hidden, num_hidden,
                                                   hidden_mask), act_func(),
                                   nn.MaskedLinear(num_hidden, num_inputs * 2,
                                                   output_mask))

    def forward(self, inputs, cond_inputs=None, mode='direct'):
        if mode == 'direct':
            h = self.joiner(inputs, cond_inputs)
            m, a = self.trunk(h).chunk(2, 1)
            u = (inputs - m) * torch.exp(-a)
            return u, -a  # .sum(-1, keepdim=True)

        else:
            x = torch.zeros_like(inputs)
            for i_col in range(inputs.shape[1]):
                h = self.joiner(x, cond_inputs)
                m, a = self.trunk(h).chunk(2, 1)
                m, a = m.detach(), a.detach()
                x[:, i_col] = inputs[:, i_col] * torch.exp(
                    a[:, i_col]) + m[:, i_col]
            return x  # , -a.sum(-1, keepdim=True)


class MixtureGammaMADE(nn.Module):
    """ An implementation of MADE
    (https://arxiv.org/abs/1502.03509s).
    """

    def __init__(self,
                 num_inputs,
                 num_hidden,
                 num_cond_inputs=None,
                 act='sigmoid',
                 pre_exp_tanh=False):
        super(MixtureGammaMADE, self).__init__()

        activations = {'relu': nn.ReLU, 'sigmoid': nn.Sigmoid, 'tanh': nn.Tanh}
        act_func = activations[act]

        input_mask = get_mask(
            num_inputs, num_hidden, num_inputs, mask_type='input')
        hidden_mask = get_mask(num_hidden, num_hidden, num_inputs)
        output_mask = get_mask(
            num_hidden, num_inputs * 2, num_inputs, mask_type='output')

        self.joiner = nn.MaskedLinear(num_inputs, num_hidden, input_mask,
                                      num_cond_inputs)

        self.trunk = nn.Sequential(act_func(),
                                   nn.MaskedLinear(num_hidden, num_hidden,
                                                   hidden_mask), act_func(),
                                   nn.MaskedLinear(num_hidden, num_inputs * 2,
                                                   output_mask))

    #
    def forward(self, inputs, cond_inputs=None, mode='direct'):
        if mode == 'direct':
            h = self.joiner(inputs, cond_inputs)
            gamma, alpha = self.trunk(h).chunk(2, 1)
            gamma = torch.sigmoid(gamma)
            alpha = torch.exp(alpha)  # + 1e-2

            beta = torch.ones_like(alpha)  # TODO: can change beta to other value
            ll = torch.where(inputs > 0,
                             gamma.log() + utils.gamma_log_prob(alpha, beta, inputs),
                             (1 - gamma).log()).sum(dim=-1, keepdim=True)
            self.gamma = gamma.detach().cpu().numpy()
            self.alpha = alpha.detach().cpu().numpy()
            return ll

        else:
            x = torch.zeros_like(inputs)
            for i_col in range(inputs.shape[1]):
                h = self.joiner(x, cond_inputs)
                gamma, alpha = self.trunk(h).chunk(2, 1)
                gamma = torch.sigmoid(gamma)
                alpha = torch.exp(alpha)
                x[:, i_col] = utils.MTsample(gamma=gamma[:, i_col], alpha=alpha[:, i_col],
                                             beta=1)  # inputs[:, i_col] * torch.exp(a[:, i_col]) + m[:, i_col]
            return x

    # def forward(self, inputs, cond_inputs=None, mode='direct'):
    #     if mode == 'direct':
    #         h = self.joiner(inputs, cond_inputs)
    #         gamma, alpha, beta = self.trunk(h).chunk(3, 1)
    #         gamma = torch.sigmoid(gamma)
    #         # alpha = torch.exp(alpha) #+ 1e-2
    #         ll = torch.where(inputs > 0,
    #                           gamma.log() + utils.normal_log_prob(mu=alpha, sd=beta.exp(), value=inputs),
    #                           (1-gamma).log()).sum(dim=-1, keepdim=True)
    #         self.gamma = gamma.detach().cpu().numpy()
    #         self.alpha = alpha.detach().cpu().numpy()
    #         return ll.mean()
    #
    #     else:
    #         x = torch.zeros_like(inputs)
    #         for i_col in range(inputs.shape[1]):
    #             h = self.joiner(x, cond_inputs)
    #             gamma, alpha, beta = self.trunk(h).chunk(2, 1)
    #             gamma = torch.sigmoid(gamma)
    #             z = Bernoulli(p=gamma).sample().cuda()
    #             nonzeros = inputs[:, i_col] * torch.exp(beta[:, i_col]) + alpha[:, i_col]
    #             x[:, i_col] = torch.where(z > 0, nonzeros, torch.zeros_like(nonzeros))
    #         return x


class MixtureNormalMADE(nn.Module):
    """ An implementation of mxiture of Dirac delta and normal: MADE structure
    (https://arxiv.org/abs/1502.03509s).
    """

    def __init__(self,
                 num_inputs,
                 num_hidden,
                 num_cond_inputs=None,
                 act='relu'):
        super(MixtureNormalMADE, self).__init__()

        activations = {'relu': nn.ReLU, 'sigmoid': nn.Sigmoid, 'tanh': nn.Tanh}
        act_func = activations[act]

        input_mask = get_mask(
            num_inputs, num_hidden, num_inputs, mask_type='input')
        hidden_mask = get_mask(num_hidden, num_hidden, num_inputs)
        output_mask = get_mask(
            num_hidden, num_inputs * 3, num_inputs, mask_type='output')

        self.joiner = nn.MaskedLinear(num_inputs, num_hidden, input_mask,
                                      num_cond_inputs)

        self.trunk = nn.Sequential(act_func(),
                                   nn.MaskedLinear(num_hidden, num_hidden,
                                                   hidden_mask), act_func(),
                                   nn.MaskedLinear(num_hidden, num_hidden,
                                                   hidden_mask), act_func(),
                                   # nn.MaskedLinear(num_hidden, num_hidden,
                                   #                 hidden_mask), act_func(),
                                   nn.MaskedLinear(num_hidden, num_inputs * 3,
                                                   output_mask))

        self.ParameterFilter = ParameterFilter()

    def forward(self, inputs, cond_inputs=None, mode='direct'):
        if mode == 'direct':
            h = self.joiner(inputs, cond_inputs)
            gamma, mu, log_std = self.trunk(h).chunk(3, 1)
            # gamma, mu, log_std = self.ParameterFilter(gamma, mu, log_std)

            gamma = torch.sigmoid(gamma)
            u = (inputs - mu) * torch.exp(-log_std)
            gamma = (1 - gamma) * utils.get_psi(mu, torch.exp(log_std)) + gamma
            # gamma = gamma * utils.get_psi(mu, torch.exp(log_std)) + gamma  Wrong could be >1
            # gamma = gamma - gamma * utils.get_psi(mu, torch.exp(log_std))

            # ll using reshaped truncated normal
            ll = torch.where(inputs > 0,
                             (gamma + 1e-10).log() + utils.normal_log_prob(mu=mu, sd=log_std.exp(), value=inputs),
                             (1 - gamma + 1e-10).log()).sum(dim=-1, keepdim=True)

            # ll using truncated normal
            # ll = torch.where(inputs > 0,
            #                  (gamma + 1e-10).log() + utils.trucated_normal_log_prob(mu=mu, sd=log_std.exp(), value=inputs),
            #                  (1 - gamma + 1e-10).log()).sum(dim=-1, keepdim=True)

            self.gamma = gamma.detach().cpu().numpy()
            self.alpha = mu.detach().cpu().numpy()
            self.log_std = log_std.detach().cpu().numpy()

            return u, -log_std, ll  # output gamma to compute ll

        else:
            x = torch.zeros_like(inputs)
            with torch.no_grad():
                for i_col in range(inputs.shape[1]):
                    # print(i_col)
                    h = self.joiner(x, cond_inputs)
                    gamma, mu, log_std = self.trunk(h).chunk(3, 1)
                    # gamma = (1 - gamma) * utils.get_psi(mu, torch.exp(log_std)) + gamma

                    gamma = torch.sigmoid(gamma[:, i_col])
                    # print(gamma.min(), gamma.max())

                    # gamma, mu, log_std = gamma.cpu().numpy(), mu.cpu().numpy(), log_std.cpu().numpy()
                    # z = np.random.binomial(1, gamma)

                    z = Bernoulli(probs=gamma).sample()
                    nonzeros = inputs[:, i_col] * torch.exp(log_std[:, i_col]) + mu[:, i_col]
                    x[:, i_col] = torch.where(z > 0, nonzeros, torch.zeros_like(nonzeros))

                    # nonzeros = utils.truncated_normal_sample(mu=mu[:, i_col],
                    #                                          sigma=np.exp(log_std[:, i_col]),
                    #                                          num_samples=inputs.shape[0])
                    # nonzeros = torch.tensor(nonzeros, dtype=torch.float).detach()
                    # x[:, i_col] = torch.tensor(np.where(z > 0, nonzeros, np.zeros_like(nonzeros)), dtype=torch.float).cuda()
            return x  # , -log_std.sum(-1, keepdim=True)


class Sigmoid(nn.Module):
    def __init__(self):
        super(Sigmoid, self).__init__()

    def forward(self, inputs, cond_inputs=None, mode='direct'):
        if mode == 'direct':
            s = torch.sigmoid
            return s(inputs), torch.log(s(inputs) * (1 - s(inputs))).sum(
                -1, keepdim=True)
        else:
            return torch.log(inputs /
                             (1 - inputs)), -torch.log(inputs - inputs ** 2).sum(
                -1, keepdim=True)


class Logit(Sigmoid):
    def __init__(self):
        super(Logit, self).__init__()

    def forward(self, inputs, cond_inputs=None, mode='direct'):
        if mode == 'direct':
            return super(Logit, self).forward(inputs, 'inverse')
        else:
            return super(Logit, self).forward(inputs, 'direct')


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
        self.num_inputs = inputs.size(-1)

        if logdets is None:
            logdets = torch.zeros(inputs.size(0), 625, device=inputs.device)

        assert mode in ['direct', 'inverse']
        if mode == 'direct':
            for module in self._modules.values():
                if isinstance(module, MixtureGammaMADE):
                    logdet = module(inputs, cond_inputs, mode)
                    logdets += logdet
                    return logdets
                elif isinstance(module, MixtureNormalMADE):
                    inputs, logdet, gamma = module(inputs, cond_inputs, mode)
                    logdets += logdet
                elif isinstance(module, MADE):
                    inputs, logdet = module(inputs, cond_inputs, mode)
                    logdets += logdet
            return inputs, logdets, gamma
        else:
            for module in reversed(self._modules.values()):
                inputs = module(inputs, cond_inputs, mode)
            return inputs

    def log_probs(self, inputs):
        if isinstance(self._modules['0'], MixtureGammaMADE):
            log_probs = self(inputs)
            return (log_probs).sum(-1, keepdim=True)
        else:
            u, log_jacob, ll = self(inputs)
            self.log_jacob = log_jacob
            self.u = u
            #
            # log_probs = (-0.5 * u ** 2 - 0.5 * math.log(2 * math.pi))  #.sum(-1, keepdim=True)
            # normal_log_prob = (log_probs + log_jacob)  #.sum(-1, keepdim=True)
            # ll = torch.where(inputs > 0,
            #                  gamma.log() + normal_log_prob,
            #                  (1-gamma).log()).sum(dim=-1, keepdim=True)
            return ll

    def sample(self, num_samples=None, noise=None, cond_inputs=None, input_size=625):
        # input_size =
        if noise is None:
            noise = torch.Tensor(num_samples, input_size).normal_()
        device = next(self.parameters()).device
        noise = noise.to(device)
        if cond_inputs is not None:
            cond_inputs = cond_inputs.to(device)
        samples = self.forward(noise, cond_inputs, mode='inverse')
        return samples


class BatchNormFlow(nn.Module):
    """ An implementation of a batch normalization layer from
    Density estimation using Real NVP
    (https://arxiv.org/abs/1605.08803).
    """

    def __init__(self, num_inputs, momentum=0.0, eps=1e-5):
        super(BatchNormFlow, self).__init__()

        self.log_gamma = nn.Parameter(torch.zeros(num_inputs))
        self.beta = nn.Parameter(torch.zeros(num_inputs))
        self.momentum = momentum
        self.eps = eps

        self.register_buffer('running_mean', torch.zeros(num_inputs))
        self.register_buffer('running_var', torch.ones(num_inputs))

    def forward(self, inputs, cond_inputs=None, mode='direct'):
        if mode == 'direct':
            if self.training:
                self.batch_mean = inputs.mean(0)
                self.batch_var = (
                                         inputs - self.batch_mean).pow(2).mean(0) + self.eps

                self.running_mean.mul_(self.momentum)
                self.running_var.mul_(self.momentum)

                self.running_mean.add_(self.batch_mean.data *
                                       (1 - self.momentum))
                self.running_var.add_(self.batch_var.data *
                                      (1 - self.momentum))

                mean = self.batch_mean
                var = self.batch_var
            else:
                mean = self.running_mean
                var = self.running_var

            x_hat = (inputs - mean) / var.sqrt()
            y = torch.exp(self.log_gamma) * x_hat + self.beta
            return y, (self.log_gamma - 0.5 * torch.log(var))  # .sum(-1, keepdim=True)
        else:
            if self.training:
                mean = self.batch_mean
                var = self.batch_var
            else:
                mean = self.running_mean
                var = self.running_var

            x_hat = (inputs - self.beta) / torch.exp(self.log_gamma)
            y = x_hat * var.sqrt() + mean

            return y  # , (-self.log_gamma + 0.5 * torch.log(var)).sum(-1, keepdim=True)


class Reverse(nn.Module):
    """ An implementation of a reversing layer from
    Density estimation using Real NVP
    (https://arxiv.org/abs/1605.08803).
    """

    def __init__(self, num_inputs):
        super(Reverse, self).__init__()
        self.perm = np.array(np.arange(0, num_inputs)[::-1])
        self.inv_perm = np.argsort(self.perm)

    def forward(self, inputs, cond_inputs=None, mode='direct'):
        if mode == 'direct':
            return inputs[:, self.perm], torch.zeros(inputs.size(0), 625, device=inputs.device)
        else:
            return inputs[:, self.inv_perm]  # , torch.zeros(inputs.size(0), 1, device=inputs.device)
