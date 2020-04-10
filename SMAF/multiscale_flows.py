# By Yadong Lu, Dec 2019

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Gamma, Bernoulli, Uniform
import utils
from torch.distributions.exponential import Exponential

# inverse_ind = np.loadtxt('inverse_ind.txt')
# image_size = 25
# grid = 0.5 * (np.linspace(-1.25, 1.25, image_size+1)[:-1] + np.linspace(-1.25, 1.25, image_size+1)[1:])
# eta = torch.tensor(np.tile(grid, (image_size, 1)), dtype=torch.float32).cuda()
# phi = torch.tensor(np.tile(grid[::-1].reshape(-1, 1), (1, image_size)), dtype=torch.float32).cuda()

def discrete_mass(jet_image):
    '''
    Calculates the jet mass from a pixelated jet image
    Args:
    -----
        jet_image: numpy ndarray of dim (1, 25, 25)
    Returns:
    --------
        M: float, jet mass
    '''
    Px = torch.sum(jet_image * torch.cos(phi), dim=(1, 2))
    Py = torch.sum(jet_image * torch.sin(phi), dim=(1, 2))

    Pz = torch.sum(jet_image * torch.sinh(eta), dim=(1, 2))
    E = torch.sum(jet_image * torch.cosh(eta), dim=(1, 2))

    PT2 = Px**2 + Py**2
    M2 = E**2 - (PT2 + Pz**2)
    M = torch.sqrt(M2)
    return M

def discrete_pt(jet_image):
    '''
    Calculates the jet transverse momentum from a pixelated jet image
    Args:
    -----
        jet_image: numpy ndarray of dim (1, 25, 25)
    Returns:
    --------
        float, jet transverse momentum
    '''
    Px = torch.sum(jet_image * torch.cos(phi), dim=(1, 2))
    Py = torch.sum(jet_image * torch.sin(phi), dim=(1, 2))
    return torch.sqrt(Px**2 + Py**2)

def metric_loss(pred, inputs):
    """
    Implementation of squared error loss of metric (mass and pt), mean prediction and input image
    :param pred: shape=(batch, 277, 625)
    :param inputs:
    :return:
    """
    mean = (pred * torch.linspace(0, 276, 277).unsqueeze(1).repeat(1, pred.shape[2]).cuda()).mean(dim=1)  # shape=(batch, 1, 625)
    mean, inputs = mean[:, inverse_ind].reshape(-1, 25, 25), inputs[:, inverse_ind].reshape(-1, 25, 25)

    loss = (discrete_pt(mean) - discrete_pt(inputs))**2  # + (discrete_mass(mean) - discrete_mass(inputs))**2
    return 1e-10*loss


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


def create_mask(num_inputs, num_hidden, type, softmax_latent):
    input_mask = get_mask(num_inputs, num_hidden, num_inputs, mask_type='input')
    hidden_mask = get_mask(num_hidden, num_hidden, num_inputs)
    if type == 'masked softmax':
        output_mask = get_mask(num_hidden, num_inputs * (1+softmax_latent), num_inputs, mask_type='output')
    elif type == 'masked truncated normal' or type == 'masked reshaped normal':
        output_mask = get_mask(num_hidden, num_inputs * 3, num_inputs, mask_type='output')
    elif type == 'softmax':
        output_mask = get_mask(num_hidden, num_inputs * softmax_latent, num_inputs, mask_type='output')
    elif type == 'logistic':
        output_mask = get_mask(num_hidden, num_inputs * 3, num_inputs, mask_type='output')
    elif type == 'masked exponential':
        output_mask = get_mask(num_hidden, num_inputs * 2, num_inputs, mask_type='output')
    return input_mask, hidden_mask, output_mask


def create_joiner_trunk(num_inputs, num_hidden, num_cond_inputs, act_func, num_latent_layer,
                   input_mask, hidden_mask, output_mask, type, softmax_latent):
    joiner = nn.MaskedLinear(num_inputs, num_hidden, input_mask, num_cond_inputs)
    latent_modules = []
    # hidden layers (currently using the same mask as joiner)
    for i in range(num_latent_layer):
        latent_modules.append(act_func())
        latent_modules.append(nn.MaskedLinear(num_hidden, num_hidden, hidden_mask))
    # output layers
    if type == 'masked softmax':
        latent_modules.append(nn.MaskedLinear(num_hidden, num_inputs * (1+softmax_latent), output_mask))
    elif type == 'masked truncated normal' or type == 'masked reshaped normal':
        latent_modules.append(nn.MaskedLinear(num_hidden, num_inputs * 3, hidden_mask))  # TODO output_mask
    elif type == 'softmax':
        latent_modules.append(nn.MaskedLinear(num_hidden, num_inputs * softmax_latent, output_mask))
    elif type == 'logistic':
        latent_modules.append(nn.MaskedLinear(num_hidden, num_inputs * 3, output_mask))
    elif type == 'masked exponential':
        latent_modules.append(nn.MaskedLinear(num_hidden, num_inputs * 2, output_mask))

    trunk = nn.Sequential(*latent_modules)
    return joiner, trunk


def sample_logistic(mu, log_std):
    """

    :param mu: (batch, 10, 49)
    :param log_std:
    :return: (batch, 10, 49)
    """
    u = Uniform(0, 1).sample(mu.size()).cuda()
    return mu + log_std.exp() * torch.log(u/(1-u))


def sample_onehot(pi):
    """

    :param pi: (batch, 10, 49)
    :return: (batch, 10, 49)
    """
    probs = F.softmax(pi, dim=1)  # shape=(batch, 10)
    nonzeros = torch.distributions.OneHotCategorical(probs).sample().float()  # shape=(batchsize, 10)
    return nonzeros  # shape=(batch, 10)



class ARBase(nn.Module):
    """
    Base module for ARM, support 1. masked softmax 2. softmax 3. logistic 4. masked truncated normal
    """
    def __init__(self,
                 num_inputs,
                 num_hidden,
                 num_cond_inputs=None,
                 act=None,
                 num_latent_layer=2,
                 type=None,
                 inner=False,
                 softmax_latent=100):
        super(ARBase, self).__init__()
        activations = {'relu': nn.ReLU, 'sigmoid': nn.Sigmoid, 'tanh': nn.Tanh, 'GeLU': utils.GeLU}
        act_func = activations[act]
        self.type = type
        self.inner = inner
        self.softmax_latent = softmax_latent

        # create mask
        input_mask, hidden_mask, output_mask = create_mask(num_inputs, num_hidden, type, softmax_latent)

        # create sub modules: joiner and trunk
        self.joiner, self.trunk = create_joiner_trunk(num_inputs, num_hidden, num_cond_inputs, act_func, num_latent_layer,
                    input_mask, hidden_mask, output_mask, type, softmax_latent)

        # deconvolution to match the shape
        if 'softmax' in type:
            self.conv1d = nn.Sequential(
                # nn.Conv1d(in_channels=softmax_latent, out_channels=180, stride=1, kernel_size=1), nn.BatchNorm1d(180), nn.ReLU(),
                # nn.Conv1d(in_channels=90, out_channels=180, stride=1, kernel_size=1), nn.BatchNorm1d(180), nn.ReLU(),
                nn.Conv1d(in_channels=softmax_latent, out_channels=277, stride=1, kernel_size=1)
            )
            if 'mask' in type:
                self.conv1d_gamma = nn.Conv1d(in_channels=1, out_channels=1, stride=1, kernel_size=1)

        # elif type == 'masked truncated normal':
        #     self.conv1d_mu = nn.Sequential(
        #         nn.Conv1d(in_channels=1, out_channels=5, stride=1, kernel_size=1), nn.BatchNorm1d(5), nn.Tanh(),
        #         nn.Conv1d(in_channels=5, out_channels=1, stride=1, kernel_size=1)
        #     )
        #     self.conv1d_log_std = nn.Sequential(
        #         nn.Conv1d(in_channels=1, out_channels=5, stride=1, kernel_size=1), nn.BatchNorm1d(5), nn.Tanh(),
        #         nn.Conv1d(in_channels=5, out_channels=1, stride=1, kernel_size=1)
        #     )
        #     self.conv1d_gamma = nn.Sequential(
        #         nn.Conv1d(in_channels=1, out_channels=5, stride=1, kernel_size=1), nn.BatchNorm1d(5), nn.Tanh(),
        #         nn.Conv1d(in_channels=5, out_channels=1, stride=1, kernel_size=1)
        #     )


        elif 'logistic' in type:
            self.conv1d_pi = nn.Sequential(
                nn.Conv1d(in_channels=1, out_channels=5, stride=1, kernel_size=1)
            )
            self.conv1d_mu = nn.Sequential(
                nn.Conv1d(in_channels=1, out_channels=5, stride=1, kernel_size=1)
            )
            self.conv1d_sd = nn.Sequential(
                nn.Conv1d(in_channels=1, out_channels=5, stride=1, kernel_size=1)
            )




    def forward(self, inputs, cond_inputs=None, mode='direct', epoch=0):
        input_dim = inputs.shape[1]
        # inference
        if mode == 'direct':
            h = self.joiner(inputs, cond_inputs)
            if self.type == 'softmax':
                theta = self.trunk(h)
                # return theta.view(-1, self.softmax_latent, input_dim)
                return self.conv1d(theta.view(-1, self.softmax_latent, input_dim))  # shape=(batch, 277, 625)
            elif self.type == 'masked softmax':
                h = self.trunk(h)
                gamma, theta = h[:, :input_dim], h[:, input_dim:].reshape(-1, self.softmax_latent, input_dim)
                # gamma = self.conv1d_gamma(gamma.view(-1, 1, input_dim))
                gamma = gamma.unsqueeze(1)
                gamma = torch.sigmoid(gamma)
                pred = self.conv1d(theta.view(-1, self.softmax_latent, input_dim))  # shape=(batch, 277, 625)
                return gamma, pred
            elif self.type == 'masked truncated normal' or self.type == 'masked reshaped normal':
                gamma, mu, log_std = self.trunk(h).chunk(3, 1)
                mu = F.relu(mu)
                gamma = torch.sigmoid(gamma)
                return gamma, mu, log_std
            elif self.type == 'logistic':
                pi, mu, log_s = self.trunk(h).chunk(3, 1)
                pi = self.conv1d_pi(pi.view(-1, 1, input_dim))
                pi = F.softmax(pi, dim=1)
                mu = self.conv1d_mu(mu.view(-1, 1, input_dim))
                log_s = self.conv1d_sd(log_s.view(-1, 1, input_dim))
                return pi, mu, log_s

            elif self.type == 'masked exponential':
                gamma, log_lambda = self.trunk(h).chunk(2, 1)
                gamma = torch.sigmoid(gamma)
                return gamma, log_lambda

        # sampling
        else:
            x = torch.zeros_like(inputs).float()
            noise = torch.Tensor(inputs.size()).normal_().cuda()
            use_empirical = True
            with torch.no_grad():
                for i in range(input_dim):
                    if i == 0 and self.inner and use_empirical:
                        x[:, i] = inputs[:, i]
                    else:
                        h = self.joiner(x.view(-1, input_dim), cond_inputs)
                        if self.type == 'softmax':
                            theta = self.trunk(h)
                            out = self.conv1d(theta.view(-1, self.softmax_latent, input_dim))
                            probs = F.softmax(out[:, :, i], dim=1).data  # shape=(batchsize, 277)
                            nonzeros = torch.multinomial(probs, 1).float().view(-1)  # shape=(batchsize)
                            x[:, i] = nonzeros

                        elif self.type == 'masked softmax':
                            h = self.trunk(h)
                            gamma, theta = h[:, :input_dim], h[:, input_dim:].reshape(-1, self.softmax_latent, input_dim)
                            # gamma, theta = self.trunk(h).chunk(2, 1)
                            # gamma = self.conv1d_gamma(gamma.view(-1, 1, input_dim)).squeeze()
                            gamma = torch.sigmoid(gamma[:, i])
                            z = Bernoulli(probs=gamma).sample()
                            out = self.conv1d(theta.view(-1, self.softmax_latent, input_dim))
                            probs = F.softmax(out[:, :, i], dim=1)  # shape=(batchsize, 277)
                            nonzeros = torch.multinomial(probs, 1).view(-1)  # shape=(batchsize)
                            x[:, i] = torch.where(z > 0, nonzeros, torch.zeros_like(nonzeros))


                        elif self.type == 'masked truncated normal':
                            gamma, mu, log_std = self.trunk(h).chunk(3, 1)
                            log_std = log_std.clamp(min=np.log(1e-3), max=np.log(1e3))
                            gamma = torch.sigmoid(gamma[:, i])
                            # gamma = (gamma / (1 - utils.get_psi(mu[:, i], torch.exp(0.5 * log_std[:, i])))).clamp(max=0.999)
                            z = Bernoulli(probs=gamma).sample()
                            # below zero to noise
                            nonzeros = noise[:, i] * torch.exp(0.5*log_std[:, i]) + mu[:, i]
                            # unif_noise = Uniform(torch.zeros_like(nonzeros), torch.ones_like(nonzeros)).sample()
                            # nonzeros = torch.where(nonzeros>0, nonzeros, unif_noise)
                            x[:, i] = torch.where(z > 0, nonzeros, torch.zeros_like(nonzeros)).clamp(min=0)
                            # sparsity = (x[:, i]>0).float().mean()
                            # sparsity_true = (inputs[:, i]>0).float().mean()

                            # if i == 0 and self.inner:
                            #     x[:, i] = inputs[:, i]

                        elif self.type == 'masked reshaped normal':
                            gamma, mu, log_std = self.trunk(h).chunk(3, 1)
                            mu = F.relu(mu)
                            gamma = torch.sigmoid(gamma[:, i])
                            z = Bernoulli(probs=gamma).sample()
                            # if i == 563:
                            #     print(i, inputs.shape[0], mu.mean(), log_std.mean())
                            nonzeros = utils.truncated_normal_sample(mu=mu[:, i],
                                                                     sigma=log_std[:, i].clamp(min=np.log(1e-3), max=np.log(1e3)).exp(),
                                                                     num_samples=inputs.shape[0])
                            x[:, i] = torch.where(z > 0, nonzeros, torch.zeros_like(nonzeros))
                            # if i == 0 and inputs.shape[1] == 225:
                            #     x[:, i] = inputs[:, i]

                        elif self.type == 'logistic':
                            pi, mu, log_s = self.trunk(h).chunk(3, 1)
                            pi = self.conv1d_pi(pi.view(-1, 1, input_dim)).clamp(min=1e-5)   # shape=(batch, 10, 1)
                            mu = self.conv1d_mu(mu.view(-1, 1, input_dim))
                            log_s = self.conv1d_sd(log_s.view(-1, 1, input_dim)).clamp(min=np.log(1e-2), max=np.log(1e2))
                            pi, mu, log_s = pi[:, :, i], mu[:, :, i], log_s[:, :, i]
                            x[:, i] = (sample_onehot(pi) * sample_logistic(mu, log_s)).sum(dim=1)  # shape=(batch, 1)

                        elif self.type == 'masked exponential':
                            gamma, log_lambda = self.trunk(h).chunk(2, 1)
                            gamma = torch.sigmoid(gamma[:, i])
                            z = Bernoulli(probs=gamma).sample()
                            nonzeros = Exponential(rate=log_lambda[:, i].exp().clamp(min=np.log(1e-2), max=np.log(1e2))).sample()
                            x[:, i] = torch.where(z > 0, nonzeros, torch.zeros_like(nonzeros))

            # x = torch.floor(x+0.5)
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
                 num_latent_layer=2,
                 type ='softmax'):  # logistic, masked softmax, masked truncated normal, softmax, masked reshaped normal, masked exponential, mixed
        super(MultiscaleAR, self).__init__()

        self.ARinner = ARBase(window_area, num_hidden[0], None, act, num_latent_layer, type=type, inner=True)
        self.ARouter = ARBase(num_inputs-window_area, num_hidden[1], window_area, act, num_latent_layer, type=type)
        self.window_area = window_area
        self.type = type

    def forward(self, inputs, mode='direct'):
        # inference
        if mode == 'direct':
            if self.type == 'masked softmax':
                gamma_i, pred_i = self.ARinner(inputs[:, :self.window_area], mode='direct')
                # inner_mean = gamma_i.squeeze() * \
                #              (pred_i*torch.linspace(0, 276, 277).unsqueeze(1).repeat(1, gamma_i.shape[2]).cuda()).mean(dim=1)  # shape=(batch, 277, 49)
                inner_mean = inputs[:, :self.window_area]
                gamma_o, pred_o = self.ARouter(inputs[:, self.window_area:], cond_inputs=inner_mean, mode='direct')
                gamma, pred = torch.cat([gamma_i, gamma_o], -1), torch.cat([pred_i, pred_o], -1)
                self.gamma = gamma.detach().cpu().numpy()
                nll_positive = F.cross_entropy(pred.view(-1, 277, 625), inputs.long(),
                                               reduction="none")  # shape=(batchsize, 625)
                ll = torch.where(inputs > 0,
                                 (gamma + 1e-10).log() - nll_positive,
                                 (1 - gamma + 1e-10).log()).sum(dim=-1, keepdim=True)

            elif self.type == 'softmax':
                pred_i = self.ARinner(inputs[:, :self.window_area], mode='direct')
                inner_mean = inputs[:, :self.window_area]
                # inner_mean = (pred_i*torch.linspace(0, 276, 277).unsqueeze(1).repeat(1, pred_i.shape[2]).cuda()).mean(dim=1) # shape=(batch, 277, 49)
                pred_o = self.ARouter(inputs[:, self.window_area:], cond_inputs=inner_mean, mode='direct')
                pred = torch.cat([pred_i, pred_o], -1)
                self.gamma = torch.tensor([0.])
                nll_positive = F.cross_entropy(pred.view(-1, 277, 625), inputs.long(), reduction="none")  # shape=(batchsize, 625)
                ll = -nll_positive.mean(dim=1)  # - metric_loss(pred, inputs)

            elif self.type == 'masked truncated normal':
                gamma_i, mu_i, log_std_i = self.ARinner(inputs[:, :self.window_area], mode='direct')
                # inner_mean = gamma_i * mu_i  # TODO 1: think about how to add std_i into cond input
                inner_mean = inputs[:, :self.window_area]
                gamma_o, mu_o, log_std_o = self.ARouter(inputs[:, self.window_area:], cond_inputs=inner_mean, mode='direct')

                gamma = torch.cat([gamma_i, gamma_o], -1)

                mu = torch.cat([mu_i, mu_o], -1)
                log_std = torch.cat([log_std_i, log_std_o], -1)

                log_std = log_std.clamp(min=np.log(1e-3), max=np.log(1e3))
                # gamma = (1 - gamma) * utils.get_psi(mu, torch.exp(0.5*log_std)) + gamma  # here gamma = p(z=0)
                gamma = (gamma / (1 - utils.get_psi(mu, torch.exp(0.5*log_std)))).clamp(max=0.999)
                self.gamma = gamma
                ll = torch.where(inputs > 0,
                                 (gamma + 1e-10).log() + utils.normal_log_prob(mu=mu, log_std=0.5*log_std, value=inputs), # -0.05*mu**2
                                 (1-gamma + 1e-10).log()).sum(dim=-1, keepdim=True)

            elif self.type == 'masked reshaped normal':
                gamma_i, mu_i, log_std_i = self.ARinner(inputs[:, :self.window_area], mode='direct')
                inner_mean = inputs[:, :self.window_area]
                gamma_o, mu_o, log_std_o = self.ARouter(inputs[:, self.window_area:], cond_inputs=inner_mean, mode='direct')
                gamma = torch.cat([gamma_i, gamma_o], -1)
                mu = torch.cat([mu_i, mu_o], -1)
                log_std = torch.cat([log_std_i, log_std_o], -1).clamp(min=np.log(1e-3), max=np.log(1e2))
                self.log_std = log_std
                self.gamma = gamma
                self.ll_trunc_norm, self.log_phi, self.log_denominator = utils.trucated_normal_log_prob_stable(mu=mu, log_std=log_std, value=inputs)
                ll = torch.where(inputs > 0,
                                 (1 - gamma + 1e-10).log() + self.ll_trunc_norm,
                                 (gamma + 1e-10).log()).sum(dim=-1, keepdim=True)  # (128, 1)

            elif self.type == 'logistic':
                pi_i, mu_i, log_s_i = self.ARinner(inputs[:, :self.window_area], mode='direct')  # shape=(batch, 10, 49)
                inner_mean = inputs[:, :self.window_area]
                # inner_mean = (pi_i * mu_i).mean(dim=1)
                pi_o, mu_o, log_s_o = self.ARouter(inputs[:, self.window_area:], cond_inputs=inner_mean, mode='direct')

                pi = torch.cat([pi_i, pi_o], -1)
                mu = torch.cat([mu_i, mu_o], -1)
                s = torch.cat([log_s_i, log_s_o], -1).exp()

                inputs = inputs.unsqueeze(1).repeat(1, pi.shape[1], 1)  # shape=(batch, 10, 625)
                nonzeros = pi*(torch.sigmoid((inputs+0.5-mu)/s) - torch.sigmoid((inputs-0.5-mu)/s))
                zeros = pi*torch.sigmoid((inputs+0.5-mu)/s)
                ll = torch.where(inputs > 0, nonzeros.log(), zeros.log())
                ll = ll.sum(dim=[1, 2])
                self.gamma = torch.tensor([0.])

            elif self.type == 'masked exponential':
                gamma_i, log_lambda_i = self.ARinner(inputs[:, :self.window_area], mode='direct')
                inner_mean = inputs[:, :self.window_area]
                gamma_o, log_lambda_o = self.ARouter(inputs[:, self.window_area:], cond_inputs=inner_mean,
                                                        mode='direct')
                gamma = torch.cat([gamma_i, gamma_o], -1)
                log_lambda = torch.cat([log_lambda_i, log_lambda_o], -1).clamp(min=np.log(1e-3), max=np.log(1e3))

                self.gamma = gamma
                ll = torch.where(inputs > 0,
                                 (1 - gamma + 1e-10).log() + log_lambda - log_lambda.exp()*inputs,
                                 (gamma + 1e-10).log()).sum(dim=-1, keepdim=True)

            elif self.type == 'softmax + masked truncated normal':
                pred = self.ARinner(inputs[:, :self.window_area], mode='direct')
                nll_positive = F.cross_entropy(pred.view(-1, 277, self.window_area), inputs[:, :self.window_area].long(),
                                               reduction="none")  # shape=(batchsize, 625)
                ll_i = -nll_positive.sum(dim=1)


                inner_mean = inputs[:, :self.window_area]
                gamma, mu, log_std = self.ARouter(inputs[:, self.window_area:], cond_inputs=inner_mean,
                                                        mode='direct')
                log_std = log_std.clamp(min=np.log(1e-3), max=np.log(1e3))
                gamma = (1 - gamma) * utils.get_psi(mu, torch.exp(0.5 * log_std)) + gamma  # here gamma = p(z=0)
                self.gamma = gamma
                ll_o = torch.where(inputs[:, self.window_area:] > 0,
                                 (gamma + 1e-10).log() + utils.normal_log_prob(mu=mu, log_std=0.5 * log_std,
                                                                               value=inputs[:, self.window_area:]),  # -0.05*mu**2
                                 (1 - gamma + 1e-10).log()).sum(dim=-1, keepdim=True)
                ll = ll_i + ll_o
            return ll

        # sampling
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

    def sample(self, inputs, num_samples=None, noise=None, cond_inputs=None, input_size=1024):
        num_samples = inputs.shape[0]
        if noise is None:
            noise = torch.Tensor(num_samples, input_size).normal_()
        device = next(self.parameters()).device
        noise = noise.to(device)
        if cond_inputs is not None:
            cond_inputs = cond_inputs.to(device)
        samples = self.forward(inputs, cond_inputs, mode='inverse')
        return samples
