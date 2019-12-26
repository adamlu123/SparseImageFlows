import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal, Uniform, Gamma, Bernoulli
import torchvision
import h5py
from scipy.stats import norm

def save_moons_plot(epoch, best_model, dataset):
    # generate some examples
    best_model.eval()
    with torch.no_grad():
        x_synth = best_model.sample(500).detach().cpu().numpy()

    fig = plt.figure()

    ax = fig.add_subplot(121)
    ax.plot(dataset.val.x[:, 0], dataset.val.x[:, 1], '.')
    ax.set_title('Real data')

    ax = fig.add_subplot(122)
    ax.plot(x_synth[:, 0], x_synth[:, 1], '.')
    ax.set_title('Synth data')

    try:
        os.makedirs('plots')
    except OSError:
        pass

    plt.savefig('plots/plot_{:03d}.png'.format(epoch))
    plt.close()


batch_size = 100
fixed_noise = torch.Tensor(batch_size, 28 * 28).normal_()
y = torch.arange(batch_size).unsqueeze(-1) % 10
y_onehot = torch.FloatTensor(batch_size, 10)
y_onehot.zero_()
y_onehot.scatter_(1, y, 1)


def save_images(epoch, best_model, cond):
    best_model.eval()
    with torch.no_grad():
        if cond:
            imgs = best_model.sample(batch_size, noise=fixed_noise, cond_inputs=y_onehot).detach().cpu()
        else:
            imgs = best_model.sample(batch_size, noise=fixed_noise).detach().cpu()

        imgs = torch.sigmoid(imgs.view(batch_size, 1, 28, 28))

    try:
        os.makedirs('images')
    except OSError:
        pass

    torchvision.utils.save_image(imgs, 'images/img_{:03d}.png'.format(epoch), nrow=10)


def load_data_LAGAN(subset='signal'):
    img_dir = "/baldig/physicsprojects/lagan"
    with h5py.File(img_dir+'/lagan-jet-images.hdf5', 'r') as f:
        image = np.asarray(f['image'][:])
        real_labels = np.asarray(f['signal'][:])  # 10000
    real_imagebg = image[real_labels == 0]
    real_imagesg = image[real_labels == 1]
    print(real_imagebg.shape, real_imagesg.shape)

    if subset == 'background':
        print('return background')
        return real_imagebg
    elif subset == 'signal':
        print('return signal')
        return real_imagesg
    elif subset == 'All':
        print('return all')
        return image

def lagan_disretized_loader(subset='concatenate'):
    img_dir = "/baldig/physicsprojects/lagan"
    with h5py.File(img_dir + '/discretized_lagan.h5', 'r') as f:
        image = np.asarray(f[subset][:])
    print('image shape', image.shape)
    return image


def load_jet_image(num=10000, signal=0):
    img_dir = '/baldig/physicsprojects/jetvision/data/download4/datasets/test_no_pile_5000000.h5'
    with h5py.File(img_dir, 'r') as f:
        image = np.asarray(f['features'][:num, :, :])
        real_labels = np.asarray(f['targets'][:num])

    real_imagebg = image[real_labels == 0]
    real_imagesg = image[real_labels == 1]
    print(real_imagebg.shape, real_imagesg.shape)

    if signal == 0:
        print('return background')
        return real_imagebg
    elif signal == 1:
        print('return signal')
        return real_imagesg
    elif signal == 'All':
        print('return all')
        return image


def gamma_log_prob(concentration, rate, value):
    value = torch.where(value > 0, value, value+1e-4)  # to avoid nan in torch.log(val)
    logprob = (concentration * torch.log(rate) +
            (concentration - 1) * torch.log(value) -
            rate * value - torch.lgamma(concentration))
    return logprob


def normal_log_prob(mu, log_std, value):
    return np.log(1/np.sqrt(2*np.pi)) - log_std - (mu-value)**2/((2*log_std.exp()).clamp(min=1e-5, max=1e5)**2)
    #.clamp(min=-10, max=10)


def get_condition(Z, U, c, V, d):
    condition1 = Z>-1/c
    condition2 = torch.log(U) < 0.5 * Z**2 + d - d*V + d*torch.log(V)
    condition = condition1 * condition2
    return condition

def MTsample(gamma, alpha, beta=1):
    """ To generate Gamma samples using Marsaglia and Tsang’s Method: https://dl.acm.org/citation.cfm?id=358414
    1. create alpha_mod > 1
    2. generate Gamma(alpha_mod, 1): processed_out
    3. when the location is alpha<1, multiply with U_alpha**(1/alpha): mod_out

    :param gamma: 0,1 prob
    :param alpha: shape: [batchsize]
    :param beta: 1
    :return: mod_out
    """
    alpha = alpha.detach()
    batch_size = 1000
    num_samples = 5
    alpha_mod = torch.where(alpha>1, alpha, alpha+1)
    U_alpha = Uniform(0, 1).sample([batch_size]).cuda()  # for each element of alpha sample 3 times to ensure at least one is accepted.

    d = (alpha_mod - 1 / 3).repeat(num_samples, 1).t()
    c = 1. / torch.sqrt(9. * d)
    Z = Normal(0, 1).sample([batch_size, num_samples]).cuda()
    U = Uniform(0, 1).sample([batch_size, num_samples]).cuda()
    V = (1 + c * Z) ** 3

    condition = get_condition(Z, U, c, V, d).type(torch.float)
    out = condition * d * V
    processed_out = torch.stack([out[p, :][out[p, :] > 0][0] for p in range(batch_size)]).squeeze()

    mod_out = torch.where(alpha > 1, processed_out, processed_out * U_alpha**(1/alpha))

    z = Bernoulli(probs=gamma).sample()
    samples = torch.where(z > 0., mod_out, torch.zeros_like(z))

    return samples


class MarsagliaTsampler(nn.Module):
    """
    Implement Marsaglia and Tsang’s method as a Gamma variable sampler: https://www.hongliangjie.com/2012/12/19/how-to-generate-gamma-random-variables/
    """
    def __init__(self, size):
        super().__init__()
        self.gamma_alpha = nn.Parameter(2.*torch.ones(size))

        self.size = size
    def forward(self, batch_size):
        self.alpha = torch.relu(self.gamma_alpha) + 1  # right now only for alpha > 1
        d = self.alpha - 1/3
        c = 1. / torch.sqrt(9. * d)
        Z = Normal(0, 1).sample([batch_size, self.size])
        U = Uniform(0, 1).sample([batch_size, self.size])
        V = (1+c*Z)**3

        condition = get_condition(Z, U, c, V, d).type(torch.float)
        out = condition * d*V
        processed_out = torch.stack([out[:,p][out[:,p]>0][:10] for p in range(self.size)], dim=0).t()
        # out = out[out>0]
        detached_gamma_alpha = self.alpha  #.detach()
        return processed_out, detached_gamma_alpha


def get_psi(mu, std):
    value = (-mu)/(std*np.sqrt(2)).clamp(min=-1e10, max=1e10)
    return 0.5 * (1 + 2*torch.sigmoid(2.5*value)-1)


# spiral permutation
def spiral_perm(A):
    """

    :param A: a square matrix of index
    :return: a list of numbers corresponding to spiral permutation
    """
    index_list = []
    num_cols = A.shape[0]
    for row in range(num_cols):
        index_list = index_list + list(A[row, row:num_cols - row])
        index_list = index_list + list(A[row + 1:num_cols - row, num_cols - row - 1])

        index_list = index_list + list(A[num_cols - row - 1, row:len(A) - row - 1][::-1])
        index_list = index_list + list(A[row + 1:len(A) - row - 1, row][::-1])
    return index_list


def vector_spiral_perm(data, dim, start='center'):
    """
    wrapper function for spiral_perm
    :param data:
    :param dim:
    :param start: the order of spiral
    :return: permutated data matrix
    """

    A = np.arange(data.shape[1]).reshape(dim, dim)
    perm_spiral = spiral_perm(A)
    if start == 'center':
        perm_spiral = perm_spiral[::-1]
    return data[:, perm_spiral], np.asarray(perm_spiral)


# reparameterizable truncated normal approximation:

def sigmoid(x):
    return 1/(1+torch.exp(-x))


def erf_approx(value):
    return 2 * sigmoid(2.5 * value) - 1


def standard_normal_cdf(value):
    return 0.5 * (1 + erf_approx(value / np.sqrt(2))).clamp(min=0, max=1)


def trucated_normal_log_prob(mu, sd, value):
    log_phi = -np.log(np.sqrt(2 * np.pi)) - (value - mu) ** 2 / (2 * sd ** 2)
    log_denominator = sd.log() + (1 - standard_normal_cdf(-mu / sd)).log()
    # phi = 1 / np.sqrt(2 * np.pi) * torch.exp(-(value - mu) ** 2 / (2 * sd ** 2))
    # denominator = sd * (1 - standard_normal_cdf(-mu / sd))
    return log_phi - log_denominator


# def truncated_normal_sample(mu, sigma, num_samples):
#     epsilon = np.random.uniform(0, 1, num_samples)
#     phi_a_bar = norm.cdf(-mu/sigma)
#     u = (1-phi_a_bar) * epsilon + phi_a_bar
#     x_bar = norm.ppf(u)
#     return sigma * x_bar + mu


def truncated_normal_sample(mu, sigma, num_samples):
    epsilon = Uniform(torch.tensor([0.0]), torch.tensor([1.0])).sample(torch.Size([num_samples])).squeeze()
    standard_norm = Normal(torch.tensor([0.0]), torch.tensor([1.0]))
    phi_a_bar = standard_norm.cdf(-mu/sigma)
    u = (1-phi_a_bar) * epsilon + phi_a_bar
    x_bar = standard_norm.icdf(u)
    return sigma * x_bar + mu




class ChiSquareTest(object):
    def __init__(self, bins, truth):
        self.bins = bins
        self.expectation = self.get_bin_count(truth, bins)

    def getChisquare(self, image):
        count = self.get_bin_count(image, self.bins)
        distance = 0
        for i in range(self.bins):
            for j in range(self.bins):
                if self.expectation[i,j] != 0.0:
                    distance += (count[i,j] - self.expectation[i,j])**2 / self.expectation[i,j]
        return distance

    def get_bin_count(self, image, bins):
        count = np.zeros((bins, bins))
        mass, pt = image[0], image[1]
        min_mass, max_mass, min_pt, max_pt = mass.min(), mass.max(), pt.min(), pt.max()
        bin_size_mass, bin_size_pt = (max_mass - min_mass) / bins + 1e-5, (max_pt - min_pt) / bins + 1e-5
        for i in range(image[0].shape[0]):
            id_mass, id_pt = int((mass[i] - min_mass) // bin_size_mass), int((pt[i] - min_pt) // bin_size_pt)
            count[id_mass, id_pt] += 1
        return count










