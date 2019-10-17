import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal, Uniform, Gamma, Bernoulli
import torchvision
import h5py

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
        real_labels = np.asarray(f['signal'][:])
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


def gamma_log_prob(concentration, rate, value):
    value = torch.where(value > 0, value, value+1e-4)  # to avoid nan in torch.log(val)
    logprob = (concentration * torch.log(rate) +
            (concentration - 1) * torch.log(value) -
            rate * value - torch.lgamma(concentration))
    return logprob

def normal_log_prob(mu, sd, value):
    return np.log(1/np.sqrt(2*np.pi)) - sd.log() - (mu-value)**2/2*sd**2




def get_condition(Z, U, c, V, d):
    condition1 = Z>-1/c
    condition2 = torch.log(U) < 0.5 * Z**2 + d - d*V + d*torch.log(V)
    condition = condition1 * condition2
    return condition

def MTsample(alpha, beta=1):
    """ To generate Gamma samples using Marsaglia and Tsang’s Method: https://dl.acm.org/citation.cfm?id=358414
    1. create alpha_mod > 1
    2. generate Gamma(alpha_mod, 1): processed_out
    3. when the location is alpha<1, multiply with U_alpha**(1/alpha): mod_out

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

    return mod_out


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