import torch
from torch.distributions import Normal, Bernoulli
import h5py
import numpy as np
from scipy.stats import wasserstein_distance
from Notebooks import plot_utils


def safe_log(z):
    return torch.log(z + 1e-7)


def random_normal_samples(n, dim=2):
    return torch.zeros(n, dim).normal_(mean=0, std=1)


def load_batch_x(x, batchsize=256, start=0, stop=None):
    iexample = 0
    while True:
        batch = slice(iexample, iexample + batchsize)
        batch_x = x[batch, :]
        yield batch_x
        iexample += batchsize
        if iexample + batchsize >= x.shape[0]:
            iexample = 0


def get_img_sample(config, pi, beta, std):
    binaries = Bernoulli(pi).sample().view(-1,config['width'],config['width'])
    binaries = binaries.view(-1,config['width'],config['width'])

    zeros = torch.zeros_like(binaries)
    betas = torch.max(zeros, Normal(loc=beta, scale=std).sample().squeeze().view(-1,config['width'],config['width']))
    generated = (binaries * betas).view(-1,config['width'],config['width']).cpu().data.numpy()

    img = np.zeros_like(generated)
    binaries = binaries.cpu().data.numpy()
    img[binaries>0] = np.exp(generated[binaries>0]) # scale it back
    return generated

def get_shuffled_indices(num_samples):
    indices = list(range(num_samples))
    import random
    random.seed(5)
    random.shuffle(indices)
    return indices


def load_data(subset, dataset='train'):
    load_path = '/baldig/physicsprojects/muon/unscaled_unrotated_images/'
    # load_path = '/home/baldig-projects/julian/sisr/muon/data/'
    with h5py.File(load_path + "images_bg.h5", "r") as hf:
        x1 = hf.get('low/%s' % dataset)[:].squeeze()
        # w1 = hf.get('weights/%s' % dataset)[:]
        # x1 = np.outer(w1, np.ones([32, 32])).reshape(w1.shape[0], 32, 32) * x1
        y1 = np.zeros((x1.shape[0],))

    with h5py.File(load_path + 'images_signal.h5', "r") as hf:
        x2 = hf.get('low/%s' % dataset)[:].squeeze()
        # w2 = hf.get('weights/%s' % dataset)[:]
        # x2 = np.outer(w2, np.ones([32, 32])).reshape(w2.shape[0], 32, 32) * x2
        y2 = np.ones((x2.shape[0],))

    if subset == 'background':
        print('loading background image!')
        indices = get_shuffled_indices(x1.shape[0])
        return [x1[indices, :, :], y1[indices]]
        # return [x1[indices, :, :], y1[indices], w1[indices]]
    elif subset == 'signal':
        print('loading signal image')
        indices = get_shuffled_indices(x2.shape[0])
        return [x2[indices, :, :], y2[indices]]
        # return [x2[indices, :, :], y2[indices], w2[indices]]
    elif subset == 'all':
        print('loading both bg and signal')
        x = np.vstack((x1, x2))
        y = np.hstack((y1, y2))
        # w = np.hstack((w1, w2))
        indices = get_shuffled_indices(x.shape[0])
        return [x[indices, :, :], y[indices]]
        # return [x[indices, :, :], y[indices], w[indices]]
    else:
        print('subset has to be bg/signal/all')



def load_data_LAGAN(subset='signal'):
    img_dir = "/baldig/physicsprojects/lagan"
    with h5py.File(img_dir+'/lagan-jet-images.hdf5', 'r') as f:
        image = np.asarray(f['image'])
        real_labels = np.asarray(f['signal'])
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


def load_LAGAN(num=10000, signal=1):
    img_dir = "/baldig/physicsprojects/lagan"
    with h5py.File(img_dir + '/lagan-jet-images.hdf5', 'r') as f:
        image = np.asarray(f['image'][:num, :, :])
        real_labels = np.asarray(f['signal'][:num])

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


def get_distance(image, samples):
    # image = load_LAGAN(num=10000, signal=1)[:1000]
    samples[samples < 0] = 0  # -samples_bg[samples_bg<0]
    print('Wasserstein distance Pt:',
          wasserstein_distance(plot_utils.discrete_pt(image),
                               plot_utils.discrete_pt(np.asarray(samples.tolist()).reshape(-1, 25, 25))))
    print('Wasserstein distance Mass:',
          wasserstein_distance(plot_utils.discrete_mass(image),
                               plot_utils.discrete_mass(np.asarray(samples.tolist()).reshape(-1, 25, 25))))
