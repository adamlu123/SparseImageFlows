import torch
from torch.distributions import Normal, Bernoulli
import h5py
import numpy as np

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
    binaries = Bernoulli(pi).sample()
    betas = Normal(loc=beta, scale=std).sample()
    img = (binaries * betas).view(-1,config['width'],config['width']).cpu().data.numpy()
    img[img<0]=0
    return img

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

    if subset == 'bg':
        print('loading bg')
        indices = get_shuffled_indices(x1.shape[0])
        return [x1[indices, :, :], y1[indices]]
        # return [x1[indices, :, :], y1[indices], w1[indices]]
    elif subset == 'signal':
        print('signal')
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



def load_data_LAGAN():
    img_dir = "/baldig/physicsprojects/lagan"
    with h5py.File(img_dir+'/lagan-jet-images.hdf5', 'r') as f:
        image = np.asarray(f['image'])
    # image[image>0] = np.log(image[image>0])
    return image
