import os

import numpy as np
import torch
from torch.autograd import Variable
from matplotlib import pyplot as plt

def plot_histogram(img, dir,epoch):
    _, bins, _ = plt.hist(img.mean(axis=0).ravel(),
                                      bins=np.linspace(0, 0.05, 50), histtype='step', label='truth', color='blue')
    plt.title('mean histogram of generated image')
    plt.savefig(dir+'mean_hist_epoch{}.png'.format(epoch))
    plt.close()

def density_plots(img, directory, epoch, flow_length, config):
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111)
    ax.imshow(img[0])
    ax.set_title("Flow length: {}\n Samples on epoch #{}".format(flow_length, epoch))

    fig.savefig(os.path.join(directory, "PlainGenerator_{}_{}.png".format(config['subset'], epoch)))
    plt.close()

    mean_img = np.mean(img, axis=0)
    plt.imshow(mean_img)
    plt.savefig(os.path.join(directory, "PlainGenerator_mean_{}_{}.png".format(config['subset'], epoch)))


def plot_density(density, directory):

    X_LIMS = (-7, 7)
    Y_LIMS = (-7, 7)

    x1 = np.linspace(*X_LIMS, 300)
    x2 = np.linspace(*Y_LIMS, 300)
    x1, x2 = np.meshgrid(x1, x2)
    shape = x1.shape
    x1 = x1.ravel()
    x2 = x2.ravel()

    z = np.c_[x1, x2]
    z = torch.FloatTensor(z)
    z = Variable(z)

    density_values = density(z).data.numpy().reshape(shape)

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111)
    ax.imshow(density_values, extent=(*X_LIMS, *Y_LIMS), cmap="summer")
    ax.set_title("True density")

    fig.savefig(os.path.join(directory, "density.png"))
    plt.close()