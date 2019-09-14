import argparse
import os
import pickle as pkl

import torch
from torch import nn
from torch import optim
from torch.distributions import Normal
import torch.utils.data

from visualizations import plot_density, density_plots, plot_histogram
# from flow import NormalizingFlow, PlainGenerator, PlainDeconvGenerator
from losses import FreeEnergyBound, SparseCE
from densities import p_z
import utils

import matplotlib.pyplot as plt
import numpy as np
import sys
PATH = '/home/baldig-projects/julian/sisr/muon'
sys.path.append(PATH)
from utils import load_data, load_data_LAGAN


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
        x = torch.exp(self.linear4(x))
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
        # pi = torch.max(torch.ones_like(pi), pi)
        beta = torch.exp(self.linear_beta(x_beta))
        # beta = torch.min(20*torch.ones_like(pi), beta)
        std = torch.exp(self.linear_std(x_beta))  #np.sqrt(0.5) * torch.ones_like(beta)#
        std = torch.min(0.5 * torch.ones_like(pi), std)

        return pi, beta, std


def train(args, config, model, train_loader, optimizer, epoch, device, scheduler):
    print('Epoch: {}'.format(epoch))
    model.train()

    for iteration, data in enumerate(train_loader):
        data = data.type(torch.cuda.FloatTensor)#.to(device)
        data = data.view(config['batch_size'], config['width']**2)
        optimizer.zero_grad()

        noisesamples = Normal(loc=0, scale=1).sample([config['batch_size']*2, 32]).cuda()
        # noisesamples = Normal(loc=0, scale=1).sample([config['batch_size'] * 2, 1, 16, 16]).cuda()
        pi, beta, std = model(noisesamples[:config['batch_size']],noisesamples[config['batch_size']:])

        loss = SparseCE()(pi, beta, std, data)
        loss.backward()
        optimizer.step()

        if iteration % args.log_interval == 0:
            print("Loss on iteration {}: {}".format(iteration , loss.tolist()))
    mean_pi = pi.view(config['batch_size'], config['width'], config['width']).mean(dim=0).data ###
    print('mean_pi', mean_pi[16,:])
    print('mean_data', data.view(-1, config['width'], config['width']).mean(dim=0)[16,:])  ###
    scheduler.step()
    return model




def test(args, config, model, epoch):
    model.eval()
    numsamples = 10000
    # noisesamples = Normal(loc=0, scale=1).sample([numsamples * 2, 1, 16, 16]).cuda()
    noisesamples = Normal(loc=0, scale=1).sample([numsamples*2, 32]).cuda()  # Variable(random_normal_samples(args.plot_points))
    pi, beta, std = model(noisesamples[:numsamples], noisesamples[numsamples:])
    # print('std', std)
    img = utils.get_img_sample(config, pi, beta, std)

    if epoch % 10 == 0:
        save_values = True
        if save_values:
            with open(config['subset_dir'] + '/img_samples_{}.pkl'.format(epoch), 'wb') as f:
                pkl.dump(img.tolist(), f)
            with open(config['subset_dir'] + '/pi_{}.pkl'.format(epoch), 'wb') as f:
                pkl.dump(pi.view(numsamples,32,32).cpu().data.numpy(), f)
                print('results saved on epoch {} !'.format(epoch))
            plot_histogram(img, dir=config['subset_dir'],epoch=epoch)





def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--log_interval", type=int, default=30,
        help="How frequenlty to print the training stats."
    )

    parser.add_argument(
        "--plot_interval", type=int, default=300,
        help="How frequenlty to plot samples from current distribution."
    )

    parser.add_argument(
        "--plot_points", type=int, default=1000,
        help="How many to points to generate for one plot."
    )

    parser.add_argument(
        "--result_dir", type=str, default='/extra/yadongl10/BIG_sandbox/SparseImageFlows_result/unscaled_plaingen',
        help="How many to points to generate for one plot."
    )

    args = parser.parse_args()
    device = torch.device("cuda")
    torch.manual_seed(42)
    config = {
        "batch_size": 32,
        "epochs": 200,
        "initial_lr": 0.01,
        "lr_decay": 0.999,
        "flow_length": 16,
        "name": "planar",
        "subset": "background",
        "width": 32
    }

    if not os.path.isdir(args.result_dir):
        os.mkdir(args.result_dir)

    subset_dir = args.result_dir + '/' + config['subset']
    config['subset_dir'] = subset_dir
    if not os.path.isdir(subset_dir):
        print('create dir', subset_dir)
        os.mkdir(subset_dir)



    x_32, _ = load_data(config['subset'], dataset='train')  # [x, y, w]
    x_32 = x_32 #* 1e2

    # x_32 = load_data_LAGAN()

    print('data_shape', x_32.shape)
    # print(x_32.mean(dim=0)[16, :])
    train_loader = torch.utils.data.DataLoader(x_32, batch_size=config['batch_size'], num_workers=2, drop_last=True)
    model = PlainGenerator(base_dim=32, img_dim=config['width']**2).to(device)
    # model = PlainDeconvGenerator(base_dim=32, img_dim=32 * 32).to(device)

    optimizer = optim.SGD(model.parameters(), lr=config['initial_lr'], momentum=0.9)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20, 30])

    # plot_density(p_z, directory=args.result_dir)



    for epoch in range(1, config['epochs'] + 1):
        model = train(args, config, model, train_loader, optimizer, epoch, device, scheduler)
        test(args, config, model, epoch)



if __name__ == "__main__":
    main()


