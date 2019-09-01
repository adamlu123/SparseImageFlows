import argparse
import os

import torch
from torch.autograd import Variable
from torch import optim
from torch.distributions import Normal
import torch.utils.data

from visualizations import plot_density, density_plots
from flow import NormalizingFlow, PlainGenerator
from losses import FreeEnergyBound, SparseCE
from densities import p_z
import utils

import numpy as np
import sys
PATH = '/home/baldig-projects/julian/sisr/muon'
sys.path.append(PATH)
from data_loader import load_data



def is_log(iteration):
    return iteration % args.log_interval == 0

def is_plot(iteration):
    return iteration % args.plot_interval == 0


def train(args, config, model, train_loader, optimizer, epoch, device, scheduler):
    print('Epoch: {}'.format(epoch))
    model.train()

    for iteration, data in enumerate(train_loader):
        data = data.to(device)
        data = data.view(config['batch_size'], -1)
        optimizer.zero_grad()

        noisesamples = Normal(loc=0, scale=1).sample([config['batch_size'], 32]).cuda()
        pi, beta = model(noisesamples)

        loss = SparseCE()(pi, beta, data)
        loss.backward()
        optimizer.step()

        if iteration % args.log_interval == 0:
            print("Loss on iteration {}: {}".format(iteration , loss.tolist()))

    scheduler.step()



def test(args, config, model, epoch):
    model.eval()
    noisesamples = Normal(loc=0, scale=1).sample([1, 32]).cuda()  # Variable(random_normal_samples(args.plot_points))
    pi, beta = model(noisesamples)
    img = utils.get_img_sample(pi, beta)

    density_plots(
        img.tolist(),
        directory=args.result_dir,
        epoch=epoch,
        flow_length=config['flow_length']
    )

    with



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
        "--result_dir", type=str, default='/extra/yadongl10/BIG_sandbox/SparseImageFlows_result/exp_test',
        help="How many to points to generate for one plot."
    )

    args = parser.parse_args()
    device = torch.device("cuda")
    torch.manual_seed(42)

    if not os.path.isdir(args.result_dir):
        os.mkdir(args.result_dir)

    config = {
        "batch_size": 128,
        "epochs": 100,
        "initial_lr": 0.001,
        "lr_decay": 0.999,
        "flow_length": 16,
        "name": "planar"
    }

    x_32, _, _ = load_data(name='low', dataset='train')  # [x, y, w]
    print('data_shape', x_32.shape)

    train_loader = torch.utils.data.DataLoader(x_32, batch_size=config['batch_size'], num_workers=2, drop_last=True)
    model = PlainGenerator(base_dim=32, img_dim=32*32).to(device)

    optimizer = optim.SGD(model.parameters(), lr=config['initial_lr'], momentum=0.9)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20, 30])

    plot_density(p_z, directory=args.result_dir)



    for epoch in range(1, config['epochs'] + 1):
        train(args, config, model, train_loader, optimizer, epoch, device, scheduler)
        test(args, config, model, epoch)



if __name__ == "__main__":
    main()


