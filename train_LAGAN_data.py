import argparse
import os
import pickle as pkl

import torch
from torch.autograd import Variable
from torch import optim
from torch.distributions import Normal
import torch.utils.data

from visualizations import plot_density, density_plots
from flow import PlainGenerator, PlainDeconvGenerator, PixelwiseGenerator, JointConvGenerator, JointLinearGenerator
from losses import FreeEnergyBound, SparseCE
from densities import p_z
import utils
import numpy as np
import sys
PATH = '/home/baldig-projects/julian/sisr/muon'
sys.path.append(PATH)
from utils import load_data, load_data_LAGAN


def train(args, config, model, train_loader, optimizer, epoch, device, scheduler):
    print('Epoch: {}'.format(epoch))
    model.train()

    for iteration, data in enumerate(train_loader):
        data = data.type(torch.cuda.FloatTensor)#.to(device)
        data = data.view(config['batch_size'], config['width']**2)
        # data = data.view(config['batch_size'], config['width'],config['width'])
        optimizer.zero_grad()

        noisesamples = Normal(loc=0, scale=1).sample([config['batch_size'], 25**2]).cuda()
        # noisesamples_beta = Normal(loc=0, scale=1).sample([config['batch_size'], 1,
        #                                               config['width'], config['width']]).cuda()
        pi, beta, std = model(noisesamples)
        beta = beta.view(config['batch_size'], -1)
        std = std.view(config['batch_size'], -1)
        pi = pi.view(config['batch_size'], -1)
        # pi, beta, std = model(noisesamples[:config['batch_size']],noisesamples[config['batch_size']:])

        loss = SparseCE()(pi, beta, std, data)
        loss.backward()
        optimizer.step()

        if iteration % args.log_interval == 0:
            print("Loss on iteration {}: {}, beta.max: {}, beta.mean:{} pi.min:{}, pi.max{}".format(iteration , loss.tolist(),
                                                                                   beta.max().tolist(), beta.mean().tolist(),
                                                                                   pi.min().tolist(),pi.max().tolist()))

    mean_pi = pi.view(config['batch_size'], config['width'], config['width']).mean(dim=0).data ###
    print('beta mean, max and min',beta.mean(), beta.max(), beta.min())
    print('data mean', data.mean())
    scheduler.step()
    return model




def test(args, config, model, epoch):
    model.eval()
    numsamples = 10000
    noisesamples = Normal(loc=0, scale=1).sample([numsamples, 25**2]).cuda()
    # noisesamples_beta = Normal(loc=0, scale=1).sample([numsamples, 1,
    #                                                    config['width'], config['width']]).cuda()
    pi, beta, std = model(noisesamples)

    img = utils.get_img_sample(config, pi, beta, std)
    print('std.max', std.max().tolist(), 'img.max', img.max())
    if epoch % config['save_result_intervel'] == 0:

        torch.save(model.state_dict(), args.result_dir+'/best_checkpoints.pt')
        print('model saved!')
        save_values = False
        if save_values:
            with open(args.result_dir + '/img_samples_{}.pkl'.format(epoch), 'wb')  as f:
                pkl.dump(img.tolist(), f)
            with open(args.result_dir + '/pi_{}.pkl'.format(epoch), 'wb') as f:
                pkl.dump(pi.view(numsamples, config['width'], config['width']).cpu().data.numpy(), f)




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
        "--result_dir", type=str, default='/extra/yadongl10/BIG_sandbox/SparseImageFlows_result/LAGAN_JointConvGen/background',
        help="How many to points to generate for one plot."
    )
    parser.add_argument(
        "--subset", type=str, default='background', help="training on which subset"
    )


    args = parser.parse_args()
    device = torch.device("cuda")
    torch.manual_seed(42)
    config = {
        "batch_size": 256,
        "epochs": 30,
        "initial_lr": 0.01,
        "lr_decay": 0.999,
        "flow_length": 16,
        "name": "planar",
        "width": 25,
        "save_result_intervel": 1
    }

    if not os.path.isdir(args.result_dir):
        print('create dir', args.result_dir)
        os.mkdir(args.result_dir)

    print('start to build model')
    # model = PlainDeconvGenerator(base_dim=32, img_dim=32 * 32).to(device)
    # model = PlainGenerator(base_dim=32, img_dim=config['width']**2).to(device)
    # model = PixelwiseGenerator(base_dim=32, img_dim=config['width'] ** 2).to(device)
    model = JointLinearGenerator(base_dim=25**2, img_dim=config['width'] ** 2).to(device)

    print('start to load data')
    raw_img = load_data_LAGAN(subset=args.subset)
    # log_img = np.zeros_like(raw_img)
    # log_img[raw_img>0] = np.log(raw_img[raw_img>0])
    print('data_shape', raw_img.shape)
    train_loader = torch.utils.data.DataLoader(raw_img, batch_size=config['batch_size'], num_workers=2, drop_last=True)


    for key, param in model.named_parameters():
        print(key)

    optimizer = optim.SGD(model.parameters(), lr=config['initial_lr'], momentum=0.9)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20, 30])


    for epoch in range(1, config['epochs'] + 1):
        model = train(args, config, model, train_loader, optimizer, epoch, device, scheduler)
        test(args, config, model, epoch)



if __name__ == "__main__":
    main()


