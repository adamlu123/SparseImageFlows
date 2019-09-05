import argparse
import os
import pickle as pkl

import torch
from torch.autograd import Variable
from torch import optim
from torch.distributions import Normal
import torch.utils.data

from visualizations import plot_density, density_plots
from flow import NormalizingFlow, PlainGenerator, PlainDeconvGenerator, FlowGenerator
from losses import FreeEnergyBound, SparseCE, SparseCE_mixture_flow
import utils

import sys
PATH = '/home/baldig-projects/julian/sisr/muon'
sys.path.append(PATH)
from utils import load_data


def train(args, config, model, train_loader, optimizer, epoch, device, scheduler):
    print('Epoch: {}'.format(epoch))
    model.train()

    for iteration, data in enumerate(train_loader):
        data = data.type(torch.cuda.FloatTensor)#.to(device)
        data = data.view(config['batch_size'], 32*32)
        optimizer.zero_grad()

        noisesamples_pi = Normal(loc=0, scale=1).sample([config['batch_size'], 32]).cuda()
        noisesamples_beta = Normal(loc=0, scale=1).sample([config['batch_size'], 32*32]).cuda()
        pi, beta, std = model(noisesamples_pi, noisesamples_beta)

        loss = SparseCE_mixture_flow(pi, beta,std, data)
        loss.backward()
        optimizer.step()

        if iteration % args.log_interval == 0:
            print("Loss on iteration {}: {}".format(iteration , loss.tolist()))
    mean_pi = pi.view(config['batch_size'], 32, 32).mean(dim=0).data ###
    print(mean_pi[16,:])
    print(data.view(-1, 32, 32).mean(dim=0)[16,:])  ###
    scheduler.step()
    return model




def test(args, config, model, epoch):
    model.eval()
    numsamples = 10000
    noisesamples_pi = Normal(loc=0, scale=1).sample([numsamples, 32]).cuda()
    noisesamples_beta = Normal(loc=0, scale=1).sample([numsamples, 32 * 32]).cuda()
    pi, beta, std = model(noisesamples_pi, noisesamples_beta)
    img = utils.get_img_sample(pi, beta, std)

    if epoch % 10 == 0:
        density_plots(
                    img.tolist(),
                    directory=config['subset_dir'],
                    epoch=epoch,
                    flow_length=config['flow_length'],
                    config=config)

        # with open(args.result_dir + '/pi.txt', 'a') as f:
        #     f.write('\n epoch:{} \n'.format(epoch))
        #     f.write(str(pi.view(32,32).cpu().data.numpy()))
        # with open(args.result_dir + '/img_samples.txt', 'a') as f:
        #     f.write('\n epoch:{} \n'.format(epoch))
        #     f.write(str(img[0]))

        save_values = True
        if save_values:
            with open(config['subset_dir'] + '/img_samples_{}.pkl'.format(epoch), 'wb') as f:
                pkl.dump(img.tolist(), f)
            with open(config['subset_dir'] + '/pi_{}.pkl'.format(epoch), 'wb') as f:
                pkl.dump(pi.view(numsamples,32,32).cpu().data.numpy(), f)




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
        help="How many points to generate for one plot."
    )

    parser.add_argument(
        "--result_dir", type=str, default='/extra/yadongl10/BIG_sandbox/SparseImageFlows_result/mixtureflow',
        help="result_dir"
    )

    args = parser.parse_args()
    device = torch.device("cuda")
    torch.manual_seed(42)
    config = {
        "batch_size": 128,
        "epochs": 100,
        "initial_lr": 0.001,
        "lr_decay": 0.999,
        "flow_length": 16,
        "name": "planar",
        "subset": "signal"
    }

    if not os.path.isdir(args.result_dir):
        os.mkdir(args.result_dir)

    subset_dir = args.result_dir + '/' + config['subset']
    config['subset_dir'] = subset_dir
    if not os.path.isdir(subset_dir):
        os.mkdir(subset_dir)



    x_32, _, _ = load_data(config['subset'], dataset='train')  # [x, y, w]
    x_32 = x_32 #* 1e2

    print('data_shape', x_32.shape)
    train_loader = torch.utils.data.DataLoader(x_32, batch_size=config['batch_size'], num_workers=2, drop_last=True)
    model = FlowGenerator(base_dim=32, img_dim=32 * 32).to(device)

    optimizer = optim.SGD(model.parameters(), lr=config['initial_lr'], momentum=0.9)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20, 30])


    for epoch in range(1, config['epochs'] + 1):
        model = train(args, config, model, train_loader, optimizer, epoch, device, scheduler)
        test(args, config, model, epoch)



if __name__ == "__main__":
    main()


