import sys
# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1,2'
import argparse
import copy
import math
import pickle as pkl
import time
import numpy as np
import utils
from utils import load_data_LAGAN, lagan_disretized_loader

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from tqdm import tqdm
# from tensorboardX import SummaryWriter
import datasets
import mixflows as fnn
import multiscale_flows as multiscale
from scipy.stats import wasserstein_distance
import plot_utils




parser = argparse.ArgumentParser(description='Sparse Auto-regressive Flows')
parser.add_argument(
    '--batch-size',
    type=int,
    default=128,
    help='input batch size for training (default: 100)')
parser.add_argument(
    '--test-batch-size',
    type=int,
    default=1000,
    help='input batch size for testing (default: 1000)')
parser.add_argument(
    '--epochs',
    type=int,
    default=100,
    help='number of epochs to train (default: 1000)')
parser.add_argument(
    '--dataset',
    default='JetImages',
    help='POWER | GAS | HEPMASS | MINIBONE | BSDS300 | MOONS | JetImages')
parser.add_argument(
    '--no-cuda',
    action='store_true',
    default=False,
    help='disables CUDA training')
parser.add_argument(
    '--cond',
    action='store_true',
    default=False,
    help='train class conditional flow (only for MNIST)')
parser.add_argument(
    '--num-blocks',
    type=int,
    default=3,
    help='number of invertible blocks (default: 5)')
parser.add_argument(
    '--seed', type=int, default=1, help='random seed (default: 1)')
parser.add_argument(
    '--log-interval',
    type=int,
    default=2048,
    help='how many batches to wait before logging training status')
parser.add_argument(
    '--jet_images',
    action='store_true',
    default=True,
    help='whether to use the Jet images in LAGAN paper')
parser.add_argument(
    "--subset",
    type=str, default='signal',
    help="training on which subset"
    )
parser.add_argument(
    "--result_dir", type=str, default='/extra/yadongl10/BIG_sandbox/SparseImageFlows_result/lagan_pixelcnn/Mixture',
    help="result directory"
    )
parser.add_argument(
    "--activation", type=str, default='GeLU',
    help="activation"
    )
parser.add_argument(
    "--latent", type=int, default=5,
    help="number of latent layer in the flow"
    )
parser.add_argument(
    "--input_permute", type=str, default='spiral from center',
    help='type of permute: none, spiral from center',
    )
parser.add_argument(
    '--lr', type=float, default=1e-4, help='learning rate (default: 0.0001)')
parser.add_argument(
    '--flow', default='multiscale AR',
    help='flow to use: mixture-maf, multiscale AR, maf | realnvp | glow')


args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

kwargs = {'num_workers': 16, 'pin_memory': True} if args.cuda else {}


if args.jet_images == True:
    print('start to load data')
    # train_dataset = lagan_disretized_loader(subset='signal')
    # train_dataset = train_dataset.reshape(-1, 625)
    train_dataset = load_data_LAGAN(subset='signal')
    train_dataset = train_dataset.reshape(-1, 625)
    image_size = 25

    # train_dataset = load_jet_image(num=50000, signal=1)
    # train_dataset = train_dataset.reshape(-1, 1024)
    # image_size = 32

    if args.input_permute == 'spiral from center':
        train_dataset, ind = utils.vector_spiral_perm(train_dataset, dim=image_size)
    print('data_shape', train_dataset.shape)
    num_cond_inputs = None


train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)



num_inputs = train_dataset.shape[1]  # dataset.n_dims
num_hidden = {
    'MNIST': 1024,
    'JetImages': 1024
}[args.dataset]


modules = []
assert args.flow in ['multiscale AR', 'mixture-maf', 'maf']
if args.flow == 'mixture-maf':
    modules += [fnn.MixtureNormalMADE(num_inputs, num_hidden, num_cond_inputs,
                                      act=args.activation, num_latent_layer=args.latent)]
    model = fnn.FlowSequential(*modules)
    # for _ in range(args.num_blocks):
    #     modules += [
    #         fnn.MADE(num_inputs, num_hidden, num_cond_inputs, act='sigmoid'),
    #         fnn.BatchNormFlow(num_inputs),
    #         fnn.Reverse(num_inputs)
    #     ]
    print('flow structure: {}'.format(modules))

elif args.flow == 'multiscale AR':
    window_area = 9
    num_hidden = [window_area*5, (625-window_area)*1]
    modules += [multiscale.MultiscaleAR(window_area, num_inputs, num_hidden, act=args.activation, num_latent_layer=args.latent)]
    model = multiscale.FlowSequential(*modules)
    print('model structure: {}'.format(modules))


for module in model.modules():
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight)
        if hasattr(module, 'bias') and module.bias is not None:
            nn.init.uniform_(module.bias, a=0.0, b=1.0)  # uniform init of bias to avoid zero in alpha
            # module.bias.data.fill_(0)

model = model.cuda()
model = nn.DataParallel(model)

if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    # model = nn.DataParallel(model)

optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0)
# optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-5, last_epoch=-1)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100], gamma=0.5, last_epoch=-1)  # 10, 20, 30
# writer = SummaryWriter(comment=args.flow + "_" + args.dataset)
global_step = 0



def train(epoch):
    # global global_step, writer
    model.train()
    train_loss = 0

    pbar = tqdm(total=len(train_loader.dataset))
    for batch_idx, data in enumerate(train_loader):
        if isinstance(data, list):
            if len(data) > 1:
                cond_data = data[1].float()
                cond_data = cond_data.to(device)
            else:
                cond_data = None

            data = data[0]
        data = data.cuda().float()
        optimizer.zero_grad()
        # loss = -model(data, mode='direct').mean()
        loss = -model.module.log_probs(data).mean()
        gamma = model.module._modules['0'].gamma


        if batch_idx % args.log_interval == 0:
            print('\n gamma min:{}, gamma max:{}, gamma mean:{}'.format(gamma.min(), gamma.max(), gamma.mean()))
            # print('log_std min:{}, max:{}, mean:{}'.format(log_std.min(), log_std.max(), log_std.mean()))
            # print('log_phi min:{}, max:{}, mean:{}'.format(log_phi.min(), log_phi.max(), log_phi.mean()))
            # print('log_denominator min:{}, max:{}, mean:{}'.format(log_denominator.min(), log_denominator.max(), log_denominator.mean()))

            # print('mu min:{}, mu max:{}, mu mean:{}'.format(mu.min(), mu.max(), mu.mean()))
            # print('u min:{}, max:{}, mean:{}'.format(u.min(), u.max(), u.mean()))
            # print('log_jacob min:{}, max:{}, mean:{}'.format(log_jacob.min(), log_jacob.max(), log_jacob.mean()))
        train_loss += loss.item()
        loss.backward()
        optimizer.step()

        pbar.update(data.size(0))
        pbar.set_description('Train, Log likelihood in nats: {:.6f}'.format(
            -train_loss / (batch_idx + 1)))
    pbar.close()
    scheduler.step()
        # writer.add_scalar('training/loss', loss.item(), global_step)
        # global_step += 1



def get_distance(image, samples, image_size):
    samples[samples < 0] = 0  # -samples_bg[samples_bg<0]
    pt_dist = wasserstein_distance(plot_utils.discrete_pt(image),
                                plot_utils.discrete_pt(np.asarray(samples.tolist()).reshape(-1, image_size, image_size)))
    mass_dist = wasserstein_distance(plot_utils.discrete_mass(image),
                               plot_utils.discrete_mass(np.asarray(samples.tolist()).reshape(-1, image_size, image_size)))
    print('Wasserstein distance Pt:', pt_dist)
    print('Wasserstein distance Mass:', mass_dist)
    return [pt_dist, mass_dist]


best_validation_loss = float('inf')
best_validation_epoch = 0
best_model = model
dist_list = []

if args.input_permute == 'spiral from center':
    inverse_ind = []
    for i in range(625):
        inverse_ind.append(np.where(ind==i))
    inverse_ind = np.asarray(inverse_ind).squeeze()

for epoch in range(args.epochs):
    print('\nEpoch: {}'.format(epoch))
    train(epoch)
    if epoch % 1 == 0:
        model.eval()
        print('start sampling')
        start = time.time()
        inputs = torch.tensor(train_dataset[:200, :]).cuda() #  torch.randn((200, 625)).cuda()
        samples = model.module.sample(inputs, input_size=image_size**2)
        eval_data = train_dataset[:samples.shape[0], :]
        if args.input_permute == 'spiral from center':
            samples = samples[:, inverse_ind]
            eval_data = eval_data[:, inverse_ind]

        duration = time.time() - start
        print('end sampling, duration:{}, max {}'.format(duration, samples.max()))
        print(samples.shape)
        #
        dist = get_distance(eval_data.reshape(-1, image_size, image_size),
                            samples.reshape(-1, image_size, image_size), image_size=image_size)
        with open(args.result_dir + '/distance_list.txt', 'a') as f:
            f.write(str(dist) + ', \n')


        if epoch % 5 == 0:
            torch.save(model.state_dict(), args.result_dir + '/model_{}.pt'.format(epoch))

            with open(args.result_dir + '/background_Mix_discretized_sample_{}.pkl'.format(epoch), 'wb') as f:
                pkl.dump(samples.tolist(), f)
                print('generated images saved!')

