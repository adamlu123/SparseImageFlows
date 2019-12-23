import argparse
import copy
import math
import sys
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1,2'
import pickle as pkl
import time
import numpy as np
import utils
from utils import load_data_LAGAN, load_jet_image, lagan_disretized_loader

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



if sys.version_info < (3, 6):
    print('Sorry, this code might need Python 3.6 or higher')

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Flows')
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
    default=1000,
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
    default=2,
    help='number of invertible blocks (default: 5)')
parser.add_argument(
    '--seed', type=int, default=1, help='random seed (default: 1)')
parser.add_argument(
    '--log-interval',
    type=int,
    default=4096,
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
    "--activation", type=str, default='relu',
    help="activation"
    )
parser.add_argument(
    "--latent", type=int, default=2,
    help="number of latent layer in the flow"
    )
parser.add_argument(
    "--input_permute", type=str, default='spiral from center',
    help='type of permute: none, spiral from center',
    )
parser.add_argument(
    '--lr', type=float, default=0.0001, help='learning rate (default: 0.0001)')
parser.add_argument(
    '--flow', default='multiscale AR',
    help='flow to use: mixture-maf, multiscale AR, maf | realnvp | glow')


args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

kwargs = {'num_workers': 4, 'pin_memory': True} if args.cuda else {}


if args.jet_images == True:
    print('start to load data')
    train_dataset = lagan_disretized_loader(subset='signal')
    train_dataset = train_dataset.reshape(-1, 625)
    # train_dataset = load_data_LAGAN(subset='signal')
    # train_dataset = train_dataset.reshape(-1, 625)
    image_size = 25

    # train_dataset = load_jet_image(num=50000, signal=1)
    # train_dataset = train_dataset.reshape(-1, 1024)
    # image_size = 32
    if args.input_permute == 'spiral from center':
        train_dataset, ind = utils.vector_spiral_perm(train_dataset, dim=image_size)
    print('data_shape', train_dataset.shape)
    num_cond_inputs = None

# else:
#     assert args.dataset in [
#         'POWER', 'GAS', 'HEPMASS', 'MINIBONE', 'BSDS300', 'MOONS', 'MNIST'
#     ]
#     dataset = getattr(datasets, args.dataset)()
#
#     if args.cond:
#         assert args.flow in ['maf', 'realnvp'] and args.dataset == 'MNIST', \
#             'Conditional flows are implemented only for maf and MNIST'
#
#         train_tensor = torch.from_numpy(dataset.trn.x)
#         train_labels = torch.from_numpy(dataset.trn.y)
#         train_dataset = torch.utils.data.TensorDataset(train_tensor, train_labels)
#
#         valid_tensor = torch.from_numpy(dataset.val.x)
#         valid_labels = torch.from_numpy(dataset.val.y)
#         valid_dataset = torch.utils.data.TensorDataset(valid_tensor, valid_labels)
#
#         test_tensor = torch.from_numpy(dataset.tst.x)
#         test_labels = torch.from_numpy(dataset.tst.y)
#         test_dataset = torch.utils.data.TensorDataset(test_tensor, test_labels)
#         num_cond_inputs = 10
#     else:
#         train_tensor = torch.from_numpy(dataset.trn.x)
#         train_dataset = torch.utils.data.TensorDataset(train_tensor)
#
#         valid_tensor = torch.from_numpy(dataset.val.x)
#         valid_dataset = torch.utils.data.TensorDataset(valid_tensor)
#
#         test_tensor = torch.from_numpy(dataset.tst.x)
#         test_dataset = torch.utils.data.TensorDataset(test_tensor)
#         num_cond_inputs = None

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)

valid_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=args.test_batch_size,
    shuffle=False,
    drop_last=False,
    **kwargs)

test_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=args.test_batch_size,
    shuffle=False,
    drop_last=False,
    **kwargs)

num_inputs = train_dataset.shape[1]  # dataset.n_dims
num_hidden = {
    'POWER': 100,
    'GAS': 100,
    'HEPMASS': 512,
    'MINIBOONE': 512,
    'BSDS300': 512,
    'MOONS': 64,
    'MNIST': 1024,
    'JetImages': 625
}[args.dataset]

act = 'tanh' if args.dataset is 'GAS' else 'relu'

modules = []

assert args.flow in ['multiscale AR', 'mixture-maf', 'maf', 'maf-split', 'maf-split-glow', 'realnvp', 'glow']
if args.flow == 'glow':
    mask = torch.arange(0, num_inputs) % 2
    mask = mask.to(device).float()

    print("Warning: Results for GLOW are not as good as for MAF yet.")
    for _ in range(args.num_blocks):
        modules += [
            fnn.BatchNormFlow(num_inputs),
            fnn.LUInvertibleMM(num_inputs),
            fnn.CouplingLayer(
                num_inputs, num_hidden, mask, num_cond_inputs,
                s_act='tanh', t_act='relu')
        ]
    mask = 1 - mask
elif args.flow == 'realnvp':
    mask = torch.arange(0, num_inputs) % 2
    mask = mask.to(device).float()

    for _ in range(args.num_blocks):
        modules += [
            fnn.CouplingLayer(
                num_inputs, num_hidden, mask, num_cond_inputs,
                s_act='tanh', t_act='relu'),
            fnn.BatchNormFlow(num_inputs)
        ]
        mask = 1 - mask
elif args.flow == 'maf':
    for _ in range(args.num_blocks):
        modules += [
            fnn.MADE(num_inputs, num_hidden, num_cond_inputs, act=act),
            fnn.BatchNormFlow(num_inputs),
            fnn.Reverse(num_inputs)
        ]
elif args.flow == 'maf-split':
    for _ in range(args.num_blocks):
        modules += [
            fnn.MADESplit(num_inputs, num_hidden, num_cond_inputs,
                          s_act='tanh', t_act='relu'),
            fnn.BatchNormFlow(num_inputs),
            fnn.Reverse(num_inputs)
        ]
elif args.flow == 'maf-split-glow':
    for _ in range(args.num_blocks):
        modules += [
            fnn.MADESplit(num_inputs, num_hidden, num_cond_inputs,
                          s_act='tanh', t_act='relu'),
            fnn.BatchNormFlow(num_inputs),
            fnn.InvertibleMM(num_inputs)
        ]
elif args.flow == 'mixture-maf':
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
    modules += [multiscale.MultiscaleAR(225, num_inputs, [225, 625-225], act=args.activation, num_latent_layer=args.latent)]
    model = multiscale.FlowSequential(*modules)
    print('model structure: {}'.format(modules))



for module in model.modules():
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight)
        if hasattr(module, 'bias') and module.bias is not None:
            nn.init.uniform_(module.bias, a=0.0, b=1.0)  # uniform init of bias to avoid zero in alpha
            # module.bias.data.fill_(0)

model = model.cuda()
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model)

optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20, 30], gamma=0.5, last_epoch=-1)

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
        # mu = model._modules['0'].mu
        # log_std = model._modules['0'].log_std
        # u = model.u
        # log_jacob = model.log_jacob

        if batch_idx % args.log_interval == 0:
            print('\n gamma min:{}, gamma max:{}, gamma mean:{}'.format(gamma.min(), gamma.max(), gamma.mean()))
            # print('mu min:{}, mu max:{}, mu mean:{}'.format(mu.min(), mu.max(), mu.mean()))
            # print('log_std min:{}, max:{}, mean:{}'.format(log_std.min(), log_std.max(), log_std.mean()))
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
        samples = model.module.sample(torch.tensor(train_dataset[:1000, :]).cuda(), num_samples=1000, input_size=image_size**2)
        eval_data = train_dataset[:samples.shape[0], :]
        if args.input_permute == 'spiral from center':
            # print(ind[inverse_ind])
            samples = samples[:, inverse_ind]
            eval_data = eval_data[:, inverse_ind]

        duration = time.time() - start
        print('end sampling, duration:{}'.format(duration))

        dist = get_distance(eval_data.reshape(-1, image_size, image_size),
                            samples.reshape(-1, image_size, image_size), image_size=image_size)
        with open(args.result_dir + '/distance_list.txt', 'a') as f:
            f.write(str(dist) + ', \n')

        if epoch % 5 == 0:
            # distance = np.asarray(dist_list)
            # print('min pt:{}, min mass: {}'.format(distance[:, 0].min(), distance[:, 1].min()))
            torch.save(model.state_dict(), args.result_dir + '/laganjet_model_{}.pt'.format(epoch))
            with open(args.result_dir + '/Mix_discretized_sample_{}.pkl'.format(epoch), 'wb') as f:
                pkl.dump(samples.tolist(), f)
                print('generated images saved!')

