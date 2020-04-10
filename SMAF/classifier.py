import sys
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import argparse
import copy
import math
import pickle as pkl
import time
import numpy as np
import utils
from utils import load_data_LAGAN, lagan_disretized_loader
import h5py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torch.utils.data import DataLoader, Dataset

import numpy as np

from sklearn import metrics


from tqdm import tqdm

# load -> assign label -> mix

# def permute(data, label):
#     ind = np.random.permutation(data.shape[0])
#     return data[ind], label[ind]


def get_auc(pred, label):
    pred, label = pred.cpu().numpy(), label.cpu().numpy()
    auc = metrics.roc_auc_score(label, pred)
    return auc



def get_acc(pred, label):
    pred_class = torch.where(pred > 0.5, torch.ones_like(pred), torch.zeros_like(pred))
    acc = (pred_class.squeeze() == label).sum().float() / label.shape[0]
    return acc


def flip_label(label, portion):
    n = label.shape[0]
    ind = np.random.choice(n, int(n*portion), replace=False)
    label[ind] = 1 - label[ind]
    return label


def prepare_data(dataset, device='cuda'):

    result_dir = '/baldig/physicsprojects/lagan'
    assert dataset in ['sarm', 'lagan', 'truth', 'truth-train', 'sarm-reshapenorm', 'augument_sarm-reshapenorm']

    if dataset == 'sarm':
        with h5py.File(result_dir + '/sg_softmax.h5', 'r') as f:
            sg = np.asarray(f['sg'][:, :, :])
        with h5py.File(result_dir + '/bg_softmax.h5', 'r') as f:
            bg = np.asarray(f['bg'][:, :, :])


    elif dataset == 'lagan':
        sg = np.load(result_dir + '/lagan_generated_data_400K/lagan_generated_signal.npy')
        bg = np.load(result_dir + '/lagan_generated_data_400K/lagan_generated_bg.npy')

    elif dataset == 'truth':
        with h5py.File(result_dir + '/lagan-jet-images_tiny.h5', 'r') as f:
            sg = np.asarray(f['sg'][:, :, :])
            bg = np.asarray(f['bg'][:, :, :])

    elif dataset == 'truth-train':
        with h5py.File(result_dir + '/lagan-jet-images_train.h5', 'r') as f:
            sg = np.asarray(f['sg'][:, :, :])
            bg = np.asarray(f['bg'][:, :, :])

    elif dataset == 'sarm-reshapenorm':
        with h5py.File(result_dir + '/sparse_arm_generated_combined_v2.h5', 'r') as f:
            sg = np.asarray(f['sg'][:, :, :])
            bg = np.asarray(f['bg'][:, :, :])

    elif dataset == 'augument_sarm-reshapenorm':
        with h5py.File(result_dir + '/sparse_arm_generated_combined_v2.h5', 'r') as f:
            sg = np.asarray(f['sg'][:, :, :])
            bg = np.asarray(f['bg'][:, :, :])
        with h5py.File(result_dir + '/lagan-jet-images_train.h5', 'r') as f:
            sg = np.concatenate([np.asarray(f['sg'][:, :, :]),sg], axis=0)
            bg = np.concatenate([np.asarray(f['bg'][:, :, :]),bg], axis=0)

    elif dataset == 'augument_sarm':
        with h5py.File(result_dir + '/sparse_arm_generated_combined_v2.h5', 'r') as f:
            sg = np.asarray(f['sg'][:, :, :])
            bg = np.asarray(f['bg'][:, :, :])
        with h5py.File(result_dir + '/sg_softmax.h5', 'r') as f:
            sg = np.concatenate([np.asarray(f['sg'][:, :, :]),sg], axis=0)
        with h5py.File(result_dir + '/bg_softmax.h5', 'r') as f:
            bg = np.concatenate([np.asarray(f['bg'][:, :, :]),bg], axis=0)





    n = bg.shape[0]
    label = torch.tensor(np.concatenate([np.zeros(n), np.ones(n)]), dtype=torch.float).to(device)
    # label = flip_label(label, portion=0.1)
    data = torch.tensor(np.concatenate([bg, sg], axis=0), dtype=torch.float).to(device)
    np.random.seed(seed=123)
    ind = np.random.permutation(2*n)
    print('load from dataset {}, sg shape {}, bg shape {}'.format(dataset, sg.shape, bg.shape))
    # return data, label
    return data[ind], label[ind]




class ClassficationNet(nn.Module):
    def __init__(self, input_dim):
        super(ClassficationNet, self).__init__()
        hidden_channel = 30
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=hidden_channel, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(in_channels=hidden_channel, out_channels=hidden_channel, kernel_size=3, stride=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(in_channels=hidden_channel, out_channels=hidden_channel, kernel_size=3, stride=1)
        self.conv4 = nn.Conv2d(in_channels=hidden_channel, out_channels=1, kernel_size=3, stride=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=1)
        self.linear1 = nn.Linear(169, 100)
        self.linear2 = nn.Linear(100, 1)

    def forward(self, x):
        x = x.view(-1, 1, 25, 25)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.maxpool1(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.maxpool2(x)
        x = x.view(x.shape[0], -1)
        x = self.linear1(x)
        x = self.linear2(x)
        return torch.sigmoid(x).squeeze()


def train(epoch, model, optimizer, train_loader, true_data, true_label, config):
    model.train()
    trainauc_ls, testauc_ls = [], []

    # forward pass
    for batch_idx, (data, label) in enumerate(train_loader):
        data, label = data.to(config['device']), label.to(config['device'])
        optimizer.zero_grad()

        pred = model(data)
        acc = get_acc(pred, label)

        # loss
        criterion = torch.nn.BCELoss(reduction='sum')
        loss = criterion(pred, label)
        loss.backward()
        optimizer.step()

        # print
        if batch_idx % 500 == 0:
            with torch.no_grad():
                pred_test = model(true_data)
                acc_test = get_acc(pred_test, true_label)
                test_auc = get_auc(pred_test, true_label)

                pred_train = model(train_loader.dataset.data[:true_data.shape[0]].cuda())
                train_label = train_loader.dataset.label[:true_label.shape[0]].cuda()
                auc_train = get_auc(pred_train, train_label)

                trainauc_ls.append(auc_train)
                testauc_ls.append(test_auc)

                print('epoch {}, loss {}, batch acc {}, train auc {}, test acc {}, test auc {}'.format(epoch, loss,
                                                                                                       acc, auc_train,
                                                                                                       acc_test, test_auc))
    return trainauc_ls, testauc_ls

def test(model, data, label):
    model.eval()
    with torch.no_grad():
        pred = model(data)
        acc = get_acc(pred, label)
    return pred.cpu().detach().numpy()


class MyDataset(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data_val = self.data[index]
        label = self.label[index]
        return data_val, label


def main(config):

    data, label = prepare_data(config['training data'], device='cpu')
    true_data, true_label = prepare_data('truth', device=config['device'])

    data, label = data, label
    train_set = MyDataset(data, label)
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=config['batch_size'], shuffle=True)

    # true_data, true_label = true_data, true_label


    # define model
    model = ClassficationNet(input_dim=data.shape[0]).to(config['device'])
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    print(model.parameters)


    # start training
    trainauc, testauc = [], []
    for epoch in range(config['epochs'] + 1):
        trainauc_ls, testauc_ls = train(epoch, model, optimizer, train_loader, true_data, true_label, config)
        trainauc.extend(trainauc_ls), testauc.extend(testauc_ls)

    with open(config['save_dir'] + '/{}trainauc.pkl'.format(config['training data']), 'wb') as f:
        pkl.dump(trainauc, f)
    with open(config['save_dir'] + '/{}testauc.pkl'.format(config['training data']), 'wb') as f:
        pkl.dump(testauc, f)

    pred = test(model, true_data, true_label)

    # with open(config['save_dir'] + '/pred_{}.pkl'.format(config['training data']), 'wb') as f:
    #     pkl.dump(pred, f)
    #
    # torch.save(model.state_dict(), config['save_dir'] + '/ep{}_trainset{}_lr{}_batch{}.pt'.format(config['epochs'],
    #                                                                               config['training data'],
    #                                                                                    config['lr'],
    #                                                                                            config['batch_size']))



if __name__ == "__main__":
    config = {'device': 'cuda',
              'lr': 0.001, # 0.0001
              'epochs': 20, # 1
              'batch_size': 128,
              'training data': 'sarm',  #lagan truth sarm, sarm-reshapenorm, truth-train, augument_sarm-reshapenorm
              'save_dir': '/extra/yadongl10/BIG_sandbox/SparseImageFlows_result/classifer/saved'}

    main(config)



