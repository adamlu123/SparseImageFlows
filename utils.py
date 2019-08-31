import torch
from torch.distributions import Normal, Bernoulli


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


def get_img_sample(pi, beta):
    binaries = Bernoulli(pi).samples()
    betas = Normal(loc=beta, scale=1).sample()
    return binaries * betas