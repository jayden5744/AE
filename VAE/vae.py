# -*- coding: utf-8 -*-
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as weight_init
from torch.autograd import Variable

reconstruction_function = nn.BCELoss(size_average=False)


def loss_function(recon_x, x, mu, logvar, size_average=False):
    mask = recon_x != 0
    num_ratings = torch.sum(mask.float())
    BCE = reconstruction_function(recon_x, x)

    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5)

    return BCE + KLD, Variable(torch.Tensor([1.0])) if size_average else num_ratings


class Encoder(nn.Module):
    """ Three-layer Encoder """
    def __init__(self, n_input=737149, n_hidden=256, n_code=512, normalize_inputs=True, dropout=(.2, .2),
                 activation='SeLU'):
        super(Encoder, self).__init__()
        self.mu_lin1 = nn.Linear(n_input, n_hidden)
        self.mu_lin2 = nn.Linear(n_hidden, n_hidden)
        self.mu_lin3 = nn.Linear(n_hidden, n_code)

        self.var_lin1 = nn.Linear(n_input, n_hidden)
        self.var_lin2 = nn.Linear(n_hidden, n_hidden)
        self.var_lin3 = nn.Linear(n_hidden, n_code)

        self.act = getattr(nn, activation)()

        if activation == 'SELU':
            self.drop1 = nn.AlphaDropout(dropout[0])
            self.drop2 = nn.AlphaDropout(dropout[1])
        else:
            self.drop1 = nn.Dropout(dropout[0])
            self.drop2 = nn.Dropout(dropout[1])

        self.val_mu = 0
        self.val_sigma = 0
        self.normalize_inputs = normalize_inputs

    def forward(self, x):
        if self.normalize_inputs:
            x = F.normalize(x, 1)
        mu = self.mu_lin1(x)
        mu = self.drop1(mu)
        mu = self.act(mu)

        mu = self.mu_lin2(mu)
        mu = self.drop2(mu)
        mu = self.act(mu)
        self.val_mu = self.mu_lin3(mu)

        var = self.var_lin1(x)
        var = self.drop1(var)
        var = self.act(var)

        var = self.var_lin2(var)
        var = self.drop2(var)
        var = self.act(var)
        self.val_sigma = self.var_lin3(var)

        reparam = self.reparametrize(self.val_mu, self.val_sigma)
        return self.val_mu, self.val_sigma, reparam

    def reparametrize(self, mean, var):
        std = var.mul(0.5).exp_()
        eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps).cuda()

        return eps.mul(std).add_(mean)


class Decoder(nn.Module):
    """ Decoder """
    def __init__(self, n_output=737149, n_hidden=256, n_code=512, dropout=(.2,.2), activation='SELU'):
        super(Decoder, self).__init__()
        self.lin1 = nn.Linear(n_code, n_hidden)
        self.lin2 = nn.Linear(n_hidden, n_hidden)
        self.lin3 = nn.Linear(n_hidden, n_output)
        if activation == 'SELU':
            self.drop1 = nn.AlphaDropout(dropout[0])
            self.drop2 = nn.AlphaDropout(dropout[1])
        else:
            self.drop1 = nn.Dropout(dropout[0])
            self.drop2 = nn.Dropout(dropout[1])

        self.act = getattr(nn, activation)()

    def forward(self, z):
        z = self.lin1(z)
        z = self.drop1(z)
        z = self.act(z)

        z = self.lin2(z)
        z = self.drop2(z)
        z = self.act(z)

        z = self.lin3(z)
        output_x = torch.sigmoid(z)
        return output_x


class VAE(nn.Module):
    def __init__(self, n_input=737149, n_hidden=256, n_code=512, dropout=(.2,.2), activation='SELU',
                 normalize_inputs=False):
        super(VAE, self).__init__()
        self.encoder = Encoder(n_input=n_input, n_hidden=n_hidden, n_code=n_code,
                               dropout=dropout, activation=activation, normalize_inputs=normalize_inputs)
        self.decoder = Decoder(n_output=n_input, n_hidden=n_hidden, n_code=n_code, dropout=dropout,
                               activation=activation)

    def forward(self, input_x):
        mu, var, reparam = self.encoder(input_x)
        output = self.decoder(reparam)
        return output, mu, var