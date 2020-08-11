# -*- coding: utf-8 -*-
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR

TINY = 1e-12
STATUS_FORMAT = "Train Loss : [ R: {:.4f} | D: {:.4f} | G: {:.4f} ]"


def log_losses(*losses):
    print('\r'+STATUS_FORMAT.format(*losses), end='', flush=True)


def BCEloss(inputs, targets, reduction='mean'):
    mask = targets == 0
    weights = 0.55 * mask.float()
    mask = mask == 0
    weights += mask.float()
    loss = nn.BCELoss(weight=weights, reduction=reduction)
    return loss(inputs, targets)


def sample_categorical(size):
    batch_size, n_classes = size
    cat = np.random.randint(0, n_classes, batch_size)
    cat = np.eye(n_classes)[cat].astype('float32')
    cat = torch.from_numpy(cat)
    return cat


def sample_bernoulli(size):
    ber = np.random.randint(0, 1, size).astype('float32')
    return torch.from_numpy(ber)


PRIOR_SAMPLERS = {
    'categorical': sample_categorical,
    'bernoulli': sample_bernoulli,
    'gauss': torch.randn
}

PRIOR_ACTIVATIONS = {
    'categorical': 'softmax',
    'bernoulli': 'sigmoid',
    'gauss': 'linear'
}

TORCH_OPTIMIZERS = {
    'sgd': optim.SGD,
    'adam': optim.Adam
}

reconstruction_function = nn.BCELoss(size_average=False)


class Encoder(nn.Module):
    """ Three-layer Encoder """
    def __init__(self, n_input=149603, n_hidden=512, n_code=1024, final_activation=None,
                 normalize_inputs=True, dropout=(.2,.2), activation='ReLU'):
        super(Encoder, self).__init__()
        self.lin1 = nn.Linear(n_input, n_hidden)
        self.act1 = getattr(nn, activation)()
        self.lin2 = nn.Linear(n_hidden, n_hidden)
        self.act2 = getattr(nn, activation)()
        if activation == 'SELU':
            self.drop1 = nn.AlphaDropout(dropout[0])
            self.drop2 = nn.AlphaDropout(dropout[1])
        else:
            self.drop1 = nn.Dropout(dropout[0])
            self.drop2 = nn.Dropout(dropout[1])
        self.lin3 = nn.Linear(n_hidden, n_code)
        self.normalize_inputs = normalize_inputs
        if final_activation == 'linear' or final_activation is None:
            self.final_activation = None
        elif final_activation == 'softmax':
            self.final_activation = nn.Softmax(dim=1)
        elif final_activation == 'sigmoid':
            self.final_activation = nn.Sigmoid()
        else:
            raise ValueError("Final activation unknown:", activation)

    def forward(self, inp):
        """ Forward method implementation of 3-layer encoder """
        # if self.normalize_inputs:
        #     inp = F.normalize(inp, 1)
        # first layer
        act = self.lin1(inp)
        act = self.drop1(act)
        act = self.act1(act)
        # second layer
        act = self.lin2(act)
        act = self.drop2(act)
        act = self.act2(act)
        # third layer
        act = self.lin3(act)
        if self.final_activation:
            act = self.final_activation(act)
        return act


class Decoder(nn.Module):
    """ Decoder """
    def __init__(self, n_output=737149, n_hidden=256, n_code=512, dropout=(.2,.2), activation='ReLU'):
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
        self.act1 = getattr(nn, activation)()
        self.act2 = getattr(nn, activation)()

    def forward(self, inp):
        """ Forward implementation of 3-layer decoder """
        # first layer
        act = self.lin1(inp)
        act = self.drop1(act)
        act = self.act1(act)
        # second layer
        act = self.lin2(act)
        act = self.drop2(act)
        act = self.act2(act)
        # final layer
        act = self.lin3(act)
        act = torch.sigmoid(act)
        return act


class Discriminator(nn.Module):
    """ Discriminator """
    def __init__(self, n_hidden=256, n_code=512, dropout=(.2,.2), activation='ReLU'):
        super(Discriminator, self).__init__()
        self.lin1 = nn.Linear(n_code, n_hidden)
        self.lin2 = nn.Linear(n_hidden, n_hidden)
        self.lin3 = nn.Linear(n_hidden, 1)
        if activation == 'SELU':
            self.drop1 = nn.AlphaDropout(dropout[0])
            self.drop2 = nn.AlphaDropout(dropout[1])
        else:
            self.drop1 = nn.Dropout(dropout[0])
            self.drop2 = nn.Dropout(dropout[1])
        self.act1 = getattr(nn, activation)()
        self.act2 = getattr(nn, activation)()

    def forward(self, inp):
        """ Forward of 3-layer discriminator """
        act = self.lin1(inp)
        act = self.drop1(act)
        act = self.act1(act)

        act = self.lin2(act)
        act = self.drop2(act)
        act = self.act2(act)
        return torch.sigmoid(self.lin3(act))


class AAE(nn.Module):
    def __init__(self, n_input=149603, n_hidden=512, n_code=1024, gen_lr=0.001, reg_lr=0.001, prior='gauss',
                 optimizer='adam', normalize_inputs=True, activation='SELU',
                 dropout=(.2, .2), verbose=True):
        super(AAE, self).__init__()
        # Build models
        self.prior = prior.lower()
        # Encoder final activation depends on prior distribution
        self.prior_sampler = PRIOR_SAMPLERS[self.prior]
        self.encoder_activation = PRIOR_ACTIVATIONS[self.prior]
        self.optimizer = optimizer.lower()
        self.n_hidden = n_hidden
        self.n_code = n_code
        self.gen_lr, self.reg_lr = gen_lr, reg_lr

        self.enc = Encoder(n_input=n_input, n_hidden=n_hidden, n_code=n_code, final_activation=self.encoder_activation,
                           normalize_inputs=normalize_inputs, dropout=dropout, activation=activation)
        self.dec = Decoder(n_output=n_input, n_hidden=n_hidden, n_code=n_code, dropout=dropout, activation=activation)
        self.disc = Discriminator(n_hidden=n_hidden, n_code=n_code, dropout=dropout, activation=activation)
        self.normalize_inputs = normalize_inputs

        optimizer_gen = TORCH_OPTIMIZERS[optimizer]
        # Reconstruction
        self.enc_optim = optimizer_gen(self.enc.parameters(), lr=self.gen_lr)
        self.dec_optim = optimizer_gen(self.dec.parameters(), lr=self.gen_lr)
        # Regularization
        self.gen_optim = optimizer_gen(self.enc.parameters(), lr=self.reg_lr)
        self.disc_optim = optimizer_gen(self.disc.parameters(), lr=self.reg_lr)

        self.enc_scheduler = MultiStepLR(self.enc_optim, milestones=[24, 36, 48, 66, 72], gamma=0.5)
        self.dec_scheduler = MultiStepLR(self.dec_optim, milestones=[24, 36, 48, 66, 72], gamma=0.5)
        self.gen_scheduler = MultiStepLR(self.gen_optim, milestones=[24, 36, 48, 66, 72], gamma=0.5)
        self.disc_scheduler = MultiStepLR(self.disc_optim, milestones=[24, 36, 48, 66, 72], gamma=0.5)

        self.dropout = dropout
        self.activation = activation
        self.verbose = verbose

    def forward(self, input_x):
        output, recon_loss = self.ae_step(input_x)
        disc_loss = self.disc_step(input_x)
        gen_loss = self.gen_step(input_x)
        return output, recon_loss, disc_loss, gen_loss

    def predict(self, input_x):
        z_sample = self.enc(input_x)
        output = self.dec(z_sample)
        return output

    def ae_step(self, input_x):
        z_sample = self.enc(input_x)
        x_sample = self.dec(z_sample)
        x_sample = x_sample + TINY

        recon_loss = F.binary_cross_entropy(x_sample + TINY, input_x + TINY)

        # Clear all related gradients
        self.enc.zero_grad()
        self.dec.zero_grad()

        # Compute gradients
        recon_loss.backward()

        # Update parameters
        self.enc_optim.step()
        self.dec_optim.step()

        self.enc_scheduler.step()
        self.dec_scheduler.step()
        return x_sample, recon_loss.data.item()

    def disc_step(self, input_x):   # Discriminator
        self.enc.eval()
        z_real = Variable(self.prior_sampler((input_x.size(0), self.n_code)))
        if torch.cuda.is_available():
            z_real = z_real.cuda()
        z_fake = self.enc(input_x)
        disc_real_out, disc_fake_out = self.disc(z_real), self.disc(z_fake)

        disc_loss = -torch.mean(torch.log(disc_real_out + TINY) + torch.log(1 - disc_fake_out + TINY))
        self.disc_optim.zero_grad()
        disc_loss.backward()
        self.disc_optim.step()
        self.disc_scheduler.step()
        return disc_loss.data.item()

    def gen_step(self, input_x):    # Gan
        self.enc.train()
        z_fake_dist = self.enc(input_x)
        disc_fake_out = self.disc(z_fake_dist) + TINY
        # Loss
        gen_loss = -torch.mean(torch.log(disc_fake_out + TINY))
        self.gen_optim.zero_grad()
        gen_loss.backward()
        self.gen_optim.step()
        self.gen_scheduler.step()
        return gen_loss.data.item()
