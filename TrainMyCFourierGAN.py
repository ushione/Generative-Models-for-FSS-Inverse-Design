# -*- coding: utf-8 -*-
# @Python  ：python 3.6
# @Time    : 2022/7/1 14:00
# @Author  : Zheming Gu / 顾哲铭
# @Email   : guzheming@zju.edu.cn
# @File    : TrainMyCGAN.py
# @Software: PyCharm
# @Remark  : 
# ---------------------------------------
import os
import torch
import random
import itertools
import numpy as np
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.autograd import Variable
from FourierLayer import FourierLayer
from torch.utils.data import DataLoader, Dataset, TensorDataset

cuda = True if torch.cuda.is_available() else False
device = torch.device("cuda" if cuda else "cpu")
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

BATCH_SIZE = 128

seed = 888
random.seed(seed)
np.random.seed(seed)
if cuda:
    torch.cuda.manual_seed(seed)
else:
    torch.manual_seed(seed)


class Generator(nn.Module):
    def __init__(self, cut_dfs):
        super(Generator, self).__init__()

        self.label_emb = nn.Embedding(101, 10)
        self.cut_dfs = cut_dfs

        def block(in_feat, out_feat, normalize=False):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(10 + 20, 128, normalize=False),
            *block(128, 128),
            *block(128, 128),
            *block(128, 128),
            nn.Linear(128, 2 * self.cut_dfs - 1),
            FourierLayer(cut_dfs=self.cut_dfs, batch_size=BATCH_SIZE, device=device),
            # nn.Sigmoid()
        )

    def forward(self, noise, labels):
        # Concatenate label embedding and image to produce input
        labels_emb = self.label_emb(labels)
        labels_input = labels_emb.view(labels_emb.size(0), -1)
        gen_input = torch.cat((labels_input, noise), -1)
        img = self.model(gen_input)
        # img = img.view(img.size(0), *img_shape)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.label_embedding = nn.Embedding(101, 10)

        self.model = nn.Sequential(
            nn.Linear(20 + 101, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, curves, labels):
        # Concatenate label embedding and image to produce input
        labels_embedding = self.label_embedding(labels)
        labels_input = labels_embedding.view(labels_embedding.size(0), -1)
        d_in = torch.cat((curves, labels_input), -1)
        # print('labels_input:{} ,d_in:{}'.format(labels_input.size(), d_in.size()))
        validity = self.model(d_in)
        return validity


TrainDataSet = np.load("./Data/DataWithPassBandLabel.npz")['arr_0']

# label to Long Tensor
data_curve, pass_band = Tensor(TrainDataSet[:, :-7]), TrainDataSet[:, -2:]

# label to Long Tensor
pass_band = LongTensor(np.round(pass_band * 5))

DataSet = TensorDataset(data_curve, pass_band)
dataloader = DataLoader(dataset=DataSet, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

G = Generator(cut_dfs=51)
D = Discriminator()

optimizer_G = torch.optim.Adam(G.parameters(), lr=0.0001, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(D.parameters(), lr=0.0001, betas=(0.5, 0.999))
adversarial_loss = torch.nn.BCELoss()
# adversarial_loss = torch.nn.MSELoss()


if cuda:
    G.cuda()
    D.cuda()
    adversarial_loss.cuda()

all_class = np.unique(pass_band.cpu().numpy(), axis=0)

Recoder_d_loss = []
Recoder_g_loss = []

for iterations in itertools.count():
    for i, (curve, label) in enumerate(dataloader):
        true_curves = Variable(curve.type(Tensor))
        gen_labels = Variable(label.type(LongTensor))

        valid = Variable(Tensor(BATCH_SIZE, 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(BATCH_SIZE, 1).fill_(0.0), requires_grad=False)

        # ---------------------
        #  Train Discriminator
        # ---------------------

        # Sample noise and labels as generator input
        z = Variable(Tensor(np.random.normal(0, 1, (BATCH_SIZE, 10))))

        # Generate a batch of curves
        gen_curves = G(z, gen_labels)

        optimizer_D.zero_grad()

        # Loss for real images
        validity_real = D(true_curves, gen_labels)
        d_real_loss = adversarial_loss(validity_real, valid)

        # Loss for fake images
        validity_fake = D(gen_curves.detach(), gen_labels)
        d_fake_loss = adversarial_loss(validity_fake, fake)

        # Total discriminator loss
        d_loss = (d_real_loss + d_fake_loss) / 2

        # if d_loss >1:
        #     print(d_real_loss , d_fake_loss)

        d_loss.backward()
        optimizer_D.step()

        # -----------------
        #  Train Generator
        # -----------------
        if i % 1 == 0:
            optimizer_G.zero_grad()

            # Sample noise and labels as generator input
            z = Variable(Tensor(np.random.normal(0, 1, (BATCH_SIZE, 10))))

            # Generate a batch of curves
            gen_curves = G(z, gen_labels)

            # Loss measures generator's ability to fool the discriminator
            validity = D(gen_curves, gen_labels)
            g_loss = adversarial_loss(validity, valid)

            g_loss.backward()
            optimizer_G.step()

        if i == 0:
            gen_label = (gen_labels[0]/5).cpu().detach().numpy()
            true_curve = true_curves[0]
            gen_curve = gen_curves[0].cpu().detach().numpy()

            x_list = np.linspace(0, 20, 101)
            # print(gen_label[0], gen_label[1])

            plt.figure(1)

            plt.cla()
            plt.title(
                'iterations:{} || label: ({},{})'.format(iterations, '%.3f' % gen_label[0], '%.3f' % gen_label[1]))
            plt.axis([-1, 21, 0, 1.1])
            plt.plot(x_list, gen_curve)

            plt.hlines(y=0.326, xmin=-1, xmax=21, linewidth=2, color='g')
            plt.vlines(x=gen_label[0], ymin=-1, ymax=21, linewidth=2, color='g')
            plt.vlines(x=gen_label[1], ymin=-1, ymax=21, linewidth=2, color='g')

            # plt.pause(0.01)
            plt.savefig('CFourierGAN')

            plt.pause(0.01)

            torch.save(G.state_dict(), 'params_G_conditional_Fourier_network.pkl')
            torch.save(D.state_dict(), 'params_D_conditional_Fourier_network.pkl')

            print(
                "[iterations %d] [D loss: %f] [G loss: %f][D real loss: %f][D fake loss: %f]"
                % (iterations, d_loss.item(), g_loss.item(), d_real_loss.item(), d_fake_loss.item())
            )

            plt.figure(2)

            plt.cla()
            Recoder_d_loss.append(d_loss.item())
            Recoder_g_loss.append(g_loss.item())
            plt.title('D_loss vs. epochs.png', size=20)
            plt.grid()
            # plt.yticks(np.linspace(start=0.00, stop=1.00, num=11))
            plt.plot(range(len(Recoder_d_loss)), Recoder_d_loss, label='D loss', color='red', linewidth=2, marker="v", markersize=6)
            plt.xlabel('Training Epochs', size=20)
            plt.ylabel('D Loss', size=20)
            plt.tick_params(labelsize=20)
            plt.legend(loc='best', fontsize='15', handlelength=4)
            plt.subplots_adjust(bottom=0.2)
            plt.pause(0.01)
            plt.savefig('D_loss vs. epochs[CFourierGAN].png')

            Recoder = pd.DataFrame(np.array([Recoder_d_loss, Recoder_g_loss]).T)
            Recoder.to_csv("Recoder_Fourier_d_and_g_loss.csv")

