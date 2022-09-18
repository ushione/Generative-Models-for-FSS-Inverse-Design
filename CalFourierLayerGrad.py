# -*- coding: utf-8 -*-
# @Python  ：python 3.6
# @Time    : 2022/7/2 15:17
# @Author  : Zheming Gu / 顾哲铭
# @Email   : guzheming@zju.edu.cn
# @File    : CalFourierLayerGrad.py
# @Software: PyCharm
# @Remark  : 
# ---------------------------------------
import os
import copy
import torch
import numpy as np
import matplotlib.pyplot as plt
import time
from FourierLayer import FourierLayer

np.set_printoptions(suppress=True)
torch.set_printoptions(precision=4, sci_mode=False)

cuda = True if torch.cuda.is_available() else False
print("use GPU ? ：", cuda)
device = torch.device("cuda" if cuda else "cpu")
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor


def convert_dfs_to_re_and_im(fft_data_complex):
    # print(type(fft_data_complex), fft_data_complex)
    fft_data_re = []
    fft_data_im = []
    for item_th, item in enumerate(fft_data_complex):
        if cuda:
            item = item.cpu().item()
        else:
            item = item.item()
        # print(item, type(item))
        # print(item.real, item.imag)
        fft_data_re.append(item.real)
        if item_th != 0:
            fft_data_im.append(item.imag)
    print(len(fft_data_re), len(fft_data_im))
    fft_data_re_and_im = fft_data_re + fft_data_im
    return fft_data_re_and_im


def restructure_complex_coefficients(separate_coefficients):
    segmentation = int((len(separate_coefficients) + 1) / 2)
    real_list = separate_coefficients[0:segmentation]
    imag_list = [0, ]
    imag_list.extend(separate_coefficients[segmentation:])

    real = torch.tensor(real_list, dtype=torch.float32)
    imag = torch.tensor(imag_list, dtype=torch.float32)
    complex_coefficients = torch.complex(real, imag)

    return complex_coefficients


# 取一条曲线
sample_index = 141
# 曲线的数据集
DataSet = np.load("./src/pass_dataset.npy")
train_data_curve = DataSet[:, 6:]
train_data_curve = train_data_curve[:, ::10]

sample_curve = Tensor(train_data_curve[sample_index])
print(type(sample_curve), np.shape(sample_curve))
print(sample_curve)

N = 101
M = 20

fourier_layer_input = torch.ones((2*M-1,)).requires_grad_(True).unsqueeze(0).to(device)
print(fourier_layer_input)

myFourierLayer = FourierLayer(cut_dfs=M, batch_size=1, device=device)
my_fourier_layer_result = myFourierLayer(fourier_layer_input)
print(my_fourier_layer_result, my_fourier_layer_result.size())

w = Tensor(torch.ones_like(sample_curve)).requires_grad_(False)

# my_fourier_layer_result.backward(w)

jacobian_array = torch.autograd.functional.jacobian(func=myFourierLayer,
                                                    inputs=fourier_layer_input).squeeze().cpu().numpy() * 101

print(jacobian_array)
jacobian_array_sum_grad = np.sum(jacobian_array, axis=1)
print(jacobian_array_sum_grad)

row_th = 1
col_th = 100

print(np.cos(2 * np.pi * row_th * col_th / N) + np.cos(2 * np.pi * row_th * (N - col_th) / N))
print(-(np.sin(2 * np.pi * row_th * (col_th - M + 1) / N) - np.sin(2 * np.pi * row_th * (N - col_th + M - 1) / N)))

my_jacobian_array = np.zeros_like(jacobian_array)

for row_th in range(N):
    for col_th in range(2 * M - 1):
        if col_th == 0:
            my_jacobian_array[row_th, col_th] = np.cos(2 * np.pi * row_th * col_th / N)
        elif col_th < M:
            my_jacobian_array[row_th, col_th] = np.cos(2 * np.pi * row_th * col_th / N) + np.cos(
                2 * np.pi * row_th * (N - col_th) / N)
        else:
            my_jacobian_array[row_th, col_th] = -(np.sin(2 * np.pi * row_th * (col_th - M + 1) / N) - np.sin(
                2 * np.pi * row_th * (N - col_th + M - 1) / N))

compare_result = my_jacobian_array - jacobian_array
print('over')


def compute_gradient_penalty(D, real_samples, fake_samples, real_labels):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = Tensor(np.random.random((real_samples.size(0), 1)))
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    # interpolates_labels = (alpha * real_labels + ((1 - alpha) * fake_labels)).requires_grad_(True)
    d_interpolates = D(interpolates, real_labels)
    fake = Variable(Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)
    # Get gradient w.r.t. interpolates
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty
