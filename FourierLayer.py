# -*- coding: utf-8 -*-
# @Python  ：python 3.6
# @Time    : 2021/11/22 11:49
# @Author  : Zheming Gu / 顾哲铭
# @Email   : guzheming@zju.edu.cn
# @File    : FourierLayer.py
# @Software: PyCharm
# @Remark  : 自定义的FourierLayer
# ---------------------------------------
import torch
import numpy as np
from torch import nn


class FourierLayer(nn.Module):
    def __init__(self, cut_dfs, batch_size, device):
        super(FourierLayer, self).__init__()
        self.device = device
        self.BATCH_SIZE = batch_size
        self.N = 101
        self.M = cut_dfs  # M<=(N+1)//2  this must be satisfied
        assert self.M <= (self.N + 1) // 2
        self.DFS_WIDTH = 2 * self.M - 1  # x.shape[1] must be equal DFS_WIDTH
        self.Wre, self.Wim = self.getW()
        self.Re_Join_Array, self.Im_Join_Array = self.getJoinArray()
        self.Coefficients_Re, self.Coefficients_Im, self.my_ifft_result = None, None, None
        self.Wre, self.Wim, self.Re_Join_Array, self.Im_Join_Array = self.Wre.repeat(self.BATCH_SIZE, 1, 1), \
                                                                     self.Wim.repeat(self.BATCH_SIZE, 1, 1), \
                                                                     self.Re_Join_Array.repeat(self.BATCH_SIZE, 1, 1), \
                                                                     self.Im_Join_Array.repeat(self.BATCH_SIZE, 1, 1)

    def getW(self):
        W_exp = complex(0, -2 * np.pi / self.N)

        W_array = np.zeros((self.N, self.N), dtype=complex)

        for row in range(self.N):
            for col in range(self.N):
                W_array[row, col] = np.exp(-W_exp * row * col)

        Wre, Wim = torch.from_numpy(W_array.real), torch.from_numpy(W_array.imag)
        return Wre.to(self.device), Wim.to(self.device)

    def getJoinArray(self):
        Re_Join_Array = torch.zeros((self.N, self.DFS_WIDTH))
        Im_Join_Array = torch.zeros((self.N, self.DFS_WIDTH))
        # print(Re_Join_Array) (N+1)/2
        Re_Join_Array[:self.M, :self.M] = torch.eye(self.M)
        Re_Join_Array[-self.M+1:, 1:self.M] = torch.rot90(torch.eye(self.M - 1), -1)
        Im_Join_Array[1:self.M, -(self.M - 1):] = torch.eye(self.M - 1)
        Im_Join_Array[-(self.M - 1):, -(self.M - 1):] = -1 * torch.rot90(torch.eye(self.M - 1), -1)

        return Re_Join_Array.to(self.device), Im_Join_Array.to(self.device)

    def forward(self, x):
        x = x.unsqueeze(-1)
        # print(self.Re_Join_Array.size(), x.size())
        self.Coefficients_Re = torch.bmm(self.Re_Join_Array, x)
        self.Coefficients_Im = torch.bmm(self.Im_Join_Array, x)
        self.my_ifft_result = torch.bmm(self.Wre.to(torch.float32), self.Coefficients_Re.to(torch.float32)) - torch.bmm(
            self.Wim.to(torch.float32), self.Coefficients_Im.to(torch.float32))
        # print(my_ifft_result/self.DFS_WIDTH)
        return (self.my_ifft_result / self.N).squeeze()


class FourierLayerOriginal(nn.Module):
    def __init__(self, cut_dfs, batch_size, ):
        super(FourierLayerOriginal, self).__init__()
        self.BATCH_SIZE = batch_size
        self.N = 101
        self.M = cut_dfs  # M<=(N+1)//2  this must be satisfied
        assert self.M <= (self.N + 1) // 2
        self.DFS_WIDTH = 2 * self.M - 1  # x.shape[1] must be equal DFS_WIDTH
        self.Wre, self.Wim = self.getW()
        self.Re_Join_Array, self.Im_Join_Array = self.getJoinArray()
        self.Wre, self.Wim, self.Re_Join_Array, self.Im_Join_Array = self.Wre.repeat(self.BATCH_SIZE, 1, 1), \
                                                                     self.Wim.repeat(self.BATCH_SIZE, 1, 1), \
                                                                     self.Re_Join_Array.repeat(self.BATCH_SIZE, 1, 1), \
                                                                     self.Im_Join_Array.repeat(self.BATCH_SIZE, 1, 1)

    def getW(self):
        W_exp = complex(0, -2 * np.pi / self.N)

        W_array = np.zeros((self.N, self.N), dtype=complex)

        for row in range(self.N):
            for col in range(self.N):
                W_array[row, col] = np.exp(-W_exp * row * col)

        Wre, Wim = torch.from_numpy(W_array.real), torch.from_numpy(W_array.imag)
        return Wre, Wim

    def getJoinArray(self):
        Re_Join_Array = torch.zeros((self.N, self.DFS_WIDTH))
        Im_Join_Array = torch.zeros((self.N, self.DFS_WIDTH))
        # print(Re_Join_Array) (N+1)/2
        Re_Join_Array[:self.M, :self.M] = torch.eye(self.M)
        Re_Join_Array[-self.M+1:, 1:self.M] = torch.rot90(torch.eye(self.M - 1), -1)
        Im_Join_Array[1:self.M, -(self.M - 1):] = torch.eye(self.M - 1)
        Im_Join_Array[-(self.M - 1):, -(self.M - 1):] = -1 * torch.rot90(torch.eye(self.M - 1), -1)

        return Re_Join_Array, Im_Join_Array

    def forward(self, x):
        x = x.unsqueeze(-1)
        # print(self.Re_Join_Array.size(), x.size())
        Coefficients_Re = torch.bmm(self.Re_Join_Array, x.cpu())
        Coefficients_Im = torch.bmm(self.Im_Join_Array, x.cpu())
        my_ifft_result = torch.bmm(self.Wre.to(torch.float32), Coefficients_Re.to(torch.float32)) - torch.bmm(
            self.Wim.to(torch.float32), Coefficients_Im.to(torch.float32))
        # print(my_ifft_result/self.DFS_WIDTH)
        return my_ifft_result / self.N