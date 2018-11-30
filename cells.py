# -*- coding: utf-8 -*-
"""
@copyright: Bigeye Lab
this file contains cells to build 2 network: resnet and nasnet
"""
from __future__ import print_function
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
# from pygreedy.ops import ConvBNReLU
from easydict import EasyDict as edict
import math
import copy
import operator
from functools import reduce


class CellResNet(nn.Module):
    def __init__(self,
                 num_layer,
                 in_channel,
                 out_channel_list,
                 kernel_size_list,
                 stride=1,
                 skip_connect=True,
                 skip_kernel_size=None,
                 **kwargs):
        """
        :param num_layers: type int, how many layers in the cell
        :param in_channel: type int, input channles of the cell
        :param out_channel_list: type int list, 1D, output channels of each layer in the cell
        :param kernel_size_list: type int list, 1D, kenel size of each layer in the cell
        :param stride: type int, stride of the first layer in the cell
        :param skip_connect: type bool, default 'True', indicates a skip connection in the cell
        :param skip_kenel_size: type int or None, None--Indendity connect if output channels are the same
                                 else conv1x1
        """
        super(CellResNet, self).__init__()
        assert len(out_channel_list) == num_layer, 'lenght of out_channels_list should equal to num_layers'
        self.num_layers = num_layer
        self.init_channel = copy.deepcopy(in_channel)
        self.out_channel_list = out_channel_list
        self.kernel_size_list = kernel_size_list
        self.stride = stride
        self.skip_connect = skip_connect
        self.skip_kernel_size = skip_kernel_size

        self.cell = nn.Sequential()

        # build cell
        for i in range(num_layer):
            if i < num_layer - 1:
                layer = ConvBnReLu(in_channel,
                                   out_channel_list[i],
                                   stride=stride if i == 0 else 1,
                                   kernel_size=kernel_size_list[i],
                                   padding=int(math.floor(kernel_size_list[i] / 2)))
                self.cell.add_module('layer_%d' % i, layer)
                in_channel = out_channel_list[i]
            else:
                layer = ConvBn(in_channel,
                               out_channel_list[i],
                               stride=stride if i == 0 else 1,
                               kernel_size=kernel_size_list[i],
                               padding=int(math.floor(kernel_size_list[i] / 2)))
                self.cell.add_module('layer_%d' % i, layer)
                in_channel = out_channel_list[i]
        self.cell_out_channel = in_channel
        self.downsample = None
        self.out_nolinear = nn.ReLU(inplace=True)

        if stride != 1 or self.skip_connect:
            if self.skip_kernel_size is None:
                if (self.init_channel != self.cell_out_channel) or (stride != 1):
                    self.downsample = ConvBn(self.init_channel,
                                             self.cell_out_channel,
                                             kernel_size=1,
                                             bias=False,
                                             stride=stride)
            else:
                ks = self.skip_kernel_size
                self.downsample = ConvBn(self.init_channel,
                                         self.cell_out_channel,
                                         kernel_size=ks,
                                         bias=False,
                                         stride=stride)

    def forward(self, x):
        x_ = x
        out = self.cell(x)
        if self.skip_connect:
            if self.downsample is not None:
                x_ = self.downsample(x_)
            out += x_
        out = self.out_nolinear(out)
        return out


class CellNAS(nn.Module):
    def __init__(self):
        super(CellNAS, self).__init__()
        pass

    def forward(self):
        pass


# basic functions
class BnReLuConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, affine=True, bias=False):
        super(BnReLuConv, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(in_channels, affine=affine))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(in_channels, out_channels,
                                          kernel_size=kernel_size,
                                          stride=stride,
                                          padding=padding, bias=bias,
                                          groups=groups))


class ConvBnReLu(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, affine=True, bias=False):
        super(ConvBnReLu, self).__init__()
        self.add_module('conv', nn.Conv2d(in_channels, out_channels,
                                          kernel_size=kernel_size,
                                          stride=stride,
                                          padding=padding, bias=bias,
                                          groups=groups))
        self.add_module('norm', nn.BatchNorm2d(out_channels, affine=affine))
        self.add_module('relu', nn.ReLU(inplace=True))


class ConvBn(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, affine=True,
                 bias=False):
        super(ConvBn, self).__init__()
        self.add_module('conv', nn.Conv2d(in_channels, out_channels,
                                          kernel_size=kernel_size,
                                          stride=stride,
                                          padding=padding, bias=bias,
                                          groups=groups))
        self.add_module('norm', nn.BatchNorm2d(out_channels, affine=affine))


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class SepConv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(SepConv, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_in, affine=affine),
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=1, padding=padding, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
        )

    def forward(self, x):
        return self.op(x)


class FactorizedReduce(nn.Module):
      def __init__(self, C_in, C_out, affine=True):
        super(FactorizedReduce, self).__init__()
        assert C_out % 2 == 0
        self.relu = nn.ReLU(inplace=False)
        self.conv_1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.conv_2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(C_out, affine=affine)

      def forward(self, x):
        x = self.relu(x)
        out = torch.cat([self.conv_1(x), self.conv_2(x[:,:,1:,1:])], dim=1)
        out = self.bn(out)
        return out
if __name__ == '__main__':
    pass
