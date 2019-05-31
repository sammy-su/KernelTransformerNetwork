#!/usr/bin/env python
# -*- coding: utf-8 -*-

import abc
import torch
import torch.nn as nn
import torch.nn.functional as F

from cfg import Config
from SphereProjection import SphereProjection


class RowBilinear(nn.Module):

    def __init__(self, n_in, kernel_shapes, pad=0):
        super(RowBilinear, self).__init__()

        n_transform = kernel_shapes.size(0)
        weights = []
        self.pad = pad
        for i in xrange(n_transform):
            kH = kernel_shapes[i,0].item()
            kW = kernel_shapes[i,1].item()
            n_out = (kH + 2 * pad) * (kW + 2 * pad)
            weight = nn.Parameter(torch.Tensor(n_out, n_in))
            weights.append(weight)
        self.weights = nn.ParameterList(weights)

    def forward(self, x, row):
        weight = self.weights[row]
        return F.linear(x, weight)


class KTN(nn.Module):
    __metaclass__ = abc.ABCMeta

    def __init__(self, kernel, bias, kernel_shapes, **kwargs):
        super(KTN, self).__init__()

        dtype = Config["FloatType"]
        self.src_kernel = nn.Parameter(kernel).type(dtype)
        self.src_bias = nn.Parameter(bias).type(dtype)

        self.activation = torch.tanh
        self.register_buffer("kernel_shapes", kernel_shapes)

        n_out, n_in, kH, kW = kernel.size()
        self.n_in = n_in
        self.n_out = n_out
        self.n_maps = n_out * n_in

        kernel_size = kH * kW
        self.initialize_ktn(kernel_size)

    @abc.abstractmethod
    def initialize_ktn(self, kernel_size):
        pass

    def forward(self, row):
        x = self.src_kernel.view(self.n_maps, -1)
        x = self.apply_ktn(x, row)

        okH, okW = self.kernel_shapes[row]
        kernel = x.view(self.n_out, self.n_in, okH, okW)
        bias = self.src_bias
        return kernel, bias

    @abc.abstractmethod
    def apply_ktn(self, x, row):
        pass

    def initialize_weight(self):
        for name, param in self.named_parameters():
            if ".bias" in name:
                param.data.zero_()
            elif ".weight" in name:
                param.data.normal_(std=0.01)

    def update_group(self, group):
        for name, param in self.named_parameters():
            param.requires_grad = False

        if group == "kernel":
            self.src_kernel.requires_grad = True
            self.src_bias.requires_grad = True
        elif group == "transform":
            for name, param in self.named_parameters():
                if ".weight" in name or ".bias" in name:
                    param.requires_grad = True
        elif group == "all":
            for name, param in self.named_parameters():
                param.requires_grad = True
        else:
            raise ValueError("Unknown parameter group")


class BilinearKTN(KTN):

    def initialize_ktn(self, kernel_size):
        self.bilinear = RowBilinear(kernel_size, self.kernel_shapes)

    def apply_ktn(self, x, row):
        x = self.bilinear(x, row)
        return x

    def initialize_weight(self, **kwargs):
        for name, param in self.named_parameters():
            if name[-5:] == ".bias":
                param.data.zero_()
            elif name[-7:] == ".weight":
                param.data.normal_(std=0.01)
        self.initialize_bilinear(self.bilinear, **kwargs)

    def initialize_bilinear(self,
                            bilinear,
                            sphereH=320,
                            fov=65.5,
                            imgW=640,
                            dilation=1,
                            tied_weights=5):
        kH = self.src_kernel.size(2)
        sphereW = sphereH * 2
        projection = SphereProjection(kernel_size=kH,
                                      sphereH=sphereH,
                                      sphereW=sphereW,
                                      view_angle=fov,
                                      imgW=imgW)
        center = sphereW / 2
        for i, param in enumerate(bilinear.weights):
            param.data.zero_()
            tilt = i * tied_weights + tied_weights / 2
            P = projection.buildP(tilt=tilt).transpose()
            okH = self.kernel_shapes[i,0].item()
            okW = self.kernel_shapes[i,1].item()
            okH += bilinear.pad * 2
            okW += bilinear.pad * 2

            sH = tilt - okH / 2
            sW = center - okW / 2
            for y in xrange(okH):
                row = y + sH
                if row < 0 or row >= sphereH:
                    continue
                for x in xrange(okW):
                    col = x + sW
                    if col < 0 or col >= sphereW:
                        continue
                    pixel = row * sphereW + col
                    p = P[pixel]
                    if p.nnz == 0:
                        continue
                    j = y * okW + x
                    for k in xrange(p.shape[1]):
                        param.data[j,k] = p[0,k]


class ResidualKTN(BilinearKTN):

    def initialize_ktn(self, kernel_size):
        self.bilinear = RowBilinear(kernel_size, self.kernel_shapes)

        self.res1 = RowBilinear(kernel_size, self.kernel_shapes, pad=2)
        self.res2 = nn.Conv2d(self.n_in, self.n_in, 1)
        self.res3 = nn.Conv2d(1, 1, 3, padding=0)
        self.res4 = nn.Conv2d(self.n_in, self.n_in, 1)
        self.res5 = nn.Conv2d(1, 1, 3, padding=0)

    def apply_ktn(self, x, row):
        base = self.bilinear(x, row)

        okH, okW = self.kernel_shapes[row]
        x = self.res1(x, row)

        x = x.view(-1, self.n_in, okH+4, okW+4)
        x = self.res2(self.activation(x))
        x = x.view(-1, 1, okH+4, okW+4)
        x = self.res3(self.activation(x))

        x = x.view(-1, self.n_in, okH+2, okW+2)
        x = self.res4(self.activation(x))
        x = x.view(-1, 1, okH+2, okW+2)
        x = self.res5(self.activation(x))

        x = x.view(base.size())
        x = x + base
        return x

    def initialize_weight(self, **kwargs):
        for name, param in self.named_parameters():
            if name[-5:] == ".bias":
                param.data.zero_()
            elif name[-7:] == ".weight":
                param.data.normal_(std=0.01)
        self.initialize_bilinear(self.bilinear, **kwargs)
        self.initialize_bilinear(self.res1, **kwargs)


KTN_ARCHS = {
    "bilinear": BilinearKTN,
    "residual": ResidualKTN,
}

