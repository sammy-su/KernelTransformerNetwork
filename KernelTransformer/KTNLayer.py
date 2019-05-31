#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

from KTN import KTN_ARCHS
from SphereProjection import SphereProjection
from util import create_variable


KERNEL_SHAPE_TYPES = ["dilated", "full"]


class KTNConv(nn.Module):

    def __init__(self,
                 kernel,
                 bias,
                 sphereH=320,
                 imgW=640,
                 fov=65.5,
                 dilation=1,
                 tied_weights=5,
                 arch="residual",
                 kernel_shape_type="dilated"):
        super(KTNConv, self).__init__()

        self.sphereH = sphereH
        self.sphereW = sphereH * 2
        self.imgW = imgW
        self.fov = fov
        self.tied_weights = tied_weights

        pad_ind = torch.arange(sphereH,-1,-1).long()
        self.register_buffer("pad_ind", pad_ind)

        kernel_shapes, dilations = compute_kernelshape(sphereH=sphereH,
                                                       fov=fov,
                                                       imgW=imgW,
                                                       dilation=dilation,
                                                       tied_weights=tied_weights,
                                                       kernel_shape_type=kernel_shape_type)
        self.register_buffer("dilations", dilations)
        KTN_CLS = KTN_ARCHS[arch]
        self.ktn = KTN_CLS(kernel, bias, kernel_shapes)
        self.ktn.initialize_weight(sphereH=sphereH,
                                   fov=fov,
                                   imgW=imgW,
                                   dilation=dilation,
                                   tied_weights=tied_weights)
        self.n_transform = kernel_shapes.size(0)

    def forward(self, X, rows=None):
        def process_padding(pad):
            pad_ind = self.pad_ind[-pad.size(2):]
            pad = pad.index_select(2, Variable(pad_ind, requires_grad=False))
            pad = torch.cat([pad[:,:,:,self.sphereW/2:], pad[:,:,:,:self.sphereW/2]], dim=3)
            return pad

        batch_size, n_in, iH, iW = X.size()
        if rows is None:
            rows = range(self.n_transform)

        # manual create outputs
        oH = len(rows) * self.tied_weights
        size = (batch_size, self.ktn.n_out, oH, iW)
        outputs = create_variable(size)

        for i, row in enumerate(rows):
            # prepare kernel
            kernel, bias = self.ktn(row)
            kH, kW = kernel.size()[-2:]

            dilation_h, dilation_w = self.dilations[row]

            # crop input for convolution
            pad_height = (kH - 1) / 2 * dilation_h
            pad_width = (kW - 1) / 2 * dilation_w
            top = row * self.tied_weights - pad_height
            bot = (row+1) * self.tied_weights + pad_height
            if top < 0:
                spill = -top
                pad = X[:,:,:spill,:]
                pad = process_padding(pad)
                x = torch.cat([pad, X[:,:,:bot,:]], dim=2)
            elif bot > self.sphereH:
                spill = bot - self.sphereH
                pad = X[:,:,-spill:,:]
                pad = process_padding(pad)
                x = torch.cat([X[:,:,top:,:], pad], dim=2)
            else:
                x = X[:,:,top:bot,:]
            x = torch.cat([x[:,:,:,-pad_width:], x, x[:,:,:,:pad_width]], dim=3)

            t = i * self.tied_weights
            b = t + self.tied_weights
            outputs[:,:,t:b,:] = F.conv2d(x, kernel, bias, dilation=(dilation_h, dilation_w))
        return outputs

    def update_group(self, group):
        self.ktn.update_group(group)


def compute_kernelshape(kernel_size=3, sphereH=320, fov=65.5, imgW=640, dilation=1, tied_weights=5, kernel_shape_type="dilated"):
    sphereW = sphereH * 2
    projection = SphereProjection(kernel_size=kernel_size,
                                  sphereH=sphereH,
                                  sphereW=sphereW,
                                  view_angle=fov,
                                  imgW=imgW)

    n_transform = (sphereH - 1) / tied_weights + 1
    kernel_shapes = numpy.zeros((n_transform, 2), dtype=numpy.int64)
    dilations = numpy.zeros((n_transform, 2), dtype=numpy.int64)
    center = sphereW / 2
    for y in xrange(n_transform):
        kernel_shape = numpy.zeros((tied_weights, 2), dtype=numpy.int64)
        for dy in xrange(tied_weights):
            row = y * tied_weights + dy
            Px, Py = projection.generate_grid(tilt=row)

            left = numpy.floor(Px.min())
            right = numpy.ceil(Px.max())
            top = numpy.floor(Py.min())
            bot = numpy.ceil(Py.max())

            kW = 2 * max(center-left, right-center) + 1
            kH = 2 * max(row-top, bot-row) + 1
            kernel_shape[dy] = kH, kW
        kH, kW = kernel_shape.max(axis=0)
        if kernel_shape_type == "dilated":
            kH, dilation_h = round_kernelshape(kH, dilation, sphereH)
            kW, dilation_w = round_kernelshape(kW, dilation, sphereW)
        else:
            kH = dilate_kernelshape(dilation, kH)
            kW = dilate_kernelshape(dilation, kW)
            dilation_h = dilation_w = dilation
        kernel_shapes[y] = kH, kW
        dilations[y] = dilation_h, dilation_w
    kernel_shapes = torch.from_numpy(kernel_shapes).type(torch.IntTensor)
    dilations = torch.from_numpy(dilations).type(torch.IntTensor)
    return kernel_shapes, dilations

def round_kernelshape(kW, dilation, sphereW):
    MAX_RADIUS = 32
    MAX_KERNEL_SIZE = 2 * MAX_RADIUS + 1
    dilated_w = dilate_kernelshape(dilation, kW)
    if dilated_w > MAX_KERNEL_SIZE:
        radius = (kW - 1) / 2
        dilation = min((radius - 1) / MAX_RADIUS + 1,
                       (sphereW - 1) / (2 * MAX_RADIUS))
        kW = MAX_KERNEL_SIZE
    else:
        kW = dilated_w
    return kW, dilation

def dilate_kernelshape(dilation, kW):
    half_W = (kW - 1) / 2
    n_w = (half_W - 1) / dilation + 1
    kW = n_w * 2 + 1
    return kW

if __name__ == "__main__":
    kernel_shapes = compute_kernelshape(imgW=40, dilation=2)
    print kernel_shapes

