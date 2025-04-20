#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Author ：Yunfan Lu (yunfanlu@ust.hk)
# Date   ：2023/2/6 10:16
import torch
from torch import nn


class GridGradientCentralDiff:
    def __init__(self, nc, padding=True, diagonal=False):
        self.conv_x = nn.Conv2d(nc, nc, kernel_size=2, stride=1, bias=False)
        self.conv_y = nn.Conv2d(nc, nc, kernel_size=2, stride=1, bias=False)
        self.conv_xy = None
        if diagonal:
            self.conv_xy = nn.Conv2d(nc, nc, kernel_size=2, stride=1, bias=False)
        self.padding = None
        if padding:
            self.padding = nn.ReplicationPad2d([0, 1, 0, 1])
        fx = torch.zeros(nc, nc, 2, 2).float().cuda()
        fy = torch.zeros(nc, nc, 2, 2).float().cuda()
        if diagonal:
            fxy = torch.zeros(nc, nc, 2, 2).float().cuda()
        fx_ = torch.tensor([[1, -1], [0, 0]]).cuda()
        fy_ = torch.tensor([[1, 0], [-1, 0]]).cuda()
        if diagonal:
            fxy_ = torch.tensor([[1, 0], [0, -1]]).cuda()
        for i in range(nc):
            fx[i, i, :, :] = fx_
            fy[i, i, :, :] = fy_
            if diagonal:
                fxy[i, i, :, :] = fxy_
        self.conv_x.weight = nn.Parameter(fx)
        self.conv_y.weight = nn.Parameter(fy)
        if diagonal:
            self.conv_xy.weight = nn.Parameter(fxy)

    def __call__(self, grid_2d):
        _image = grid_2d
        if self.padding is not None:
            _image = self.padding(_image)
        dx = self.conv_x(_image)
        dy = self.conv_y(_image)
        if self.conv_xy is not None:
            dxy = self.conv_xy(_image)
            return dx, dy, dxy
        return dx, dy


class VariationLoss(nn.Module):
    def __init__(self, nc):
        super(VariationLoss, self).__init__()
        self.grad_fn = GridGradientCentralDiff(nc)

    def forward(self, image, weight=None, mean=True):
        dx, dy = self.grad_fn(image)
        variation = dx**2 + dy**2
        if weight is not None:
            variation = variation * weight.float()
            if mean:
                return variation.sum() / weight.sum()
        if mean:
            return variation.mean()
        return variation.sum()


class DisplacementFieldLoss(nn.Module):
    def __init__(self, nc=2):
        super(DisplacementFieldLoss, self).__init__()
        self.loss = VariationLoss(nc)

    def forward(self, batch):
        # [B, 2, T, H, W]
        displacement_field_3d = batch["displacement_field"]
        B, C, T, H, W = displacement_field_3d.shape
        loss = 0
        for i in range(T):
            # [B, 2, H, W]
            displacement_field_2d = displacement_field_3d[:, :, i, :, :]
            loss = loss + self.loss(displacement_field_2d)
        loss = loss / T
        return loss
