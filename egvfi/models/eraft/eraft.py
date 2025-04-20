#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Author ：Yunfan Lu (yunfanlu@ust.hk)
# Date   ：2023/2/4 18:50

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast

from egvfi.models.eraft.corr import CorrBlock
from egvfi.models.eraft.extractor import BasicEncoder
from egvfi.models.eraft.image_padder import ImagePadder
from egvfi.models.eraft.update import BasicUpdateBlock
from egvfi.models.eraft.utils import coords_grid, upflow8


class ERAFT(nn.Module):
    def __init__(self, subtype, n_first_channels, mixed_precision=False):
        super(ERAFT, self).__init__()
        self.mixed_precision = mixed_precision
        self.image_padder = ImagePadder(min_size=32)
        self.subtype = subtype
        assert self.subtype == "standard" or self.subtype == "warm_start"
        self.hidden_dim = hdim = 128
        self.context_dim = cdim = 128
        self.corr_levels = 4
        self.corr_radius = 4
        # feature network, context network, and update block
        self.fnet = BasicEncoder(
            output_dim=256,
            norm_fn="instance",
            dropout=0,
            n_first_channels=n_first_channels,
        )
        self.cnet = BasicEncoder(
            output_dim=hdim + cdim,
            norm_fn="batch",
            dropout=0,
            n_first_channels=n_first_channels,
        )
        self.update_block = BasicUpdateBlock(self.corr_levels, self.corr_radius, hidden_dim=hdim)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def initialize_flow(self, img):
        """Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H // 8, W // 8).to(img.device)
        coords1 = coords_grid(N, H // 8, W // 8).to(img.device)
        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1

    def upsample_flow(self, flow, mask):
        """Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination"""
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)
        up_flow = F.unfold(8 * flow, [3, 3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)
        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, 8 * H, 8 * W)

    def forward(self, image1, image2, iters=12, flow_init=None, upsample=True):
        """Estimate optical flow between a pair of frames"""
        # Pad Image (for flawless up & downsampling)
        image1 = self.image_padder.pad(image1)
        image2 = self.image_padder.pad(image2)
        image1 = image1.contiguous()
        image2 = image2.contiguous()
        hdim = self.hidden_dim
        cdim = self.context_dim
        # run the feature network
        with autocast(enabled=self.mixed_precision):
            fmap1, fmap2 = self.fnet([image1, image2])
        fmap1 = fmap1.float()
        fmap2 = fmap2.float()
        corr_fn = CorrBlock(fmap1, fmap2, radius=self.corr_radius)
        # run the context network
        with autocast(enabled=self.mixed_precision):
            if self.subtype == "standard" or self.subtype == "warm_start":
                cnet = self.cnet(image2)
            else:
                raise Exception
            net, inp = torch.split(cnet, [hdim, cdim], dim=1)
            net = torch.tanh(net)
            inp = torch.relu(inp)
        # Initialize Grids. First channel: x, 2nd channel: y. Image is just used to get the shape
        coords0, coords1 = self.initialize_flow(image1)
        if flow_init is not None:
            coords1 = coords1 + flow_init
        flow_predictions = []
        for itr in range(iters):
            coords1 = coords1.detach()
            corr = corr_fn(coords1)  # index correlation volume
            flow = coords1 - coords0
            with autocast(enabled=self.mixed_precision):
                net, up_mask, delta_flow = self.update_block(net, inp, corr, flow)
            # F(t+1) = F(t) + \Delta(t)
            coords1 = coords1 + delta_flow
            # upsample predictions
            if up_mask is None:
                flow_up = upflow8(coords1 - coords0)
            else:
                flow_up = self.upsample_flow(coords1 - coords0, up_mask)
            flow_predictions.append(self.image_padder.unpad(flow_up))
        return coords1 - coords0, flow_predictions
