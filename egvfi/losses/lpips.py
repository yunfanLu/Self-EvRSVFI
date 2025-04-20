#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Author ：Yunfan Lu (yunfanlu@ust.hk)
# Date   ：2023/2/9 16:27
import lpips
from torch.nn.modules.loss import _Loss


class LPIPS(_Loss):
    def __init__(self, net="alex"):
        super(LPIPS, self).__init__()
        self.net = net
        if net == "alex":
            self.lpips = lpips.LPIPS(net="alex")
        elif net == "vgg":
            self.lpips = lpips.LPIPS(net="vgg")
        else:
            raise ValueError(f"Unknown net: {net}")
        self.lpips = self.lpips.cuda()

    def forward(self, x, y):
        return self.lpips(x, y).mean()


class GlobalShutterReconstructionLPIPS(LPIPS):
    def forward(self, batch):
        gs_gt = batch["ground_truth_global_shutter_frames"]
        gs_recon = batch["reconstructed_global_shutter_frames"]
        B, N, C, H, W = gs_gt.shape
        loss = 0
        for i in range(N):
            i1 = gs_recon[i]
            i2 = gs_gt[:, i, :, :, :]
            loss = loss + super().forward(i1, i2)
        return loss / N
