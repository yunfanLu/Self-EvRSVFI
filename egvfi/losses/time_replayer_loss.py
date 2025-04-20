#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Author ：Yunfan Lu (yunfanlu@ust.hk)
# Date   ：2023/3/2 21:49
import torch
from torch.nn.modules.loss import _Loss

from egvfi.losses.lpips import LPIPS
from egvfi.losses.psnr import _PSNR
from egvfi.losses.ssim import SSIM


class TimeReplayerReconstructionLoss(_Loss):
    def __init__(self):
        super(TimeReplayerReconstructionLoss, self).__init__()

    def forward(self, batch):
        GS_list = batch["gs_gt"]
        B, Ngs, C, H, W = GS_list.shape
        I0 = GS_list[:, 0, :, :, :]
        I1 = GS_list[:, -1, :, :, :]
        #
        out_I0_list = batch["gs_0_from_gs_trp"]
        out_I1_list = batch["gs_1_from_gs_trp"]
        #
        loss = 0
        count = 0
        for i0 in out_I0_list:
            loss = loss + torch.mean(torch.abs(i0 - I1))
            count = count + 1
        for i1 in out_I1_list:
            loss = loss + torch.mean(torch.abs(i1 - I0))
            count = count + 1
        loss = loss / count
        return loss


class TimeReplayerReconstructionPSNR(_PSNR):
    def forward(self, batch):
        GS_list = batch["gs_gt"]
        B, Ngs, C, H, W = GS_list.shape
        out_gs_list = batch["gs_trp"]
        psnr = 0
        count = 0
        for i in range(1, Ngs - 1):
            psnr = psnr + super().forward(out_gs_list[i - 1], GS_list[:, i, :, :, :])
            count = count + 1
        psnr = psnr / count
        return psnr


class TimeReplayerReconstructionSSIM(SSIM):
    def forward(self, batch):
        GS_list = batch["gs_gt"]
        B, Ngs, C, H, W = GS_list.shape
        out_gs_list = batch["gs_trp"]
        ssim = 0
        count = 0
        for i in range(1, Ngs - 1):
            ssim = ssim + super().forward(out_gs_list[i - 1], GS_list[:, i, :, :, :])
            count = count + 1
        ssim = ssim / count
        return ssim


class TimeReplayerReconstructionLPIPS(LPIPS):
    def forward(self, batch):
        GS_list = batch["gs_gt"]
        B, Ngs, C, H, W = GS_list.shape
        out_gs_list = batch["gs_trp"]
        lpips = 0
        count = 0
        for i in range(1, Ngs - 1):
            lpips = lpips + super().forward(out_gs_list[i - 1], GS_list[:, i, :, :, :])
            count = count + 1
        lpips = lpips / count
        return lpips
