# Reference: https://github.com/Po-Hsun-Su/pytorch-ssim
from math import exp

import torch
import torch.nn.functional as F
from absl.logging import info
from torch import nn
from torch.autograd import Variable


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-((x - window_size // 2) ** 2) / float(2 * sigma**2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    if img1.is_cuda:
        window = window.cuda(img1.get_device())

    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class SSIM(nn.Module):
    def __init__(
        self,
        value_range=1.0,
        window_size=11,
        size_average=True,
    ):
        super(SSIM, self).__init__()
        self.value_range = value_range
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 3
        self.window = create_window(self.window_size, self.channel)
        self.eps = 0.00001
        info(f"Init SSIM:")
        info(f"  value_range    : {value_range}")
        info(f"  window_size    : {window_size}")
        info(f"  size_average   : {size_average}")

    def forward(self, img1, img2):
        if img1.dim() == 5:
            img1 = torch.flatten(img1, start_dim=0, end_dim=1)
            img2 = torch.flatten(img2, start_dim=0, end_dim=1)

        img1 = img1 / self.value_range
        img2 = img2 / self.value_range
        (_, channel, _, _) = img1.size()
        if self.channel != channel:
            self.channel = channel
            self.window = create_window(self.window_size, self.channel)
        return _ssim(
            img1,
            img2,
            self.window,
            self.window_size,
            channel,
            self.size_average,
        )


def ssim(img1, img2, window_size=11, size_average=True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)
    window = window.type_as(img1)
    return _ssim(img1, img2, window, window_size, channel, size_average)


class GlobalShutterReconstructionSSIM(SSIM):
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


class RollingShutterReconstructionFromRollingShutterSSIM(SSIM):
    def forward(self, batch):
        rs_0_from_rs = batch["rs_0_from_rs"]
        rs_1_from_rs = batch["rs_1_from_rs"]
        rs = batch["input_rolling_frames"]
        rs_0, rs_1 = rs[:, 0, ...], rs[:, 1, ...]
        loss = super().forward(rs_0_from_rs, rs_0) + super().forward(rs_1_from_rs, rs_1)
        return loss / 2


class RollingShutterReconstructionFromGlobalShutterSSIM(SSIM):
    def forward(self, batch):
        rs_0_from_gs = batch["rs_0_from_gs"]
        rs_1_from_gs = batch["rs_1_from_gs"]
        rs = batch["input_rolling_frames"]
        rs_0, rs_1 = rs[:, 0, ...], rs[:, 1, ...]
        loss = 0
        for rs_0_ in rs_0_from_gs:
            loss += super().forward(rs_0_, rs_0)
        for rs_1_ in rs_1_from_gs:
            loss += super().forward(rs_1_, rs_1)
        return loss / (len(rs_0_from_gs) + len(rs_1_from_gs))
