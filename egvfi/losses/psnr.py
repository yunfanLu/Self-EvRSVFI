import torch
from absl.logging import info
from torch import nn


def almost_equal(x, y):
    return abs(x - y) < 1e-6


class _PSNR(nn.Module):
    def __init__(self):
        super(_PSNR, self).__init__()
        self.eps = torch.tensor(1e-10)

        info(f"Init PSNR:")
        info(f"  Note: the psnr max value is {-10 * torch.log10(self.eps)}")

    def forward(self, x, y):
        d = x - y
        mse = torch.mean(d * d) + self.eps
        psnr = -10 * torch.log10(mse)
        return psnr


class GlobalShutterReconstructionPSNR(_PSNR):
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


class RollingShutterReconstructionFromRollingShutterPSNR(_PSNR):
    def forward(self, batch):
        rs_0_from_rs = batch["rs_0_from_rs"]
        rs_1_from_rs = batch["rs_1_from_rs"]
        rs = batch["input_rolling_frames"]
        rs_0, rs_1 = rs[:, 0, ...], rs[:, 1, ...]
        loss = super().forward(rs_0_from_rs, rs_0) + super().forward(rs_1_from_rs, rs_1)
        return loss / 2


class RollingShutterReconstructionFromGlobalShutterPSNR(_PSNR):
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
