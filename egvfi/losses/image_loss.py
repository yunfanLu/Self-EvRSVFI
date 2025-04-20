import torch
from pudb import set_trace
from torch.nn.modules.loss import _Loss

from egvfi.losses.perceptual_loss import PerceptualLoss


class L1CharbonnierLossColor(_Loss):
    def __init__(self):
        super(L1CharbonnierLossColor, self).__init__()
        self.eps = 1e-6

    def forward(self, x, y):
        diff = torch.add(x, -y)
        diff_sq = diff * diff
        diff_sq_color = torch.mean(diff_sq, 1, True)
        error = torch.sqrt(diff_sq_color + self.eps)
        loss = torch.mean(error)
        return loss


class GlobalShutterReconstructionLoss(_Loss):
    def __init__(self, loss_type="l1-charbonnier"):
        super(GlobalShutterReconstructionLoss, self).__init__()
        if loss_type == "l1-charbonnier":
            self.loss = L1CharbonnierLossColor()
        elif loss_type == "perceptual-vgg-loss":
            self.loss = PerceptualLoss()
        else:
            raise NotImplementedError(f"Loss type {loss_type} is not implemented.")

    def forward(self, batch):
        gs_gt = batch["ground_truth_global_shutter_frames"]
        gs_recon = batch["reconstructed_global_shutter_frames"]
        B, N, C, H, W = gs_gt.shape
        loss = 0
        for i in range(N):
            i1 = gs_recon[i]
            i2 = gs_gt[:, i, :, :, :]
            loss = loss + self.loss(i1, i2)
        return loss / N


class RollingShutterReconstructionFromRollingShutter(_Loss):
    def __init__(self, loss_type="l1-charbonnier"):
        super(RollingShutterReconstructionFromRollingShutter, self).__init__()
        if loss_type == "l1-charbonnier":
            self.loss = L1CharbonnierLossColor()
        elif loss_type == "perceptual-vgg-loss":
            self.loss = PerceptualLoss()
        else:
            raise NotImplementedError(f"Loss type {loss_type} is not implemented.")

    def forward(self, batch):
        rs_0_from_rs = batch["rs_0_from_rs"]
        rs_1_from_rs = batch["rs_1_from_rs"]
        rs = batch["input_rolling_frames"]
        rs_0, rs_1 = rs[:, 0, ...], rs[:, 1, ...]
        loss = self.loss(rs_0_from_rs, rs_0) + self.loss(rs_1_from_rs, rs_1)
        return loss / 2


class RollingShutterReconstructionFromGlobalShutter(_Loss):
    def __init__(self, loss_type="l1-charbonnier"):
        super(RollingShutterReconstructionFromGlobalShutter, self).__init__()
        if loss_type == "l1-charbonnier":
            self.loss = L1CharbonnierLossColor()
        elif loss_type == "perceptual-vgg-loss":
            self.loss = PerceptualLoss()
        else:
            raise NotImplementedError(f"Loss type {loss_type} is not implemented.")

    def forward(self, batch):
        rs_0_from_gs = batch["rs_0_from_gs"]
        rs_1_from_gs = batch["rs_1_from_gs"]
        rs = batch["input_rolling_frames"]
        rs_0, rs_1 = rs[:, 0, ...], rs[:, 1, ...]
        loss = 0
        for rs_0_ in rs_0_from_gs:
            loss += self.loss(rs_0_, rs_0)
        for rs_1_ in rs_1_from_gs:
            loss += self.loss(rs_1_, rs_1)
        return loss / (len(rs_0_from_gs) + len(rs_1_from_gs))


class RS0toGSandRS1toGSLoss(_Loss):
    def __init__(self, loss_type="l1-charbonnier"):
        super(RS0toGSandRS1toGSLoss, self).__init__()
        if loss_type == "l1-charbonnier":
            self.loss = L1CharbonnierLossColor()
        elif loss_type == "perceptual-vgg-loss":
            self.loss = PerceptualLoss()
        else:
            raise NotImplementedError(f"Loss type {loss_type} is not implemented.")

    def forward(self, batch):
        gs_i_from_rs0_list = batch["gs_frame_from_rs0_list"]
        gs_i_from_rs1_list = batch["gs_frame_from_rs1_list"]
        loss = 0
        for gs_i_from_rs0, gs_i_from_rs1 in zip(gs_i_from_rs0_list, gs_i_from_rs1_list):
            loss += self.loss(gs_i_from_rs0, gs_i_from_rs1)
        return loss / len(gs_i_from_rs0_list)
