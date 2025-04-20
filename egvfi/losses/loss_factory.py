from absl.logging import info
from torch.nn.modules.loss import _Loss

from egvfi.losses.image_loss import (
    GlobalShutterReconstructionLoss,
    RS0toGSandRS1toGSLoss,
    RollingShutterReconstructionFromGlobalShutter,
    RollingShutterReconstructionFromRollingShutter,
)
from egvfi.losses.time_replayer_loss import TimeReplayerReconstructionLoss
from egvfi.losses.veriation_loss import DisplacementFieldLoss


def get_single_loss(config):
    if config.NAME == "GlobalShutterReconstructionLoss":
        return GlobalShutterReconstructionLoss(config.LOSS_TYPE)
    elif config.NAME == "RollingShutterReconstructionFromRollingShutter":
        return RollingShutterReconstructionFromRollingShutter(config.LOSS_TYPE)
    elif config.NAME == "RollingShutterReconstructionFromGlobalShutter":
        return RollingShutterReconstructionFromGlobalShutter(config.LOSS_TYPE)
    elif config.NAME == "RS0toGSandRS1toGS":
        return RS0toGSandRS1toGSLoss(config.LOSS_TYPE)
    elif config.NAME == "rs_to_rs-perceptual-loss":
        return RollingShutterReconstructionFromRollingShutter(config.LOSS_TYPE)
    elif config.NAME == "rs_gs_rs-perceptual-loss":
        return RollingShutterReconstructionFromGlobalShutter(config.LOSS_TYPE)
    elif config.NAME == "DisplacementFieldLoss":
        return DisplacementFieldLoss()
    elif config.NAME == "time_replayer_reconstruction":
        return TimeReplayerReconstructionLoss()
    else:
        raise ValueError(f"Unknown config: {config}")


class MixedLoss(_Loss):
    def __init__(self, configs):
        super(MixedLoss, self).__init__()
        self.loss = []
        self.weight = []
        self.criterion = []
        for item in configs:
            self.loss.append(item.NAME)
            self.weight.append(item.WEIGHT)
            self.criterion.append(get_single_loss(item))
        info(f"Init Mixed Loss: {configs}")

    def forward(self, batch):
        name_to_loss = []
        total = 0
        for n, w, fun in zip(self.loss, self.weight, self.criterion):
            tmp = fun(batch)
            name_to_loss.append((n, tmp))
            total = total + tmp * w
        return total, name_to_loss
