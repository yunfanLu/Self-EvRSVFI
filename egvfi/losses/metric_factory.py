from absl.logging import info
from torch import nn

from egvfi.losses.lpips import GlobalShutterReconstructionLPIPS
from egvfi.losses.psnr import (
    GlobalShutterReconstructionPSNR,
    RollingShutterReconstructionFromGlobalShutterPSNR,
    RollingShutterReconstructionFromRollingShutterPSNR,
)
from egvfi.losses.ssim import (
    GlobalShutterReconstructionSSIM,
    RollingShutterReconstructionFromGlobalShutterSSIM,
    RollingShutterReconstructionFromRollingShutterSSIM,
)
from egvfi.losses.time_replayer_loss import (
    TimeReplayerReconstructionLPIPS,
    TimeReplayerReconstructionPSNR,
    TimeReplayerReconstructionSSIM,
)


def get_single_metric(config):
    if config.NAME == "GlobalShutterReconstructionPSNR":
        return GlobalShutterReconstructionPSNR()
    elif config.NAME == "RollingShutterReconstructionFromRollingShutterPSNR":
        return RollingShutterReconstructionFromRollingShutterPSNR()
    elif config.NAME == "RollingShutterReconstructionFromGlobalShutterPSNR":
        return RollingShutterReconstructionFromGlobalShutterPSNR()
    elif config.NAME == "GlobalShutterReconstructionLPIPS":
        return GlobalShutterReconstructionLPIPS()
    elif config.NAME == "RollingShutterReconstructionFromRollingShutterSSIM":
        return RollingShutterReconstructionFromRollingShutterSSIM()
    elif config.NAME == "RollingShutterReconstructionFromGlobalShutterSSIM":
        return RollingShutterReconstructionFromGlobalShutterSSIM()
    elif config.NAME == "GlobalShutterReconstructionSSIM":
        return GlobalShutterReconstructionSSIM()
    elif config.NAME == "time_replayer_psnr":
        return TimeReplayerReconstructionPSNR()
    elif config.NAME == "time_replayer_ssim":
        return TimeReplayerReconstructionSSIM()
    elif config.NAME == "time_replayer_lpips":
        return TimeReplayerReconstructionLPIPS()
    else:
        raise ValueError(f"Unknown config: {config}")


class MixedMetric(nn.Module):
    def __init__(self, configs):
        super(MixedMetric, self).__init__()
        self.metric = []
        self.eval = []
        for config in configs:
            self.metric.append(config.NAME)
            self.eval.append(get_single_metric(config))
        info(f"Init Mixed Metric: {configs}")

    def forward(self, batch):
        r = []
        for m, e in zip(self.metric, self.eval):
            r.append((m, e(batch)))
        return r
