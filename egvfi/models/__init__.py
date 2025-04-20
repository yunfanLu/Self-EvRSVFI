from egvfi.models.cgsvr import ContinueGlobalShutterVideoReconstruction
from egvfi.models.timereplayer.timereplayer import TimeReplayerMultiFrameTrainer


def get_model(config):
    if config.NAME == "CGSVR":
        return ContinueGlobalShutterVideoReconstruction(
            events_moments=config.events_moments,
            image_count=config.image_count,
            image_high=config.image_high,
            image_width=config.image_width,
            e_raft_in_channels=config.e_raft_in_channels,
            time_bin_size=config.time_bin_size,
            optical_flow_estimator=config.optical_flow_estimator,
            erfat_iter=config.erfat_iter,
            e_raft_pretrained=config.e_raft_pretrained,
            invert_event_in_raft_flow_and_fusion=config.invert_event_in_raft_flow_and_fusion,
            rs_timestamps=config.rs_timestamps,
            gs_timestamps=config.gs_timestamps,
            mask_high_sample_count=config.mask_high_sample_count,
            mask_time_sample_count=config.mask_time_sample_count,
            intermediate_visualization=config.intermediate_visualization,
        )
    elif config.NAME == "TimeReplayerTrainer":
        return TimeReplayerMultiFrameTrainer(
            height=config.image_high,
            width=config.image_width,
            event_voxel_grid_moments=config.time_replayer_events_moments,
            is_train=config.is_train,
        )
