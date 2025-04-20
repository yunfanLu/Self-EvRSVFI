#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project :EG-VFI
@Author  :Yunfan Lu (yunfanlu@ust.hk)
@Date    :9/12/2022 15:02
"""
from egvfi.datasets.alpx_vfi_dataset import (
    get_alpx_vfi_dataset,
    get_alpx_vfi_rotaing_fan_dataset,
)
from egvfi.datasets.bs_ergb import get_timelen_pp
from egvfi.datasets.fastec_rs import (
    get_2x_frame_interpolate_3_frame_fastec,
    get_4x_frame_interpolate_5_frame_fastec,
    get_8x_frame_interpolate_9_frame_fastec,
    get_16x_frame_interpolate_17_frame_fastec,
    get_24x_frame_interpolate_25_frame_fastec,
    get_32x_frame_interpolate_33_frame_fastec,
)
from egvfi.datasets.gev_rs import (
    get_1x_frame_interpolate_1_frame,
    get_2x_frame_interpolate_3_frame,
    get_4x_frame_interpolate_5_frame,
    get_8x_frame_interpolate_9_frame,
    get_12x_frame_interpolate_13_frame,
    get_16x_frame_interpolate_17_frame,
    get_24x_frame_interpolate_25_frame,
    get_32x_frame_interpolate_33_frame,
    get_128x_frame_interpolate_33_frame,
)


def get_dataset(config):
    if config.NAME == "aplex_vfi_dataset":
        return get_alpx_vfi_dataset(
            alpx_vfi_root=config.alpx_vfi_root,
            moments=config.moments,
            random_crop_resolution=config.random_crop_resolution,
            low_resolution=config.low_resolution,
            is_test=config.is_test,
        )
    elif config.NAME == "get_alpx_vfi_rotaing_fan_dataset":
        return get_alpx_vfi_rotaing_fan_dataset(
            alpx_vfi_root=config.alpx_vfi_root,
            moments=config.moments,
            random_crop_resolution=config.random_crop_resolution,
            low_resolution=config.low_resolution,
        )
    elif config.NAME == "get_1x_frame_interpolate_1_frame":
        return get_1x_frame_interpolate_1_frame(
            root=config.root,
            center_cropped_height=config.center_cropped_height,
            random_cropped_width=config.random_cropped_width,
            sample_step=config.sample_step,
            events_moment_count=config.events_moment_count,
        )
    elif config.NAME == "get_2x_frame_interpolate_3_frame":
        return get_2x_frame_interpolate_3_frame(
            root=config.root,
            center_cropped_height=config.center_cropped_height,
            random_cropped_width=config.random_cropped_width,
            sample_step=config.sample_step,
            events_moment_count=config.events_moment_count,
        )
    elif config.NAME == "get_4x_frame_interpolate_5_frame":
        return get_4x_frame_interpolate_5_frame(
            root=config.root,
            center_cropped_height=config.center_cropped_height,
            random_cropped_width=config.random_cropped_width,
            sample_step=config.sample_step,
            events_moment_count=config.events_moment_count,
        )
    elif config.NAME == "get_8x_frame_interpolate_9_frame":
        return get_8x_frame_interpolate_9_frame(
            root=config.root,
            center_cropped_height=config.center_cropped_height,
            random_cropped_width=config.random_cropped_width,
            sample_step=config.sample_step,
            events_moment_count=config.events_moment_count,
        )
    elif config.NAME == "get_12x_frame_interpolate_13_frame":
        return get_12x_frame_interpolate_13_frame(
            root=config.root,
            center_cropped_height=config.center_cropped_height,
            random_cropped_width=config.random_cropped_width,
            sample_step=config.sample_step,
            events_moment_count=config.events_moment_count,
        )
    elif config.NAME == "get_16x_frame_interpolate_17_frame":
        return get_16x_frame_interpolate_17_frame(
            root=config.root,
            center_cropped_height=config.center_cropped_height,
            random_cropped_width=config.random_cropped_width,
            sample_step=config.sample_step,
            events_moment_count=config.events_moment_count,
        )
    elif config.NAME == "get_24x_frame_interpolate_25_frame":
        return get_24x_frame_interpolate_25_frame(
            root=config.root,
            center_cropped_height=config.center_cropped_height,
            random_cropped_width=config.random_cropped_width,
            sample_step=config.sample_step,
            events_moment_count=config.events_moment_count,
        )
    elif config.NAME == "get_32x_frame_interpolate_33_frame":
        return get_32x_frame_interpolate_33_frame(
            root=config.root,
            center_cropped_height=config.center_cropped_height,
            random_cropped_width=config.random_cropped_width,
            sample_step=config.sample_step,
            events_moment_count=config.events_moment_count,
        )
    elif config.NAME == "get_2x_frame_interpolate_3_frame_fastec":
        return get_2x_frame_interpolate_3_frame_fastec(
            root=config.root,
            center_cropped_height=config.center_cropped_height,
            random_cropped_width=config.random_cropped_width,
            sample_step=config.sample_step,
            events_moment_count=config.events_moment_count,
        )
    elif config.NAME == "get_4x_frame_interpolate_5_frame_fastec":
        return get_4x_frame_interpolate_5_frame_fastec(
            root=config.root,
            center_cropped_height=config.center_cropped_height,
            random_cropped_width=config.random_cropped_width,
            sample_step=config.sample_step,
            events_moment_count=config.events_moment_count,
        )
    elif config.NAME == "get_8x_frame_interpolate_9_frame_fastec":
        return get_8x_frame_interpolate_9_frame_fastec(
            root=config.root,
            center_cropped_height=config.center_cropped_height,
            random_cropped_width=config.random_cropped_width,
            sample_step=config.sample_step,
            events_moment_count=config.events_moment_count,
        )
    elif config.NAME == "get_16x_frame_interpolate_17_frame_fastec":
        return get_16x_frame_interpolate_17_frame_fastec(
            root=config.root,
            center_cropped_height=config.center_cropped_height,
            random_cropped_width=config.random_cropped_width,
            sample_step=config.sample_step,
            events_moment_count=config.events_moment_count,
        )
    elif config.NAME == "get_24x_frame_interpolate_25_frame_fastec":
        return get_24x_frame_interpolate_25_frame_fastec(
            root=config.root,
            center_cropped_height=config.center_cropped_height,
            random_cropped_width=config.random_cropped_width,
            sample_step=config.sample_step,
            events_moment_count=config.events_moment_count,
        )
    elif config.NAME == "get_32x_frame_interpolate_33_frame_fastec":
        return get_32x_frame_interpolate_33_frame_fastec(
            root=config.root,
            center_cropped_height=config.center_cropped_height,
            random_cropped_width=config.random_cropped_width,
            sample_step=config.sample_step,
            events_moment_count=config.events_moment_count,
        )
    elif config.NAME == "get_128x_frame_interpolate_33_frame":
        return get_128x_frame_interpolate_33_frame(
            root=config.root,
            center_cropped_height=config.center_cropped_height,
            random_cropped_width=config.random_cropped_width,
            sample_step=config.sample_step,
            events_moment_count=config.events_moment_count,
        )
    else:
        raise ValueError(f"Unknown dataset: {config.NAME}")
