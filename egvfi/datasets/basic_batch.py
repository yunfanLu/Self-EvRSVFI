#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Author ：Yunfan Lu (yunfanlu@ust.hk)
# Date   ：2022/12/24 15:57


def get_vfi_empty_batch():
    return {
        "input_rolling_frames": None,
        "input_rolling_frames_normal_timestamps": None,
        "target_global_frames": None,
        "target_global_frames_normal_timestamps": None,
        "events": None,
        "video_name": None,
        "first_rolling_frame_name": None,
    }


def get_vfi_output_batch():
    return {
        "displacement_field": None,
        "input_rolling_frames": None,
        "rs_0_from_rs": None,
        "field_rs0_to_gs": None,
        "rs_1_from_rs": None,
        "gs_frame_from_rs0_list": None,
        "gs_frame_from_rs1_list": None,
        "field_rs1_to_gs": None,
        "confidence_rs0_to_gs": None,
        "rs_0_from_gs": None,
        "flow_rs_0_to_1": None,
        "rs_1_from_gs": None,
        "flow_rs_1_to_0": None,
        "reconstructed_global_shutter_frames": None,
        "ground_truth_global_shutter_frames": None,
        # TimeReplayer
        "gs_0_from_gs_trp": None,
        "gs_1_from_gs_trp": None,
        "gs_trp": None,
    }


def get_vfi_inference_batch():
    return {
        "reconstructed_rolling_frames": None,
        "ground_truth_rolling_frames": None,
        "reconstructed_global_frames": None,
        "ground_truth_global_frames": None,
        "reconstructed_events": None,
        "ground_truth_events": None,
    }
