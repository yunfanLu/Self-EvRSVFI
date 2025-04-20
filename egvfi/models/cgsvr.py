#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Author ：Yunfan Lu (yunfanlu@ust.hk)
# Date   ：2023/2/5 09:33
from os.path import dirname, join

import torch
from absl.logging import info
from torch import nn
from torchvision.models.optical_flow import Raft_Large_Weights, raft_large

from egvfi.datasets.basic_batch import get_vfi_output_batch
from egvfi.functions.backward_warp import backward_warp
from egvfi.functions.rolling_temporal_mask import (
    rolling_shutter_to_global_shutter_temporal_mask,
    rolling_shutter_to_rolling_shutter_temporal_mask,
)
from egvfi.models.eraft.eraft import ERAFT
from egvfi.models.unet.unet_2d import UNet


class ContinueGlobalShutterVideoReconstruction(nn.Module):
    def __init__(
        self,
        events_moments,
        image_count,
        image_high,
        image_width,
        e_raft_in_channels,
        time_bin_size,
        optical_flow_estimator,
        erfat_iter,
        e_raft_pretrained,
        invert_event_in_raft_flow_and_fusion,
        rs_timestamps,
        gs_timestamps,
        mask_high_sample_count,
        mask_time_sample_count,
        intermediate_visualization,
    ):
        """
        this model is used to reconstruct the global shutter video from the rolling shutter video and the event stream.
        :param events_moments: the event moments, e.g. 520.
        :param image_count: the image count, e.g. 5.
        :param image_high: the image high, e.g. 240.
        :param image_width: the image width, e.g. 320.
        :param e_raft_in_channels: the input channel of e_raft, e.g. 4.
        :param time_bin_size: the time bin size, e.g. 104.
        :param rs_timestamps: N, 2 (start, end)
        :param gs_timestamps: M, 1 (start)
        """
        super(ContinueGlobalShutterVideoReconstruction, self).__init__()
        # 0. Check the input
        if optical_flow_estimator == "E-RAFT":
            assert events_moments // e_raft_in_channels == time_bin_size
        assert len(rs_timestamps) == 2, "Not support two rolling shutter image now."
        # 1. Global configuration
        self.image_count = image_count
        self.events_moments = events_moments
        self.image_high = image_high
        self.image_width = image_width
        self.e_raft_in_channels = e_raft_in_channels
        self.time_bin_size = time_bin_size
        # rolling shutter exposure timestamps and global shutter exposure timestamps
        self.rs_timestamps = rs_timestamps
        self.rs_count = len(rs_timestamps)
        self.gs_timestamps = gs_timestamps
        self.gs_count = len(gs_timestamps)
        # 1.1 Monte-Carlo sampling and generate the temporal mask
        self.mask_high_sample_count = mask_high_sample_count
        self.mask_time_sample_count = mask_time_sample_count
        (self.rs2gs_mask, self.rs2gs_mask_2d) = self._generate_rs_to_gs_temporal_mask()
        (self.rs2rs_mask, self.rs2rs_mask_2d) = self._generate_rs_to_rs_temporal_mask()
        self.intermediate_visualization = intermediate_visualization
        # 1.2 eraft
        self.erfat_iter = erfat_iter
        # 2. Build the model
        # self.raft = raft_large(weights=Raft_Large_Weights.DEFAULT, progress=True)
        self.optical_flow_estimator = optical_flow_estimator
        if self.optical_flow_estimator == "E-RAFT":
            self.e_raft_pretrained = e_raft_pretrained
            self.e_raft = ERAFT(subtype="standard", n_first_channels=self.e_raft_in_channels)
            self.invert_event_in_raft_flow_and_fusion = invert_event_in_raft_flow_and_fusion
            self.construct_displacement_field = nn.Conv3d(4, 2, kernel_size=1, stride=1, padding=0)
            # 5. load e_raft
            if self.e_raft_pretrained:
                checkpoint = torch.load(join(dirname(__file__), "dsec.tar"))
                self.e_raft.load_state_dict(checkpoint["model"])
        elif self.optical_flow_estimator == "U-Net":
            self.unet_optical_flow = UNet(in_channels=self.events_moments + 6, out_channels=2 * self.time_bin_size - 2)
        # 3. Generate the rolling shutter to global shutter temporal mask
        # self._generate_displacement_mask()
        # 4. Build the context model
        self.refine = UNet(in_channels=16, out_channels=6)
        self._info()

    def _info(self):
        info("ContinueGlobalShutterVideoReconstruction")
        info(f"  image_count: {self.image_count}")
        info(f"  events_moments: {self.events_moments}")
        info(f"  image_high: {self.image_high}")
        info(f"  image_width: {self.image_width}")
        info(f"  time_bin_size: {self.time_bin_size}")
        info(f"  optical_flow_estimator: {self.optical_flow_estimator}")
        if self.optical_flow_estimator == "E-RAFT":
            info(f"  e_raft_in_channels: {self.e_raft_in_channels}")
            info(f"  erfat_iter: {self.erfat_iter}")
            info(f"  invert_event_in_raft_flow_and_fusion: {self.invert_event_in_raft_flow_and_fusion}")
            info(f"  e_raft_pretrained: {self.e_raft_pretrained}")
        info(f"  rs_timestamps: {self.rs_timestamps}")
        info(f"  gs_timestamps: {self.gs_timestamps}")
        info(f"  mask_high_sample_count: {self.mask_high_sample_count}")
        info(f"  mask_time_sample_count: {self.mask_time_sample_count}")
        info(f"  rs2gs_mask:")
        for i in range(self.rs_count):
            for j in range(self.gs_count):
                info(f"    rs{i} to gs{j}: {self.rs2gs_mask[i][j].shape}")
        info(f"  rs2rs_mask:")
        for i in range(self.rs_count):
            for j in range(self.rs_count):
                info(f"    rs{i} to rs{j}: {self.rs2rs_mask[i][j].shape}")
        info(f"  intermediate_visualization: {self.intermediate_visualization}")

    def _rolling_temporal_mask_for_3d_field(self, mask):
        # H, T -> 2, W, H, T -> 2, T, H, W
        return mask.unsqueeze(0).unsqueeze(0).expand(2, self.image_width, -1, -1).permute(0, 3, 2, 1).cuda()

    def _generate_rs_to_gs_temporal_mask(self):
        info(f"Generate the rolling image to global shutter temporal mask")
        rs_to_gs_mask = [[None for _ in range(self.gs_count)] for _ in range(self.rs_count)]
        rs_to_gs_mask_2d = [[None for _ in range(self.gs_count)] for _ in range(self.rs_count)]
        for i in range(self.rs_count):
            rs_start, rs_end = self.rs_timestamps[i]
            for j in range(self.gs_count):
                gs_time = self.gs_timestamps[j]
                info(f"  rs{i} to gs{j}: [{rs_start} -> {rs_end}] -> {gs_time}")
                rs_i_to_gs_j = rolling_shutter_to_global_shutter_temporal_mask(
                    rolling_start_time=rs_start,
                    rolling_end_time=rs_end,
                    global_time=gs_time,
                    high=self.image_high,
                    time_bins=self.time_bin_size - 1,
                    high_sample_count=self.mask_high_sample_count,
                    time_sample_count=self.mask_time_sample_count,
                )
                rs_to_gs_mask_2d[i][j] = rs_i_to_gs_j
                rs_to_gs_mask[i][j] = self._rolling_temporal_mask_for_3d_field(rs_i_to_gs_j)
        return rs_to_gs_mask, rs_to_gs_mask_2d

    def _generate_rs_to_rs_temporal_mask(self):
        info(f"Generate the rolling image to rolling shutter temporal mask")
        rs_to_rs_mask = [[None for _ in range(self.rs_count)] for _ in range(self.rs_count)]
        rs_to_rs_mask_2d = [[None for _ in range(self.rs_count)] for _ in range(self.rs_count)]
        for i in range(self.rs_count):
            for j in range(self.rs_count):
                rs1_start, rs1_end = self.rs_timestamps[i]
                rs2_start, rs2_end = self.rs_timestamps[j]
                rs_i_to_rs_j_mask = rolling_shutter_to_rolling_shutter_temporal_mask(
                    rolling_start_time_begin=rs1_start,
                    rolling_end_time_begin=rs1_end,
                    rolling_start_time_end=rs2_start,
                    rolling_end_time_end=rs2_end,
                    high=self.image_high,
                    time_bins=self.time_bin_size - 1,
                    high_sample_count=self.mask_high_sample_count,
                    time_sample_count=self.mask_time_sample_count,
                )
                rs_to_rs_mask_2d[i][j] = rs_i_to_rs_j_mask
                rs_to_rs_mask[i][j] = self._rolling_temporal_mask_for_3d_field(rs_i_to_rs_j_mask)
        return rs_to_rs_mask, rs_to_rs_mask_2d

    def forward(self, batch):
        # 1. load data from batch
        rs_images = batch["input_rolling_frames"]  # B, N, C, H, W. N = 2 for left and right.
        B, N, C, H, W = rs_images.shape
        events = batch["events"]  # - B, NE, H, W. NE is the event moment count.
        # 2. extract displacement field
        displacement_field = self._estimate_3d_displacement_field(events, rs_images)
        # 3. rolling to global
        (
            gs_refined,
            field_rs0_to_gs,
            field_rs1_to_gs,
            confidence_rs0_to_gs,
            gs_frame_from_rs0_list,
            gs_frame_from_rs1_list,
        ) = self._reconstruct_global_from_rolling(B, displacement_field, rs_images)
        # 4. rolling to rolling
        (
            rs_0_from_rs,
            rs_1_from_rs,
            flow_rs_0_to_1,
            flow_rs_1_to_0,
        ) = self._reconstruct_rolling_from_rolling(B, displacement_field, rs_images)
        # 5. cycle gs to rs
        RS0_from_gs, RS1_from_gs = self._reconstruct_rolling_from_global(B, displacement_field, gs_refined)
        # 5. construct the output
        vfi_output = get_vfi_output_batch()
        vfi_output["input_rolling_frames"] = rs_images
        vfi_output["displacement_field"] = displacement_field # type: ignore
        # 5.1 rs to gs
        vfi_output["reconstructed_global_shutter_frames"] = gs_refined
        if self.intermediate_visualization:
            vfi_output["field_rs0_to_gs"] = field_rs0_to_gs
            vfi_output["field_rs1_to_gs"] = field_rs1_to_gs
            vfi_output["confidence_rs0_to_gs"] = confidence_rs0_to_gs
        # 5.2 rs to rs
        vfi_output["rs_0_from_rs"] = rs_0_from_rs
        vfi_output["rs_1_from_rs"] = rs_1_from_rs
        if self.intermediate_visualization:
            vfi_output["flow_rs_0_to_1"] = flow_rs_0_to_1
            vfi_output["flow_rs_1_to_0"] = flow_rs_1_to_0
        # 5.3 rs to gs to rs
        vfi_output["rs_0_from_gs"] = RS0_from_gs
        vfi_output["rs_1_from_gs"] = RS1_from_gs
        if "target_global_frames" in batch.keys():
            vfi_output["ground_truth_global_shutter_frames"] = batch["target_global_frames"]
        # 5.4 rs0 to gs, rs1 to gs
        vfi_output["gs_frame_from_rs0_list"] = gs_frame_from_rs0_list
        vfi_output["gs_frame_from_rs1_list"] = gs_frame_from_rs1_list
        return vfi_output

    def _estimate_optical_flow_from_events(self, events):
        """
        :param events: B, NE, H, W
        :return: B, 2, T, H, W
        """
        ec = self.e_raft_in_channels
        flow_3d = []
        e_ls_flow = None
        for i in range(self.time_bin_size - 1):
            ll, mm, rr = i * ec, (i + 1) * ec, (i + 2) * ec
            e0, e1 = events[:, ll:mm, :, :], events[:, mm:rr, :, :]
            e_ls_flow, flows = self.e_raft(e0, e1, iters=self.erfat_iter, flow_init=e_ls_flow)
            flow_3d.append(flows[-1])
        flow_3d = torch.stack(flow_3d, dim=1)
        # B, T, 2, H, W -> B, 2, T, H, W
        flow_3d = flow_3d.permute(0, 2, 1, 3, 4)
        return flow_3d

    def _estimate_3d_displacement_field(self, events, rs_images):
        # Frames: todo(2023-02-12): use the rolling shutter images to estimate the 3d displacement field.
        # # 2.1. extract optical flow from rolling shutter images
        # flow_l_to_r = self.raft(left_rs_image, right_rs_image)[-1]
        # flow_r_to_l = self.raft(right_rs_image, left_rs_image)[-1]
        # Events
        if self.optical_flow_estimator == "E-RAFT":
            flow_3d_l_to_r = self._estimate_optical_flow_from_events(events)
            if self.invert_event_in_raft_flow_and_fusion == "ONE-FLOW":
                return flow_3d_l_to_r
            elif self.invert_event_in_raft_flow_and_fusion == "INV-Event-CONV3D":
                # Invert the events and estimate the flow again.
                events = torch.flip(events, dims=[1]) * -1
                flow_3d_r_to_l = self._estimate_optical_flow_from_events(events)
                flow_3d_r_to_l = torch.flip(flow_3d_r_to_l, dims=[1]) * -1
                # B, 2, T, H, W -> B, 4, T, H, W
                flow_3d = torch.concat([flow_3d_l_to_r, flow_3d_r_to_l], dim=1)
                # B, 4, T, H, W -> B, 2, T, H, W
                displacement_field = self.construct_displacement_field(flow_3d)
                return displacement_field
            else:
                raise NotImplementedError
        elif self.optical_flow_estimator == "U-Net":
            flow_3d_l_to_r = self._estimate_optical_flow_by_unet(events, rs_images)
            return flow_3d_l_to_r

    def _estimate_optical_flow_by_unet(self, events, rs_images):
        """
        :param events: B, NE, H, W
        :param rs_images: B, 2, C, H, W
        :return: B, 2, T, H, W
        """
        # B, 2, C, H, W -> B, 2C, H, W
        B, _, C, H, W = rs_images.shape
        rs_images = rs_images.reshape(B, 2 * C, H, W)
        # B, NE, H, W -> B, NE + 2C, H, W
        rs_event = torch.cat([rs_images, events], dim=1)
        flow = self.unet_optical_flow(rs_event)
        # B, 2T, H, W -> B, 2, T, H, W
        B, T2, H, W = flow.shape
        flow = flow.reshape(B, 2, T2 // 2, H, W)
        return flow

    def _reconstruct_rolling_from_rolling(self, batch_size, displacement_field, rs_images):
        """
        :param batch_size:
        :param displacement_field:
        :param rs_images:
        :return:
            RS0_from_RS1: B, C, H, W. RS0 reconstructed from RS1
            RS1_from_RS0: B, C, H, W. RS1 reconstructed from RS0
            F_0_to_1: B, 2, H, W. Flow RS0 -> RS1
            F_1_to_0: B, 2, H, W. Flow RS1 -> RS0
        """
        RS0, RS1 = rs_images[:, 0, ...], rs_images[:, 1, ...]
        # RS0 -> RS1
        F_0_to_1 = self.rs2rs_mask[0][1]
        F_0_to_1 = F_0_to_1.unsqueeze(0).repeat(batch_size, 1, 1, 1, 1)
        F_0_to_1 = F_0_to_1 * displacement_field
        F_0_to_1 = torch.sum(F_0_to_1, dim=2)
        RS1_from_RS0 = backward_warp(RS0, F_0_to_1)
        # RS1 -> RS0
        F_1_to_0 = self.rs2rs_mask[1][0]
        F_1_to_0 = F_1_to_0.unsqueeze(0).repeat(batch_size, 1, 1, 1, 1)
        F_1_to_0 = F_1_to_0 * displacement_field
        F_1_to_0 = torch.sum(F_1_to_0, dim=2)
        RS0_from_RS1 = backward_warp(RS1, F_1_to_0)
        return RS0_from_RS1, RS1_from_RS0, F_0_to_1, F_1_to_0

    def _reconstruct_global_from_rolling(self, batch_size, displacement_field, rs_images):
        """
        :param batch_size:
        :param displacement_field:
        :param rs_images:
        :return: rs_to_gs, rs i to gs j.
        """
        RS0, RS1 = rs_images[:, 0, ...], rs_images[:, 1, ...]
        gs = []
        field_rs0_to_gs = []
        field_rs1_to_gs = []
        gs_frame_from_rs0_list = []
        gs_frame_from_rs1_list = []
        confidence_rs0_to_gs = []
        for j in range(self.gs_count):
            # RS0 -> GS
            gs_i_to_rs0_mask = self.rs2gs_mask[0][j].unsqueeze(0).repeat(batch_size, 1, 1, 1, 1)
            gs_i_to_rs0_field = gs_i_to_rs0_mask * displacement_field
            # B, 2, T, H, W -> B, 2, H, W
            gs_i_to_rs0_field = torch.sum(gs_i_to_rs0_field, dim=2)
            gs_i_from_rs0 = backward_warp(RS0, gs_i_to_rs0_field)
            # RS1 -> GS
            gs_i_to_rs1_mask = self.rs2gs_mask[1][j].unsqueeze(0).repeat(batch_size, 1, 1, 1, 1)
            gs_i_to_rs1_field = gs_i_to_rs1_mask * displacement_field
            # B, 2, T, H, W -> B, 2, H, W
            gs_i_to_rs1_field = torch.sum(gs_i_to_rs1_field, dim=2)
            gs_i_from_rs1 = backward_warp(RS1, gs_i_to_rs1_field)
            # Refine
            concat = torch.cat(
                [
                    RS0,
                    RS1,
                    gs_i_to_rs0_field,
                    gs_i_from_rs0,
                    gs_i_to_rs1_field,
                    gs_i_from_rs1,
                ],
                dim=1,
            )
            output = self.refine(concat)
            # Residual optical flow
            gs_i_to_rs0_field_delta = output[:, 0:2, ...]
            gs_i_to_rs1_field_delta = output[:, 2:4, ...]
            gs_i_to_rs0_field = gs_i_to_rs0_field + gs_i_to_rs0_field_delta
            gs_i_to_rs1_field = gs_i_to_rs1_field + gs_i_to_rs1_field_delta
            gs_i_from_rs0_refined = backward_warp(RS0, gs_i_to_rs0_field)
            gs_i_from_rs1_refined = backward_warp(RS1, gs_i_to_rs1_field)
            # Context confidence
            context_confidence = output[:, 4:5, ...]
            context_confidence = torch.sigmoid(context_confidence)
            gs_i = gs_i_from_rs0_refined * context_confidence + gs_i_from_rs1_refined * (1 - context_confidence)
            # Append
            gs.append(gs_i)
            gs_frame_from_rs0_list.append(gs_i_from_rs0)
            gs_frame_from_rs1_list.append(gs_i_from_rs1)
            field_rs0_to_gs.append(gs_i_to_rs0_field)
            field_rs1_to_gs.append(gs_i_to_rs1_field)
            confidence_rs0_to_gs.append(context_confidence)
        return (
            gs,
            field_rs0_to_gs,
            field_rs1_to_gs,
            confidence_rs0_to_gs,
            gs_frame_from_rs0_list,
            gs_frame_from_rs1_list,
        )

    def _reconstruct_rolling_from_global(self, batch_size, displacement_field, gs_refined):
        # GS to RS1
        RS1_from_gs = [None for _ in range(self.gs_count)]
        for i in range(self.gs_count):
            gs_i_from_RS0 = gs_refined[i]
            # rs2gs * -1 = gs2rs
            gs_i_to_RS1_mask = -1.0 * self.rs2gs_mask[1][i]
            gs_i_to_RS1_mask = gs_i_to_RS1_mask.unsqueeze(0).repeat(batch_size, 1, 1, 1, 1)
            field = gs_i_to_RS1_mask * displacement_field
            field = torch.sum(field, dim=2)
            RS1_from_gs[i] = backward_warp(gs_i_from_RS0, field)
        # GS to RS0
        RS0_from_gs = [None for _ in range(self.gs_count)]
        for i in range(self.gs_count):
            gs_i_from_RS1 = gs_refined[i]
            # rs2gs * -1 = gs2rs
            gs_i_to_RS0_mask = -1.0 * self.rs2gs_mask[0][i]
            gs_i_to_RS0_mask = gs_i_to_RS0_mask.unsqueeze(0).repeat(batch_size, 1, 1, 1, 1)
            field = gs_i_to_RS0_mask * displacement_field
            field = torch.sum(field, dim=2)
            RS0_from_gs[i] = backward_warp(gs_i_from_RS1, field)
        return RS0_from_gs, RS1_from_gs
