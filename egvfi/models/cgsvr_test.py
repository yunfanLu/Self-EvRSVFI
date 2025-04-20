#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Author ：Yunfan Lu (yunfanlu@ust.hk)
# Date   ：2023/2/6 11:32
from os import makedirs
from os.path import dirname, join

import numpy as np
import torch
from absl.logging import info
from absl.testing import absltest
from easydict import EasyDict
from thop import profile
from torch.utils.data import DataLoader

from config import global_path as gp
from egvfi.core.launch import Visualization
from egvfi.datasets.gev_rs import get_2x_frame_interpolate_3_frame
from egvfi.functions.rolling_temporal_mask import rolling_mask_to_image
from egvfi.models.cgsvr import ContinueGlobalShutterVideoReconstruction


def dict_batch_to_cuda(batch):
    for k in batch.keys():
        if isinstance(batch[k], torch.Tensor):
            batch[k] = batch[k].cuda()
    return batch


class ContinueGlobalShutterVideoReconstructionTest(absltest.TestCase):
    def setUp(self):
        self.test_folder = join(dirname(__file__), "testdata", "cgvsr_test-v2")
        info(f"Test folder: {self.test_folder}")
        makedirs(self.test_folder, exist_ok=True)
        # build model
        self.H = 256
        self.W = 320
        self.events_moments = 90
        self.cgsvr = ContinueGlobalShutterVideoReconstruction(
            events_moments=self.events_moments,
            image_count=2,
            image_high=self.H,
            image_width=self.W,
            e_raft_in_channels=15,
            time_bin_size=6,
            erfat_iter=12,
            e_raft_pretrained=True,
            invert_event_in_raft_flow_and_fusion="ONE-FLOW",
            rs_timestamps=[[0, 0.5], [0.5, 1]],
            gs_timestamps=[0.25, 0.5, 0.75],
            mask_high_sample_count=10,
            mask_time_sample_count=20,
            intermediate_visualization=True,
        )
        self.cgsvr = self.cgsvr.cuda()
        # config visualization
        visualization_config = EasyDict(
            {
                "tag": "unit-test",
                "intermediate_visualization": True,
                "folder": "vis",
            }
        )
        self.visualizer = Visualization(visualization_config)

    # def test_mask(self):
    #     # mask visualization
    #     rs2gs_mask_2d = self.cgsvr.rs2gs_mask_2d
    #     rs2gs_mask = self.cgsvr.rs2gs_mask
    #     rs2rs_mask = self.cgsvr.rs2rs_mask
    #     gs_count = self.cgsvr.gs_count
    #     rs_count = self.cgsvr.rs_count
    #     folder = join(self.test_folder, "mask")
    #     makedirs(folder, exist_ok=True)
    #     info(f"gs_count: {gs_count}")
    #     info(f"rs_count: {rs_count}")
    #     # rs2gs
    #     for i in range(rs_count):
    #         for j in range(gs_count):
    #             # mask 2d
    #             # H, T
    #             mask_2d = rs2gs_mask_2d[i][j].cpu().numpy()
    #             info(f"rs2gs_mask_2d_[{i}][{j}]: RS{i} -> GS{j}/{gs_count}, shape: {mask_2d.shape}")
    #             rolling_mask_to_image(f"rs2gs_mask_2d__rs{i}_to_gs{j}", mask_2d, folder)
    #             # mask 3d
    #             mask = rs2gs_mask[i][j].cpu().numpy()
    #             C, T, H, W = mask.shape
    #             info(f"rs2gs_mask_3d_[{i}][{j}]: RS{i} -> GS{j}/{gs_count}, shape: {mask.shape}")
    #             for t in range(1):
    #                 # C, T, H, W -> 2, H, W -> H, W
    #                 mask_at_t = mask[0, t, :, :]
    #                 rolling_mask_to_image(f"rs2gs_mask_3d_rs{i}_to_gs{j}_t{t}-{T}", mask_at_t, folder)
    #             for w in range(1):
    #                 # C, T, H, W -> 2, T, H -> T, H
    #                 mask_at_w = mask[0, :, :, w]
    #                 # T, H -> H, T
    #                 mask_at_w = np.transpose(mask_at_w, (1, 0))
    #                 rolling_mask_to_image(f"rs2gs_mask_3d_rs{i}_to_gs{j}_w{w}-{W}", mask_at_w, folder)

    def test_inference_in_gev_dataset(self):
        root = gp.fps5000_video_folder
        train, test = get_2x_frame_interpolate_3_frame(
            root=root,
            center_cropped_height=self.H,
            random_cropped_width=self.W,
            sample_step=130,
            events_moment_count=self.events_moments,
        )
        dataloader = DataLoader(test, batch_size=1, shuffle=False, num_workers=0)
        for batch in dataloader:
            batch = dict_batch_to_cuda(batch)
            info("Input:")
            self._out(batch)
            info("Inference:")
            out = self.cgsvr(batch)
            self._out(out)
            flops, params = profile(self.cgsvr, (batch,))
            info(f"flops : {flops / 1000 ** 3}G")
            info(f"params: {params / 1000 ** 2}M")
            # visual
            self.visualizer.visualize(batch, out)
            break

    def _out(self, item):
        if isinstance(item, torch.Tensor):
            info(f"{item.shape}")
        elif isinstance(item, list):
            for i, t in enumerate(item):
                info(f"{i}:")
                self._out(t)
        elif isinstance(item, dict):
            for k in item.keys():
                info(f"{k}:")
                self._out(item[k])


if __name__ == "__main__":
    absltest.main()
