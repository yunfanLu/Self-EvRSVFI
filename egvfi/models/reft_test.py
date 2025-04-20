#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Author ：Yunfan Lu (yunfanlu@ust.hk)
# Date   ：2023/2/1 16:32
from os.path import join

import cv2
import numpy as np
import torch
import torchvision
from absl.logging import info
from absl.testing import absltest
from torchvision.models.optical_flow import Raft_Large_Weights, Raft_Small_Weights, raft_large, raft_small
from torchvision.transforms import transforms

from egvfi.utils.flow_viz import flow_to_image
from egvfi.utils.padder import InputPadder


class TorchVisionRAFTTest(absltest.TestCase):
    def setUp(self):
        self.to_tensor = transforms.ToTensor()
        self.padder = InputPadder(dims=[260, 346])
        # load two image.
        self.test_folder = "testdata/optical_flow_test/two_image/"
        left = join(self.test_folder, "0000001560_0000001820_T0.5_GTGS.png")
        right = join(self.test_folder, "0000001820_0000002080_T0.5_GTGS.png")
        left = cv2.cvtColor(cv2.imread(left), cv2.COLOR_BGR2RGB)
        right = cv2.cvtColor(cv2.imread(right), cv2.COLOR_BGR2RGB)
        left = torch.from_numpy(left).permute(2, 0, 1).float().cuda() / 255
        right = torch.from_numpy(right).permute(2, 0, 1).float().cuda() / 255
        left, right = self.padder.pad(left, right)
        left = left[None]
        right = right[None]
        self.transforms = Raft_Large_Weights.DEFAULT.transforms()
        left, right = self.transforms(left, right)
        info(f"left.shape: {left.shape}, {left.max()}, {left.min()}")
        info(f"right.shape: {right.shape}, {right.max()}, {right.min()}")
        self.left = left
        self.right = right

    def test_raft_small_inference(self):
        # large
        weights = Raft_Small_Weights.DEFAULT
        raft = raft_small(weights=weights, progress=True)
        raft = raft.cuda()
        raft.eval()
        flows = raft(self.left, self.right)
        info(f"len(flows): {len(flows)}")
        for i, flow in enumerate(flows):
            info(f"  flow[{i}]: {flow.shape}")
            flow = flows[-1]
            flo = flow_to_image(flow[0].permute(1, 2, 0).cpu().detach().numpy())
            flow_path = join(
                self.test_folder,
                "flow-small",
                f"flow_{str(i).zfill(2)}.png",
            )
            cv2.imwrite(flow_path, flo[:, :, [2, 1, 0]])
            intensity = np.sqrt(np.power(flo[:, :, 1], 2) + np.power(flo[:, :, 0], 2))
            intensity = (intensity / intensity.max() * 255).astype(np.uint8)
            flow_path = join(
                self.test_folder,
                "flow-small",
                f"flow_intensity_{str(i).zfill(2)}.png",
            )
            cv2.imwrite(flow_path, intensity)

    def test_raft_inference(self):
        # large
        weights = Raft_Large_Weights.DEFAULT
        self.raft = raft_large(weights=weights, progress=True)
        self.raft = self.raft.cuda()
        self.raft.eval()
        flows = self.raft(self.left, self.right)
        info(f"len(flows): {len(flows)}")
        for i, flow in enumerate(flows):
            info(f"  flow[{i}]: {flow.shape}")
            flow = flows[-1]
            flo = flow_to_image(flow[0].permute(1, 2, 0).cpu().detach().numpy())
            flow_path = join(self.test_folder, "flow", f"flow" f"_{str(i).zfill(2)}.png")
            cv2.imwrite(flow_path, flo[:, :, [2, 1, 0]])
            intensity = np.sqrt(np.power(flo[:, :, 1], 2) + np.power(flo[:, :, 0], 2))
            intensity = (intensity / intensity.max() * 255).astype(np.uint8)
            flow_path = join(
                self.test_folder,
                "flow",
                f"flow_intensity_{str(i).zfill(2)}.png",
            )
            cv2.imwrite(flow_path, intensity)


if __name__ == "__main__":
    absltest.main()
