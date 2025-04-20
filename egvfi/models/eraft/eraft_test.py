#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Author ：Yunfan Lu (yunfanlu@ust.hk)
# Date   ：2023/2/4 19:45
import os
from os.path import dirname, join

import cv2
import numpy as np
import torch
from absl.logging import info
from absl.testing import absltest
from thop import profile

from egvfi.models.eraft.eraft import ERAFT
from egvfi.utils.flow_viz import flow_to_image


def _load_events(event_path):
    events = np.load(event_path)
    events = _render(shape=[260, 346], **events)
    return events


def _render(x, y, t, p, shape):
    events = np.zeros(shape=shape)
    events[y, x] = p
    return events


class ERAFTTest(absltest.TestCase):
    def setUp(self):
        self.testdata = join(dirname(__file__), "testdata")
        pretrain_model = join(self.testdata, "dsec.tar")
        self.eraft = ERAFT(subtype="standard", n_first_channels=15)
        checkpoint = torch.load(pretrain_model)
        self.eraft.load_state_dict(checkpoint["model"])
        self.eraft = self.eraft.cuda()

    def test_eraft_flops_params(self):
        eframe0 = torch.rand(1, 15, 128, 128).cuda()
        eframe1 = torch.rand(1, 15, 128, 128).cuda()
        macs, params = profile(self.eraft, inputs=(eframe0, eframe1))
        info(f"E-RAFT: macs={macs / (1024 ** 3)}G, params={params / (1024 ** 2)}M")

    def test_eraft_at_train_demo(self):
        self.events_folder = join(self.testdata, "events-train-demo")
        event_list = []
        for file in sorted(os.listdir(self.events_folder)):
            if file.endswith(".npz"):
                events = _load_events(join(self.events_folder, file))
                event_list.append(torch.from_numpy(events).float())
        length = len(event_list)
        event_list = torch.stack(event_list, dim=0)
        info(f"event length={length}")
        for i in range(0, length - 30, 15):
            left = event_list[i : i + 15]
            right = event_list[i + 15 : i + 30]
            left = left.unsqueeze(0).cuda()
            right = right.unsqueeze(0).cuda()
            flow, flow_predictions = self.eraft(left, right)
            info(f"flow shape={flow.shape}")
            info(f"len(flow_predictions)={len(flow_predictions)}")
            info(f"flow_predictions shape={flow_predictions[-1].shape}")
            last_flow = flow_predictions[-1]
            info(f"last_flow shape={last_flow.shape}")
            info(f"last_flow min={last_flow.min()}")
            info(f"last_flow max={last_flow.max()}")
            flo = flow_to_image(
                last_flow[0].permute(1, 2, 0).cpu().detach().numpy(),
                normalize=True,
            )
            flow_path = join(self.testdata, f"flow-{i}.png")
            cv2.imwrite(flow_path, flo[:, :, [2, 1, 0]])


if __name__ == "__main__":
    absltest.main()
