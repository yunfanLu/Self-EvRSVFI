#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Author ：Yunfan Lu (yunfanlu@ust.hk)
# Date   ：2022/12/24 17:04

import random

import torch
from absl import logging
from absl.logging import debug, info
from absl.testing import absltest

from config import global_path as gp
from egvfi.datasets.gev_rs import get_1x_frame_interpolate_1_frame


class HighSpeedVideoSimulatedDatasetTest(absltest.TestCase):
    def setUp(self):
        logging.set_verbosity(logging.INFO)
        self.root = gp.fps5000_video_folder

    def test_2i1_get_item_samples(self):
        #  root, center_cropped_height, random_cropped_width, sample_step, events_moment_count
        train, test = get_1x_frame_interpolate_1_frame(
            root=self.root,
            center_cropped_height=256,
            random_cropped_width=256,
            sample_step=130,
            events_moment_count=60)
        # I0526 21:33:47.327337 140030146372416 gev_rs_test.py:30] Train Set Size : 1568
        # I0526 21:33:47.327375 140030146372416 gev_rs_test.py:31] Test  Set Size : 882
        info(f"Train Set Size : {len(train)}")
        info(f"Test  Set Size : {len(test)}")

        index = random.randint(0, len(train))
        info(f"Select item[{index}]:")
        item = train.samples[index]
        items_rollings, items_events, items_global = item
        info(f"  items_rollings : {items_rollings}")
        info(f"  items_global   : {items_global}")
        info(f"  items_events   : {len(items_events)}")
        info(f"    From {items_events[0]}")
        info(f"    To   {items_events[-1]}")

    # def test_2i1_get_item(self):
    #     train, test = get_1x_frame_interpolate_1_frame(root=self.root, sample_step=130, events_moment_count=60)
    #     index = random.randint(0, len(train))
    #     info(f"Select item[{index}]:")
    #     item = train[index]
    #     info(f"item: {item.keys()}")
    #     for k in item.keys():
    #         if isinstance(item[k], torch.Tensor):
    #             info(f"    {k}: {item[k].shape}, {type(item[k])}, {item[k].dtype}")
    #         else:
    #             info(f"    {k}: {item[k]}")

    # def test_2i1_all_test_items(self):
    #     _, test = get_1x_frame_interpolate_1_frame(root=self.root, sample_step=130, events_moment_count=60)
    #     for i in range(len(test)):
    #         items_rollings, items_events, items_global = test.samples[i]
    #         debug(f"  items_rollings[{i}] :")
    #         debug(f"    Left  {items_rollings[0]}")
    #         debug(f"    Right {items_rollings[1]}")
    #         debug(f"  items_global[{i}]   : {items_global}")
    #         debug(f"  items_events[{i}]   : {len(items_events)}")
    #         debug(f"    From {items_events[0]}")
    #         debug(f"    To   {items_events[-1]}")


if __name__ == "__main__":
    # import pudb

    # pudb.set_trace()
    absltest.main()
