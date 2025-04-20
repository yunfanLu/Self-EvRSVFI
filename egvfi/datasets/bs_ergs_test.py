#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Author ：Yunfan Lu (yunfanlu@ust.hk)
# Date   ：2022/10/16 10:31
from os import makedirs
from os.path import join

import cv2
import numpy as np
from absl.logging import info
from absl.testing import absltest

from config import global_path as gp
from egvfi.datasets.bs_ergb import get_timelen_pp


def _event_to_image(event, path):
    b, h, w = event.shape
    image = np.zeros((h, w, 3)) + 255
    image[event[0] > 0] = [0, 0, 255]
    image[event[1] > 0] = [255, 0, 0]
    cv2.imwrite(path, image)


def _tensor_to_image(tensor, path):
    image = tensor.permute(1, 2, 0).numpy() * 255
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, image)


class TimeLensPPDatasetTest(absltest.TestCase):
    def setUp(self):
        self.key_frames = 2
        self.skip_frames = 1
        self.step = 1
        self.moments = 4
        folder = (
            f"keyframes_{self.key_frames}_skipframe_{self.skip_frames}" + f"_step_{self.step}_moments_{self.moments}"
        )
        self.testdata = gp.test_data
        self.folder = join(self.testdata, "bs_ergb_dataset_visual_demo", folder)
        makedirs(self.folder, exist_ok=True)
        self.random_crop_resolution = (400, 300)

    def test_load_dataset(self):
        train, test, val = get_timelen_pp(
            key_frames=self.key_frames,
            skip_frames=self.skip_frames,
            step=self.step,
            moments=self.moments,
            random_crop=False,
        )
        self._run(train, "full-image-train", is_random_crop=False)

    def test_load_dataset_with_random_crop(self):
        train, test, val = get_timelen_pp(
            key_frames=self.key_frames,
            skip_frames=self.skip_frames,
            step=self.step,
            moments=self.moments,
            random_crop=True,
            random_crop_resolution=self.random_crop_resolution,
        )
        self._run(train, "random-crop-train", is_random_crop=True)

    def _run(self, dataset, tag, is_random_crop):
        info(f"{tag} dataset size: {len(dataset)}")
        h, w = self.random_crop_resolution
        if not is_random_crop:
            h, w = 625, 970
        for _ in range(0, 10):
            index = np.random.randint(0, len(dataset))
            folder = join(self.folder, tag, f"{str(index).zfill(4)}")
            makedirs(folder, exist_ok=True)
            sample = dataset[index]
            key_frames = sample["key_frames"]
            target_frames = sample["target_frames"]
            target_frame_coords = sample["target_frame_coords"]
            target_feature_t = sample["target_feature_t"]
            events = sample["events"]
            events_feature_t = sample["events_feature_t"]
            events_coords = sample["events_coords"]

            info(f"Load sample {index}")
            self.assertEqual(key_frames.shape, (self.key_frames, 3, h, w))
            info(f"    Keyframe val in({key_frames.min()}, {key_frames.max()})")
            self.assertEqual(
                target_frames.shape,
                (self.skip_frames * (self.key_frames - 1), 3, h, w),
            )
            info(f"    Target frame coords  : {target_frame_coords.shape}")
            self.assertEqual(
                target_frame_coords.shape,
                (self.skip_frames * (self.key_frames - 1), h, w, 3),
            )
            info(f"    Target feature t     : {target_feature_t}")
            info(f"    Events.shape         : {events.shape}")
            self.assertEqual(events.shape, (self.moments, 2, h, w))
            info(f"        Max({events.max()}), Min({events.min()})")
            info(f"    Events feature t     : {events_feature_t}")
            info(f"    Events coords        : {events_coords.shape}")

            for j, key in enumerate(key_frames):
                _tensor_to_image(key, join(folder, f"key-{j}.png"))
            for j, tar in enumerate(target_frames):
                _tensor_to_image(tar, join(folder, f"tag-{j}.png"))
            for j in range(dataset.moments):
                _event_to_image(
                    events[j, :, :, :],
                    join(folder, f"event-{j}.png"),
                )
            for k in sample.keys():
                info(f"{k} : {sample[k].shape}")


if __name__ == "__main__":
    import pudb

    pudb.set_trace()
    absltest.main()
