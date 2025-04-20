#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Author ：Yunfan Lu (yunfanlu@ust.hk)
# Date   ：2022/11/2 11:12
import logging
from os import listdir
from os.path import isdir, join

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from absl.logging import info
from torch.utils.data import Dataset

from egvfi.datasets.basic_batch import get_vfi_empty_batch

logging.getLogger("PIL").setLevel(logging.WARNING)


def get_alpx_vfi_rotaing_fan_dataset(
    alpx_vfi_root,
    moments,
    random_crop_resolution,
    low_resolution,
):
    """Get the alpx vfi rotaing fan dataset.

    Args:
        alpx_vfi_root (str): the root of the alpx vfi dataset.
        moments (tuple): the moments of events segmentation.
        random_crop_resolution (tuple): the resolution of the random crop.
        low_resolution (tuple): the low resolution of the input.

    Returns:
        torch.data.Dataset: Dataset for training and testing.
    """
    all_videos = sorted(listdir(alpx_vfi_root))
    # 6 test video and 20 train video
    test_videos = [
        "20230210104136493", "20230224151350007", "20230224151405372", "20230307112649786"
    ]
    train_videos = []
    for v in all_videos:
        if v in test_videos:
            continue
        if not isdir(join(alpx_vfi_root, v)):
            continue
        train_videos.append(v)

    info(f"train_videos ({len(train_videos)}): {train_videos}")
    info(f"test_videos  ({len(test_videos)}): {test_videos}")

    train_dataset = AlpxVideoFrameInterpolationDataset(
        alpx_vfi_root,
        train_videos,
        moments,
        random_crop_resolution,
        low_resolution,
        is_test=False,
    )
    test_dataset = AlpxVideoFrameInterpolationDataset(
        alpx_vfi_root,
        test_videos,
        moments,
        random_crop_resolution,
        low_resolution,
        is_test=True,
    )
    return train_dataset, test_dataset



def get_alpx_vfi_dataset(
    alpx_vfi_root,
    moments,
    random_crop_resolution,
    low_resolution,
    is_test,
):
    """Get the alpx vfi dataset. 

    Args:
        alpx_vfi_root (str): the root of the alpx vfi dataset.
        moments (int): the moments of events segmentation.
        random_crop_resolution (tuple): the resolution of the random crop.
        low_resolution (tuple): low resolution of the input.
        is_test (bool): the switch of testing or training. 

    Returns:
        torch.data.Dataset: The train dataset and test dataset.
    """
    all_videos = sorted(listdir(alpx_vfi_root))
    # 6 test video and 20 train video
    test_videos = [
        "20221202165340657",
        "20221202165603704",
        "20221202165920476",
        "20221202164318309",
        "20221202164547101",
        "20221202170034076",
        "20221203133414042",
        "20221203132821109",
        "20221203131505755",
        "20221203132245710",
    ]
    train_videos = []
    for v in all_videos:
        if v in test_videos:
            continue
        if not isdir(join(alpx_vfi_root, v)):
            continue
        train_videos.append(v)

    info(f"train_videos ({len(train_videos)}): {train_videos}")
    info(f"test_videos  ({len(test_videos)}): {test_videos}")

    train_dataset = AlpxVideoFrameInterpolationDataset(
        alpx_vfi_root,
        train_videos,
        moments,
        random_crop_resolution,
        low_resolution,
        is_test,
    )
    test_dataset = AlpxVideoFrameInterpolationDataset(
        alpx_vfi_root,
        test_videos,
        moments,
        random_crop_resolution,
        low_resolution,
        is_test,
    )
    return train_dataset, test_dataset


class AlpxVideoFrameInterpolationDataset(Dataset):
    def __init__(
        self,
        alpx_vsr_root,
        videos,
        moments,
        random_crop_resolution,
        low_resolution,
        is_test,
    ):
        super(AlpxVideoFrameInterpolationDataset, self).__init__()
        self.moments = moments
        self.alpx_vsr_root = alpx_vsr_root
        self.videos = videos
        self.in_frame = 3
        self.future_frame = 1
        self.past_frame = 1
        self.random_crop_resolution = random_crop_resolution
        self.low_resolution = low_resolution
        # static values
        self.event_resolution = (1224, 1632)
        self.positive = 2
        self.negative = 1
        self.is_test = is_test

        self.items = self._generate_items()
        self._info()

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        image_paths, event_paths = self.items[index]
        RS_0_1 = [image_paths[0], image_paths[1]]
        h, w = self.event_resolution
        images = []
        for path in RS_0_1:
            image = cv2.imread(path)
            if image is None:
                raise ValueError(f"image is None: {path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (w, h))
            images.append(image)
        events = []
        for path in event_paths:
            event = np.load(path)
            event[event == self.negative] = -1
            event[event == self.positive] = 1
            event = np.fliplr(cv2.rotate(event, cv2.ROTATE_90_CLOCKWISE))
            events.append(event)
        if len(events) < self.moments:
            for i in range(self.moments - len(events)):
                events.append(np.zeros_like(events[0]))
        lr, lr_events = self._random_crop(images, events)
        # generate vfi input
        batch = {
            "input_rolling_frames": lr,
            "events": lr_events,
            "video_name": image_paths[0].split("/")[-2],
            "first_rolling_frame_name": image_paths[0].split("/")[-1].split(".")[0],
        }
        return batch

    def _random_crop(self, images, events):
        h, w = self.event_resolution
        crop_h, crop_w = self.random_crop_resolution
        lr_h, lr_w = self.low_resolution[0], self.low_resolution[1]

        if self.is_test:
            x, y = 0, 0
        else:
            x = np.random.randint(0, h - crop_h - 1)
            y = np.random.randint(0, w - crop_w - 1)
        crop_images = []
        for i in range(len(images)):
            image = torch.from_numpy(images[i]).float()
            image = image.permute(2, 0, 1)
            crop_image = image[:, x : x + crop_h, y : y + crop_w]
            crop_images.append(crop_image / 255.0)

        crop_events = []
        for i in range(len(events)):
            event = torch.from_numpy(np.ascontiguousarray(events[i])).float()
            event = event.unsqueeze(0)
            crop_event = event[:, x : x + crop_h, y : y + crop_w]
            crop_events.append(crop_event)
        # resize
        crop_images = torch.stack(crop_images, dim=0)

        lr_images = F.interpolate(
            crop_images,
            size=(lr_h, lr_w),
            mode="bilinear",
            align_corners=False,
        )
        crop_events = torch.stack(crop_events, dim=0)
        lr_events = F.interpolate(
            crop_events,
            size=(lr_h, lr_w),
            mode="bilinear",
            align_corners=False,
        )
        lr_events = lr_events.squeeze(1)
        return lr_images, lr_events

    def _generate_items(self):
        samples = []
        for video in self.videos:
            video_path = join(self.alpx_vsr_root, video)
            samples += self._generate_video_items(video_path)
        return samples

    def _generate_video_items(self, video_path):

        items = []
        files = sorted(listdir(video_path))
        for i in range(len(files)):
            if not files[i].endswith("frame.png"):
                continue
            frame_indexes = []
            for j in range(i + 1, len(files)):
                if not files[j].endswith("frame.png"):
                    continue
                frame_indexes.append(j)
                if len(frame_indexes) == self.in_frame:
                    break
            # 1. Check has enough frames.
            if len(frame_indexes) != self.in_frame:
                break
            # 2. Check the event in neighbor frames is average.
            is_average = True
            for k in range(self.in_frame - 1):
                index_step = frame_indexes[k + 1] - frame_indexes[k]
                # 20, 50, -> 31, 34
                if index_step <= 20 or index_step >= 50:
                    is_average = False
                    break
            if not is_average:
                continue
            # 3. Generate items.
            item = [[], []]
            for k in range(min(frame_indexes), max(frame_indexes) + 1):
                frame_name = files[k]
                if frame_name.endswith("frame.png"):
                    item[0].append(join(video_path, frame_name))
                elif frame_name.endswith("events.npy"):
                    item[1].append(join(video_path, frame_name))
            items.append(item)
        return items

    def _info(self):
        info(f"Init AlpxVideoSRDataset.")
        info(f"  alpx_vsr_root: {self.alpx_vsr_root}")
        info(f"  videos: {self.videos}")
        info(f"  in_frame: {self.in_frame}")
        info(f"  future_frame: {self.future_frame}")
        info(f"  past_frame: {self.past_frame}")
        info(f"  random_crop_resolution: {self.random_crop_resolution}")
        info(f"  low_resolution: {self.low_resolution}")
        info(f"  Length: {len(self.items)}")
