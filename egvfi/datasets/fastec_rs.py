#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Author ：Yunfan Lu (yunfanlu@ust.hk)
# Date   ：2022/12/24 16:23
import random
from os import listdir
from os.path import isdir, join

import numpy as np
import torch
from absl.logging import debug, info, warning
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision.transforms import transforms

from egvfi.datasets.basic_batch import get_vfi_empty_batch


def get_1x_frame_interpolate_1_frame_fastec(
    root,
    center_cropped_height,
    random_cropped_width,
    sample_step,
    events_moment_count,
):
    item_samples = 520
    input_rs_frame_indexes = [0, 260]
    input_rs_frame_timestamps = [[0, 0.5], [0.5, 1]]
    output_gs_frame_indexes = [260]
    output_gs_frame_timestamps = [0.5]
    events_moment_count = events_moment_count
    return get_fastec_rs_dataset(
        root=root,
        sample_step=sample_step,
        item_samples=item_samples,
        input_rs_frame_indexes=input_rs_frame_indexes,
        input_rs_frame_timestamps=input_rs_frame_timestamps,
        output_gs_frame_indexes=output_gs_frame_indexes,
        output_gs_frame_timestamps=output_gs_frame_timestamps,
        events_moment_count=events_moment_count,
        center_cropped_height=center_cropped_height,
        random_cropped_width=random_cropped_width,
    )


def get_2x_frame_interpolate_3_frame_fastec(
    root,
    center_cropped_height,
    random_cropped_width,
    sample_step,
    events_moment_count,
):
    item_samples = 520
    input_rs_frame_indexes = [0, 260]
    input_rs_frame_timestamps = [[0, 0.5], [0.5, 1]]
    output_gs_frame_indexes = [130, 260, 390]
    output_gs_frame_timestamps = [0.25, 0.5, 0.75]
    return get_fastec_rs_dataset(
        root=root,
        sample_step=sample_step,
        item_samples=item_samples,
        input_rs_frame_indexes=input_rs_frame_indexes,
        input_rs_frame_timestamps=input_rs_frame_timestamps,
        output_gs_frame_indexes=output_gs_frame_indexes,
        output_gs_frame_timestamps=output_gs_frame_timestamps,
        events_moment_count=events_moment_count,
        center_cropped_height=center_cropped_height,
        random_cropped_width=random_cropped_width,
    )


def get_4x_frame_interpolate_5_frame_fastec(
    root,
    center_cropped_height,
    random_cropped_width,
    sample_step,
    events_moment_count,
):
    item_samples = 520
    input_rs_frame_indexes = [0, 260]  # 0-259, 260-519
    input_rs_frame_timestamps = [[0, 0.5], [0.5, 1]]
    output_gs_frame_indexes = [130, 195, 260, 325, 390]
    output_gs_frame_timestamps = [0.25, 0.375, 0.5, 0.625, 0.75]
    return get_fastec_rs_dataset(
        root=root,
        sample_step=sample_step,
        item_samples=item_samples,
        input_rs_frame_indexes=input_rs_frame_indexes,
        input_rs_frame_timestamps=input_rs_frame_timestamps,
        output_gs_frame_indexes=output_gs_frame_indexes,
        output_gs_frame_timestamps=output_gs_frame_timestamps,
        events_moment_count=events_moment_count,
        center_cropped_height=center_cropped_height,
        random_cropped_width=random_cropped_width,
    )


def get_8x_frame_interpolate_9_frame_fastec(
    root,
    center_cropped_height,
    random_cropped_width,
    sample_step,
    events_moment_count,
):
    item_samples = 520
    input_rs_frame_indexes = [0, 260]  # 0-259, 260-519
    input_rs_frame_timestamps = [[0, 0.5], [0.5, 1]]
    output_gs_frame_indexes = [130, 162, 195, 227, 260, 292, 325, 357, 390]
    output_gs_frame_timestamps = [
        0.25,
        0.3125,
        0.375,
        0.4375,
        0.5,
        0.5625,
        0.625,
        0.6875,
        0.75,
    ]
    return get_fastec_rs_dataset(
        root=root,
        sample_step=sample_step,
        item_samples=item_samples,
        input_rs_frame_indexes=input_rs_frame_indexes,
        input_rs_frame_timestamps=input_rs_frame_timestamps,
        output_gs_frame_indexes=output_gs_frame_indexes,
        output_gs_frame_timestamps=output_gs_frame_timestamps,
        events_moment_count=events_moment_count,
        center_cropped_height=center_cropped_height,
        random_cropped_width=random_cropped_width,
    )


def get_12x_frame_interpolate_13_frame_fastec(
    root,
    center_cropped_height,
    random_cropped_width,
    sample_step,
    events_moment_count,
):
    item_samples = 520
    input_rs_frame_indexes = [0, 260]  # 0-259, 260-519
    input_rs_frame_timestamps = [[0, 0.5], [0.5, 1]]
    output_gs_frame_indexes = [
        130,
        151,
        173,
        195,
        216,
        238,
        260,
        281,
        303,
        325,
        346,
        368,
        390,
    ]
    output_gs_frame_timestamps = [
        0.25,
        0.2916666666666667,
        0.3333333333333333,
        0.375,
        0.41666666666666663,
        0.45833333333333337,
        0.5,
        0.5416666666666667,
        0.5833333333333333,
        0.625,
        0.6666666666666667,
        0.7083333333333333,
        0.75,
    ]
    return get_fastec_rs_dataset(
        root=root,
        sample_step=sample_step,
        item_samples=item_samples,
        input_rs_frame_indexes=input_rs_frame_indexes,
        input_rs_frame_timestamps=input_rs_frame_timestamps,
        output_gs_frame_indexes=output_gs_frame_indexes,
        output_gs_frame_timestamps=output_gs_frame_timestamps,
        events_moment_count=events_moment_count,
        center_cropped_height=center_cropped_height,
        random_cropped_width=random_cropped_width,
    )


def get_16x_frame_interpolate_17_frame_fastec(
    root,
    center_cropped_height,
    random_cropped_width,
    sample_step,
    events_moment_count,
):
    item_samples = 520
    input_rs_frame_indexes = [0, 260]  # 0-259, 260-519
    input_rs_frame_timestamps = [[0, 0.5], [0.5, 1]]
    output_gs_frame_indexes = [
        130,
        146,
        162,
        178,
        195,
        211,
        227,
        243,
        260,
        276,
        292,
        308,
        325,
        341,
        357,
        373,
        390,
    ]
    output_gs_frame_timestamps = [
        0.25,
        0.28125,
        0.3125,
        0.34375,
        0.375,
        0.40625,
        0.4375,
        0.46875,
        0.5,
        0.53125,
        0.5625,
        0.59375,
        0.625,
        0.65625,
        0.6875,
        0.71875,
        0.75,
    ]
    return get_fastec_rs_dataset(
        root=root,
        sample_step=sample_step,
        item_samples=item_samples,
        input_rs_frame_indexes=input_rs_frame_indexes,
        input_rs_frame_timestamps=input_rs_frame_timestamps,
        output_gs_frame_indexes=output_gs_frame_indexes,
        output_gs_frame_timestamps=output_gs_frame_timestamps,
        events_moment_count=events_moment_count,
        center_cropped_height=center_cropped_height,
        random_cropped_width=random_cropped_width,
    )


def get_24x_frame_interpolate_25_frame_fastec(
    root,
    center_cropped_height,
    random_cropped_width,
    sample_step,
    events_moment_count,
):
    item_samples = 520
    input_rs_frame_indexes = [0, 260]  # 0-259, 260-519
    input_rs_frame_timestamps = [[0, 0.5], [0.5, 1]]
    n = 24
    output_gs_frame_indexes = [
        130,
        140,
        151,
        162,
        173,
        184,
        195,
        205,
        216,
        227,
        238,
        249,
        260,
        270,
        281,
        292,
        303,
        314,
        325,
        335,
        346,
        357,
        368,
        379,
        390,
    ]
    output_gs_frame_timestamps = [
        0.25,
        0.2708333333333333,
        0.2916666666666667,
        0.3125,
        0.3333333333333333,
        0.3541666666666667,
        0.375,
        0.39583333333333337,
        0.41666666666666663,
        0.4375,
        0.45833333333333337,
        0.47916666666666663,
        0.5,
        0.5208333333333333,
        0.5416666666666667,
        0.5625,
        0.5833333333333333,
        0.6041666666666667,
        0.625,
        0.6458333333333333,
        0.6666666666666667,
        0.6875,
        0.7083333333333333,
        0.7291666666666667,
        0.75,
    ]
    return get_fastec_rs_dataset(
        root=root,
        sample_step=sample_step,
        item_samples=item_samples,
        input_rs_frame_indexes=input_rs_frame_indexes,
        input_rs_frame_timestamps=input_rs_frame_timestamps,
        output_gs_frame_indexes=output_gs_frame_indexes,
        output_gs_frame_timestamps=output_gs_frame_timestamps,
        events_moment_count=events_moment_count,
        center_cropped_height=center_cropped_height,
        random_cropped_width=random_cropped_width,
    )


def get_32x_frame_interpolate_33_frame_fastec(
    root,
    center_cropped_height,
    random_cropped_width,
    sample_step,
    events_moment_count,
):
    item_samples = 520
    input_rs_frame_indexes = [0, 260]  # 0-259, 260-519
    input_rs_frame_timestamps = [[0, 0.5], [0.5, 1]]

    n = 32
    output_gs_frame_indexes = [int(130 + (i * 260) / n) for i in range(n + 1)]
    output_gs_frame_timestamps = [0.25 + (i * 0.5) / n for i in range(n + 1)]
    return get_fastec_rs_dataset(
        root=root,
        sample_step=sample_step,
        item_samples=item_samples,
        input_rs_frame_indexes=input_rs_frame_indexes,
        input_rs_frame_timestamps=input_rs_frame_timestamps,
        output_gs_frame_indexes=output_gs_frame_indexes,
        output_gs_frame_timestamps=output_gs_frame_timestamps,
        events_moment_count=events_moment_count,
        center_cropped_height=center_cropped_height,
        random_cropped_width=random_cropped_width,
    )


def get_fastec_rs_dataset(
    root,
    sample_step,
    item_samples,
    input_rs_frame_indexes,
    input_rs_frame_timestamps,
    output_gs_frame_indexes,
    output_gs_frame_timestamps,
    events_moment_count,
    center_cropped_height,
    random_cropped_width,
):
    train = FastecRSEventHighSpeedVideoSimulatedDataset(
        root=root,
        item_samples=item_samples,
        input_rs_frame_indexes=input_rs_frame_indexes,
        input_rs_frame_timestamps=input_rs_frame_timestamps,
        output_gs_frame_indexes=output_gs_frame_indexes,
        output_gs_frame_timestamps=output_gs_frame_timestamps,
        events_moment_count=events_moment_count,
        sample_step=sample_step,
        is_train=True,
        height=260,
        width=346,
        center_cropped_height=center_cropped_height,
        random_cropped_width=random_cropped_width,
    )
    test = FastecRSEventHighSpeedVideoSimulatedDataset(
        root=root,
        item_samples=item_samples,
        input_rs_frame_indexes=input_rs_frame_indexes,
        input_rs_frame_timestamps=input_rs_frame_timestamps,
        output_gs_frame_indexes=output_gs_frame_indexes,
        output_gs_frame_timestamps=output_gs_frame_timestamps,
        events_moment_count=events_moment_count,
        sample_step=sample_step,
        is_train=False,
        height=260,
        width=346,
        center_cropped_height=center_cropped_height,
        random_cropped_width=random_cropped_width,
    )
    return train, test


class FastecRSEventHighSpeedVideoSimulatedDataset(Dataset):
    def __init__(
        self,
        root,
        item_samples,
        input_rs_frame_indexes,
        input_rs_frame_timestamps,
        output_gs_frame_indexes,
        output_gs_frame_timestamps,
        events_moment_count,
        sample_step,
        is_train,
        height,
        width,
        center_cropped_height,
        random_cropped_width,
    ):
        """
        :param root:
        :param item_samples:
        :param input_rs_frame_indexes:
        :param output_gs_frame_indexes:
        :param sample_step:
        :param is_train:
        :param height:
        :param width:
        The data folder structure.
            ./dataset/3-10000fps-videos/1-Videos-EvNurall/
                24209_1_10.avi
                24209_1_10_fps5000-frames
                24209_1_10_fps5000-resize-frames-H260-W346
                24209_1_10_fps5000-resize-frames-H260-W346-events
                24209_1_10_fps5000-resize-frames-H260-W346-rolling
                24209_1_10_fps5000-resize-frames-H260-W346-rolling-R4
        """
        super(FastecRSEventHighSpeedVideoSimulatedDataset, self).__init__()
        # assert
        assert center_cropped_height % 32 == 0, f"center_cropped_height {center_cropped_height} % 32 != 0"
        assert random_cropped_width % 32 == 0, f"random_cropped_width {random_cropped_width} % 32 != 0"
        assert center_cropped_height <= height, f"center_cropped_height {center_cropped_height} > height {height}"
        assert random_cropped_width <= width, f"random_cropped_width {random_cropped_width} > width {width}"
        # The event segments should be divisible by the event moment count.
        if item_samples % events_moment_count > 0:
            warning(f"zero padding will be appended to events for {item_samples} % {events_moment_count} != 0")
        # Set the global configuration.
        self.root = join(root, "Train") if is_train else join(root, "Test")
        self.is_train = is_train
        self.item_samples = item_samples
        self.input_rs_frame_indexes = input_rs_frame_indexes
        self.input_rs_frame_timestamps = input_rs_frame_timestamps
        self.output_gs_frame_indexes = output_gs_frame_indexes
        self.output_gs_frame_timestamps = output_gs_frame_timestamps
        self.events_moment_count = events_moment_count
        self.sample_step = sample_step
        self.H = height
        self.W = width
        self.center_cropped_height = center_cropped_height
        self.random_cropped_width = random_cropped_width
        # walk for all items
        self.samples = self._walk()
        # Transformer
        self.to_tensor = transforms.ToTensor()
        self._info()

    def _info(self):
        info(f"Dataset: {self.__class__.__name__}")
        info(f"  Root: {self.root}")
        info(f"  Item samples: {self.item_samples}")
        info(f"  Input rolling frame indexes: {self.input_rs_frame_indexes}")
        info(f"  Input rolling frame timestamps: {self.input_rs_frame_timestamps}")
        info(f"  Output global frame indexes: {self.output_gs_frame_indexes}")
        info(f"  Output global frame timestamps: {self.output_gs_frame_timestamps}")
        info(f"  Events moment count: {self.events_moment_count}")
        info(f"  Sample step: {self.sample_step}")
        info(f"  Is train: {self.is_train}")
        info(f"  Height: {self.H}")
        info(f"  Width: {self.W}")
        info(f"  Samples: {len(self.samples)}")
        info(f"  Crop height: {self.center_cropped_height}")
        info(f"  Crop width: {self.random_cropped_width}")

    def __getitem__(self, index):
        item = self.samples[index]
        input_rolling_frames, input_events, output_global = item
        # Load data
        video_name = input_rolling_frames[0].split("/")[-2]
        first_rolling_frame_name = input_rolling_frames[0].split("/")[-1].split(".")[0]
        input_frames = torch.stack([self.to_tensor(Image.open(i).convert("RGB")) for i in input_rolling_frames])
        output_global_frames = torch.stack([self.to_tensor(Image.open(i).convert("RGB")) for i in output_global])

        events = [self._load_events(e) for e in input_events]
        events = np.stack(events, axis=0)
        events = torch.from_numpy(events).float()
        events = self._to_moments(events)

        # input_frames = self._center_crop(input_frames)
        # output_global_frames = self._center_crop(output_global_frames)
        # events = self._center_crop(events)
        input_frames, output_global_frames, events = self._crop(input_frames, output_global_frames, events)

        batch = get_vfi_empty_batch()
        batch["input_rolling_frames"] = input_frames  # [B, N, C, H, W]
        batch["target_global_frames"] = output_global_frames  # [B, N, C, H, W]
        batch["input_rolling_frames_normal_timestamps"] = torch.Tensor(self.input_rs_frame_timestamps)
        batch["target_global_frames_normal_timestamps"] = torch.Tensor(self.output_gs_frame_timestamps)
        batch["events"] = events
        batch["video_name"] = video_name
        batch["first_rolling_frame_name"] = first_rolling_frame_name
        return batch

    def __len__(self):
        return len(self.samples)

    def _crop(self, input_frames, output_global_frames, events):
        # Drop the height
        drop_size_height = self.H - self.center_cropped_height
        th = drop_size_height // 2
        min_x = th
        max_x = min_x + self.center_cropped_height
        # Drop the width
        if self.is_train:
            min_y = random.randint(0, self.W - self.random_cropped_width)
        else:
            min_y = (self.W - self.random_cropped_width) // 2
        max_y = min_y + self.random_cropped_width
        # Crop
        input_frames = input_frames[:, :, min_x:max_x, min_y:max_y]
        output_global_frames = output_global_frames[:, :, min_x:max_x, min_y:max_y]
        events = events[:, min_x:max_x, min_y:max_y]
        return input_frames, output_global_frames, events

    def _to_moments(self, events):
        B, H, W = events.shape
        windows_size = B // self.events_moment_count
        if B % self.events_moment_count > 0:
            windows_size += 1
        events_zeros = torch.zeros((self.events_moment_count * windows_size, H, W))
        events_zeros[:B, :, :] = events
        events_zeros = events_zeros.view(self.events_moment_count, windows_size, H, W)
        events_zeros = events_zeros.mean(dim=1)
        return events_zeros

    def _load_events(self, event_path):
        events = np.load(event_path)
        events = self._render(shape=[260, 346], **events)
        return events

    @staticmethod
    def _render(x, y, t, p, shape):
        events = np.zeros(shape=shape)
        events[y, x] = p
        return events

    def _walk(self):
        folders = sorted(listdir(self.root))
        frame_folders = []
        event_folders = []
        rolling_folders = []
        for folder in folders:
            if not isdir(join(self.root, folder)):
                continue
            if folder.endswith("fps5000-resize-frames-H260-W346"):
                frame_folders.append(folder)
            elif folder.endswith("fps5000-resize-frames-H260-W346-events"):
                event_folders.append(folder)
            elif folder.endswith("fps5000-resize-frames-H260-W346-rolling"):
                rolling_folders.append(folder)
        assert len(frame_folders) == len(event_folders) == len(rolling_folders)

        items = []
        for frame_folder, event_folder, rolling_folder in zip(frame_folders, event_folders, rolling_folders):
            assert frame_folder[:23] == event_folder[:23] == rolling_folder[:23]
            items.extend(self._walk_video(frame_folder, event_folder, rolling_folder))
        return items

    def _walk_video(self, frame_folder, event_folder, rolling_folder):
        frames = sorted(listdir(join(self.root, frame_folder)))
        event_npz_png = sorted(listdir(join(self.root, event_folder)))
        events = [npz for npz in event_npz_png if npz.endswith(".npz")]
        rollings = sorted(listdir(join(self.root, rolling_folder)))
        # every two adjacent images will generate some events.
        assert len(frames) - 1 == len(events)
        # The first frame of rolling data is 260.png, which means that a total
        # of 260 images from images 0-256 in the global are synthesized.
        assert len(frames) == len(rollings) + 260
        items = []
        for i in range(0, len(rollings) - 261, self.sample_step):
            items_events = []
            for j in range(self.item_samples):
                items_events.append(join(self.root, event_folder, events[i + j]))
            items_rollings = []
            for j in self.input_rs_frame_indexes:
                items_rollings.append(join(self.root, rolling_folder, rollings[i + j]))
            items_global = []
            for j in self.output_gs_frame_indexes:
                items_global.append(join(self.root, frame_folder, frames[i + j]))
            items.append((items_rollings, items_events, items_global))
        return items
