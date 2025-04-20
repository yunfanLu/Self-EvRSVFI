#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project :EG-VFI 
@Author  :Yunfan Lu (yunfanlu@ust.hk)
@Date    :9/12/2022 15:30 
"""
import glob
import logging
import random
from logging import info
from os import listdir
from os.path import join

import numpy as np
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision.transforms import transforms

from config import global_path as gp
from egvfi.utils.events_to_frame import event_stream_to_frames

logging.getLogger("PIL").setLevel(logging.WARNING)


def get_WHT_fixed_timestamp_coords(t: float, h: int, w: int):
    """
    :param t: The timestamp of the sampled frame.
    :param h: The height of the sampled frame, which is not the original height.
    :param w: The width of the sampled frame, which is not the original width.
    :return: a coords tensor of shape (1, h, w, 3), where the last dimension
             is (t, w, h).
    """
    assert t >= -1 and t <= 1, f"Time t should be in [-1, 1], but got {t}."
    # (1, h, w, 3):
    #   1 means one time stamps.
    #   h and w means the height and width of the image.
    #   3 means the w, h, and t coordinates. The order is important.
    grid_map = torch.zeros(1, h, w, 3) + t
    h_coords = torch.linspace(-1, 1, h)
    w_coords = torch.linspace(-1, 1, w)
    mesh_h, mesh_w = torch.meshgrid([h_coords, w_coords])
    # The feature is H W T, so the coords order is (t, w, h)
    # grid_map \in R^{1, h, w, 3}, grid_map[:, :, :, i] is (t, w, h)
    # grid_map[:, :, :, 1:] = torch.stack((mesh_w, mesh_h), 2)
    grid_map[:, :, :, 1:] = torch.stack((mesh_h, mesh_w), 2)
    return grid_map.float()


def get_timelen_pp(skip_frames, key_frames, step, moments, random_crop, random_crop_resolution):
    timelens_pp_root_train = join(gp.timelens_pp_root, "3_TRAINING")
    train = TimeLensPPDataset(
        timelens_pp_root=timelens_pp_root_train,
        key_frames=key_frames,
        skip_frames=skip_frames,
        step=step,
        moments=moments,
        random_cropping=random_crop,
        random_crop_resolution=random_crop_resolution,
    )
    timelens_pp_root_test = join(gp.timelens_pp_root, "1_TEST")
    test = TimeLensPPDataset(
        timelens_pp_root=timelens_pp_root_test,
        key_frames=key_frames,
        skip_frames=skip_frames,
        step=step,
        moments=moments,
        random_cropping=random_crop,
        random_crop_resolution=random_crop_resolution,
    )
    timelens_pp_root_val = join(gp.timelens_pp_root, "2_VALIDATION")
    val = TimeLensPPDataset(
        timelens_pp_root=timelens_pp_root_val,
        key_frames=key_frames,
        skip_frames=skip_frames,
        step=step,
        moments=moments,
        random_cropping=random_crop,
        random_crop_resolution=random_crop_resolution,
    )
    return train, test, val


def _load_events(file):
    tmp = np.load(file, allow_pickle=True)
    (x, y, timestamp, polarity) = (
        tmp["x"].astype(np.float64).reshape((-1,)) / 32,
        tmp["y"].astype(np.float64).reshape((-1,)) / 32,
        tmp["timestamp"].astype(np.float64).reshape((-1,)),
        tmp["polarity"].astype(np.float32).reshape((-1,)) * 2 - 1,
    )
    events = np.stack((timestamp, x, y, polarity), axis=-1)
    return events


def walk_video(video_path, key_frame, skip_frames, step=1):
    image_template = join(video_path, "images", "*.png")
    event_template = join(video_path, "events", "*.npz")
    images = sorted(glob.glob(image_template))
    events = sorted(glob.glob(event_template))

    timestamps_path = join(video_path, "images", "timestamp.txt")
    timestamps = np.loadtxt(timestamps_path).tolist()
    assert len(images) == (len(events) + 1)
    assert len(timestamps) >= len(images)
    # Generate a sample.
    all_frames = key_frame + (key_frame - 1) * skip_frames
    sample = []
    for i in range(0, len(images) - all_frames, step):
        item = {}
        item["begin_timestamp"] = timestamps[i]
        item["end_timestamp"] = timestamps[i + all_frames - 1]
        item["key_frames"] = []
        item["key_frames_timestamp"] = []
        item["target_frames"] = []
        item["target_frames_timestamp"] = []
        for j in range(i, i + all_frames):
            if (j - i) % (skip_frames + 1) == 0:
                item["key_frames"].append(images[j])
                item["key_frames_timestamp"].append(timestamps[j])
            else:
                item["target_frames"].append(images[j])
                item["target_frames_timestamp"].append(timestamps[j])
        item["events"] = []
        for j in range(i, i + all_frames - 1):
            item["events"].append(events[j])
        sample.append(item)
    return sample


def walk(timelens_pp_root, key_frame, skip_frames, step=1):
    videos = listdir(timelens_pp_root)
    sample = []
    for video in videos:
        video_path = join(timelens_pp_root, video)
        sample.extend(walk_video(video_path, key_frame, skip_frames, step))
    return sample


class TimeLensPPDataset(Dataset):
    def __init__(
        self,
        timelens_pp_root,
        key_frames,
        skip_frames,
        step,
        moments,
        random_cropping,
        random_crop_resolution,
    ):
        super(TimeLensPPDataset, self).__init__()
        assert moments >= 2
        info(f"Loading TimeLensPPDataset from {timelens_pp_root} ...")
        self.timelens_pp_root = timelens_pp_root
        self.key_frames = key_frames
        self.skip_frames = skip_frames
        self.all_frame_count = key_frames + (key_frames - 1) * skip_frames
        self.step = step
        self.samples = walk(timelens_pp_root, key_frames, skip_frames, step)
        # Set the transforms.
        self.moments = moments
        self.resolution = (625, 970)
        self.positive = 1
        self.negative = -1
        self.random_cropping = random_cropping
        if random_cropping:
            self.random_crop_resolution = random_crop_resolution
        self.events_max_value = 40.0 * 8 * 4 / moments
        # Function
        self.to_tensor = transforms.ToTensor()
        self._info()

    def _info(self):
        info(f"  -key_frame= {self.key_frames}")
        info(f"  -skip_frame= {self.skip_frames}")
        info(f"    -in = {self.key_frames}")
        info(f"    -out= {self.all_frame_count}")
        info(f"  -step= {self.step}")
        info(f"  -Num samples: {len(self.samples)}")
        info(f"  -moments = {self.moments}")
        info(f"  -resolution= {self.resolution}")
        info(f"  -random_cropping= {self.random_cropping}")
        info(f"  -events_max_value= {self.events_max_value}")

    def __getitem__(self, index):
        item = self.samples[index]
        key_frames = [self.to_tensor(Image.open(i).convert("RGB")) for i in item["key_frames"]]
        target_frames = [self.to_tensor(Image.open(i).convert("RGB")) for i in item["target_frames"]]
        key_frames = torch.stack(key_frames, dim=0)
        key_frames_union_representation = torch.zeros(self.moments, 3, 625, 970)
        key_frames_union_representation[0] = key_frames[0]
        key_frames_union_representation[-1] = key_frames[-1]
        target_frames = torch.stack(target_frames, dim=0)

        # Timestamps to coords
        begin_timestamp = item["begin_timestamp"]
        end_timestamp = item["end_timestamp"]
        duration = end_timestamp - begin_timestamp
        if self.random_cropping:
            h, w = self.random_crop_resolution
        else:
            h, w = self.resolution
        target_frame_coords = []
        target_feature_t = []
        for target in item["target_frames_timestamp"]:
            t = (target - begin_timestamp) * 2 / duration - 1
            target_feature_t.append(t)
            coord = get_WHT_fixed_timestamp_coords(t=t, h=h, w=w)
            target_frame_coords.append(coord)
        target_feature_t = torch.FloatTensor(target_feature_t)
        target_frame_coords = torch.concat(target_frame_coords, dim=0)
        # For event data
        events = [_load_events(e) for e in item["events"]]
        events = np.concatenate(events, axis=0)
        event_count_images = event_stream_to_frames(
            events,
            moments=self.moments,
            resolution=self.resolution,
            positive=self.positive,
            negative=self.negative,
        )
        event_count_images = np.stack(event_count_images, axis=0)
        events = torch.from_numpy(event_count_images) / self.events_max_value

        events_coords = []
        events_feature_t = []
        for i in range(self.moments):
            t = (1.0 * i / self.moments) * 2 - 1
            events_feature_t.append(t)
            h, w = self.random_crop_resolution if self.random_cropping else self.resolution
            coord = get_WHT_fixed_timestamp_coords(t=t, h=h, w=w)
            events_coords.append(coord)
        events_feature_t = torch.FloatTensor(events_feature_t)
        events_coords = torch.stack(events_coords, dim=0)

        batch = {
            "key_frames": key_frames,
            "key_frames_union_representation": key_frames_union_representation,
            "target_frames": target_frames,
            "target_frame_coords": target_frame_coords,
            "target_feature_t": target_feature_t,
            "events": events,
            "events_feature_t": events_feature_t,
            "events_coords": events_coords,
        }
        if self.random_cropping:
            batch = self.random_crop(batch)
        return batch

    def random_crop(self, batch):
        random_crop_h, random_crop_w = self.random_crop_resolution
        x1 = random.randint(0, 625 - random_crop_h)
        y1 = random.randint(0, 970 - random_crop_w)
        x2 = x1 + random_crop_h
        y2 = y1 + random_crop_w
        batch["key_frames"] = batch["key_frames"][:, :, x1:x2, y1:y2]
        batch["target_frames"] = batch["target_frames"][:, :, x1:x2, y1:y2]
        batch["events"] = batch["events"][:, :, x1:x2, y1:y2]
        return batch

    def __len__(self):
        return len(self.samples)
