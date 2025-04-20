#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Author ：Yunfan Lu (yunfanlu@ust.hk)
# Date   ：2023/1/30 14:18

import torch


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


def get_WHT_rolling_coords(t_start: float, t_end: float, h: int, w: int):
    """
    :param t: The timestamp of the sampled frame.
    :param h: The height of the sampled frame, which is not the original height.
    :param w: The width of the sampled frame, which is not the original width.
    :return: a coords tensor of shape (1, h, w, 3), where the last dimension
             is (t, w, h).
    """
    assert t_start >= -1 and t_end <= 1 and t_end >= t_start
    # (1, h, w, 3):
    #   1 means one time stamps.
    #   h and w means the height and width of the image.
    #   3 means the w, h, and t coordinates. The order is important.
    grid_map = torch.zeros(1, h, w, 3) + t_start

    for i in range(h):
        grid_map[0, i, :, 0] = t_start + (t_end - t_start) * i / (h - 1)

    h_coords = torch.linspace(-1, 1, h)
    w_coords = torch.linspace(-1, 1, w)
    mesh_h, mesh_w = torch.meshgrid([h_coords, w_coords])
    # The feature is H W T, so the coords order is (t, w, h)
    # grid_map \in R^{1, h, w, 3}, grid_map[:, :, :, i] is (t, w, h)

    # todo: check if this is correct, the mesh_h is in the first of mesh_w.
    #  this is currently.
    # grid_map[:, :, :, 1:] = torch.stack((mesh_w, mesh_h), 2)
    grid_map[:, :, :, 1:] = torch.stack((mesh_h, mesh_w), 2)
    return grid_map.float()
