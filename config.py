#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from os.path import isdir

"""
This is the config file for the project.
You can change the path here.
"""

class GPU87LU:
    @property
    def test_data(self):
        return "./testdata/"

    @property
    def timelens_pp_root(self):
        return "./dataset/timelens_pp/"

    @property
    def alpx_event_dataset(self):
        return "./dataset/5-ALPEX-VFI/"

    @property
    def fps10000_video_folder(self):
        return "./dataset/3-10000fps-videos/0-Videos/"

    @property
    def fps5000_video_folder(self):
        return "/mnt/dev-ssd-8T/yunfanlu/workspace/dataset/3-10000fps-videos/1-Videos-EvNurall/"

    @property
    def alpx_vfi_dataset(self):
        return "/mnt/dev-ssd-8T/yunfanlu/workspace/dataset/0-ALPEX-Real-World-Dataset/iccv23-vfi-dataset/"

    def test(self):
        assert isdir(self.fps5000_video_folder)
        assert isdir(self.fps10000_video_folder)


class HPC:
    @property
    def test_data(self):
        return "./testdata/"

    @property
    def timelens_pp_root(self):
        return "./dataset/timelens_pp/"

    @property
    def alpx_event_dataset(self):
        return "./dataset/5-ALPEX-VFI/"

    @property
    def fps10000_video_folder(self):
        return "./dataset/3-10000fps-videos/0-Videos/"

    @property
    def fps5000_video_folder(self):
        return "/hpc/users/CONNECT/ethanliang/evunroll_whole_dataset/"

    @property
    def fastec_video_folder(self):
        return "./dataset/3-10000fps-videos/2-Fastec-Simulated/"

    def test(self):
        assert isdir(self.fps5000_video_folder)
        assert isdir(self.fps10000_video_folder)


global_path = GPU87LU()
