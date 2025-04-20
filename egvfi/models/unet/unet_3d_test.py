#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Author ：Yunfan Lu (yunfanlu@ust.hk)
# Date   ：2023/2/5 10:03
import torch
from absl.logging import info
from absl.testing import absltest
from thop import profile

from egvfi.models.unet.unet_3d import UNet3D


class UNet3DTest(absltest.TestCase):
    def test_inference(self):
        image_hight = 260 // 32 * 32
        image_width = 346 // 32 * 32
        time_bins = 32
        x = torch.Tensor(1, 2, time_bins, image_hight, image_width).cuda()
        info("x size: {}".format(x.size()))
        model = UNet3D(in_dim=2, out_dim=2, num_filters=2, employ_bn=False).cuda()
        out = model(x)
        info("out size: {}".format(out.size()))
        flops, params = profile(model, inputs=(x,))
        info("flops: {}G".format(flops / 1024**3))
        info("params: {}M".format(params / 1024**2))


if __name__ == "__main__":
    absltest.main()
