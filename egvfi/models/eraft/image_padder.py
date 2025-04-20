#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Author ：Yunfan Lu (yunfanlu@ust.hk)
# Date   ：2023/2/4 19:32
from torch import nn


class ImagePadder(object):
    # =================================================================== #
    # In some networks, the image gets downsized. This is a problem, if   #
    # the to-be-downsized image has odd dimensions ([15x20]->[7.5x10]).   #
    # To prevent this, the input image of the network needs to be a       #
    # multiple of a minimum size (min_size)                               #
    # The ImagePadder makes sure, that the input image is of such a size, #
    # and if not, it pads the image accordingly.                          #
    # =================================================================== #

    def __init__(self, min_size=64):
        # --------------------------------------------------------------- #
        # The min_size additionally ensures, that the smallest image      #
        # does not get too small                                          #
        # --------------------------------------------------------------- #
        self.min_size = min_size
        self.pad_height = None
        self.pad_width = None

    def pad(self, image):
        # --------------------------------------------------------------- #
        # If necessary, this function pads the image on the left & top    #
        # --------------------------------------------------------------- #
        height, width = image.shape[-2:]
        if self.pad_width is None:
            self.pad_height = (self.min_size - height % self.min_size) % self.min_size
            self.pad_width = (self.min_size - width % self.min_size) % self.min_size
        else:
            pad_height = (self.min_size - height % self.min_size) % self.min_size
            pad_width = (self.min_size - width % self.min_size) % self.min_size
            if pad_height != self.pad_height or pad_width != self.pad_width:
                raise
        return nn.ZeroPad2d((self.pad_width, 0, self.pad_height, 0))(image)

    def unpad(self, image):
        # --------------------------------------------------------------- #
        # Removes the padded rows & columns                               #
        # --------------------------------------------------------------- #
        return image[..., self.pad_height :, self.pad_width :]
