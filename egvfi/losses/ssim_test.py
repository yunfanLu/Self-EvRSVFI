from os.path import dirname, join

import torch
from absl.logging import info
from absl.testing import absltest
from egvsr.losses.ssim import SSIM, ssim
from torch.autograd import Variable


class SSIMTest(absltest.TestCase):
    def setUp(self):
        self.testdata = join(dirname(__file__), "testdata")

    def test_basic_ssim(self):
        img1 = Variable(torch.rand(1, 1, 256, 256))
        img2 = Variable(torch.rand(1, 1, 256, 256))
        ssim_loss = SSIM(value_range=1, window_size=11)

        if torch.cuda.is_available():
            img1 = img1.cuda()
            img2 = img2.cuda()
            ssim_loss = ssim_loss.cuda()

        ssim_21 = ssim(img2, img1)
        ssim_loss_21 = ssim_loss(img2, img1)
        ssim_12 = ssim(img1, img2)
        ssim_loss_12 = ssim_loss(img1, img2)
        ssim_11 = ssim(img1, img1)
        ssim_loss_11 = ssim_loss(img1, img1)
        ssim_22 = ssim(img2, img2)
        ssim_loss_22 = ssim_loss(img2, img2)
        info(f"ssim_11: {ssim_11}")
        info(f"ssim_12: {ssim_12}")
        info(f"ssim_21: {ssim_21}")
        info(f"ssim_22: {ssim_22}")
        info(f"ssim_loss_11: {ssim_loss_11}")
        info(f"ssim_loss_12: {ssim_loss_12}")
        info(f"ssim_loss_21: {ssim_loss_21}")
        info(f"ssim_loss_22: {ssim_loss_22}")

        self.assertEqual(ssim_21, ssim_12)
        self.assertEqual(ssim_21, ssim_loss_21)
        self.assertEqual(ssim_21, ssim_loss_12)
        self.assertEqual(ssim_11, 1)
        self.assertEqual(ssim_loss_11, 1)
        self.assertEqual(ssim_22, 1)
        self.assertEqual(ssim_loss_22, 1)


if __name__ == "__main__":
    absltest.main()
