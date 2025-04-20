from os.path import dirname, join

import cv2
import torch
from absl import logging
from absl.logging import info
from absl.testing import absltest
from egvsr.losses.psnr import PSNR
from torch.autograd import Variable


class PSNRTest(absltest.TestCase):
    def setUp(self):
        self.testdata = join(dirname(__file__), "testdata")
        self.einstein = cv2.imread(join(self.testdata, "einstein.png"))
        logging.set_verbosity(logging.DEBUG)

    def test_basic_psnr(self):
        img1 = Variable(torch.rand(1, 1, 256, 256))
        img2 = Variable(torch.rand(1, 1, 256, 256))
        psnr_loss = PSNR(value_range=1)

        if torch.cuda.is_available():
            img1 = img1.cuda()
            img2 = img2.cuda()
            psnr_loss = psnr_loss.cuda()

        psnr_11 = psnr_loss(img1, img1)
        psnr_12 = psnr_loss(img1, img2)
        psnr_21 = psnr_loss(img2, img1)
        psnr_22 = psnr_loss(img2, img2)

        info(f"psnr_11: {psnr_11}")
        info(f"psnr_12: {psnr_12}")
        info(f"psnr_21: {psnr_21}")
        info(f"psnr_22: {psnr_22}")
        self.assertGreater(psnr_11, 40)
        self.assertEqual(psnr_12, psnr_21)
        self.assertGreater(psnr_22, 40)


if __name__ == "__main__":
    absltest.main()
