# -*- coding: utf-8 -*-

import unittest
import torch

from Code.lib import ops


class CropOrPadAsTest(unittest.TestCase):
    def test_crop(self):
        input = torch.rand(64, 256, 13, 24)
        other = torch.rand(32, 8, 6)
        output = ops.crop_or_pad_as(input, other)
        self.assertEqual(output.shape[:-2], input.shape[:-2])
        self.assertEqual(output.shape[-2:], other.shape[-2:])
        torch.testing.assert_allclose(
            output, input[..., 2:10, 9:15])

    def test_pad(self):
        input = torch.rand(64, 256, 7, 9)
        other = torch.rand(32, 18, 19)
        output = ops.crop_or_pad_as(input, other, pad_val=0.5)
        self.assertEqual(output.shape[:-2], input.shape[:-2])
        self.assertEqual(output.shape[-2:], other.shape[-2:])
        torch.testing.assert_allclose(
            output[..., 5:12, 5:14], input)
        self.assertEqual(output[0, 0, 0, 0], 0.5)

    def test_crop_and_pad(self):
        input = torch.rand(64, 256, 20, 9)
        other = torch.rand(32, 18, 19)
        output = ops.crop_or_pad_as(input, other, pad_val=0.5)
        self.assertEqual(output.shape[:-2], input.shape[:-2])
        self.assertEqual(output.shape[-2:], other.shape[-2:])
        torch.testing.assert_allclose(
            output[..., :, 5:14], input[..., 1:19, :])
        self.assertEqual(output[0, 0, 0, 0], 0.5)


if __name__ == '__main__':
    unittest.main()
