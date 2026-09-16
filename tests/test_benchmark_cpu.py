"""CPU-only checks for benchmark reporting logic."""
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench_pyopt_vs_trt import errors  # noqa: E402


class ErrorMetricsTest(unittest.TestCase):
    def test_identical_pixels(self):
        image = torch.zeros((1, 3, 1, 2, 2))
        result = errors(image, image, pixels=True)
        self.assertTrue(result["shape_match"])
        self.assertEqual(result["rmse"], 0)
        self.assertIsNone(result["psnr_db"])

    def test_encoder_latents_do_not_get_pixel_psnr(self):
        reference = torch.zeros((1, 24, 1, 2, 2))
        candidate = torch.ones_like(reference)
        result = errors(reference, candidate, pixels=False)
        self.assertEqual(result["mae"], 1)
        self.assertIsNone(result["psnr_db"])

    def test_shape_mismatch(self):
        result = errors(torch.zeros((1, 3, 1, 2, 2)),
                        torch.zeros((1, 3, 2, 2, 2)), pixels=True)
        self.assertFalse(result["shape_match"])


if __name__ == "__main__":
    unittest.main()
