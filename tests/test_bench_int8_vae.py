"""CPU contracts for the production-path INT8 benchmark helpers."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import bench_int8_vae as bench  # noqa: E402


class BenchInt8VaeCpuTest(unittest.TestCase):
    def test_rgb_layout_is_bcthw_and_float32_reference(self):
        frames = torch.zeros((2, 2, 3, 3), dtype=torch.uint8)
        frames[1, 1, 2] = torch.tensor([0, 127, 255], dtype=torch.uint8)
        reference = bench._rgb_uint8_to_reference(frames)
        self.assertEqual(tuple(reference.shape), (1, 3, 2, 2, 3))
        self.assertEqual(reference.dtype, torch.float32)
        self.assertAlmostEqual(float(reference[0, 0, 1, 1, 2]), 0.0)
        self.assertAlmostEqual(float(reference[0, 1, 1, 1, 2]), 127 / 255)
        self.assertAlmostEqual(float(reference[0, 2, 1, 1, 2]), 1.0)

    def test_mean_and_global_psnr_are_distinct(self):
        reference = torch.full((1, 3, 2, 2, 2), 0.5)
        candidate = reference.clone()
        candidate[:, :, 0] += 0.01
        candidate[:, :, 1] += 0.1
        metrics = bench._quality_metrics(reference, candidate)
        self.assertAlmostEqual(metrics["mean_frame_psnr_db"], 30.0, places=5)
        self.assertAlmostEqual(metrics["global_psnr_db"], 22.967086, places=5)
        self.assertFalse(metrics["all_frames_pass_30db"])
        self.assertEqual(metrics["below_30db_count"], 1)

    def test_zero_error_frame_marks_mean_infinite(self):
        reference = torch.full((1, 3, 2, 2, 2), 0.5)
        candidate = reference.clone()
        candidate[:, :, 1] += 0.1
        metrics = bench._quality_metrics(reference, candidate)
        self.assertIsNone(metrics["mean_frame_psnr_db"])
        self.assertTrue(metrics["mean_frame_psnr_infinite"])
        self.assertFalse(metrics["global_psnr_infinite"])
        self.assertTrue(metrics["per_frame"][0]["psnr_infinite"])

    def test_exact_threshold_passes(self):
        record = bench._psnr_record(bench.MSE_AT_30_DB)
        self.assertAlmostEqual(record["psnr_db"], 30.0, places=6)
        self.assertFalse(record["below_30db"])

    def test_tensor_error_is_finite_and_shape_checked(self):
        reference = torch.ones((2, 3))
        candidate = reference + 0.25
        result = bench._tensor_error(reference, candidate)
        self.assertTrue(result["finite"])
        self.assertAlmostEqual(result["mae"], 0.25)
        with self.assertRaisesRegex(RuntimeError, "shape mismatch"):
            bench._tensor_error(reference, torch.ones((3, 2)))


if __name__ == "__main__":
    unittest.main()
