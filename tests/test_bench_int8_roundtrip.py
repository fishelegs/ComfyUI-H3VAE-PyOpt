"""CPU contracts for the four-way INT8 round-trip benchmark."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import bench_int8_roundtrip as bench  # noqa: E402


class _FakeEncoderBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.int8_first = nn.Identity()
        self.int8_second = nn.Identity()


class _FakeEncoderPrefix(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList(_FakeEncoderBlock() for _ in range(4))

    def forward(self, value):
        for block in self.blocks:
            value = block.int8_first(value)
            value = block.int8_second(value)
        return value


class _FakeCompiledPrefix(nn.Module):
    def __init__(self, original):
        super().__init__()
        self._orig_mod = original

    def forward(self, value):
        return self._orig_mod(value)


class _FakeRuntime:
    def __init__(self, *, fail=False):
        self._raw_prefix = _FakeEncoderPrefix()
        self.prefix = _FakeCompiledPrefix(self._raw_prefix)
        self.suffix = nn.Identity()
        self.int8_encoder_metadata = {"count": 8, "policy": {"stages": [0, 1]}}
        self.fail = fail

    def _device(self):
        return torch.device("cpu")

    def encode(self, value):
        self.prefix(value)
        if self.fail:
            raise RuntimeError("fake encoder failure")
        marker = 1.0 if self.prefix is self._raw_prefix else 2.0
        return torch.full((1, 24, 1, 1, 1), marker, dtype=torch.float32)


class BenchInt8RoundTripCpuTest(unittest.TestCase):
    def test_path_mapping_covers_the_four_real_combinations(self):
        self.assertEqual(
            bench.PATH_CONFIG,
            {
                "default": {"encode": "E0", "decode": "D0", "runtime": "default"},
                "d_only": {"encode": "E0", "decode": "D1", "runtime": "both"},
                "e_only": {"encode": "E1", "decode": "D0", "runtime": "default"},
                "both": {"encode": "E1", "decode": "D1", "runtime": "both"},
            },
        )
        self.assertEqual(set(bench.PATH_CONFIG), set(bench.DECODE_PATHS))

    def test_latent_error_includes_relative_rmse(self):
        reference = torch.tensor([[[[[3.0, 4.0]]]]])
        candidate = reference + 0.5
        result = bench._latent_error(reference, candidate)
        self.assertTrue(result["finite"])
        self.assertAlmostEqual(result["mae"], 0.5)
        self.assertAlmostEqual(result["rmse"], 0.5)
        self.assertAlmostEqual(result["reference_rms"], 12.5**0.5)
        self.assertAlmostEqual(result["relative_rmse"], 0.5 / (12.5**0.5))

    def test_zero_reference_relative_rmse_is_json_safe(self):
        reference = torch.zeros((1, 2))
        self.assertEqual(
            bench._latent_error(reference, reference.clone())["relative_rmse"], 0.0
        )
        candidate = torch.ones_like(reference)
        self.assertIsNone(bench._latent_error(reference, candidate)["relative_rmse"])

    def test_quality_headline_exposes_requested_statistics(self):
        reference = torch.full((1, 3, 2, 2, 2), 0.5)
        candidate = reference.clone()
        candidate[:, :, 0] += 0.01
        candidate[:, :, 1] += 0.1
        metrics = bench._base._quality_metrics(reference, candidate)
        headline = bench._quality_headline(metrics)
        self.assertAlmostEqual(headline["mean_frame_psnr_db"], 30.0, places=5)
        self.assertAlmostEqual(headline["global_psnr_db"], 22.967086, places=5)
        self.assertAlmostEqual(headline["min_frame_psnr_db"], 20.0, places=5)
        self.assertEqual(headline["below_30db_count"], 1)
        self.assertEqual(headline["frame_count"], 2)

    def test_rotate_order_records_ab_ba_and_all_four_paths(self):
        self.assertEqual(bench._rotate_order(bench.ENCODER_VARIANTS, 0), ["E0", "E1"])
        self.assertEqual(bench._rotate_order(bench.ENCODER_VARIANTS, 1), ["E1", "E0"])
        self.assertEqual(
            bench._rotate_order(bench.DECODE_PATHS, 2),
            ["e_only", "both", "default", "d_only"],
        )

    def test_parser_accepts_quality_only_and_csv_without_media_imports(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            videos = root / "videos"
            videos.mkdir()
            model = root / "model"
            model.mkdir()
            weights = root / "weights.safetensors"
            weights.write_bytes(b"placeholder")
            args, _ = bench._parse_args(
                [
                    "--videos-dir",
                    str(videos),
                    "--model-code-dir",
                    str(model),
                    "--weights",
                    str(weights),
                    "--output-dir",
                    str(root / "out"),
                    "--quality-only",
                    "--csv",
                ]
            )
            self.assertTrue(args.quality_only)
            self.assertTrue(args.per_frame_csv)

    def test_hook_diagnostic_latent_is_not_public_compiled_latent(self):
        runtime = _FakeRuntime()
        original_prefix = runtime.prefix
        reference = torch.zeros((1, 3, 1, 16, 16), dtype=torch.float32)
        prepared = torch.zeros_like(reference, dtype=torch.float16)
        hooked, _hook_ms, validation = bench._validate_int8_encoder_inputs(
            runtime, prepared, reference
        )
        public, _public_ms = bench._encode_quality_once(
            runtime, prepared, reference, name="fake public E1"
        )
        self.assertIs(runtime.prefix, original_prefix)
        self.assertTrue(torch.equal(hooked, torch.ones_like(hooked)))
        self.assertTrue(torch.equal(public, torch.full_like(public, 2.0)))
        self.assertEqual(validation["module_hook_count"], 8)
        self.assertTrue(validation["all_inputs_finite"])

    def test_hook_restores_compiled_prefix_and_removes_hooks_on_failure(self):
        runtime = _FakeRuntime(fail=True)
        original_prefix = runtime.prefix
        reference = torch.zeros((1, 3, 1, 16, 16), dtype=torch.float32)
        prepared = torch.zeros_like(reference, dtype=torch.float16)
        with self.assertRaisesRegex(RuntimeError, "fake encoder failure"):
            bench._validate_int8_encoder_inputs(runtime, prepared, reference)
        self.assertIs(runtime.prefix, original_prefix)
        for _name, module in bench._int8_encoder_targets(runtime):
            self.assertFalse(module._forward_pre_hooks)


if __name__ == "__main__":
    unittest.main()
