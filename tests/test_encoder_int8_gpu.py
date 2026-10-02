"""Opt-in GPU numerical contracts: H3VAE_TEST_CUDA=1 python -m unittest discover -s tests -p test_encoder_int8_gpu.py."""

import os
import unittest

import torch

from opt.encoder_int8 import (
    _triton_kernels,
    int8_valid_conv3d,
    prepare_int8_weight,
    quantize_activation_tensor,
    triton_cdiv,
)


@unittest.skipUnless(
    os.environ.get("H3VAE_TEST_CUDA") == "1" and torch.cuda.is_available(),
    "set H3VAE_TEST_CUDA=1 on a CUDA host",
)
class EncoderInt8GpuTests(unittest.TestCase):
    def legacy_quantize(self, x):
        x = x.contiguous(memory_format=torch.channels_last_3d)
        scale = x.abs().amax().float().div(127.0)
        scale = torch.where(scale > 0, scale, torch.ones_like(scale))
        q = torch.empty_like(x, dtype=torch.int8)
        kernel, _ = _triton_kernels()
        kernel[(triton_cdiv(x.numel(), 256),)](
            x,
            q,
            x.numel(),
            *x.shape[1:],
            *x.stride(),
            scale,
            BLOCK=256,
            num_warps=4,
        )
        return q, scale

    def test_quantization_matches_legacy_layouts_and_grid_tail(self):
        torch.manual_seed(20261002)
        for shape in [(2, 3, 5, 7, 11), (1, 128, 19, 258, 258), (1, 256, 19, 130, 130)]:
            x = torch.randn(shape, dtype=torch.float16, device="cuda")
            layouts = [x, x.contiguous(memory_format=torch.channels_last_3d)]
            if shape[1] == 3:
                layouts.append(x[..., ::2])
            for value in layouts:
                q, s = quantize_activation_tensor(value)
                expected_q, expected_s = self.legacy_quantize(value)
                self.assertTrue(torch.equal(s, expected_s))
                self.assertTrue(torch.equal(q, expected_q))
                self.assertTrue(q.is_contiguous(memory_format=torch.channels_last_3d))
                self.assertEqual(s.dtype, torch.float32)

    def test_zero_subnormal_extreme_and_half_away_rounding(self):
        for maximum in [0.0, 2**-24, 2**-14, 0.5, 127.0, 65504.0]:
            x = torch.full((1, 1, 1, 1, 9), maximum, dtype=torch.float16, device="cuda")
            x[..., 0] = -maximum
            q, s = quantize_activation_tensor(x)
            expected_q, expected_s = self.legacy_quantize(x)
            self.assertTrue(torch.equal(s, expected_s))
            self.assertTrue(torch.equal(q, expected_q))
        x = torch.tensor(
            [-127.0, -0.5, -1.5, 0.0, 0.5, 1.5, 127.0],
            dtype=torch.float16,
            device="cuda",
        ).view(1, 1, 1, 1, 7)
        q, s = quantize_activation_tensor(x)
        self.assertEqual(s.item(), 1.0)
        self.assertEqual(q.flatten().tolist(), [-127, -1, -2, 0, 1, 2, 127])

    def test_optional_finite_validation_and_empty_guard(self):
        for v in [float("nan"), float("inf")]:
            x = torch.full((1, 1, 1, 1, 1), v, dtype=torch.float16, device="cuda")
            with self.assertRaisesRegex(ValueError, "non-finite"):
                quantize_activation_tensor(x, check_finite=True)
        with self.assertRaisesRegex(ValueError, "nonempty"):
            quantize_activation_tensor(
                torch.empty((1, 1, 0, 1, 1), dtype=torch.float16, device="cuda")
            )

    def test_tile_equivalence_channels_tails_strides_and_bias(self):
        torch.manual_seed(20261002)
        for ci, co, shape, stride, with_bias in [
            (128, 128, (1, 128, 3, 19, 23), (1, 1, 1), True),
            (128, 256, (1, 128, 19, 17, 19), (1, 1, 1), False),
            (256, 256, (1, 256, 19, 17, 19), (1, 1, 1), True),
            (7, 19, (2, 7, 5, 9, 11), (1, 2, 2), True),
        ]:
            x = torch.randn(shape, dtype=torch.float16, device="cuda")
            w = (
                torch.randn((co, ci, 3, 3, 3), dtype=torch.float16, device="cuda")
                / (ci * 27) ** 0.5
            )
            bias = (
                torch.randn(co, dtype=torch.float16, device="cuda")
                if with_bias
                else None
            )
            qw, ws, config = prepare_int8_weight(w, stride=stride)
            original = int8_valid_conv3d(
                x, qw, ws, config=config, bias=bias, tile_variant="128x64x64"
            )
            candidate = int8_valid_conv3d(
                x, qw, ws, config=config, bias=bias, tile_variant="128x128x64"
            )
            self.assertTrue(torch.equal(original, candidate))
            self.assertTrue(torch.isfinite(candidate).all())


if __name__ == "__main__":
    unittest.main()
