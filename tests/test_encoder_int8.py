"""CPU contracts for the standalone Triton encoder INT8 prototype."""

from __future__ import annotations

import unittest

import torch
from torch import nn

from opt.encoder_int8 import (
    TritonInt8Conv3d,
    int32_accumulator_bound,
    int8_valid_conv3d_cpu_reference,
    prepare_int8_weight,
    validate_int8_weight_buffers,
)


class EncoderInt8Tests(unittest.TestCase):
    def test_packed_weight_shape_scale_and_overflow_bound(self) -> None:
        weight = torch.randn(5, 3, 3, 1, 3)
        qweight, scale, config = prepare_int8_weight(
            weight, stride=(2, 1, 2)
        )
        self.assertEqual(qweight.dtype, torch.int8)
        self.assertEqual(tuple(qweight.shape), (5, 27))
        self.assertEqual(tuple(scale.shape), (5,))
        self.assertTrue(scale.isfinite().all())
        self.assertTrue((scale > 0).all())
        validate_int8_weight_buffers(qweight, scale, config)
        self.assertLess(int32_accumulator_bound(config), 2**31)

    def test_cpu_reference_rectangular_stride2_bias(self) -> None:
        torch.manual_seed(31)
        x = torch.randn(1, 3, 5, 8, 10)
        weight = torch.randn(4, 3, 3, 1, 3)
        bias = torch.randn(4)
        got = int8_valid_conv3d_cpu_reference(
            x, weight, bias, stride=(2, 1, 2)
        )
        self.assertEqual(tuple(got.shape), (1, 4, 2, 8, 4))
        self.assertTrue(torch.isfinite(got).all())

    def test_cpu_reference_zero_weight_preserves_bias(self) -> None:
        x = torch.zeros(1, 2, 3, 4, 5)
        weight = torch.zeros(3, 2, 1, 1, 1)
        bias = torch.tensor([1.0, -2.0, 0.5])
        got = int8_valid_conv3d_cpu_reference(x, weight, bias)
        expected = bias.view(1, 3, 1, 1, 1).expand_as(got)
        self.assertTrue(torch.equal(got, expected))

    def test_module_does_not_silently_fallback_on_cpu(self) -> None:
        source = nn.Conv3d(2, 3, 1, bias=True).eval()
        candidate = TritonInt8Conv3d(source, assume_padded=True).eval()
        self.assertIsInstance(candidate.qweight, nn.Parameter)
        self.assertFalse(candidate.qweight.requires_grad)
        candidate.to(dtype=torch.float16)
        self.assertEqual(candidate.qweight.dtype, torch.int8)
        with self.assertRaisesRegex(RuntimeError, "requires a CUDA"):
            candidate(torch.randn(1, 2, 2, 3, 4))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
