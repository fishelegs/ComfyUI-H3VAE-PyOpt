"""Exact producer tests; CUDA cases require H3VAE_TEST_CUDA=1."""
import os
from pathlib import Path
import sys
import unittest

import torch

from opt.encoder_int8_norm import _norm_pack_with_maxima, quantized_temporal_norm_pad
from opt.encoder_int8 import int8_valid_conv3d, prepare_int8_weight, quantize_activation_tensor
from opt.encoder_int8_integration import int8_norm_conv3d


class NormProducerCpuTests(unittest.TestCase):
    def test_rejects_cpu_before_loading_cuda_modules(self):
        x = torch.empty((1, 128, 1, 2, 2), dtype=torch.float16)
        with self.assertRaisesRegex(ValueError, 'CUDA rank-5'):
            quantized_temporal_norm_pad(x, torch.ones(128), torch.zeros(128), 1e-6)


@unittest.skipUnless(
    os.environ.get('H3VAE_TEST_CUDA') == '1' and torch.cuda.is_available(),
    'set H3VAE_TEST_CUDA=1 on a CUDA host',
)
class NormProducerGpuTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Runtime installs this same opt import path for the original kernels.
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'opt'))

    def old_pack(self, x, w, b, pre_bias):
        from encoder_fused_temporal_norm import fused_temporal_norm_pad
        from encoder_fused_norm_bias import fused_bias_temporal_norm_pad
        if pre_bias is None:
            return fused_temporal_norm_pad(x, w, b, 1e-6, (128, 64, 8))
        return fused_bias_temporal_norm_pad(x, pre_bias, w, b, 1e-6)

    @torch.inference_mode()
    def test_exact_norm_quant_and_scale_with_padding_tails_and_layouts(self):
        for seed, c, d, h, w in [(41, 128, 1, 2, 3), (42, 256, 3, 13, 19), (43, 128, 17, 256, 256)]:
            torch.manual_seed(seed)
            x = torch.randn((1, c, d, h, w), device='cuda', dtype=torch.float16)
            weight = torch.randn(c, device='cuda', dtype=torch.float16)
            bias = torch.randn_like(weight)
            layouts = [x, x.contiguous(memory_format=torch.channels_last_3d)]
            if w > 3:
                layouts.append(x[..., ::2])
            for value in layouts:
                for pre_bias in (None, bias):
                    expected_y = self.old_pack(value, weight, bias, pre_bias)
                    y, maxima = _norm_pack_with_maxima(value, weight, bias, 1e-6, pre_bias)
                    self.assertTrue(torch.equal(y, expected_y))
                    self.assertEqual(maxima.max().item(), y.abs().max().item())
                    expected_q, expected_s = quantize_activation_tensor(expected_y)
                    q, s = quantized_temporal_norm_pad(value, weight, bias, 1e-6, pre_bias=pre_bias)
                    self.assertTrue(torch.equal(q, expected_q))
                    self.assertTrue(torch.equal(s, expected_s))
                    self.assertEqual(s.dtype, torch.float32)
                    self.assertTrue(q.is_contiguous(memory_format=torch.channels_last_3d))

    @torch.inference_mode()
    def test_zero_subnormal_and_constant_norm_outputs(self):
        x = torch.zeros((1, 128, 1, 3, 5), device='cuda', dtype=torch.float16)
        weight = torch.ones(128, device='cuda', dtype=torch.float16)
        for value in [0., 2**-24, 2**-14, -10., 2.]:
            bias = torch.full_like(weight, value)
            expected = quantize_activation_tensor(self.old_pack(x, weight, bias, None))
            actual = quantized_temporal_norm_pad(x, weight, bias, 1e-6)
            self.assertTrue(all(torch.equal(a, b) for a, b in zip(expected, actual)))

    @torch.inference_mode()
    def test_exact_custom_op_bias_and_fullgraph(self):
        torch.manual_seed(44)
        x = torch.randn((1, 128, 3, 13, 19), device='cuda', dtype=torch.float16)
        weight = torch.randn(128, device='cuda', dtype=torch.float16)
        bias = torch.randn_like(weight)
        conv = torch.randn((256, 128, 3, 3, 3), device='cuda', dtype=torch.float16) / 64
        qw, ws, config = prepare_int8_weight(conv)
        conv_bias = torch.randn(256, device='cuda', dtype=torch.float16)
        compiled = torch.compile(int8_norm_conv3d, fullgraph=True)
        for pre_bias in (None, bias):
            for cb in (None, conv_bias):
                expected = int8_valid_conv3d(
                    self.old_pack(x, weight, bias, pre_bias), qw, ws,
                    config=config, bias=cb, tile_variant='128x128x64_pipeline4',
                )
                actual = compiled(x, weight, bias, pre_bias, 1e-6, qw, ws, cb)
                self.assertTrue(torch.equal(expected, actual))

    @torch.inference_mode()
    def test_compiled_causal_zero_path_on_aligned_short_and_full_clips(self):
        # Separate wrapper limits compilation variants from the existing odd-
        # shape test while covering the now-optimized producer/conv boundary.
        def call(x, weight, bias, pre_bias, qw, ws, conv_bias):
            return int8_norm_conv3d(x, weight, bias, pre_bias, 1e-6, qw, ws, conv_bias)

        compiled = torch.compile(call, fullgraph=True)
        cases = [(128, 1, False, False), (128, 2, True, True), (256, 17, True, False)]
        for c, d, with_pre_bias, with_conv_bias in cases:
            torch.manual_seed(45 + d)
            x = torch.randn((1, c, d, 8, 16), device='cuda', dtype=torch.float16)
            weight = torch.randn(c, device='cuda', dtype=torch.float16)
            bias = torch.randn_like(weight)
            pre_bias = bias if with_pre_bias else None
            conv = torch.randn((256, c, 3, 3, 3), device='cuda', dtype=torch.float16) / 64
            qw, ws, config = prepare_int8_weight(conv)
            cb = torch.randn(256, device='cuda', dtype=torch.float16) if with_conv_bias else None
            expected = int8_valid_conv3d(
                self.old_pack(x, weight, bias, pre_bias), qw, ws,
                config=config, bias=cb, tile_variant='128x128x64_pipeline4',
            )
            actual = compiled(x, weight, bias, pre_bias, qw, ws, cb)
            self.assertTrue(torch.equal(expected, actual), (c, d))

    def test_rejects_unvalidated_shape_and_vector_dtype(self):
        for shape in [(2, 128, 1, 3, 3), (1, 64, 1, 3, 3), (1, 128, 1, 1, 3)]:
            x = torch.empty(shape, device='cuda', dtype=torch.float16)
            with self.assertRaisesRegex(ValueError, 'batch-one'):
                quantized_temporal_norm_pad(x, None, None, 1e-6)
        x = torch.empty((1, 128, 1, 3, 3), device='cuda', dtype=torch.float16)
        with self.assertRaisesRegex(ValueError, 'channel vectors'):
            quantized_temporal_norm_pad(x, torch.ones(128, device='cuda'), None, 1e-6)


if __name__ == '__main__':
    unittest.main()
