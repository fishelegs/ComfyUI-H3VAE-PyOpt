"""CPU contract tests for the opt-in kitchen INT8 experiment."""
from __future__ import annotations

import sys
import types
import unittest
from contextlib import contextmanager
from pathlib import Path

import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import opt.kitchen_int8 as kitchen_int8  # noqa: E402


def _fake_quantize_rowwise(weight, stochastic_rounding=0):
    del stochastic_rounding
    scale = weight.detach().abs().amax(dim=-1, keepdim=True).float()
    scale = (scale / 127.0).clamp(min=1e-30)
    qweight = (weight.detach().float() / scale).round().clamp(-128, 127).to(
        torch.int8
    )
    return qweight, scale


def _fake_int8_linear(
    x,
    qweight,
    weight_scale,
    bias=None,
    out_dtype=None,
    convrot=False,
    **kwargs,
):
    del convrot, kwargs
    dequantized = qweight.float() * weight_scale.reshape(-1, 1)
    bias_float = None if bias is None else bias.float()
    output = torch.nn.functional.linear(x.float(), dequantized, bias_float)
    return output.to(out_dtype or x.dtype)


def fake_kitchen():
    module = types.ModuleType("comfy_kitchen")
    module.quantize_int8_rowwise = _fake_quantize_rowwise
    module.int8_linear = _fake_int8_linear
    return module


@contextmanager
def fake_kitchen_import():
    """Replace one module key without rolling back lazy torch imports."""
    missing = object()
    previous = sys.modules.get("comfy_kitchen", missing)
    sys.modules["comfy_kitchen"] = fake_kitchen()
    try:
        yield
    finally:
        if previous is missing:
            sys.modules.pop("comfy_kitchen", None)
        else:
            sys.modules["comfy_kitchen"] = previous


class ToyFeedForward(nn.Module):
    def __init__(self, dim=5):
        super().__init__()
        self.use_gated = True
        self.w1 = nn.Linear(dim, dim * 2)
        self.act_fn = nn.SiLU()
        self.w2 = nn.Linear(dim, dim)


class KitchenInt8CpuTest(unittest.TestCase):
    def test_zero_row_and_nonsquare_per_output_scales(self):
        source = nn.Linear(5, 3).eval()
        with torch.no_grad():
            source.weight[1].zero_()
        original_weight = source.weight.detach().clone()
        original_bias = source.bias.detach().clone()
        with fake_kitchen_import():
            candidate = kitchen_int8.KitchenInt8Linear(source).eval()
            x = torch.randn(2, 4, 5)
            with torch.no_grad():
                output = candidate(x)
        expected_weight = candidate.qweight.float() * candidate.weight_scale.reshape(
            -1, 1
        )
        expected = torch.nn.functional.linear(x, expected_weight, source.bias)
        self.assertEqual(tuple(candidate.qweight.shape), (3, 5))
        self.assertEqual(tuple(candidate.weight_scale.shape), (3,))
        self.assertEqual(candidate.weight_scale.dtype, torch.float32)
        self.assertTrue(torch.isfinite(candidate.weight_scale).all())
        self.assertTrue((candidate.weight_scale > 0).all())
        self.assertTrue(torch.equal(candidate.qweight[1], torch.zeros(5, dtype=torch.int8)))
        self.assertEqual(float(candidate.weight_scale[1]), 1.0)
        self.assertTrue(torch.isfinite(output).all())
        self.assertTrue(torch.allclose(output, expected, rtol=1e-5, atol=1e-6))
        self.assertTrue(torch.equal(source.weight, original_weight))
        self.assertTrue(torch.equal(source.bias, original_bias))
        self.assertIsNot(candidate.qweight, source.weight)

    def test_ffn_mode_only_replaces_selected_branch(self):
        source = ToyFeedForward().eval()
        original_w1 = source.w1
        original_w2 = source.w2
        with fake_kitchen_import():
            candidate = kitchen_int8.KitchenInt8FFN(source, "INT8w1").eval()
        self.assertIsInstance(candidate.w1, kitchen_int8.KitchenInt8Linear)
        self.assertIs(candidate.w2, original_w2)
        self.assertIs(source.w1, original_w1)
        self.assertIs(source.w2, original_w2)
        self.assertGreater(candidate.extra_weight_bytes, 0)

    def test_forward_rejects_training_and_grad(self):
        source = nn.Linear(5, 3).eval()
        with fake_kitchen_import():
            candidate = kitchen_int8.KitchenInt8Linear(source)
            candidate.train()
            with self.assertRaisesRegex(RuntimeError, "inference-only"):
                candidate(torch.randn(2, 5))
            candidate.eval()
            with torch.enable_grad():
                with self.assertRaisesRegex(RuntimeError, "no_grad"):
                    candidate(torch.randn(2, 5, requires_grad=True))

    def test_nonfinite_source_weight_rejected_before_quantization(self):
        source = nn.Linear(5, 3).eval()
        with torch.no_grad():
            source.weight[0, 0] = float("nan")
        with fake_kitchen_import():
            with self.assertRaisesRegex(ValueError, "non-finite source weights"):
                kitchen_int8.KitchenInt8Linear(source)

    def test_bad_scale_rejected(self):
        source = nn.Linear(5, 3).eval()

        def bad_quantize(weight, stochastic_rounding=0):
            del stochastic_rounding
            return torch.zeros_like(weight, dtype=torch.int8), torch.zeros(
                weight.shape[0], 1
            )

        with fake_kitchen_import():
            with self.assertRaisesRegex(ValueError, "strictly positive"):
                kitchen_int8.KitchenInt8Linear(source, quantize_fn=bad_quantize)

    def test_module_dtype_move_preserves_fp32_scales_and_int8_weights(self):
        source = nn.Linear(5, 3).eval()
        with fake_kitchen_import():
            candidate = kitchen_int8.KitchenInt8Linear(source).eval()
        original_scale = candidate.weight_scale.detach().clone()
        state = candidate.state_dict()
        self.assertIn("qweight", state)
        self.assertIn("weight_scale", state)
        self.assertEqual(state["qweight"].dtype, torch.int8)
        self.assertEqual(state["weight_scale"].dtype, torch.float32)
        candidate.to(dtype=torch.float16)
        self.assertEqual(candidate.weight_scale.dtype, torch.float32)
        self.assertTrue(torch.equal(candidate.weight_scale, original_scale))
        self.assertEqual(candidate.qweight.dtype, torch.int8)


if __name__ == "__main__":
    unittest.main()
