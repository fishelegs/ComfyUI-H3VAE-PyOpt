"""CPU contracts for the explicit mixed INT8 encoder option."""
from __future__ import annotations

import inspect
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from h3vae_runtime import (  # noqa: E402
    H3VAEPyOptRuntime,
    _validate_int8_encode_options,
)
from opt.encoder_int8_integration import (  # noqa: E402
    INT8_ENCODER_POLICY,
    Int8FusedValidConv3d,
    _int8_valid_conv3d_integration_fake,
    _check_encoder_platform,
    _resolve_tile_variant,
    install_int8_encoder_prefix,
)


class _ToyConv(nn.Module):
    def __init__(self, channels: int = 4):
        super().__init__()
        self.weight = nn.Parameter(
            torch.randn(channels, channels, 3, 3, 3, dtype=torch.float16)
        )
        self.bias = nn.Parameter(torch.randn(channels, dtype=torch.float16))
        self.stride = (1, 1, 1)


class _ToyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.first = _ToyConv()
        self.second = _ToyConv()


class BiasFusedResidualBlock(_ToyBlock):
    pass


class ResidualDownsampleBlock(_ToyBlock):
    def __init__(self):
        super().__init__()
        self.down_weight = nn.Parameter(
            torch.randn(4, 4, 3, 3, 3, dtype=torch.float16)
        )


class _ToyStage(nn.Module):
    def __init__(self, final: bool = False):
        super().__init__()
        self.block = nn.ModuleList(
            [BiasFusedResidualBlock(), ResidualDownsampleBlock()]
            if final
            else [BiasFusedResidualBlock(), BiasFusedResidualBlock()]
        )


class _DeviceOnlyPrefix(nn.Module):
    def __init__(self, devices):
        super().__init__()
        self._parameter_devices = [torch.device(device) for device in devices]

    def parameters(self, recurse=True):
        del recurse
        return [type("Parameter", (), {"device": device})()
                for device in self._parameter_devices]


def _toy_prefix() -> nn.Module:
    prefix = nn.Module()
    prefix.stages = nn.ModuleList([_ToyStage(), _ToyStage(final=True)])
    return prefix


class EncoderInt8IntegrationCpuTest(unittest.TestCase):
    @staticmethod
    def _supported(**overrides):
        values = {
            "enabled": True,
            "device": "cuda",
            "dtype": torch.float16,
            "cuda_available": True,
            "hip": None,
            "capability": (8, 0),
        }
        values.update(overrides)
        return values

    def test_disabled_contract_is_noop_and_default_is_false(self):
        _validate_int8_encode_options(
            enabled=False, device="cpu", dtype=torch.float32
        )
        self.assertIs(
            inspect.signature(H3VAEPyOptRuntime).parameters["int8_encode"].default,
            False,
        )

    def test_platform_contract_rejects_unsupported_requests(self):
        with self.assertRaisesRegex(RuntimeError, "CUDA device"):
            _validate_int8_encode_options(**self._supported(device="cpu"))
        with self.assertRaisesRegex(RuntimeError, "dtype=fp16"):
            _validate_int8_encode_options(**self._supported(dtype=torch.float32))
        with self.assertRaisesRegex(RuntimeError, "ROCm"):
            _validate_int8_encode_options(**self._supported(hip="6.1"))
        with self.assertRaisesRegex(RuntimeError, "available CUDA GPU"):
            _validate_int8_encode_options(
                **self._supported(cuda_available=False)
            )
        with self.assertRaisesRegex(RuntimeError, "SM80"):
            _validate_int8_encode_options(**self._supported(capability=(7, 5)))

    def test_install_guard_rejects_cpu_and_mixed_devices(self):
        with self.assertRaisesRegex(RuntimeError, "all encoder prefix parameters"):
            _check_encoder_platform(_DeviceOnlyPrefix(["cpu"]))
        with self.assertRaisesRegex(RuntimeError, "one CUDA device"):
            _check_encoder_platform(_DeviceOnlyPrefix(["cuda:0", "cuda:1"]))

    def test_install_guard_queries_prefix_cuda_device(self):
        prefix = _DeviceOnlyPrefix(["cuda:1"])
        with (
            patch.object(torch.cuda, "is_available", return_value=True),
            patch.object(
                torch.cuda, "get_device_capability", return_value=(8, 0)
            ) as get_capability,
        ):
            _check_encoder_platform(prefix)
        get_capability.assert_called_once_with(torch.device("cuda:1"))

    def test_installation_is_exactly_eight_and_bias_policy_is_explicit(self):
        prefix = _toy_prefix()
        # The tiny CPU toy uses four channels, so select a valid tile
        # explicitly; production H3 installation uses the stage-aware 128/256
        # input-channel policy.
        metadata = install_int8_encoder_prefix(
            prefix, require_cuda=False, tile_variant="128x64x64"
        )
        self.assertEqual(metadata["count"], 8)
        self.assertEqual(metadata["tile_variant"], "stage-aware-input-channels")
        self.assertEqual(INT8_ENCODER_POLICY["quantized_conv_count"], 8)
        wrappers = []
        for stage in prefix.stages[:2]:
            for block in stage.block:
                wrappers.extend([block.int8_first, block.int8_second])
        self.assertTrue(all(isinstance(item, Int8FusedValidConv3d) for item in wrappers))
        self.assertEqual(sum(item.apply_bias for item in wrappers), 3)
        self.assertFalse(prefix.stages[1].block[1].int8_second.apply_bias)
        self.assertTrue(all(item.qweight.dtype == torch.int8 for item in wrappers))
        self.assertTrue(all(not item.qweight.requires_grad for item in wrappers))
        self.assertTrue(
            all(item.weight_scale.dtype == torch.float32 for item in wrappers)
        )
        state_keys = prefix.state_dict()
        self.assertTrue(any(key.endswith("qweight") for key in state_keys))
        self.assertTrue(any(key.endswith("weight_scale") for key in state_keys))

    def test_stage_aware_tile_policy(self):
        self.assertEqual(_resolve_tile_variant(128, "auto"), "128x64x64")
        self.assertEqual(_resolve_tile_variant(256, None), "128x64x128")
        with self.assertRaisesRegex(ValueError, "only supports input channels"):
            _resolve_tile_variant(4, "auto")

    def test_apply_preserves_int8_and_exact_fp32_scale(self):
        source = _ToyConv()
        wrapper = Int8FusedValidConv3d(source, apply_bias=True)
        expected_scale = wrapper.weight_scale.detach().clone()
        wrapper.to(dtype=torch.float16)
        self.assertEqual(wrapper.qweight.dtype, torch.int8)
        self.assertEqual(wrapper.weight_scale.dtype, torch.float32)
        self.assertTrue(torch.equal(wrapper.weight_scale, expected_scale))
        self.assertEqual(wrapper.bias.dtype, torch.float16)

    def test_prefix_offload_apply_preserves_persistent_quant_state(self):
        prefix = _toy_prefix()
        install_int8_encoder_prefix(
            prefix, require_cuda=False, tile_variant="128x64x64"
        )
        wrappers = [
            block.int8_first
            for stage in prefix.stages[:2]
            for block in stage.block
        ]
        expected_scales = [item.weight_scale.detach().clone() for item in wrappers]
        prefix.to(dtype=torch.float32)
        prefix.to(dtype=torch.float16)
        for wrapper, expected in zip(wrappers, expected_scales):
            self.assertEqual(wrapper.qweight.dtype, torch.int8)
            self.assertEqual(wrapper.weight_scale.dtype, torch.float32)
            self.assertTrue(torch.equal(wrapper.weight_scale, expected))
            self.assertIn("weight_scale", wrapper.state_dict())

    def test_fake_shape_matches_valid_kernel_contract(self):
        x = torch.empty((2, 4, 7, 10, 12), device="meta", dtype=torch.float16)
        qweight = torch.empty((6, 4 * 27), device="meta", dtype=torch.int8)
        scale = torch.empty((6,), device="meta", dtype=torch.float32)
        bias = torch.empty((6,), device="meta", dtype=torch.float16)
        output = _int8_valid_conv3d_integration_fake(
            x, qweight, scale, bias, 3, 3, 3, 1, 1, 1, True, "128x64x64"
        )
        self.assertEqual(tuple(output.shape), (2, 6, 5, 8, 10))
        self.assertEqual(output.dtype, torch.float16)
        self.assertTrue(output.is_contiguous(memory_format=torch.channels_last_3d))

    def test_loader_exposes_independent_option(self):
        source = (ROOT / "h3vae_nodes.py").read_text(encoding="utf-8")
        self.assertIn('"int8_encode"', source)
        self.assertIn("bool(int8_encode)", source)
        self.assertIn("int8_decode=bool(int8_decode)", source)


if __name__ == "__main__":
    unittest.main()
