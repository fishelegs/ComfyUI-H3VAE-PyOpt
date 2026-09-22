"""CPU policy checks for the explicit production INT8 decoder option."""
from __future__ import annotations

import inspect
import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from h3vae_runtime import (  # noqa: E402
    H3VAEPyOptRuntime,
    _validate_int8_decode_options,
)


class RuntimeInt8OptionsCpuTest(unittest.TestCase):
    @staticmethod
    def _supported(**overrides):
        values = {
            "enabled": True,
            "fast_linear": False,
            "device": "cuda",
            "dtype": torch.float16,
            "cuda_available": True,
            "hip": None,
            "capability": (8, 0),
            "ck_version": "0.2.34",
            "has_quantize_api": True,
            "has_linear_api": True,
        }
        values.update(overrides)
        return values

    def test_disabled_option_is_noop(self):
        _validate_int8_decode_options(
            enabled=False,
            fast_linear=True,
            device="cpu",
            dtype=torch.float32,
        )

    def test_supported_contract_passes_without_cuda(self):
        _validate_int8_decode_options(**self._supported())

    def test_fast_linear_is_mutually_exclusive(self):
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            _validate_int8_decode_options(**self._supported(fast_linear=True))

    def test_non_cuda_dtype_and_rocm_are_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "CUDA device"):
            _validate_int8_decode_options(**self._supported(device="cpu"))
        with self.assertRaisesRegex(RuntimeError, "dtype=fp16"):
            _validate_int8_decode_options(**self._supported(dtype=torch.float32))
        with self.assertRaisesRegex(RuntimeError, "does not support ROCm"):
            _validate_int8_decode_options(**self._supported(hip="6.1"))

    def test_gpu_and_dependency_requirements_are_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "available CUDA GPU"):
            _validate_int8_decode_options(
                **self._supported(cuda_available=False)
            )
        with self.assertRaisesRegex(RuntimeError, "SM80"):
            _validate_int8_decode_options(
                **self._supported(capability=(7, 5))
            )
        with self.assertRaisesRegex(RuntimeError, "0.2.34"):
            _validate_int8_decode_options(
                **self._supported(ck_version="0.2.33")
            )
        with self.assertRaisesRegex(RuntimeError, "quantize_int8_rowwise"):
            _validate_int8_decode_options(
                **self._supported(has_quantize_api=False)
            )

    def test_runtime_argument_defaults_off(self):
        parameter = inspect.signature(H3VAEPyOptRuntime).parameters["int8_decode"]
        self.assertIs(parameter.default, False)

    def test_loader_exposes_explicit_option_and_cache_token(self):
        source = (ROOT / "h3vae_nodes.py").read_text(encoding="utf-8")
        self.assertIn('"int8_decode"', source)
        self.assertIn("bool(int8_decode)", source)


if __name__ == "__main__":
    unittest.main()
