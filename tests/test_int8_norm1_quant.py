"""CPU guards and move/shape contracts for the SM120 norm1 integration."""
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import Mock, patch

import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from opt import decoder_final_norm as final
from opt import int8_norm1_quant as norm1


class Norm1ContractTest(unittest.TestCase):
    def test_import_and_cpu_rejection_keep_cuda_and_triton_lazy(self):
        script = '''import sys, torch
from opt import int8_norm1_quant as n
assert 'triton' not in sys.modules
h = torch.empty((4,1797,2048),dtype=torch.float16)
v = torch.empty(2048,dtype=torch.float16)
try:
    n._FINAL_HELPER.final_residual_ln_w8_impl(h,h,v,v,v)
except ValueError as e:
    assert 'NVIDIA CUDA' in str(e)
else:
    raise AssertionError('CPU dispatch accepted')
assert 'triton' not in sys.modules and not torch.cuda.is_initialized()
'''
        subprocess.run([sys.executable, '-c', script], cwd=Path(__file__).resolve().parents[1], check=True, capture_output=True)

    def test_final_adapter_preserves_parameters_and_tensor_fallback_after_move(self):
        source = torch.nn.LayerNorm(2048, eps=1e-5, dtype=torch.float16).eval()
        adapter = norm1._FinalNormAdapter(source).eval()
        self.assertEqual(set(adapter.state_dict()), set(source.state_dict()))
        self.assertIs(adapter.weight, source.weight)
        self.assertIs(adapter.bias, source.bias)
        with torch.inference_mode():
            x = torch.randn(2, 7, 2048, dtype=torch.float16)
            self.assertTrue(torch.equal(adapter(x), source(x)))
            adapter.to(dtype=torch.float32)
            self.assertIs(adapter.weight, source.weight)
            self.assertTrue(torch.equal(adapter(x.float()), source(x.float())))

    def test_final_fake_preserves_shape_and_owned_output(self):
        with FakeTensorMode():
            h = torch.empty(final.SHAPE, dtype=torch.float16)
            v = torch.empty(final.WIDTH, dtype=torch.float16)
            y = final.final_residual_ln_w8(h, h, v, v, v)
            self.assertEqual(y.shape, h.shape)
            self.assertEqual(y.dtype, h.dtype)
            self.assertTrue(y.is_contiguous())
            self.assertIsNot(y, h)

    def test_fallback_uses_current_parameters_after_overwrite_conversion(self):
        source = torch.nn.LayerNorm(2048, eps=1e-5, dtype=torch.float16).eval()
        adapter = norm1._FinalNormAdapter(source).eval()
        old_flag = torch.__future__.get_overwrite_module_params_on_conversion()
        try:
            torch.__future__.set_overwrite_module_params_on_conversion(True)
            adapter.to(dtype=torch.float32)
        finally:
            torch.__future__.set_overwrite_module_params_on_conversion(old_flag)
        self.assertIsNot(adapter.weight, source.weight)
        with torch.inference_mode():
            y = adapter(torch.randn(2, 7, 2048))
        self.assertEqual(y.dtype, torch.float32)
        self.assertTrue(torch.isfinite(y).all())

    def test_final_metadata_rejects_wrong_shape_dtype_layout(self):
        h = torch.empty(final.SHAPE, dtype=torch.float16, device='meta')
        v = torch.empty(final.WIDTH, dtype=torch.float16, device='meta')
        for bad in (h[:1], h.float(), h.transpose(0, 1)):
            with self.subTest(shape=bad.shape, dtype=bad.dtype), self.assertRaises(ValueError):
                final.validate_metadata(bad, h, v, v, v, require_cuda=False)
        with self.assertRaises(ValueError):
            final.validate_metadata(h, h, v[:1], v, v, require_cuda=False)

    def test_nonzero_padding_carry_fails_before_original_unpad(self):
        original = Mock(side_effect=lambda states, pad: states)
        unpad = norm1._carry_safe_unpad(original)
        carry = (object(), object(), object())
        self.assertIs(unpad(carry, 0), carry)
        original.reset_mock()
        with self.assertRaisesRegex(RuntimeError, 'zero sequence padding'):
            unpad(carry, 1)
        original.assert_not_called()
        tensor = torch.empty(2, 5, 3)
        self.assertIs(unpad(tensor, 1), tensor)
        original.assert_called_once_with(tensor, 1)

    def test_unpad_must_preserve_carry_identity(self):
        unpad = norm1._carry_safe_unpad(lambda s, p: tuple(list(s)))
        with self.assertRaisesRegex(RuntimeError, 'preserve final residual carry'):
            unpad((object(), object(), object()), 0)

    def test_hardware_and_triton_guard(self):
        self.assertFalse(norm1.supports_norm1_quant('cpu'))
        for version, capability, expected in [('3.6.0', (12, 0), True), ('3.6.0', (8, 0), False), ('3.5.0', (12, 0), False)]:
            norm1._device_supports_norm1_quant.cache_clear()
            with patch.object(norm1.importlib.metadata, 'version', return_value=version), patch.object(torch, '__version__', '2.11.0+cu130'), patch.object(torch.version, 'cuda', '13.0'), patch.object(torch.cuda, 'is_available', return_value=True), patch.object(torch.cuda, 'get_device_capability', return_value=capability):
                self.assertEqual(norm1.supports_norm1_quant('cuda:0'), expected)
        norm1._device_supports_norm1_quant.cache_clear()


if __name__ == '__main__':
    unittest.main()
