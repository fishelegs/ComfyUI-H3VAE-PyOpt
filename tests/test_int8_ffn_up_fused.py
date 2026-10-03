"""CPU guards and fake-output contracts; numerical claims require CUDA evidence."""
import unittest
from unittest.mock import patch
import torch
from opt.int8_ffn_up_fused import (
    fused_residual_ffn, quantize_ffn_up, supports_fused_ffn_up, validate_prequantized,
)
from opt.int8_norm_quant import NormQuantBlock


class FusedFFNContracts(unittest.TestCase):
    def inputs(self, rows=3):
        return (torch.empty(rows,2048,dtype=torch.int8,device='meta'),
                torch.empty(rows,1,dtype=torch.float32,device='meta'),
                torch.empty(16384,2048,dtype=torch.int8,device='meta'),
                torch.empty(16384,dtype=torch.float32,device='meta'),
                torch.empty(16384,dtype=torch.float16,device='meta'))

    def test_shape_dtype_layout_scale_guards(self):
        values = self.inputs()
        validate_prequantized(*values)
        validate_prequantized(*values[:-1], None)
        cases = [(0,torch.empty(3,2048,dtype=torch.float16,device='meta')),
                 (0,torch.empty(3,2047,dtype=torch.int8,device='meta')),
                 (0,torch.empty(0,2048,dtype=torch.int8,device='meta')),
                 (1,torch.empty(3,1,dtype=torch.float16,device='meta')),
                 (1,torch.empty(3,dtype=torch.float32,device='meta')),
                 (2,torch.empty(16383,2048,dtype=torch.int8,device='meta')),
                 (3,torch.empty(16384,dtype=torch.float16,device='meta')),
                 (4,torch.empty(16384,dtype=torch.float32,device='meta')),
                 (0,torch.empty(2048,3,dtype=torch.int8,device='meta').t())]
        for index, bad in cases:
            with self.subTest(index=index,shape=bad.shape,dtype=bad.dtype):
                changed=list(values); changed[index]=bad
                with self.assertRaises(ValueError): validate_prequantized(*changed)

    def test_output_indexing_boundary_without_allocating_storage(self):
        validate_prequantized(*self.inputs(rows=262144))
        with self.assertRaisesRegex(ValueError, '32-bit output indexing'):
            validate_prequantized(*self.inputs(rows=262145))

    def test_architecture_scope_and_cpu_rejection(self):
        with patch('torch.cuda.is_available',return_value=True), patch('torch.version.hip',None), \
                patch('torch.cuda.get_device_capability',return_value=(12,0)) as capability:
            self.assertFalse(supports_fused_ffn_up('cpu'))
            capability.assert_not_called()
            self.assertTrue(supports_fused_ffn_up('cuda:0'))
            for arch in ((8,0),(9,0),(12,1)):
                capability.return_value=arch
                self.assertFalse(supports_fused_ffn_up('cuda:0'))
        with self.assertRaisesRegex(ValueError,'SM120'):
            quantize_ffn_up(*self.inputs())

    def test_fake_outputs_have_honest_shape_dtype_and_no_input_alias(self):
        x=torch.empty(2,3,2048,dtype=torch.float16,device='meta')
        g=torch.empty(2048,dtype=torch.float16,device='meta')
        _,_,w,ws,b=self.inputs()
        down=torch.empty(2048,8192,dtype=torch.int8,device='meta')
        ds=torch.empty(2048,dtype=torch.float32,device='meta')
        h,out=fused_residual_ffn(x,torch.empty_like(x),g,g,w,ws,b,down,ds,g,1e-5,8)
        self.assertEqual(h.shape,x.shape); self.assertEqual(out.shape,x.shape)
        self.assertEqual(h.dtype,torch.float16); self.assertEqual(out.dtype,torch.float16)
        self.assertIsNot(h,x); self.assertIsNot(out,x); self.assertIsNot(h,out)

    def test_selection_works_for_cpu_test_double_and_refreshes_on_apply(self):
        block=torch.nn.Module()
        block.use_scale=True
        block.norm1=torch.nn.RMSNorm(2048).half()
        block.norm2=torch.nn.RMSNorm(2048,eps=1e-5).half()
        block.attn=torch.nn.Identity()
        block.scale1=torch.nn.Parameter(torch.ones(2048,dtype=torch.float16))
        block.scale2=torch.nn.Parameter(torch.ones(2048,dtype=torch.float16))
        block.ff=torch.nn.Module()
        block.ff.w1=torch.nn.Linear(4,8)
        block.ff.w2=torch.nn.Linear(4,4)
        block.eval()
        result=NormQuantBlock(block)
        self.assertFalse(result.fuse_ffn_up)
        with patch('opt.int8_norm_quant.supports_fused_ffn_up',return_value=True):
            result.to(dtype=torch.float16)
        self.assertTrue(result.fuse_ffn_up)
        result.to('cpu')
        self.assertFalse(result.fuse_ffn_up)


if __name__=='__main__': unittest.main()
