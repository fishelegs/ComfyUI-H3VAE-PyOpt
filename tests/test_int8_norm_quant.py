"""CPU contracts; numerical and speed evidence require CUDA experiments."""
import unittest
from unittest.mock import patch
import torch
from opt.int8_norm_quant import validate, residual_quant, clone_norm_quant_decoder


class Contracts(unittest.TestCase):
    def inputs(self):
        x=torch.empty(2,3,2048,dtype=torch.float16)
        return x,torch.empty_like(x),torch.ones(2048,dtype=torch.float16),torch.ones(2048,dtype=torch.float16)

    def test_valid_shapes(self):
        validate(*self.inputs(),1e-5,8)

    def test_reject_layout_shape_dtype(self):
        x,a,g,w=self.inputs()
        for bad in (x.float(),x.transpose(0,1),x[...,:1024],x[:0]):
            with self.assertRaises(ValueError):validate(bad,a,g,w,1e-5,8)
        with self.assertRaises(ValueError):validate(x,a,g,w[:1],1e-5,8)

    def test_eps_and_warps(self):
        for eps,warps in ((0.,8),(float('nan'),8),(1e-5,3)):
            with self.assertRaises(ValueError):validate(*self.inputs(),eps,warps)

    def test_cpu_rejected(self):
        with self.assertRaises(ValueError):residual_quant(*self.inputs(),1e-5)

    def test_clone_requires_fp32_norm(self):
        with patch.dict('os.environ',{'MINIMAX_H3_VAE_DECODER_VIT_FP32_NORM':'0'}):
            with self.assertRaises(ValueError):clone_norm_quant_decoder(torch.nn.Module().eval())

    def test_clone_rejects_warps_before_backend_import(self):
        with self.assertRaises(ValueError):clone_norm_quant_decoder(torch.nn.Module().eval(),3)

    def test_clone_keeps_original_block(self):
        raw=torch.nn.Module()
        block=torch.nn.Module()
        block.use_scale=True
        block.norm1=torch.nn.RMSNorm(2048).half()
        block.norm2=torch.nn.RMSNorm(2048,eps=1e-5).half()
        block.attn=torch.nn.Identity()
        block.scale1=torch.nn.Parameter(torch.ones(2048,dtype=torch.float16))
        block.scale2=torch.nn.Parameter(torch.ones(2048,dtype=torch.float16))
        ff=torch.nn.Module()
        ff.mode='INT8both'
        ff.w1=torch.nn.Linear(4,8)
        ff.w2=torch.nn.Linear(4,4)
        block.ff=ff
        raw.transformer_blocks=torch.nn.ModuleList([block])
        raw.eval()
        with patch('opt.int8_swiglu_fused.require_backend'):
            result=clone_norm_quant_decoder(raw)
        self.assertIs(raw.transformer_blocks[0],block)
        self.assertIsNot(result.transformer_blocks[0],block)
        self.assertIs(result.transformer_blocks[0].scale1,block.scale1)
        self.assertIs(result.transformer_blocks[0].ff.w1.weight,ff.w1.weight)
        with self.assertRaises(RuntimeError):result.transformer_blocks[0](torch.zeros(2,2048))


if __name__=='__main__':unittest.main()
