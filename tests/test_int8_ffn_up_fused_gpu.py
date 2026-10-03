"""H3VAE_TEST_CUDA=1 enables SM120/CK numerical checks; full video needs benchmark."""
import os
import unittest
import torch
from opt.int8_ffn_up_fused import (
    fused_residual_ffn, quantize_ffn_up, supports_fused_ffn_up,
)
from opt.int8_norm_quant import residual_rms_linear
from opt.int8_swiglu_fused import prequantized_linear, quantize_swiglu, swiglu_int8_linear


@unittest.skipUnless(os.environ.get('H3VAE_TEST_CUDA')=='1' and torch.cuda.is_available(),
                     'set H3VAE_TEST_CUDA=1 on CUDA SM120 with CK 0.2.34')
class FusedFFNGpuTests(unittest.TestCase):
    def setUp(self):
        if not supports_fused_ffn_up('cuda'):
            self.skipTest('numerical kernel is SM120-scoped')
        torch.manual_seed(20261002)

    @torch.inference_mode()
    def test_quantized_outputs_match_ck_with_row_tails_and_optional_bias(self):
        w=torch.randint(-31,32,(16384,2048),device='cuda',dtype=torch.int8)
        ws=torch.rand(16384,device='cuda',dtype=torch.float32)*0.02+0.001
        for rows,bias in ((1,None),(17,True),(129,True)):
            with self.subTest(rows=rows,bias=bias):
                q=torch.randint(-31,32,(rows,2048),device='cuda',dtype=torch.int8)
                scale=torch.rand(rows,1,device='cuda',dtype=torch.float32)*0.02+0.001
                b=torch.randn(16384,device='cuda',dtype=torch.float16) if bias else None
                hidden=prequantized_linear(q,scale,w,ws,b)
                rq,rs=quantize_swiglu(hidden,num_warps=16)
                cq,cs=quantize_ffn_up(q,scale,w,ws,b)
                self.assertTrue(torch.equal(cq,rq))
                self.assertTrue(torch.equal(cs,rs))

    @torch.inference_mode()
    def test_zero_input_and_finite_bias_quantization_contract(self):
        q=torch.zeros((3,2048),device='cuda',dtype=torch.int8)
        s=torch.ones((3,1),device='cuda',dtype=torch.float32)
        w=torch.zeros((16384,2048),device='cuda',dtype=torch.int8)
        ws=torch.ones(16384,device='cuda',dtype=torch.float32)
        for magnitude in (0.,2**-14,1.):
            b=torch.full((16384,),magnitude,device='cuda',dtype=torch.float16)
            reference=quantize_swiglu(prequantized_linear(q,s,w,ws,b),num_warps=16)
            actual=quantize_ffn_up(q,s,w,ws,b)
            self.assertTrue(torch.equal(actual[0],reference[0]))
            self.assertTrue(torch.equal(actual[1],reference[1]))

    @torch.inference_mode()
    def test_combined_op_matches_original_registered_ops(self):
        x=torch.randn(2,3,2048,device='cuda',dtype=torch.float16)
        a=torch.randn_like(x)
        g=torch.full((2048,),0.1,device='cuda',dtype=torch.float16)
        norm=torch.ones_like(g)
        up=torch.randint(-31,32,(16384,2048),device='cuda',dtype=torch.int8)
        us=torch.full((16384,),0.001,device='cuda',dtype=torch.float32)
        ub=torch.randn(16384,device='cuda',dtype=torch.float16)*0.01
        down=torch.randint(-31,32,(2048,8192),device='cuda',dtype=torch.int8)
        ds=torch.full((2048,),0.001,device='cuda',dtype=torch.float32)
        db=torch.randn(2048,device='cuda',dtype=torch.float16)*0.01
        rh,hidden=residual_rms_linear(x,a,g,norm,up,us,ub,1e-5,8)
        rout=swiglu_int8_linear(hidden,down,ds,db,16)
        ch,cout=fused_residual_ffn(x,a,g,norm,up,us,ub,down,ds,db,1e-5,8)
        self.assertTrue(torch.equal(ch,rh))
        self.assertTrue(torch.equal(cout,rout))


if __name__=='__main__': unittest.main()
