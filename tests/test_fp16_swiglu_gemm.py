import unittest
import torch
from opt.fp16_swiglu_gemm import validate, fused_w1_swiglu, FusedFP16FFN, clone_fp16_decoder


class Contracts(unittest.TestCase):
    def model(self):
        ff=torch.nn.Module();ff.use_gated=True;ff.act_fn=torch.nn.SiLU()
        ff.w1=torch.nn.Linear(4,16,dtype=torch.float16)
        ff.w2=torch.nn.Linear(8,4,dtype=torch.float16)
        block=torch.nn.Module();block.ff=ff;block.attn=torch.nn.Identity()
        raw=torch.nn.Module();raw.transformer_blocks=torch.nn.ModuleList([block]);raw.eval()
        return raw

    def test_clone_preserves_original_and_state(self):
        raw=self.model();candidate=clone_fp16_decoder(raw,(128,64,64,4,2))
        self.assertNotIsInstance(raw.transformer_blocks[0].ff,FusedFP16FFN)
        self.assertIsInstance(candidate.transformer_blocks[0].ff,FusedFP16FFN)
        self.assertIsNot(candidate.transformer_blocks[0].ff.w1,raw.transformer_blocks[0].ff.w1)
        self.assertIs(candidate.transformer_blocks[0].ff.w1.weight,raw.transformer_blocks[0].ff.w1.weight)
        self.assertEqual(set(candidate.state_dict()),set(raw.state_dict()))

    def test_module_guards(self):
        raw=self.model();raw.train()
        with self.assertRaises(ValueError):clone_fp16_decoder(raw,(128,64,64,4,2))
        raw.eval();raw.transformer_blocks[0].ff.act_fn=torch.nn.GELU()
        with self.assertRaises(ValueError):clone_fp16_decoder(raw,(128,64,64,4,2))

    def test_cpu_guard(self):
        x=torch.ones(2,4,dtype=torch.float16);w=torch.ones(8,4,dtype=x.dtype)
        validate(x,w,None,(64,64,64,4,3))
        with self.assertRaisesRegex(ValueError,'CUDA'): fused_w1_swiglu(x,w)

    def test_shape_dtype_config_guards(self):
        x=torch.ones(2,4,dtype=torch.float16);w=torch.ones(8,4,dtype=x.dtype)
        for a,b,bias,c in [(x.float(),w,None,(64,64,64,4,3)),
                (x,w[:7],None,(64,64,64,4,3)),(x,w,torch.zeros(7,dtype=x.dtype),(64,64,64,4,3)),
                (x,w,None,(64,64,64,1,3)),(x,w,None,(64,64,64,4)),
                (x[:0],w,None,(64,64,64,4,3))]:
            with self.assertRaises(ValueError): validate(a,b,bias,c)


if __name__=='__main__':unittest.main()
