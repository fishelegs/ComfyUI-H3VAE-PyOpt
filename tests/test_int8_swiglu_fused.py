"""CPU contracts only; not substitutes for actual CUDA/numerical validation."""
import unittest
from unittest.mock import patch
import torch
from opt.int8_swiglu_fused import validate_input, FusedFFN, clone_fused_decoder


class Contracts(unittest.TestCase):
    def test_valid_shape(self):
        validate_input(torch.empty(2,3,128,dtype=torch.float16))

    def test_bad_shapes_dtypes_layout(self):
        for x in (torch.empty(2,127,dtype=torch.float16),torch.empty(128,dtype=torch.float16),
                  torch.empty(2,128),torch.empty(0,128,dtype=torch.float16),
                  torch.empty(2,128,dtype=torch.float16).t()):
            with self.assertRaises(ValueError):validate_input(x)

    def test_requires_quantized_ffn(self):
        with self.assertRaises(ValueError):FusedFFN(torch.nn.Linear(4,4))

    def test_clone_isolates_containers_and_shares_weights(self):
        raw=torch.nn.Module()
        block=torch.nn.Module()
        ff=torch.nn.Module()
        ff.mode='INT8both'
        ff.w1=torch.nn.Linear(4,8)
        ff.w2=torch.nn.Linear(4,4)
        block.ff=ff
        block.register_module('absent',None)
        raw.transformer_blocks=torch.nn.ModuleList([block])
        raw.eval()
        with patch('opt.int8_swiglu_fused.require_backend'):
            clone=clone_fused_decoder(raw)
        self.assertIs(raw.transformer_blocks[0].ff,ff)
        self.assertIsNot(clone.transformer_blocks[0],block)
        self.assertIs(clone.transformer_blocks[0].ff.w1.weight,ff.w1.weight)
        self.assertIsNone(clone.transformer_blocks[0].absent)
        with self.assertRaises(RuntimeError):clone.transformer_blocks[0].ff(torch.zeros(2,4))


if __name__=='__main__':unittest.main()
