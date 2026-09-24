"""CPU contracts; actual QKV speed/quality require GPU measurements."""
import copy
import unittest
from unittest.mock import patch
import torch
from opt.int8_qkv import clone_qkv_int8_decoder, qkv_linear, selected_indices


class Contracts(unittest.TestCase):
    def test_indices(self):
        self.assertEqual(selected_indices(36), tuple(range(36)))
        self.assertEqual(selected_indices(36, [3, 7]), (3, 7))
        for bad in ([], [1, 1], [-1], [36], [True], [1.0]):
            with self.assertRaises(ValueError): selected_indices(36, bad)

    def block(self, width=2048):
        block = torch.nn.Module()
        block.attn = torch.nn.Module()
        block.attn.original = torch.nn.Module()
        block.attn.original.to_qkv = torch.nn.Linear(width, width*3, device='meta')
        block.attn.original.to_out = torch.nn.Identity()
        block.eval()
        return block

    def test_validation(self):
        block = self.block()
        self.assertIs(qkv_linear(block)[1], block.attn.original.to_qkv)
        block.train()
        with self.assertRaises(ValueError): qkv_linear(block)
        with self.assertRaises(ValueError): qkv_linear(self.block(128))

    def test_replacement_scope_original_untouched(self):
        raw = torch.nn.Module()
        raw.transformer_blocks = torch.nn.ModuleList([self.block(), self.block()])
        raw.eval()
        clone = copy.deepcopy(raw)
        replacement = torch.nn.Identity()
        with patch('opt.int8_qkv.clone_norm_quant_decoder', return_value=clone), \
                patch('opt.int8_qkv.KitchenInt8Linear', return_value=replacement) as factory:
            result = clone_qkv_int8_decoder(raw, indices=[1])
        self.assertIsInstance(raw.transformer_blocks[1].attn.original.to_qkv, torch.nn.Linear)
        self.assertIsInstance(result.transformer_blocks[0].attn.original.to_qkv, torch.nn.Linear)
        self.assertIs(result.transformer_blocks[1].attn.original.to_qkv, replacement)
        self.assertIsInstance(result.transformer_blocks[1].attn.original.to_out, torch.nn.Identity)
        self.assertEqual(factory.call_count, 1)
        self.assertTrue(factory.call_args.kwargs['require_cuda'])


if __name__ == '__main__': unittest.main()
