"""CPU contracts only; actual quality and speed require GPU measurements."""
import unittest
from unittest.mock import patch
import torch
from opt.int8_attention_out import clone_containers, out_linear, clone_attention_out_decoder


class Contracts(unittest.TestCase):
    def block(self, width=2048):
        block = torch.nn.Module()
        block.attn = torch.nn.Module()
        block.attn.original = torch.nn.Module()
        block.attn.original.to_qkv = torch.nn.Linear(width, width*3, device='meta')
        block.attn.original.to_out = torch.nn.Linear(width, width, device='meta')
        block.eval()
        return block

    def test_validation(self):
        block = self.block()
        self.assertIs(out_linear(block)[1], block.attn.original.to_out)
        block.train()
        with self.assertRaises(ValueError): out_linear(block)
        with self.assertRaises(ValueError): out_linear(self.block(128))
        block = self.block()
        block.attn.original.to_out = torch.nn.Identity()
        with self.assertRaises(ValueError): out_linear(block)

    def test_container_alias_and_tensor_sharing(self):
        raw = self.block()
        raw.alias = raw.attn
        raw.register_buffer('probe', torch.ones(2))
        clone = clone_containers(raw)
        self.assertIsNot(clone.attn, raw.attn)
        self.assertIs(clone.alias, clone.attn)
        self.assertIs(clone.probe, raw.probe)
        self.assertIs(clone.attn.original.to_out.weight, raw.attn.original.to_out.weight)
        clone.probe = torch.zeros(2)
        self.assertTrue(torch.equal(raw.probe, torch.ones(2)))

    def test_replacement_scope_original_untouched(self):
        raw = torch.nn.Module()
        raw.transformer_blocks = torch.nn.ModuleList([self.block(), self.block()])
        raw.eval()
        replacement = torch.nn.Identity()
        with patch('opt.int8_attention_out.KitchenInt8Linear', return_value=replacement) as factory:
            result = clone_attention_out_decoder(raw, indices=[1])
        for i in range(2):
            self.assertIsInstance(raw.transformer_blocks[i].attn.original.to_out, torch.nn.Linear)
            self.assertIs(result.transformer_blocks[i].attn.original.to_qkv.weight,
                          raw.transformer_blocks[i].attn.original.to_qkv.weight)
        self.assertIsInstance(result.transformer_blocks[0].attn.original.to_out, torch.nn.Linear)
        self.assertIs(result.transformer_blocks[1].attn.original.to_out, replacement)
        self.assertEqual(factory.call_count, 1)
        self.assertTrue(factory.call_args.kwargs['require_cuda'])


if __name__ == '__main__': unittest.main()
