"""CPU policy and workflow contracts; GPU evidence is documented separately."""
import inspect
import json
import os
from pathlib import Path
import unittest
from unittest.mock import patch

import torch

from h3vae_runtime import H3VAEPyOptRuntime, _validate_decode_fusions_options


ROOT = Path(__file__).resolve().parents[1]


class FusionOptions(unittest.TestCase):
    def options(self, **changes):
        args=dict(enabled=True, device='cuda', dtype=torch.float16, fast_linear=False,
                  compile_decoder=True, int8_encode=False, cuda_available=True,
                  capability=(8,0), hip=None)
        args.update(changes)
        return args

    def test_default_off_and_supported(self):
        self.assertIs(inspect.signature(H3VAEPyOptRuntime).parameters['decode_fusions'].default,False)
        with patch.dict(os.environ, {'MINIMAX_H3_VAE_DECODER_VIT_FP32_NORM':'1'}):
            _validate_decode_fusions_options(**self.options())
        _validate_decode_fusions_options(**self.options(enabled=False,device='cpu',dtype=torch.float32))

    def test_reject_unsupported_without_model_load(self):
        for values in ({'fast_linear':True},{'int8_encode':True},{'compile_decoder':False},
                       {'device':'cpu'},{'dtype':torch.float32},{'dtype':torch.bfloat16},
                       {'cuda_available':False},{'hip':'6.0'},{'capability':(7,5)}):
            with self.subTest(values=values), self.assertRaises((ValueError,RuntimeError)):
                _validate_decode_fusions_options(**self.options(**values))
        with patch.dict(os.environ, {'MINIMAX_H3_VAE_DECODER_VIT_FP32_NORM':'0'}):
            with self.assertRaisesRegex(ValueError,'FP32'):
                _validate_decode_fusions_options(**self.options())
        with patch('h3vae_runtime._load_klvae_core') as load:
            with self.assertRaisesRegex(ValueError,'mutually exclusive'):
                H3VAEPyOptRuntime(decode_fusions=True,fast_linear=True)
            load.assert_not_called()

    def test_loader_cache_and_preset_workflows(self):
        source=(ROOT/'h3vae_nodes.py').read_text()
        self.assertIn('bool(int8_encode), bool(decode_fusions)',source)
        self.assertIn('decode_fusions=bool(decode_fusions)',source)
        for mode,batch in (('fp16',8),('int8',4)):
            document=json.loads((ROOT/f'examples/minimal_h3vae_pyopt_{mode}_fused_prompt.json').read_text())
            loader=document['prompt']['1']['inputs']
            self.assertTrue(loader['decode_fusions'])
            self.assertEqual(loader['int8_decode'],mode=='int8')
            self.assertEqual(loader['tile_batch'],batch)
            self.assertEqual(loader['dtype'],'fp16')
            self.assertFalse(loader['fast_linear'])
            self.assertFalse(loader['int8_encode'])
            self.assertTrue(loader['compile_decoder'])


if __name__=='__main__':unittest.main()
