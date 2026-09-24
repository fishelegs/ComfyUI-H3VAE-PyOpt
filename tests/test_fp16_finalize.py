import unittest
from unittest.mock import patch
import torch
from opt.fp16_finalize import validate,finalize_pixels
from h3vae_runtime import H3VAEPyOptRuntime


class Core(torch.nn.Module):
    def decode(self,z):return z[:,:3]
    def decode_base(self,z,frame_num):return self.decode(z)


class Contracts(unittest.TestCase):
    def runtime(self):
        r=H3VAEPyOptRuntime.__new__(H3VAEPyOptRuntime);torch.nn.Module.__init__(r)
        r.core=Core();r.latents_mean=torch.zeros((1,24,1,1,1),dtype=torch.float16)
        r.latents_std=torch.ones_like(r.latents_mean);r.int8_decode=False;r.log_calls=False
        r._img_mean=torch.tensor([.485,.456,.406]).reshape(1,3,1,1,1)
        r._img_std=torch.tensor([.229,.224,.225]).reshape(1,3,1,1,1)
        return r

    def test_guards(self):
        x=torch.zeros((1,3,2,4,4),dtype=torch.float16);v=torch.ones(3)
        validate(x,v,v)
        with self.assertRaisesRegex(ValueError,'CUDA'):finalize_pixels(x,v,v)
        for a,b,c,o in [(x[:,:2],v,v,None),(x.double(),v,v,None),
                       (x,v.half(),v,None),(x,v,v,torch.empty_like(x))]:
            with self.assertRaises(ValueError):validate(a,b,c,o)

    def test_decode_padding_and_output_buffer_contract(self):
        def eager(x,std,mean,out=None):
            y=(x.float()*std+mean).clamp_(0,1)
            if out is not None:out.copy_(y);return out
            return y
        r=self.runtime()
        with torch.inference_mode(),patch('opt.fp16_finalize.finalize_pixels',side_effect=eager):
            for int8,t in ((i,t) for i in (False,True) for t in (1,2,7)):
                r.int8_decode=int8
                z=torch.randn((1,24,t,2,4),dtype=torch.float16)
                r.decode_fusions=False;reference=r.decode(z)
                r.decode_fusions=True;actual=r.decode(z)
                self.assertTrue(torch.equal(actual,reference))
                for out in (torch.empty_like(reference),torch.empty_like(reference).half(),
                            torch.empty((*reference.shape[:-2],4,2)).transpose(-1,-2)):
                    self.assertIs(r.decode(z,output_buffer=out),out)
                    self.assertTrue(torch.equal(out,reference.to(out.dtype)))


if __name__=='__main__':unittest.main()
