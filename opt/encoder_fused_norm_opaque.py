"""Read-only custom-op boundary for the three-kernel norm/activation pack.

The kernels write only newly allocated scratch/output buffers. Exposing that
contract prevents redundant input copies from conservative Triton capture.
"""
import torch
import torch.nn.functional as F
from encoder_fused_temporal_norm import fused_temporal_norm_pad,FusedNormPadConv

@torch.library.custom_op('h3vae_encoder::temporal_norm_pack',mutates_args=(),device_types='cuda')
def temporal_norm_pack(x:torch.Tensor,weight:torch.Tensor,bias:torch.Tensor,eps:float)->torch.Tensor:
    return fused_temporal_norm_pad(x,weight,bias,eps,(128,64,8))

@temporal_norm_pack.register_fake
def _fake(x,weight,bias,eps):
    b,c,d,h,w=x.shape
    return torch.empty((b,c,d+2,h+2,w+2),device=x.device,dtype=x.dtype,memory_format=torch.channels_last_3d)

class OpaqueFusedNormPadConv(FusedNormPadConv):
    def forward(self,x):
        y=temporal_norm_pack(x,self.norm_weight,self.norm_bias,self.eps)
        return F.conv3d(y,self.weight,self.bias,stride=self.stride) if self.with_conv else y
