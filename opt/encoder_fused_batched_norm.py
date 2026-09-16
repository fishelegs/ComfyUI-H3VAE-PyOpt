"""Batch-aware temporal GN/bias/SiLU/padding for the encoder suffix.

Reuse the validated per-frame moment kernels, but pad each batch item
independently. Dense temporal/batch storage is checked before flattening stats.
"""
from typing import Optional
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from encoder_fused_temporal_norm import _partial, _merge, FusedNormPadConv
from encoder_fused_norm_bias import _partial_bias
from encoder_staged_batch import EncoderSuffix

@triton.jit
def _normalize_pack_batched(X, PreBias, Weight, Bias, S, Y,
                            B:tl.constexpr,C:tl.constexpr,D:tl.constexpr,H:tl.constexpr,W:tl.constexpr,
                            SD:tl.constexpr,SC:tl.constexpr,SH:tl.constexpr,SW:tl.constexpr,
                            HAS_BIAS:tl.constexpr,BLOCK:tl.constexpr):
    idx=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    total:tl.constexpr=B*C*(D+2)*(H+2)*(W+2)
    c=idx%C;q=idx//C
    n=q//((D+2)*(H+2)*(W+2))
    t=q//((H+2)*(W+2))%(D+2)-2
    h=q//(W+2)%(H+2)-1;w=q%(W+2)-1
    h=tl.where(h<0,-h,tl.where(h>=H,2*H-2-h,h))
    w=tl.where(w<0,-w,tl.where(w>=W,2*W-2-w,w))
    valid=(idx<total)&(t>=0)&(t<D)
    frame=n*D+t
    v=tl.load(X+frame*SD+c*SC+h*SH+w*SW,valid,other=0).to(tl.float32)
    if HAS_BIAS:
        pb=tl.load(PreBias+c).to(tl.float32)
        v=(v+pb).to(tl.float16).to(tl.float32)
    stat=(frame*32+c//(C//32))*2
    mean=tl.load(S+stat,valid,other=0)
    rstd=tl.load(S+stat+1,valid,other=0)
    gamma=tl.load(Weight+c).to(tl.float32);beta=tl.load(Bias+c).to(tl.float32)
    z=(((v-mean)*rstd)*gamma+beta).to(tl.float16).to(tl.float32)
    z=(z/(1.+tl.exp(-z))).to(tl.float16)
    tl.store(Y+idx,tl.where(valid,z,0.),idx<total)

@torch.library.custom_op('h3vae_encoder::batched_temporal_norm_pack',mutates_args=(),device_types='cuda')
def batched_temporal_norm_pack(x:torch.Tensor,pre_bias:Optional[torch.Tensor],
                               weight:torch.Tensor,bias:torch.Tensor,eps:float)->torch.Tensor:
    b,c,d,h,w=x.shape
    if b not in (1,3,4) or c not in (256,512,1024) or min(h,w)<2 or x.dtype!=torch.float16:
        raise ValueError('Supported: FP16 B=1/3/4 C=256/512/1024 H/W>=2')
    if b>1 and x.stride(0)!=d*x.stride(2):
        raise ValueError('Batch and time must be contiguous adjacent frame spans')
    if b*c*(d+2)*(h+2)*(w+2)>=2**31 or sum((size-1)*stride for size,stride in zip(x.shape,x.stride()))>=2**31:
        raise ValueError('Tensor exceeds validated int32 indexing range')
    for v in (weight,bias,pre_bias):
        if v is not None and (v.shape!=(c,) or v.dtype!=x.dtype or v.device!=x.device or not v.is_contiguous()):
            raise ValueError('Expected FP16 contiguous channel vectors on input device')
    bs,bc,warps=128,64,8
    nparts=triton.cdiv(h*w,bs)
    p=torch.empty((b*d,nparts,32),device=x.device,dtype=torch.float32)
    q=torch.empty_like(p)
    stats=torch.empty((b*d,32,2),device=x.device,dtype=torch.float32)
    y=torch.empty((b,c,d+2,h+2,w+2),device=x.device,dtype=x.dtype,memory_format=torch.channels_last_3d)
    grid=(nparts,triton.cdiv(c,bc),b*d)
    if pre_bias is None:
        _partial[grid](x,p,q,c,h,w,x.stride(2),x.stride(1),x.stride(3),x.stride(4),32,nparts,bs,bc,num_warps=warps)
    else:
        _partial_bias[grid](x,pre_bias,p,q,c,h,w,x.stride(2),x.stride(1),x.stride(3),x.stride(4),32,nparts,bs,bc,num_warps=warps)
    _merge[(32,b*d)](p,q,stats,h,w,c,32,nparts,bs,eps,triton.next_power_of_2(nparts),num_warps=4)
    _normalize_pack_batched[(triton.cdiv(y.numel(),1024),)](x,weight if pre_bias is None else pre_bias,
        weight,bias,stats,y,b,c,d,h,w,x.stride(2),x.stride(1),x.stride(3),x.stride(4),
        pre_bias is not None,1024,num_warps=4,enable_fp_fusion=False)
    return y

@batched_temporal_norm_pack.register_fake
def _fake(x,pre_bias,weight,bias,eps):
    b,c,d,h,w=x.shape
    return torch.empty((b,c,d+2,h+2,w+2),device=x.device,dtype=x.dtype,memory_format=torch.channels_last_3d)

class BatchedNormPadConv(FusedNormPadConv):
    def forward(self,x):
        y=batched_temporal_norm_pack(x,None,self.norm_weight,self.norm_bias,self.eps)
        return F.conv3d(y,self.weight,self.bias,stride=self.stride) if self.with_conv else y

class BatchedResidualBlock(torch.nn.Module):
    def __init__(self,block):
        super().__init__()
        if block.use_fused_norm or block.conv1.bias is None:raise ValueError('Expected original encoder block')
        self.first=BatchedNormPadConv(block.norm1,block.conv1,with_conv=False)
        self.second=BatchedNormPadConv(block.norm2,block.conv2,with_conv=False)
        self.shortcut=getattr(block,'nin_shortcut',None)
    def forward(self,x,zq=None):
        if zq is not None:raise ValueError('Unconditional inference only')
        h=F.conv3d(self.first(x),self.first.weight,None,stride=self.first.stride)
        h=batched_temporal_norm_pack(h,self.first.bias,self.second.norm_weight,self.second.norm_bias,self.second.eps)
        h=F.conv3d(h,self.second.weight,self.second.bias,stride=self.second.stride)
        return (self.shortcut(x) if self.shortcut is not None else x)+h

class BatchedSuffix(EncoderSuffix):
    def __init__(self,encoder,quant_conv,all_stages=False):
        super().__init__(encoder,quant_conv,2)
        for i in range(len(self.stages) if all_stages else 1):
            original=encoder.down[i+2]
            stage=torch.nn.Module()
            stage.block=torch.nn.ModuleList([BatchedResidualBlock(block) for block in original.block])
            if hasattr(original,'downsample'):stage.downsample=original.downsample
            self.stages[i]=stage
        self.final= BatchedNormPadConv(encoder.norm_out,encoder.conv_out) if all_stages else None
    def forward(self,h):
        for stage in self.stages:
            for block in stage.block:h=block(h)
            if hasattr(stage,'downsample'):h=stage.downsample(h)
        h=self.final(h) if self.final is not None else self.conv_out(F.silu(self.norm_out(h)))
        return self.quant_conv(h)

def fused_suffix(encoder,quant_conv,all_stages=False):
    return BatchedSuffix(encoder,quant_conv,all_stages)
