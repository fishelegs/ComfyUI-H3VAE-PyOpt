"""Experimental FP16-input, FP32-accumulate GEMM with SwiGLU epilogue.

No quantization, no packed checkpoint rewrite, and no FP16 accumulation.
The first GEMM is rounded to FP16 before compiled-style FP32 SwiGLU arithmetic.
Different GEMM reduction order can still change rounding; test full outputs.
"""
from functools import lru_cache
import copy
import torch
from torch import nn


@lru_cache(maxsize=1)
def kernel():
    import triton
    import triton.language as tl
    from triton.language.extra.cuda import libdevice

    @triton.jit
    def run(X,W,B,O,M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,
            BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr,HAS_BIAS:tl.constexpr):
        pid=tl.program_id(0)
        nm,nn=tl.cdiv(M,BM),tl.cdiv(N,BN)
        group=pid//(8*nn)
        first=group*8
        size=tl.minimum(nm-first,8)
        pm=first+(pid%(8*nn))%size
        pn=(pid%(8*nn))//size
        rm=pm*BM+tl.arange(0,BM)
        pair=tl.arange(0,2*BN)
        rn=pn*BN+pair//2
        wn=rn+(pair%2)*N
        rk=tl.arange(0,BK)
        acc=tl.full((BM,2*BN),0,tl.float32)
        for step in range(tl.cdiv(K,BK)):
            kk=step*BK+rk
            x=tl.load(X+rm[:,None]*K+kk[None,:],(rm[:,None]<M)&(kk[None,:]<K),other=0)
            w=tl.load(W+wn[None,:]*K+kk[:,None],(rn[None,:]<N)&(kk[:,None]<K),other=0)
            acc=tl.dot(x,w,acc,out_dtype=tl.float32)
        if HAS_BIAS:
            acc=acc+tl.load(B+wn,rn<N,other=0)[None,:].to(tl.float32)
        rounded=acc.to(tl.float16).to(tl.float32).reshape(BM,BN,2)
        gate,up=tl.split(rounded)
        value=(gate/(1.0+libdevice.exp(-gate)))*up
        col=pn*BN+tl.arange(0,BN)
        tl.store(O+rm[:,None]*N+col[None,:],value,(rm[:,None]<M)&(col[None,:]<N))
    return run


def validate(x,w,bias,config):
    if x.dtype!=torch.float16 or w.dtype!=torch.float16:
        raise ValueError('FP16 activations and weights required')
    if x.ndim<2 or w.ndim!=2 or x.shape[-1]!=w.shape[1] or w.shape[0]%2:
        raise ValueError('expected matching K and paired gate/up output rows')
    if x.numel()==0 or w.numel()==0 or not x.is_contiguous() or not w.is_contiguous():
        raise ValueError('nonempty contiguous tensors required')
    if bias is not None and (bias.shape!=(w.shape[0],) or bias.dtype!=x.dtype or not bias.is_contiguous()):
        raise ValueError('contiguous FP16 bias with output width required')
    if len(config)!=5 or any(type(v) is not int for v in config):
        raise ValueError('five integer config values required')
    bm,bn,bk,warps,stages=config
    if bm not in (32,64,128,256) or bn not in (32,64,128) or bk not in (32,64,128) or warps not in (4,8) or stages not in (2,3,4,5):
        raise ValueError('unsupported kernel configuration')


def fused_w1_swiglu(x,w,bias=None,config=(64,64,64,4,3)):
    validate(x,w,bias,config)
    if x.device.type!='cuda' or torch.version.hip is not None:
        raise ValueError('NVIDIA CUDA required')
    if w.device!=x.device or (bias is not None and bias.device!=x.device):
        raise ValueError('same device required')
    import triton
    m,k=x.numel()//x.shape[-1],x.shape[-1]
    n=w.shape[0]//2
    bm,bn,bk,warps,stages=config
    out=torch.empty((*x.shape[:-1],n),device=x.device,dtype=x.dtype)
    kernel()[(triton.cdiv(m,bm)*triton.cdiv(n,bn),)](
        x,w,bias if bias is not None else x,out,m,n,k,bm,bn,bk,bias is not None,
        num_warps=warps,num_stages=stages,enable_fp_fusion=False)
    return out


@torch.library.custom_op('h3vae_fp16_exp::w1_swiglu',mutates_args=())
def w1_swiglu(x:torch.Tensor,w:torch.Tensor,bias:torch.Tensor|None,
              bm:int,bn:int,bk:int,warps:int,stages:int)->torch.Tensor:
    return fused_w1_swiglu(x,w,bias,(bm,bn,bk,warps,stages))


@w1_swiglu.register_fake
def fake(x,w,bias,bm,bn,bk,warps,stages):
    return x.new_empty((*x.shape[:-1],w.shape[0]//2))


class FusedFP16FFN(nn.Module):
    def __init__(self,original,config):
        super().__init__()
        if original.training or not original.use_gated or not isinstance(original.act_fn,nn.SiLU):
            raise ValueError('eval gated SiLU FFN required')
        if not isinstance(original.w1,nn.Linear) or not isinstance(original.w2,nn.Linear):
            raise ValueError('original FP16 nn.Linear FFN required')
        if original.w1.out_features!=2*original.w2.in_features:
            raise ValueError('gate/up dimensions do not match w2')
        self.w1,self.w2=original.w1,original.w2
        self.config=tuple(config)
        self.eval()

    def forward(self,x):
        if self.training or torch.is_grad_enabled():
            raise RuntimeError('inference only')
        return self.w2(w1_swiglu(x,self.w1.weight,self.w1.bias,*self.config))


def clone_fp16_decoder(raw,config):
    if raw.training:raise ValueError('eval decoder required')
    def clone(module):
        result=copy.copy(module)
        result._parameters=module._parameters.copy()
        result._buffers=module._buffers.copy()
        result._modules={k:None if v is None else clone(v) for k,v in module._modules.items()}
        return result
    result=clone(raw)
    for block in result.transformer_blocks:
        block.ff=FusedFP16FFN(block.ff,config)
    return result
