"""Exact-order FP32 pixel finalization; no model arithmetic changes."""
from functools import lru_cache
import torch


def validate(x,std,mean,out=None):
    if x.ndim!=5 or x.shape[1]!=3 or x.numel()==0:
        raise ValueError('nonempty B,3,T,H,W tensor required')
    if x.dtype not in (torch.float16,torch.float32):
        raise ValueError('FP16 or FP32 decoded pixels required')
    for v in (std,mean):
        if v.numel()!=3 or v.dtype!=torch.float32 or not v.is_contiguous() or v.device!=x.device:
            raise ValueError('three contiguous FP32 channel constants on same device required')
    if out is not None and (out.shape!=x.shape or out.dtype!=torch.float32 or out.device!=x.device or not out.is_contiguous()):
        raise ValueError('contiguous FP32 output with matching shape/device required')


@lru_cache(maxsize=1)
def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def run(X,S,A,Y,NUM:tl.constexpr,T:tl.constexpr,H:tl.constexpr,W:tl.constexpr,
            S0:tl.constexpr,S1:tl.constexpr,S2:tl.constexpr,S3:tl.constexpr,S4:tl.constexpr,
            CONTIG:tl.constexpr,BLOCK:tl.constexpr):
        i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
        c=(i//(T*H*W))%3
        if CONTIG:
            offset=i
        else:
            offset=(i//(3*T*H*W))*S0+c*S1+((i//(H*W))%T)*S2+((i//W)%H)*S3+(i%W)*S4
        x=tl.load(X+offset,i<NUM,other=0).to(tl.float32)
        std=tl.load(S+c);mean=tl.load(A+c)
        # Deliberately disable FMA at launch: match separate FP32 multiply/add.
        value=x*std+mean
        # Comparisons preserve NaN propagation and the sign of an in-range zero.
        value=tl.where(value<0.0,0.0,tl.where(value>1.0,1.0,value))
        tl.store(Y+i,value,i<NUM)
    return run


def finalize_pixels(x,std,mean,out=None):
    validate(x,std,mean,out)
    if x.device.type!='cuda' or torch.version.hip is not None:
        raise ValueError('NVIDIA CUDA required')
    if torch.is_grad_enabled():raise RuntimeError('inference only')
    import triton
    if out is None:out=torch.empty(x.shape,device=x.device,dtype=torch.float32)
    _,_,t,h,w=x.shape
    kernel()[(triton.cdiv(x.numel(),1024),)](x,std,mean,out,x.numel(),t,h,w,*x.stride(),
        x.is_contiguous(),1024,num_warps=4,enable_fp_fusion=False)
    return out
