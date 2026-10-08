"""Fixed SM120 B4 residual/norm1/INT8-QKV fusion.

Owns the original FP32 residual statistics and FP16/CK quantization rounding.
Other tile shapes retain their real INT8 path. Triton compilation stays lazy.
"""
from functools import lru_cache
import importlib.metadata
import sys

import torch
from torch import nn

from . import decoder_block0_norm as _BLOCK0_HELPER
from . import decoder_final_norm as _FINAL_HELPER
from .int8_attention_out import clone_containers
from .int8_ffn_up_fused import fused_residual_ffn, supports_fused_ffn_up
from .int8_norm_quant import residual_rms_linear
from .int8_swiglu_fused import prequantized_linear, swiglu_int8_linear
from .kitchen_int8 import KitchenInt8Linear

_EPS = 1.0e-5
_WIDTH = 2048
_B4_SHAPE = (4, 1797, 2048)
_FFN_WARPS = 8
_NORM_WARPS = 16
_CK_VERSION = "0.2.34"

block0_rms_fp16 = _BLOCK0_HELPER.rms_fp16_x2_w16


class _FinalNormAdapter(nn.Module):
    def __init__(self, original):
        super().__init__()
        self.weight, self.bias = original.weight, original.bias
        self.normalized_shape, self.eps = original.normalized_shape, original.eps
        self.train(original.training)

    def forward(self, states):
        if isinstance(states, tuple):
            if self.training or torch.is_grad_enabled():
                raise RuntimeError("eval + inference/no_grad required")
            h, out, gate = states
            return _FINAL_HELPER.final_residual_ln_w8(
                h, out, gate, self.weight, self.bias)
        return torch.nn.functional.layer_norm(
            states, self.normalized_shape, self.weight, self.bias, self.eps)


def _carry_safe_unpad(original_unpad):
    def unpad(states, pad_len):
        if isinstance(states, tuple):
            if pad_len != 0:
                raise RuntimeError("final residual carry requires zero sequence padding")
            unpadded = original_unpad(states, pad_len)
            if unpadded is not states:
                raise RuntimeError("zero unpadding must preserve final residual carry")
            return unpadded
        return original_unpad(states, pad_len)
    return unpad


@lru_cache(maxsize=4)
def _device_supports_norm1_quant(index):
    try:
        if (importlib.metadata.version("triton") != "3.6.0" or
                str(torch.__version__).split("+")[0] != "2.11.0" or
                torch.version.cuda != "13.0"):
            return False
        return torch.cuda.get_device_capability(index) == (12, 0)
    except Exception:
        return False


def supports_norm1_quant(device):
    """Select only the validated SM120 / Torch2.11 / CUDA13 / Triton3.6 recipe."""
    device = torch.device(device)
    if (device.type != "cuda" or torch.version.hip is not None or
            not torch.cuda.is_available()):
        return False
    index = torch.cuda.current_device() if device.index is None else device.index
    return _device_supports_norm1_quant(index)


@lru_cache(maxsize=1)
def _validate_ck_dependency():
    try:
        version = importlib.metadata.version("comfy-kitchen")
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError("norm1 fusion requires comfy-kitchen==0.2.34") from exc
    if version != _CK_VERSION:
        raise RuntimeError(
            f"norm1 fusion requires comfy-kitchen=={_CK_VERSION}; found {version}"
        )
    try:
        import comfy_kitchen as ck
    except Exception as exc:
        raise RuntimeError("norm1 fusion could not import comfy-kitchen 0.2.34") from exc
    if not callable(getattr(ck, "quantize_int8_rowwise", None)) or not callable(
        getattr(ck, "int8_linear", None)
    ):
        raise RuntimeError(
            "norm1 fusion requires comfy-kitchen quantize_int8_rowwise and int8_linear"
        )
    return True


@lru_cache(maxsize=1)
def _residual_norm_quant_kernel():
    from .int8_norm1_producer import residual_norm_quant
    return residual_norm_quant


def _produce_norm1(h, o, g, w, *, debug=False):
    """Run the fixed W16 producer; debug mode exists only for shadow diagnosis."""
    m = h.numel() // _WIDTH
    q = torch.empty((m, _WIDTH), device=h.device, dtype=torch.int8)
    scale = torch.empty((m, 1), device=h.device, dtype=torch.float32)
    norm = torch.empty_like(h) if debug else h
    _residual_norm_quant_kernel()[(m,)](
        h, o, g, w, q, scale, norm, m, debug,
        num_warps=_NORM_WARPS, num_stages=1, enable_fp_fusion=True,
    )
    return q, scale, (norm if debug else None)


def residual_qkv_impl(h, o, g, w, qw, ws, bias):
    """In-place H producer followed by the existing CK-compatible QKV GEMM."""
    tensors = (h, o, g, w, qw, ws) + (() if bias is None else (bias,))
    if (h.shape != _B4_SHAPE or o.shape != h.shape or g.shape != (_WIDTH,) or
            w.shape != (_WIDTH,) or qw.shape != (6144, _WIDTH) or ws.shape != (6144,) or
            h.dtype != torch.float16 or o.dtype != torch.float16 or
            g.dtype != torch.float16 or w.dtype != torch.float16 or
            qw.dtype != torch.int8 or ws.dtype != torch.float32 or
            any(t.device != h.device or not t.is_contiguous() for t in tensors) or
            (bias is not None and (bias.shape != (6144,) or bias.dtype != torch.float16))):
        raise ValueError("expected contiguous B4 [4,1797,2048] FP16 inputs and INT8 QKV")
    if torch.is_grad_enabled():
        raise RuntimeError("norm1 fusion requires inference/no_grad")
    _validate_ck_dependency()
    if h.device.type != "cuda" or torch.version.hip is not None:
        raise RuntimeError("norm1 fusion requires SM120 / Torch2.11 / CUDA13 / Triton3.6")
    index = torch.cuda.current_device() if h.device.index is None else h.device.index
    if not _device_supports_norm1_quant(index):
        raise RuntimeError("norm1 fusion requires SM120 / Torch2.11 / CUDA13 / Triton3.6")
    if h.data_ptr() == o.data_ptr():
        raise ValueError("residual H and FFN output must have distinct storage")

    q, scale, _ = _produce_norm1(h, o, g, w)
    return prequantized_linear(q, scale, qw, ws, bias).reshape(4, 1797, 6144)


@torch.library.custom_op("h3vae_norm1::residual_qkv", mutates_args=("h",))
def residual_qkv(h: torch.Tensor, o: torch.Tensor, g: torch.Tensor,
                 w: torch.Tensor, qw: torch.Tensor, ws: torch.Tensor,
                 bias: torch.Tensor | None) -> torch.Tensor:
    return residual_qkv_impl(h, o, g, w, qw, ws, bias)


@residual_qkv.register_fake
def _fake_residual_qkv(h, o, g, w, qw, ws, bias):
    del o, g, w, ws, bias
    return h.new_empty((*h.shape[:-1], qw.shape[0]))


def _qk_rope_for(attn):
    """Read the op from the actual runtime class module; do not import a copy."""
    module = sys.modules.get(type(attn).__module__)
    if (module is None or getattr(module, "FusedQKAttention", None) is not type(attn)):
        raise ValueError("expected the runtime FusedQKAttention class")
    op = type(attn).forward.__globals__.get("qk_rope")
    if op is None or getattr(module, "qk_rope", None) is not op:
        raise ValueError("runtime FusedQKAttention qk_rope op is unavailable")
    return op


def _attention_from_qkv(attn, qkv, rotary, pack_info, qk_rope):
    a = attn.original
    b, n, _ = qkv.shape
    q, k = qk_rope(qkv, *rotary, a.heads, a.norm_q.eps, attn.rows)
    v = qkv.view(b, n, a.heads, 3 * a.dim_head)[..., 2 * a.dim_head:]
    y = a.perform_attention(q, k, v, pack_info)
    return a.to_out(y.reshape(b, n, -1))


class _Norm1QuantBlock(nn.Module):
    def __init__(self, src, first, last, norm1_enabled):
        super().__init__()
        self.norm1, self.norm2, self.attn, self.ff = src.norm1, src.norm2, src.attn, src.ff
        self.scale1, self.scale2 = src.scale1, src.scale2
        self.warps, self.fuse_ffn_up = src.warps, src.fuse_ffn_up
        self.norm1_enabled = bool(norm1_enabled)
        self.fuse_norm1 = self.norm1_enabled and supports_norm1_quant(self.scale1.device)
        self.qk_rope = _qk_rope_for(self.attn)
        self.first, self.last = first, last
        self.train(src.training)

    def _apply(self, fn, recurse=True):
        out = super()._apply(fn, recurse=recurse)
        self.fuse_ffn_up = supports_fused_ffn_up(self.scale1.device)
        self.fuse_norm1 = self.norm1_enabled and supports_norm1_quant(self.scale1.device)
        return out

    def _ffn_pair(self, h, attn):
        w1, w2 = self.ff.w1, self.ff.w2
        if self.fuse_ffn_up:
            return fused_residual_ffn(h, attn, self.scale1, self.norm2.weight,
                w1.qweight, w1.weight_scale, w1.bias, w2.qweight,
                w2.weight_scale, w2.bias, self.norm2.eps, self.warps)
        h, hidden = residual_rms_linear(h, attn, self.scale1, self.norm2.weight,
            w1.qweight, w1.weight_scale, w1.bias, self.norm2.eps, self.warps)
        out = swiglu_int8_linear(hidden, w2.qweight, w2.weight_scale, w2.bias, 16)
        return h, out

    def _target_shape(self, h):
        return (self.fuse_norm1 and h.shape == _B4_SHAPE and
                h.dtype == torch.float16 and h.is_cuda and h.is_contiguous())

    def forward(self, states, rotary_pos_emb=None, pack_info=None):
        if self.training or torch.is_grad_enabled():
            raise RuntimeError("eval + inference/no_grad required")
        pack_info = {} if pack_info is None else pack_info
        chain = False
        if isinstance(states, tuple):
            h, prev_out, prev_scale2 = states
            can_fuse = (not self.first and self._target_shape(h) and
                        prev_out.shape == h.shape and prev_out.dtype == torch.float16 and
                        prev_out.is_contiguous() and rotary_pos_emb is not None)
            if can_fuse:
                linear = self.attn.original.to_qkv
                qkv = residual_qkv(h, prev_out, prev_scale2, self.norm1.weight,
                                   linear.qweight, linear.weight_scale, linear.bias)
                attn = _attention_from_qkv(
                    self.attn, qkv, rotary_pos_emb, pack_info, self.qk_rope)
                states, chain = h, True
            else:
                states = h + prev_out * prev_scale2
                normed = self.norm1(states.float()).to(states.dtype)
                attn = self.attn(normed, rotary_pos_emb, pack_info)
        else:
            if self.first and self._target_shape(states):
                normed = block0_rms_fp16(states, self.norm1.weight)
            else:
                normed = self.norm1(states.float()).to(states.dtype)
            attn = self.attn(normed, rotary_pos_emb, pack_info)
            chain = self._target_shape(states) and rotary_pos_emb is not None

        h, out = self._ffn_pair(states, attn)
        if self.last:
            if self._target_shape(h):
                return h, out, self.scale2
            return h + out * self.scale2
        return (h, out, self.scale2) if chain else h + out * self.scale2


def _validate_blocks(blocks):
    if len(blocks) != 36:
        raise ValueError("norm1 fusion expects 36 decoder blocks")
    for block in blocks:
        if (block.training or block.warps != _FFN_WARPS or
                not isinstance(block.norm1, nn.RMSNorm) or
                not isinstance(block.norm2, nn.RMSNorm) or
                block.norm1.normalized_shape != (_WIDTH,) or
                block.norm2.normalized_shape != (_WIDTH,) or
                block.norm1.weight is None or block.norm2.weight is None or
                block.norm1.eps != _EPS or block.norm2.eps != _EPS or
                block.scale1.shape != (_WIDTH,) or block.scale2.shape != (_WIDTH,) or
                block.scale1.dtype != torch.float16 or block.scale2.dtype != torch.float16 or
                block.norm1.weight.dtype != torch.float16 or
                block.norm2.weight.dtype != torch.float16):
            raise ValueError("norm1 fusion requires the validated FP16 width-2048 block contract")
        _qk_rope_for(block.attn)
        a = block.attn.original
        if (not isinstance(a.to_qkv, KitchenInt8Linear) or
                not isinstance(a.to_out, KitchenInt8Linear) or
                not isinstance(block.ff.w1, KitchenInt8Linear) or
                not isinstance(block.ff.w2, KitchenInt8Linear) or
                a.to_qkv.qweight.shape != (6144, _WIDTH) or
                a.to_qkv.weight_scale.shape != (6144,)):
            raise ValueError("expected an already-quantized INT8 decoder; refusing requantization")


def clone_norm1_quant_decoder(raw, *, enabled=False):
    """Clone an already-INT8 decoder; explicit unsupported enablement fails."""
    if raw.training:
        raise ValueError("eval decoder required")
    blocks = getattr(raw, "transformer_blocks", None)
    if blocks is None or len(blocks) != 36:
        raise ValueError("expected a 36-block decoder")
    result = clone_containers(raw)
    try:
        device = next(raw.parameters()).device
    except StopIteration as exc:
        raise ValueError("decoder has no parameters") from exc
    if not enabled:
        result._h3vae_norm1_quant_enabled = False
        return result.eval()
    if not supports_norm1_quant(device):
        raise RuntimeError("norm1 fusion requires SM120 / Torch2.11 / CUDA13 / Triton3.6")
    _validate_ck_dependency()

    # A carry tuple cannot pass sequence-parallel gather or nonzero unpadding.
    # This research candidate is restricted to the validated single-device path.
    if getattr(raw, "spatial_parallel", True):
        raise RuntimeError("authored final LayerNorm requires spatial_parallel=false")
    norm = getattr(raw, "norm_out", None)
    if (type(norm) is not nn.LayerNorm or
            getattr(norm.forward, "__func__", None) is not nn.LayerNorm.forward or
            norm.normalized_shape != (_WIDTH,) or
            norm.eps != _EPS or norm.weight is None or norm.bias is None or
            norm.weight.dtype != torch.float16 or norm.bias.dtype != torch.float16):
        raise ValueError("final LayerNorm requires affine FP16 width-2048/eps-1e-5")

    _validate_blocks(list(result.transformer_blocks))
    blocks = list(result.transformer_blocks)
    result.transformer_blocks = nn.ModuleList([
        _Norm1QuantBlock(block, i == 0, i == len(blocks) - 1, norm1_enabled=True).eval()
        for i, block in enumerate(blocks)
    ])
    result.norm_out = _FinalNormAdapter(result.norm_out).eval()
    result._unpad_for_sp = _carry_safe_unpad(result._unpad_for_sp)
    result._h3vae_norm1_quant_enabled = True
    return result.eval()
