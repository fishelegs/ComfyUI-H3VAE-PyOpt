"""Pipelined implicit INT8 convolution; selected automatically only on SM120.

The flattened integer reduction exposes a long software-pipelined loop without
materializing im2col. INT32 accumulation and the original FP32/FP16 epilogue
are unchanged. Imported lazily by encoder_int8, including on CPU-only hosts.
"""
from functools import lru_cache


@lru_cache(maxsize=1)
def pipelined_conv3d_kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def implicit_conv3d_kernel(
        qx_ptr,
        qw_ptr,
        x_scale_ptr,
        w_scale_ptr,
        bias_ptr,
        out_ptr,
        batch: tl.constexpr,
        channels: tl.constexpr,
        depth: tl.constexpr,
        height: tl.constexpr,
        width: tl.constexpr,
        out_channels: tl.constexpr,
        out_depth: tl.constexpr,
        out_height: tl.constexpr,
        out_width: tl.constexpr,
        sx0: tl.constexpr,
        sx1: tl.constexpr,
        sx2: tl.constexpr,
        sx3: tl.constexpr,
        sx4: tl.constexpr,
        so0: tl.constexpr,
        so1: tl.constexpr,
        so2: tl.constexpr,
        so3: tl.constexpr,
        so4: tl.constexpr,
        stride_d: tl.constexpr,
        stride_h: tl.constexpr,
        stride_w: tl.constexpr,
        KERNEL_D: tl.constexpr,
        KERNEL_H: tl.constexpr,
        KERNEL_W: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        HAS_BIAS: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        output_m = out_depth * out_height * out_width
        total_m = batch * output_m
        m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        m_mask = m_offsets < total_m
        n_mask = n_offsets < out_channels

        ow = m_offsets % out_width
        rem = m_offsets // out_width
        oh = rem % out_height
        rem = rem // out_height
        od = rem % out_depth
        ob = rem // out_depth

        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
        k_total = KERNEL_D * KERNEL_H * KERNEL_W * channels
        for start in range(0, k_total, BLOCK_K):
            k = start + tl.arange(0, BLOCK_K)
            c_offsets = k % channels
            kw = (k // channels) % KERNEL_W
            kh = (k // (channels * KERNEL_W)) % KERNEL_H
            kd = k // (channels * KERNEL_W * KERNEL_H)
            x_offsets = (
                ob[:, None] * sx0 + c_offsets[None, :] * sx1
                + (od[:, None] * stride_d + kd[None, :]) * sx2
                + (oh[:, None] * stride_h + kh[None, :]) * sx3
                + (ow[:, None] * stride_w + kw[None, :]) * sx4
            )
            w_offsets = k[:, None] + n_offsets[None, :] * k_total
            x_values = tl.load(qx_ptr + x_offsets,
                mask=m_mask[:, None] & (k[None, :] < k_total), other=0)
            w_values = tl.load(qw_ptr + w_offsets,
                mask=(k[:, None] < k_total) & n_mask[None, :], other=0)
            accumulator = tl.dot(x_values, w_values, accumulator, out_dtype=tl.int32)
        x_scale = tl.load(x_scale_ptr).to(tl.float32)
        w_scale = tl.load(w_scale_ptr + n_offsets, mask=n_mask, other=0.0)
        values = accumulator.to(tl.float32) * x_scale * w_scale[None, :]
        if HAS_BIAS:
            bias = tl.load(bias_ptr + n_offsets, mask=n_mask, other=0.0)
            values += bias[None, :].to(tl.float32)

        out_offsets = (
            ob[:, None] * so0
            + n_offsets[None, :] * so1
            + od[:, None] * so2
            + oh[:, None] * so3
            + ow[:, None] * so4
        )
        tl.store(
            out_ptr + out_offsets,
            values.to(tl.float16),
            mask=m_mask[:, None] & n_mask[None, :],
        )

    return implicit_conv3d_kernel
