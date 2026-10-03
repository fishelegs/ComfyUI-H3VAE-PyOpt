"""Exact sum64 tree used by the isolated QKV/RoPE experiment; no production adoption.

Matches the observed production Triton 3.6.0 / SM120 tree with enable_fp_fusion=False.
Revalidated with real layers 0/17/35; not a proof for arbitrary future compiler layouts.
"""
import triton
import triton.language as tl


@triton.jit
def production_square_sum64(x, ROWS: tl.constexpr):
    # Exact production PTX tree: four contiguous values summed serially,
    # then 16 lane-groups reduced by butterfly xor 8,4,2,1.
    group = tl.arange(0, 16)
    squared = x * x
    i0 = tl.broadcast_to((4 * group)[None, :], (ROWS, 16))
    a0 = tl.gather(squared, i0, 1)
    a1 = tl.gather(squared, i0 + 1, 1)
    a2 = tl.gather(squared, i0 + 2, 1)
    a3 = tl.gather(squared, i0 + 3, 1)
    partial = a0 + a1
    partial = a2 + partial
    partial = a3 + partial
    for shift in tl.static_range(0, 4):
        idx = tl.broadcast_to((group ^ (8 >> shift))[None, :], (ROWS, 16))
        partial = partial + tl.gather(partial, idx, 1)
    return tl.gather(partial, tl.full((ROWS, 1), 0, tl.int32), 1)
