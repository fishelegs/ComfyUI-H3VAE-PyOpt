# GEMM scheduling and fusion follow-up — 2026-10-03

**None of the kernel candidates below was adopted.** Against `c223906` (the
production kernels from `7bd9809`), persistent FFN scheduling regressed, separate
gate/up accumulators saved too little, and exact QKV/RMSNorm/RoPE fusion saved
only 0.007–0.022 ms per complete local chain. These local gains are not new
full-decoder performance records. A separate batch-8 configuration screen
failed the full-RGB exactness gate before timing; production remains at batch 4.

This follows the [earlier FFN and attention screening](decoder_followup_2026-10-03.md).
The [production Nsight profile](decoder_int8_ffn_2026-10-03.md) identified the
remaining GEMM budget. Experiments retained INT8/INT32 arithmetic, FP32 scales,
FMA bias and FP16 rounding, with all transient allocations included. They ran
serially on the same SM120 host, without stopping unrelated GPU tasks; the
software stack was PyTorch 2.11.0+cu130 / Triton 3.6.0, with comfy-kitchen 0.2.34
for CK calls. Compilation, capture and numerical checks were outside timing.
NCU counters remain unavailable, so resource and instruction counts below do
not establish measured occupancy, bandwidth or stall causes.

## Persistent FFN scheduling

The current kernel launches one CTA for each output tile. The candidate reuses
a fixed grid of CTAs for multiple tiles, retaining the production
BM128/BN64/BK64, four warps, four stages and grouped-M8 ordering. Here
“persistent” describes work within one invocation; no GPU tensor is cached
across calls.

The complete up/SwiGLU/quantization chain uses one real M7188/K2048/N16384
capture, three warmups and six alternating ABBA/BAAB blocks with ten calls per
sample. Each row has its own paired baseline.

| CTA scheduling | Production (ms) | Candidate (ms) | Latency change | Winning blocks |
| --- | ---: | ---: | ---: | ---: |
| One CTA per device SM | 1.368640 | 1.713414 | 25.19% slower | 0/6 |
| Two CTAs per device SM | 1.367887 | 1.398429 | 2.23% slower | 0/6 |
| Two CTAs per SM, flattened outer loop | 1.366509 | 1.692565 | 23.86% slower | 0/6 |

ACT, q and scale bits match on M7188 and M1/17/129 tails. Each candidate reports
232 registers and no spills; shared memory is 49,152 bytes, or 57,344 with loop
flattening. Lower register count did not improve latency.

The first timing harness accidentally returned ACT alongside q/scale, retaining
the intermediate between calls. Those timings were invalidated. The corrected
harness releases ACT at return, matching production; only its results appear
above. Both records remain in the [raw evidence](benchmarks/decoder_gemm_persistent_2026-10-03.json).

## Separate gate/up accumulators

The current FFN-up kernel interleaves gate/up columns in one INT32 accumulator,
then converts its layout before SwiGLU. Two independent accumulators avoid that
interleaving while retaining the original weights and quantizer. The experiment
tested exactly two predefined tiles on the same real M7188 capture, with three
warmups, six ABBA/BAAB blocks and five complete-chain calls per sample.

| BM/BN/BK/warps/stages | Production (ms) | Candidate (ms) | Result |
| --- | ---: | ---: | --- |
| 128/64/64/4/4 | 1.328330 | 1.321007 | 0.55% faster; 5/6 blocks faster |
| 64/64/64/4/4 | 1.351460 | 1.726693 | 27.76% slower; 0/6 blocks faster |

Both retain exact ACT/q/scale on the captured input. The BM128 compiled code
does halve the FP16 epilogue rearrangement: 16,384→8,192 elements per CTA, with
static barriers 8→4. However, registers rise from 238 to 255, and the generated
MMA warp layout changes, increasing static K-loop matrix-load instructions
16→20. These observations describe compiled code, not a causal timing breakdown.
The 0.0073 ms saving is below the predefined 0.03 ms local screening threshold.

Static gate/up weight packing was assessed without another GPU sweep. The
current K-axis loads are already vectorized; an extra packed copy across
36 layers would add 1.125 GiB and require additional offload/state handling.
Earlier ordinary transposed-weight GEMMs had already regressed. This does not
prove all packing strategies ineffective; it gives no current evidence to
justify that deployment cost. [Samples, source hashes and IR audit](benchmarks/decoder_gemm_dualacc_2026-10-03.json).

## QKV GEMM with RMSNorm/RoPE

This candidate eliminates the Q/K portion of the intermediate FP16 QKV
write/read. Q/K retain their original contiguous BNHD layout, and V retains its
original interleaved backing. The GEMM uses complete head-sized column groups,
without extra INT8 multiply-adds. The first tile contains B4, sequence1797,
32 heads of dimension64, QKV K2048/N6144 and rotary dimension48.

Capture ties the actual QKV weight pointer to its layer, verifies captured q/s
reconstruct that same GEMM output, and checks its output pointer against the
immediately following RoPE input. It observes all 36 QKV/RoPE calls and leaves
the full tile output exactly equal. The input latent is cached; this capture is
not fresh full-video validation.

The initial fused kernels preserve raw FP16 GEMM output and V but change Q/K
by up to 0.00390625. Production PTX first adds each group of four consecutive
FP32 squared values serially, then reduces 16 groups with butterfly xor
8/4/2/1. The GEMM-derived layout makes plain `tl.sum` choose another tree.
These numerically failing kernels were not timed.

One explicit-tree repair was tested. Layer-0 FP32 sums and pre-RoPE FP16
normalization match bitwise; the actual timed fusion and a separate debug
specialization then match Q/K/V on layers 0/17/35. Strides, actual Torch Flash
dispatch, Flash output and downstream INT8 projection remain exact.

The complete local chain includes dynamic activation quantization, QKV GEMM,
normalization/RoPE, Flash, sanitation/reshape, INT8 projection and allocations.
Each layer uses three warmups and six ABBA/BAAB blocks, ten calls per sample.

| Layer | Production chain (ms) | Repaired BM128 chain (ms) | Reduction | Winning blocks |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 1.510388 | 1.488006 | 1.48% | 6/6 |
| 17 | 1.493467 | 1.486302 | 0.48% | 3/6 |
| 35 | 1.489638 | 1.470651 | 1.27% | 5/6 |

BM128 uses 255 registers, no reported spills and 49,152 shared bytes. Repaired
BM64 is also exact, but its layer-0 complete chain is 6.53% slower and was not
expanded to other layers. Short, unpaired preparation-only diagnostics are
retained separately and do not replace the complete-chain results above.

The first capture attempt correctly stopped at the CK version guard because
the default environment contained 0.2.31. All successful QKV captures, quality
checks and timings used the isolated 0.2.34 environment. The rejection and
subsequent valid measurements are preserved separately.

The repaired BM128 savings are small and inconsistent across paired blocks,
so this candidate was not promoted to full decode. The
[original explicit-tree helper](benchmarks/qkv_production_sum64_helper.py)
is retained for research; it must be revalidated when compiler or layout changes
and is not imported by the runtime.
[Capture, quality, raw timings, resources and hashes](benchmarks/decoder_gemm_qkv_2026-10-03.json).

## Batch 4 versus batch 8 configuration screen

We also tested decoder tile batching with the current fused FFN-up kernel,
keeping tile256, the same weights, precision, compiled decoder object and cached
FP16 latent fixed. Each batching closure wrapped the same original task method.
The tile inventory and order matched exactly. Actual execution changed from 49
batch-4 calls (1,764 FFN calls) to 21 batch-8 plus 7 batch-4 calls (1,008 FFN
calls), including 756 real FFN inputs with M=14,376; all 36 weights were covered.

The complete 124-frame RGB outputs were finite but differed: RMSE
0.00215067815 and maximum absolute error 0.158203125. The configuration was
rejected before warmup or timing, so there is no batch-8 speedup claim. This used
a cached input for screening, not a fresh video encode. Per-frame errors and the
first differing output element are preserved. The numerical difference was not
localized to an operator; no cross-shape normalization bindings, precision
relaxation or batch-16 search followed. Production remains at batch 4.

This establishes a numerical difference on this input, not a ranking of visual
quality. [Schedule coverage and per-frame RGB errors](benchmarks/decoder_gemm_batch8_2026-10-03.json).

## 中文结论

本轮没有采用新的 kernel 改动。持久化 FFN 调度的最好结果仍慢 2.23%；双累加器虽将
epilogue 重排减半，却增加了寄存器和矩阵加载指令，局部仅快 0.55%。这说明消除一项
开销后，还必须检查它是否增加其他成本。

QKV GEMM 与 RMSNorm/RoPE 融合最初改变了浮点归约顺序。显式恢复生产归约树后，
三层真实输入的 Q/K/V、Flash 和 INT8 投影均精确一致，但完整局部链仅节省
0.007–0.022 ms，尚不足以支持替换生产实现。这些 kernel 实验没有新的完整 decode
成绩，也没有新的多视频画质结论；README 的生产成绩不据此更新。

此外，当前 INT8 decoder 的 batch 4→8 配置筛选虽减少了实际 tile 调用次数，
124 帧完整 RGB 却出现 RMSE 0.002150678、最大绝对差 0.158203125。两组输出均有限，
但未通过精确性检查，因此没有测速，也没有继续 fresh 视频验证。这不是 batch 8
画质更差或速度更慢的证据，差异来源尚未定位；当前仍保留 batch 4。
