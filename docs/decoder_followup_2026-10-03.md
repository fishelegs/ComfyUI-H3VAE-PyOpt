# Decoder follow-up screening — 2026-10-03

**No new production change was adopted.** This round screened the remaining FFN
and attention hotspots against `7bd9809`. The current full-stage records remain
INT8 encode **8.458 s**, INT8 decode **6.509 s**, and fused FP16 decode
**10.923 s**. They are earlier validated results, not new measurements in this
round. Competitor charts remain unchanged.

## Scope and method

The [latest Nsight Systems profile](decoder_int8_ffn_2026-10-03.md) attributed
2.269 s to FFN-up/SwiGLU, 1.171 s to FlashAttention, and 0.273 s to activation
quantization. These durations guided bounded local experiments using the ACA
workflow. They do not establish bandwidth, occupancy or warp-stall bottlenecks:
NCU hardware counters remain unavailable (`ERR_NVGPUCTRPERM`).

The machine and stack match that report: historical RTX PRO 5000 72GB identity,
SM120 (driver 580.82.07 reports NVIDIA Graphics Device), Linux, Python 3.12.14,
PyTorch 2.11.0+cu130, CUDA 13.0, Triton 3.6.0 and comfy-kitchen 0.2.34.
Experiments used one GPU lock and did not stop unrelated GPU tasks. All results
below are CUDA-event **local component or chain** timings, excluding compilation,
capture and numerical checks. Different timing scopes must not be combined.

FFN replay uses one real captured input: M=7188, K=2048, N=16384. Attention uses
layers 0/17/35 of one real decoder tile, captured from the unchanged compiled
INT8 decoder using a cached FP16-encoded latent. Its SDPA shape is
`[B,H,S,D]=[4,32,1797,64]`, scale 0.125, without mask, dropout or causality.
The actual FP16 Q/K/V values and strides were preserved. This is not a separate
full FP16 decoder benchmark or a new multi-video quality validation.

## FFN results

The complete up/SwiGLU/quantization chain includes transient allocations.
Each variant was warmed up three times and measured 12 times with rotated and
reversed execution order; these are screening samples, not full-stage ACA
confirmation blocks.

| Candidate | Production mean (ms) | Candidate mean (ms) | Decision |
| --- | ---: | ---: | --- |
| Source-copy control | 1.187613 | 1.188000 | Control only |
| GROUP_M=4 | 1.187613 | 1.186192 | Insufficient local saving |
| GROUP_M=16 | 1.187613 | 1.189107 | Slower |
| Streaming activation store | 1.187613 | 1.193533 | Slower |
| Up GEMM with 8 warps | 1.187613 | 1.181907 | Only 0.48% local saving; not promoted |
| FP32 SiLU lookup, 4 warps | 1.191448 | 2.058368 | Slower |
| FP32 SiLU lookup, 8 warps | 1.191448 | 2.069053 | Slower |

All candidates produced exactly matching INT8 q and FP32 scales on this input.
The lookup variants generate a fresh 256 KiB table per call; allocation,
generation and lookup are included, without a persistent GPU tensor cache.
An additional check covered every 63,488 finite FP16 gate encoding paired with
five up constants (`0, -1, 1, 0.03125, 65504`): all 317,440 final FP16 outputs
matched bitwise. This does **not** exhaust every gate/up combination or prove
FP32 intermediate equality.

The separate quantizer experiment preallocates input/output and excludes
FFN-up. Each candidate uses three ABBA/BAAB blocks, six samples per segment and
20 launches per sample. Each row has its own paired baseline.

| Quantizer candidate | Production mean (ms) | Candidate mean (ms) | Decision |
| --- | ---: | ---: | --- |
| One row / 4 warps | 0.170225 | 0.168851 | 0.81% saving; one of three blocks slower |
| One row / 16 warps | 0.169830 | 0.172293 | Slower |
| Two rows / 8 warps | 0.170412 | 0.172970 | Slower |
| Four rows / 16 warps | 0.170842 | 0.176829 | Slower |

All retain FP32 scales, FP16 denominator/division rounding, `tl.div_rn`,
nearbyint and NaN→-128 handling. q/scale checks pass on real M7188, a repeated-row
M14376 extension and synthetic M17 tail/tiny/nonfinite cases. M14376 is not an
independent real capture. None was promoted to a full decoder experiment.

## Attention results

Layout candidates pack V into compact BNHD storage, or store all Q/K/V in BHND
order. Timing includes QK/RoPE preparation, packing, the same actual Torch Flash
kernel, `nan_to_num`, output reshape/copy and INT8 output projection. Q/K/V,
attention and projected outputs matched exactly and were finite on all three
layers. A validation-only profiler confirmed the same Flash kernel; no profiler
hook was active in timed samples.

A final native FA2 candidate reduces query tile M128→M64 while holding
FP16/D64/key tile N128/4 warps fixed. The native M128 control first matched
Torch exactly; M64 and downstream projection also matched exactly. Native
timing includes per-call output/LSE allocation and current-stream launch.
Each comparison uses six ABBA/BAAB blocks per layer, ten calls per sample.

| Candidate | Complete local-chain latency change across three layers | Winning blocks |
| --- | ---: | ---: |
| Compact V in BNHD | 9.01–9.90% slower | 0/18 |
| Head-contiguous Q/K/V in BHND | 13.62–14.24% slower | 0/18 |
| Native FA2 query tile M64 | 3.48–3.90% slower | 0/18 |

Packing approximately doubled preparation time while prepacked Flash-only
timing barely changed. The BHND variant also requires output layout conversion.
Native M64 reduced registers from 255 to 166 and shared memory from 49,152 to
40,960 bytes, but its attention-only time was 10.35–10.95% slower. The native
M128 complete-chain control was 0.04–0.68% slower than Torch.

The local native M128 build reported a 40-byte stack frame, 72-byte spill stores
and 60-byte spill loads; M64 reported no spill. These are **this experiment's
compiler reports**, not measured spill traffic in the production Torch kernel.
Lower resource counts did not translate into lower latency.

Native compilation used the existing offline FlashAttention
`e2743ab5b3803bb672b16437ba98a3b1d4576c50` and CUTLASS
`0d2b201e8c1c4a03efa6e9c468161916e2334725` sources, their BSD-3-Clause notices,
CUDA 13.0.88, the original FA2 entry and `UNFUSE_FMA`. No native extension,
third-party source, dependency or binary was added to the production package.

The initial capture hook targeted the wrong module import and observed zero
calls. That capture was invalidated before timing; the corrected capture
observed all 36 calls and preserved the original tile output exactly. Both
attempts are retained in the evidence.

## Decision and remaining work

These candidates do not justify replacing the current implementation. No full
decode speedup or new full-video quality result is claimed. The small FFN
scheduling savings are insufficient local evidence; the attention variants
consistently regress. Production precision defaults and quality trade-offs
remain unchanged.

The largest remaining measured region is still GEMM: fused FFN-up, FFN-down and
QKV. A further round needs a different mechanism that reduces their measured
cost, while retaining exact quantization/rounding and counting all preparation
costs. Any local success still needs full decode pairing and video validation.
This round does not establish that further optimization is impossible.

[FFN raw samples, contracts and source hashes](benchmarks/decoder_followup_ffn_2026-10-03.json)
· [Attention capture, raw samples, build and source metadata](benchmarks/decoder_followup_attention_2026-10-03.json).
These are sanitized experiment records, not a standalone replay distribution.

## 中文结论

本轮以 `7bd9809` 为基线，按 ACA 的有界候选流程，针对上一轮 Nsight 中实际剩余的
FFN 和 attention 时间做局部回放，**没有采用新的生产改动**。

- FFN-up 调度、streaming store 和 SiLU 查表均未取得值得晋升的收益；8 warps 仅在
  单份真实输入上减少约 0.48% 的局部时间，不能视为完整 decode 提升。
- 独立量化器最好的候选仅快约 0.81%，三个配对块中还有一块回退；其余候选更慢。
- V 紧凑布局和 Q/K/V head 连续布局，计入准备及输出处理后分别慢 9.01–9.90% 和
  13.62–14.24%。预先排好布局的 Flash 单独计时几乎不变。
- Native FA2 M64 虽然降低寄存器/shared 并消除本次编译报告中的 spill，局部完整链
  仍慢 3.48–3.90%。资源数减少不代表实际加速；没有 NCU 硬件计数器证据。

FFN 候选的真实输入 q/scale、attention 三层的中间结果和下游投影均通过相应精确检查。
这些检查只覆盖已记录的局部输入，不是新的多视频画质结论；也没有重新测出更快的
完整阶段。因此 README 和竞品柱状图仍采用此前验证的 **INT8 encode 8.458 s / decode
6.509 s / FP16 decode 10.923 s**。后续优先寻找能降低 FFN-up、FFN-down、QKV GEMM
实际成本的新机制，通过局部筛选后再做完整视频验证。
