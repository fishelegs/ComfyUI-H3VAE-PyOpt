# Residual/norm1/INT8 quantization fusion — 2026-10-03

**No runtime change was adopted in this round.** Current production records
remain INT8 encode **8.458108 s**, INT8 decode **6.508537 s**, and fused FP16
decode **10.923379 s**. README performance tables and competitor charts are
unchanged. The cleaned candidate failed exact quantization under a newly
selected 16-warp normalization schedule, before timing.

Under its recorded eight-warp schedule, a private fused producer reduced the
**complete cached-latent decode** from
**6.513790 to 6.430177 s (1.28% lower host-wall latency)** in six paired blocks.
The first tile group's 35 replaced layers passed exact shadow checks, and the
full 124-frame RGB output remained bitwise equal. This is a same-graph research
result. A separate private module subsequently passed eight-video checks, but
the cleaned module did not pass the final natural-baseline gate.

## Why this boundary

The [previous batch diagnosis](decoder_batch_diagnosis_2026-10-03.md) located a
sensitive boundary between the preceding FFN residual, norm1 and QKV activation
quantization. The saved Nsight profile attributes 127.511 ms to norm1 and
64.717 ms to QKV activation quantization. Their 192.229 ms sum is an optimistic
activity budget, not an expected saving. NCU counters remain unavailable;
this experiment makes no measured occupancy, bandwidth or stall claim.

Fusion avoids materializing the normalized FP16 activation before quantization.
The QKV GEMM, INT8 weights, FP32 scales and existing quantization rounding remain
unchanged. Batch 4 and tile256 are retained. Layers 1–35 are targeted; layer 0
has no preceding FFN residual and keeps its existing path.

## Arithmetic and rejected attempts

Statistics must use the **unrounded FP32 residual** `fma(output, scale2, H)`.
The residual is then rounded to FP16 before normalization, and norm is rounded
to FP16 before CK-compatible rowwise quantization. Normalizing only the stored
FP16 residual would lose part of the original compiled operation's contract.

The archived B4 norm used XBLOCK2/R0_BLOCK2048, eight warps and one stage.
Tests use captured `[4,1797,2048]` FP16 inputs from the first two tile groups.

| Attempt | H | FP16 norm differences | Q / scale / QKV differences | Decision |
| --- | --- | ---: | --- | --- |
| Ordinary Triton fusion | Exact | 732 | 2 / 1 / 5,785 | Reject before timing |
| Explicit Gluon layout | Exact | 16 | 0 / 0 / 0 | Reject before timing |
| Explicit layout and local arithmetic | Exact | 0 | 0 / 0 / 0 | Pass local screen |

The INT8 store initially changed the reduction layout. Fixing that layout
still allowed different FMA contraction. PTX showed a rounded square at
position 1, followed by FMA for positions 0 and 2, then alternating separately
rounded odd squares/additions and even-position FMA through position 7.
The final helper specifies that sequence explicitly before the observed
inter-lane/inter-warp reduction. It was a separately recorded additional
experiment after the initial candidate and one layout repair; no exactness
threshold was relaxed and both failures are retained.

Both captured groups and M1/17/129 row slices passed bitwise H, norm, Q, FP32
scale and QKV checks. Tail slices have no production capture anchor. Debug
and non-debug specializations were checked separately. The reference uses
37 registers versus the candidate's 40; both report 64 shared bytes and no
spills. These observations apply to the recorded compiler and layout.

## Local chain measurement

Three warmups per variant precede six alternating ABBA/BAAB blocks, ten calls
per sample. Every call includes the same H clone, allocations, producer,
quantization and existing QKV GEMM. Loading, compilation and validation are
excluded. Clone is needed because the replay's original norm mutates H.

| Measurement | Baseline | Candidate |
| --- | ---: | ---: |
| CUDA event, complete local chain | 0.604345 ms | 0.562133 ms |
| Synchronized host wall, complete local chain | 0.625358 ms | 0.584390 ms |
| Shared clone-only control, CUDA event / wall | 0.020941 / 0.038891 ms | Same control |

The local CUDA-event reduction is 6.98%, with six of six blocks faster.
It exceeds the declared 0.02 ms/call and five-winning-block local screen.
This local percentage is not the full-decoder result below.

## Full decoder with its natural schedule

All variants use one compiled decoder, unchanged tile order and cached FP16
latent. Temporary routing connects each fused norm producer to its immediately
following QKV call. Pointer, layer, stream, missing-consumer and request-boundary
checks reject incorrect routing. Both timed variants include routing and call
counters; numerical shadow work stays outside timing. The graph still allocates
its original FP16 norm buffer, so this experiment does not remove that allocation.

Independent process starts selected different norm schedules despite identical
source: X1/warps4, X1/warps8 and X2/warps8, all with R0_BLOCK2048. Early attempts
stopped at a narrow configuration gate; another stopped because the diagnostic
script incorrectly expected a `launcher.bin` attribute. These attempts produced
no numerical or timing conclusion. The corrected script reads the actual
`launcher.cache_hash`, records any natural configuration and retains all direct
bitwise gates. It never binds or overrides the baseline's norm schedule.

The completed run naturally selected X2/warps8. All 35 replaced layers in the
first tile group matched residual H, normalized FP16 values, Q, FP32 scale and
QKV output exactly. Observer and candidate tile outputs also matched. Across the
full decode, each replaced layer executed 49 times (1,715 replacements); all
36 QKV layers executed 49 times each. The full `[1,3,124,768,1344]` FP32 RGB
was bitwise equal and the input latent hash remained unchanged.

After two warmups per variant, six ABBA/BAAB blocks each timed four full decodes.

| Complete runtime decode | Baseline | Candidate |
| --- | ---: | ---: |
| Synchronized host wall | 6.513790 s | 6.430177 s |
| CUDA event | 6.513585 s | 6.429990 s |
| Peak allocated bytes | 5,120,049,152 | 5,120,049,152 |

Each block saved 79.219–86.739 ms; mean saving was 83.613 ms, or 1.2836%.
All six blocks were faster, passing the declared 0.5% / five-block screen.
These peaks include the resident benchmark runtime and outputs. They are not
isolated model memory. This one cached input does not establish fresh-video
quality, behavior on other shapes or equivalence to every natural norm schedule.

## Real-module integration

A separate module carries the previous FFN residual as a short-lived tuple
between transformer blocks. Its registered custom op declares the mutation of
H and returns an independent QKV tensor. Weights are shared from the already
quantized decoder without requantization; there is no pointer-based activation
cache. The last block returns the ordinary Tensor expected by the decoder.

The first module attempt stopped at a class-identity check: importing the QK
module under both its package and runtime top-level names creates distinct
classes. The corrected clone resolves QK RoPE from the actual runtime attention
class. A second attempt passed 35 shadows, tile and full RGB checks and recorded
a provisional 1.28% timing gain, but its offload diagnostic still keyed layers by
pre-offload pointers and failed after reload. That attempt also reset Dynamo
between variants without rechecking the old RGB after timing; its timing is not
used as the final production record.

An independent quality run passed all replaced-operation shadows but failed
whole-tile equality: 22,302 of 22,020,096 FP16 values differed, RMSE 1.19341e-5,
maximum absolute error 0.001953125. The generated sources of the remaining norms
were unchanged. The existing benchmark-only `ReferenceReductionBindings` can
reuse original autotuners only after checking source, compiler metadata,
arguments, alignment and alias contracts. Binding the two unchanged reduction
norms restored tile and complete RGB equality in the next controlled attempt.
That attempt then rejected a baseline schedule/argument identity change on its
first fresh video; no quality result was claimed for it.

The final private-module quality run removed the unnecessary global Dynamo
reset while retaining every identity and numerical gate. Actual 35-op coverage
ensured the new graph was used. Its natural tile, controlled tile and full cached
RGB were exact. All **8 fresh videos / 992 frames** had bitwise-equal old/new RGB,
with 49 tile callbacks and 35 × 49 fused calls per video. B1 latent T1/2/7 checks
retained the existing INT8 path, used zero new fusion calls and were exact.
CPU offload and CUDA reload preserved 144 INT8 leaves and FP32 scales; the 36
fusion flags changed 36 → 0 → 36 and the reloaded tile was exact. These are
private-module results; the cleaned production candidate is validated separately.

The no-reset change addressed a plausible cause of recompilation; the failed
attempt did not log enough argument descriptors to prove it was the sole cause.
All failed reports remain in the evidence, and no exactness threshold was relaxed.
Independent compilations still have no general bitwise-equality guarantee.

## Cleaned production candidate

The cleaned candidate restricts selection to SM120 / Triton 3.6.0 and
contiguous FP16 `[4,1797,2048]` blocks with the measured weight and norm contract.
Its seven CPU contract tests passed, covering lazy imports, target selection,
cloning without requantization, custom-op mutation/fake-output behavior, input
guards, offload flags and non-target Tensor behavior. These tests do not prove
GPU arithmetic or integration, and the candidate was not installed in `opt/`.

The first GPU driver incorrectly compared the prequantized GEMM's intermediate
2D `[7188,6144]` output with CK's final 3D `[4,1797,6144]` result. H, Q and scale
were exact, but that shape check necessarily failed. CPU inspection confirmed
both APIs' shape contracts. The failed record is retained as a harness error;
no QKV numerical conclusion follows from it. The next driver compares the
actual final 3D custom-op output without changing the kernel or threshold.

That corrected run encountered a **real numerical failure**. The original norm
had the same generated source hash but naturally selected
`XBLOCK=1, R0_BLOCK=2048, num_warps=16, num_stages=1`. Layer 1 passed every check.
At layer 2, against the original norm operating on the exact same current inputs:

| Actual production-candidate boundary | Result |
| --- | --- |
| Residual H | Bitwise equal |
| INT8 Q | 3 / 14,721,024 values differ; maximum absolute difference 1 |
| FP32 row scale | 2 / 7,188 values differ; maximum absolute difference 0.000123039 |
| Final QKV `[4,1797,6144]` | 11,587 / 44,163,072 values differ; RMSE 0.0000456246, max 0.017578125 |

Execution stopped immediately. This final run produced **no tile/full-RGB,
latency, eight-video or offload conclusion**. The QKV error above is an internal
activation error, not an RGB or visual-quality measurement. Its candidate kernel
uses the explicitly fixed eight-warp tree; matching a source formula and GPU
architecture does not establish equivalence to a different reduction schedule.
The previous controlled eight-video pass remains valid for its own private
module and schedules, but it does not validate this final attempt.

CPU inspection of the final baseline's cached TTGIR confirms
`sizePerThread=[1,4], threadsPerWarp=[1,32], warpsPerCTA=[1,16]`, compared with
the validated eight-warp `[1,8] / [1,32] / [1,8]` layout. PTX's cross-warp
butterfly stages change from XOR 4/2/1 to 8/4/2/1; both metadata records
report 64 shared bytes.
This establishes different reduction organization, without claiming a measured
hardware bottleneck or proof that this is the sole cause of the layer-2 error.

## Retained research and next step

The original [Gluon helper](benchmarks/norm1_quant_research_helper.py) is archived
byte-for-byte with the S1 source fingerprint. It is **not imported by the runtime**
and is not a drop-in replacement for arbitrary normalization schedules. Its
launch contract is `grid=(ceil(M/2),)`, eight warps, one stage and FP fusion
enabled, K2048, contiguous FP16 H/O/G/W, INT8 Q and FP32 S. It mutates H;
`DEBUG=True` also writes the normalized FP16 tensor. Captured inputs, checkpoint
weights and generated third-party reference kernels are not distributed.

The next useful experiment must first address reduction-schedule dependence.
It needs schedule-specific variants with a reliable way to select the matching
four/eight/sixteen-warp tree, or an explicitly defined fixed numerical contract
whose effect on full RGB quality is independently validated. Repeatedly recompiling until an
eight-warp baseline appears would not solve this problem. No further kernel
repair or timing was performed after the final numerical rejection.

The experiment uses ACA's budget, local gates, paired timing and failure
retention ideas. Its timings concern the complete **VAE decode stage**, with one
seed/cached input; they do not constitute a multi-seed, full-generation S4
workflow confirmation. NCU counters remain unavailable.

## Repository checks

The clean tracked-source snapshot, including the archived authored helper,
passed `compileall`, critical Ruff (`E9,F63,F7,F82`) and all **91 CPU tests**;
**14 opt-in CUDA tests were skipped**. These repository checks are separate
from the candidate's seven private CPU contract tests and the GPU experiments.

## Environment and evidence

Baseline repository `5750098` (production kernel implementation from `7bd9809`),
RTX PRO 5000 72GB / SM120 (driver label NVIDIA Graphics Device), Linux,
Python 3.12.14, PyTorch 2.11.0+cu130, CUDA 13.0, Triton 3.6.0,
comfy-kitchen 0.2.34. Seed20261003, FP32 norm enabled, FP16 accumulation disabled,
SDPA auto, cuDNN benchmark enabled/limit5, four CPU threads. All GPU experiments
ran serially under the shared benchmark lock; unrelated GPU work was retained.
Media I/O, FP16 encoding, loading and compilation are outside decoder timing.
This direct-runtime experiment does not exercise a running ComfyUI server.

[All attempts, raw samples, numerical gates and source fingerprints](benchmarks/decoder_norm1_quant_2026-10-03.json).
Local paths are removed; weights, media, captured tensors and generated
third-party kernel source are not distributed.

## 中文结论

**本轮没有把新方案接入运行时，也没有更新首页性能数字和柱状图。** 当前正式成绩仍是
INT8 encode **8.458 s**、INT8 decode **6.509 s**、融合 FP16 decode **10.923 s**。

residual/norm1/QKV 量化融合原型在已记录的 8-warps 调度下，局部完整链降低 6.98%，
完整 cached-latent decode 为 **6.513790 → 6.430177 s（降低 1.28%）**，六组均更快。
私有模块在统一未改动 norm 的调度后，也通过 8 段视频 / 992 帧逐位一致、小尺寸回退
和 CPU→CUDA 重载检查。这些研究结果均保留，但不能直接作为正式生产成绩。

最终整理后的候选遇到了相同 norm 源码的 **16-warps** 自然调度。第 2 层 residual H
仍一致，但有 **3 个 INT8 值、2 个 scale** 不同，并传递到 QKV 输出。这里比较的是
同一组实际输入，且修正了第一版诊断脚本误比二维/三维输出的错误；因此这次是真实
数值失败。已在完整 RGB 和测速之前停止，没有放宽精度门槛或反复重编译挑选配置。

下一步应先解决归约树随调度变化的问题，再验证完整视频；只固定新 kernel 的线程
布局还不足以保证它与不同的 Inductor 编译结果等价。前两次 kernel 失败、后续脚本
错误、私有模块成功结果和最终真实失败都包含在[原始证据](benchmarks/decoder_norm1_quant_2026-10-03.json)中。
