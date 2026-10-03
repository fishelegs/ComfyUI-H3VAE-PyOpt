# Norm1 numerical-contract follow-up — 2026-10-03

**No new runtime optimization is enabled by this follow-up.** A separately
implemented 16-warp producer now matches the original 16-warp operation
exactly and passes a local performance screen. Complete-decoder validation
then stopped at a newly selected, unsupported two-warp reduction schedule,
before numerical shadows or timing. The current records remain INT8 encode
**8.458108 s**, INT8 decode **6.508537 s** and fused FP16 decode **10.923379 s**.

This extends the [previous norm1 fusion experiment](decoder_norm1_quant_2026-10-03.md).
The research helper is not imported by the runtime and does not add a Loader
option. FP16 defaults, quantization scope and competitor charts stay unchanged.

## Fixed eight-warp contract: quality screen rejected

The first experiment assessed the existing eight-warp fusion as an explicit
fixed numerical contract. Its canonical reference was the independently
archived original generated norm expression, launched with XBLOCK2,
R0_BLOCK2048, eight warps, one stage and FP fusion. Public comfy-kitchen
quantization and linear APIs checked the reference; helper debug output was
not the sole oracle. All 35 target layers passed exact H, Q, FP32 scale and
QKV shadows. Cached RGB checks passed.

The first fresh video used 124 frames at 768×1376 with no resize, crop or frame
drop. Natural old/new graphs passed the declared screen. Binding only the two
unchanged reductions, norm0 and the final norm, made the candidate RGB exact
against the natural eight-warp baseline. These benchmark bindings are not a
production feature.

A stress comparison changed only the original target norm1 launch to a
16-warp or four-warp schedule and compared it with that controlled candidate:

| Comparison against fixed-W8 candidate | Mean / global / worst-frame RGB PSNR | Max absolute RGB delta | Temporal delta RMSE | Mean source PSNR change, candidate minus reference |
| --- | --- | ---: | ---: | ---: |
| Natural independently compiled graphs | 58.861 / 58.473 / 54.705 dB | 0.100188 | 0.001158 | −0.000546 dB |
| Same unchanged reductions, original W8 target | Exact | 0 | 0 | 0 dB |
| Original target forced to W16 | 53.435 / 53.323 / 51.472 dB | 0.117400 | 0.002106 | −0.000417 dB |
| Original target forced to W4 | 53.625 / 53.508 / 51.595 dB | 0.109375 | 0.002073 | −0.000143 dB |

The matched-unchanged-norm/stress screen required mean and global comparison
PSNR ≥65 dB, worst-frame ≥60 dB, max absolute RGB delta ≤0.02 and temporal
RMSE ≤0.00075. The primary natural-graph screen used 55/55/50 dB, 0.15 and
0.003 respectively. Both also required no more than 0.01 dB mean or 0.05 dB
per-frame source-PSNR loss and no additional frames below 30 dB. Natural and
controlled screens passed; both cross-schedule stress screens failed.

These are pairwise screening criteria, not a perceptual-quality guarantee.
The very small source-PSNR changes do not establish visible degradation.
Nevertheless, the predeclared rejection was retained: the eight-video suite
stopped after this first video and **no timing ran**. No threshold was lowered.
It is not valid to advertise fixed W8 as lossless against arbitrary old schedules.

## An exact, separate W16 recipe

Static PTX tracing established the original W16 layout as
`[1,4] / [1,32] / [1,16]`: one row, four consecutive values per thread and
512 threads per CTA. For lane-local raw residuals x0–x3, the low packed lane is:

```text
s1 = mul_rn(x1, x1)
a  = fma_rn(x0, x0, s1)
b  = fma_rn(x2, x2, a)
s3 = mul_rn(x3, x3)
p  = add_rn(s3, b)
```

The warp butterfly uses offsets 16/8/4/2/1. Sixteen warp partials then use
shared memory and offsets 8/4/2/1. The old packed PTX's unused high lanes
contain undefined values; the final low-lane extraction makes them irrelevant.
The recipe does not assume those registers are zero.

The new Gluon helper pins this layout and local arithmetic explicitly.
Statistics use the unrounded FP32 FMA residual; H and norm are each rounded
to FP16 at their original boundaries. CK0.2.34 rowwise rounding and FP32
scales are retained, followed by the existing INT8 QKV GEMM. It is a distinct
W16 recipe, not a threshold repair of the rejected fixed-W8 experiment.

Two real captured B4 tile groups, plus M1/17/129 row slices, passed exact
H, FP16 norm, Q, FP32 scale and QKV output checks against the independently
archived original W16 expression and public CK APIs. Both debug and non-debug
specializations were tested; owned inputs and original weights were unchanged.
Historical W8 capture hashes were diagnostic only, since W8 is not the W16
oracle. The archived candidate IR/PTX also confirms the intended tree.

| W16 local chain | Original W16 + public quantization | Fused W16 |
| --- | ---: | ---: |
| CUDA event | 0.596367 ms | 0.569963 ms |
| Synchronized host wall | 0.616892 ms | 0.592661 ms |
| Registers, original / non-debug candidate | 26 | 27 |
| Shared bytes / spills | 64 / 0 | 64 / 0 |

Each call includes an identical H ownership clone, allocations, residual/norm,
quantization and QKV GEMM. Three warmups per variant precede six alternating
ABBA/BAAB blocks, ten calls per sample. The event saving is **0.026405 ms
(4.43%)**, with **five of six** blocks faster, meeting the declared ≥0.02 ms
and ≥five-winning-block local screen. Host-wall results are reported
separately; a local gain is not a full-decoder gain.

## Complete decoder: unsupported schedule, no timing

The next plan first required natural-schedule shadows using the matching W8
or W16 producer, then intended to compare original W16 and fused W16 within
one unchanged compiled decoder. All eight fresh-video RGB checks had to be
bitwise exact before six paired full-decode timing blocks. Target W16 would
be controlled in both variants; the remaining norms would retain the same
original graph. Even success would be a controlled research measurement,
not a natural, unhooked production latency record.

The first observed natural launcher had the same generated source SHA,
FP-fusion setting and one stage, but selected:

```text
XBLOCK=1, R0_BLOCK=1024, num_warps=2
```

This is outside the declared supported schedules. The identity/configuration
gate stopped immediately: **zero numerical shadows, zero fresh videos and
zero timed samples**. It is not evidence that the W16 kernel is numerically
wrong or slower. The selected launch records 56 registers, zero spills and
8 shared bytes; these are resources, not measured occupancy or stall data.

R0_BLOCK1024 introduces two source-loop segments for width2048. It must be
validated as a separate arithmetic schedule. The current evidence shows that
matching only the source hash or supporting eight and sixteen warps is
insufficient for runtime adoption. Independent compilation can also change
the two unchanged reductions, as the first experiment demonstrates.

## Evidence and remaining work

CPU verification passed in a clean tracked-source snapshot: compileall, critical
Ruff and 105 unit tests (91 passed; 14 CUDA tests skipped). These CPU checks do
not establish full-decoder or ComfyUI compatibility for the new helper.

The [redacted machine-readable archive](benchmarks/decoder_norm1_contract_2026-10-03.json)
retains plans, quality failures, static audits, local timings and the complete
validation's early rejection, with numeric/boolean/SHA leaf preservation.
The [authored W16 research helper](benchmarks/norm1_quant_w16_research_helper.py)
has SHA-256 `e410a15bce556e6e38017fae14874d32182b0e0defbfca7c70c0ee1b336bd63b`.
Private captures/media, weights, local absolute paths and generated third-party
norm source are excluded.

Environment: RTX PRO 5000 72GB / SM120, Linux, Python3.12.14,
PyTorch2.11.0+cu130, CUDA13.0, Triton3.6.0, comfy-kitchen0.2.34,
FP32 norm enabled, FP16 accumulation disabled, tile256 / batch4, SDPA auto,
seed20261003. Baseline repository commit: `9131353`. Compilation, loading,
static weight preparation, media I/O and validation are outside local timing.
GPU runs held a shared benchmark lock; unrelated GPU work was left running.
This round did not measure ComfyUI wrapper/offload behavior for a new runtime.

The [latest saved Nsight attribution](decoder_int8_ffn_2026-10-03.md) still
places about 35.34% of decoder kernel activity in FFN-up/SwiGLU, 18.24% in
attention and 15.39% in FFN-down; recurrent norm contributes about 1.97%.
These activity fractions are not theoretical speedups. NCU counters remain
unavailable, so no hardware stall or occupancy cause is claimed here.
Before deploying norm1 fusion, the remaining requirement is a defined
schedule-selection contract that handles or explicitly rejects additional
reductions without silently changing INT8 arithmetic, followed by natural
real-module, video, performance and ComfyUI offload/reload validation.
