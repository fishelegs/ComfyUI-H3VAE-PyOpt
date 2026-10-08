# INT8 decode: exact norm1 fusion and owned normalization (2026-10-08)

The fixed SM120 candidate reduces complete decode from **6.518516 to
6.440266 seconds (1.20%)**, winning all six paired blocks. Finalized RGB is
bitwise identical on the retained natural full-video input and peak allocated
memory is unchanged. This result includes the residual/norm1/QKV-quantization
fusion and two independently implemented normalization operations; it does
not isolate the contribution of norm1 alone.

Production acceptance passed, and the runtime now selects this recipe on
the bounded hardware/software contract below. Eight native videos, three
public-factory holdout blocks and two actual ComfyUI offload/reload smokes
passed in addition to the primary performance screen.

## Complete decode performance

Each block contains two calls per variant, in APPA or PAAP order. A is the
previous real 144-linear INT8 decoder; P has the norm1 bundle. Both models
decode exactly the same retained FP16 channels-last latent. Loading, initial
compilation, media I/O, encoding and quality measurement are excluded;
allocation, dynamic activation quantization, tile stitching and RGB
finalization are included in CUDA Event timing.

| Block | Order | A (s) | P (s) | Lower latency |
| --- | --- | ---: | ---: | ---: |
| 1 | APPA | 6.494505 | 6.418752 | 1.166% |
| 2 | PAAP | 6.511747 | 6.433896 | 1.196% |
| 3 | APPA | 6.521232 | 6.443660 | 1.190% |
| 4 | PAAP | 6.527458 | 6.448269 | 1.213% |
| 5 | APPA | 6.526813 | 6.449230 | 1.189% |
| 6 | PAAP | 6.529341 | 6.447787 | 1.249% |
| **Mean, 12 calls per variant** | | **6.518516** | **6.440266** | **1.200%** |

The gain is **78.250 ms** per complete decode. Peak allocated memory is
**5,120,063,488 bytes** for both variants. All 24 timed outputs are finite;
the post-timing A/P RGB comparison is exact, A still reproduces the retained
RGB hash, and latent/weight contents are unchanged. There are no observers,
counter wrappers, shadows or normalization dispatch bindings during timing.
Dynamo's captured-call and unique-graph counts remain 1519 and 2.

The acceptance threshold was at least 0.5% lower latency and at least five
of six winning blocks. This is a standalone VAE screen on one fixed latent,
not a complete generation workflow result or a three-seed ACA S4 result.
The historical **6.508537 s** FFN record is a different experiment; it is
not the baseline used to compute this round's 1.20% gain.

## What changed

- Blocks 1–35 carry the preceding FFN residual into a fixed W16 producer.
  It updates the FP16 residual and produces INT8 QKV input plus FP32 row
  scales in one kernel, avoiding materializing and rereading normalized FP16
  activations. The existing prequantized real INT8 GEMM consumes them.
- Block 0 uses an independently authored FP16 RMS kernel with the retained
  X2/R2048/W16 arithmetic. Its norm, INT8 input, scales and QKV are exact
  against the independently compiled original path.
- The last block carries its residual to an independently authored W8 final
  LayerNorm. It computes unrounded FP32 FMA residual statistics, Welford
  variance, reciprocal square root and affine FMA, then writes its own FP16
  output. Torch Inductor's Welford helper remains a lazy dependency; no
  third-party implementation body or captured compiled producer is copied.

The producer preserves the original packed square-sum reduction, FP16
residual rounding, FP16 normalized-input rounding, comfy-kitchen's FP16
quantization denominator and rounded INT8 values. An apparently equivalent
RMS expression with another reduction order did not pass earlier quality
checks. These arithmetic details are part of the measured contract.

The selected final kernel has **33 registers, zero spills and 96 bytes of
shared memory**. Its compiled TTGIR SHA-256 is
`b63a115d8403dbd3750ba4eea0976105e2a23f75eb2b30d1b2646a0793acc13a`;
the active compiled object matches the artifact in P's independent cache.
These are compiler resource records, not measured occupancy or bandwidth.

## Selection and integration contracts

The runtime selects this recipe for `int8_decode=true`,
`decode_fusions=true`, single-device decoding on **SM120 / PyTorch 2.11.0 /
CUDA 13.0 / Triton 3.6.0 / comfy-kitchen 0.2.34**. The fused kernels require
contiguous FP16 `[4,1797,2048]` decoder states. Other tile shapes keep the
existing real INT8 implementation; other supported software/hardware keeps
the preceding decoder implementation.

All **585 decoder tensors and 144 real INT8 weight storages** retain their
contents. Weights are reused without requantization and original checkpoints
are untouched. The final carry requires zero sequence padding and rejects
nonzero padding. Ordinary LayerNorm fallback reads the adapter's current
parameters, including after dtype/device conversion. GPU/CPU transitions
refresh both norm1 and FFN fusion selection.

FP16 remains the default. This change does not remove the original INT8
versus FP16 quality loss or introduce support for other platforms.

## Correctness evidence

The independent authored-kernel window passed its helper/NaN probe, exact
first-tile and block0 norm/Q/scale/QKV checks, and natural full RGB equality
for 124 frames. P returns its own helper outputs. Original-kernel shadows
operate only on clones during validation, and do not supply P's output.
Full target coverage is 49 block0 calls, 49 final calls and 1715 residual
calls: 35 norm1 layers for each of 49 tile groups.

The required CPU checks passed on a clean public-source snapshot:
compileall, critical Ruff checks and **99 tests passed / 14 CUDA tests
skipped**. Eight new CPU tests cover lazy imports, device/stack selection,
metadata/fake-output contracts, parameter sharing and conversion, and safe
unpadding. CPU checks do not prove GPU numerical equivalence.

## Public integration acceptance

A and P are constructed through the actual public runtime factory. A only
disables the new recipe during construction and retains all 144 real INT8
linears. P uses the unmodified new selection policy. They have independent
compiled graphs and cache ownership, and identical decoder weight contents.
The public kernel function/launch definitions match the private candidate
used in the primary timing window; source fingerprints distinguish the
private experiment and final public integration.

Each native video is encoded once by the FP16 encoder. The same freshly
encoded FP16 channels-last latent is decoded by A and P. No frames are
dropped and no resolution is changed. All **992 frames** are bitwise equal,
finite and within RGB [0,1]; the primary 55 dB, strict 65 dB and source-delta
gates pass. Each video covers exactly **49 / 49 / 1715** block0/final/residual
calls. Against the source video, both variants have these identical scores:

| Video index | Native H×W×frames | Mean frame PSNR | Worst frame PSNR | Below 30 dB |
| --- | --- | ---: | ---: | ---: |
| 0 | 768×1344×124 | 34.683 dB | 33.559 dB | 0 |
| 1 | 768×1376×124 | 35.334 dB | 34.000 dB | 0 |
| 2 | 768×1376×124 | 35.319 dB | 34.567 dB | 0 |
| 3 | 768×1376×124 | 34.629 dB | 33.546 dB | 0 |
| 4 | 768×1376×124 | 35.097 dB | 33.582 dB | 0 |
| 5 | 768×1344×124 | 34.991 dB | 34.266 dB | 0 |
| 6 | 768×1344×124 | 34.079 dB | 32.590 dB | 0 |
| 7 | 768×1344×124 | 35.386 dB | 34.397 dB | 0 |

Mean per-frame source PSNR is **34.939796 dB** for both. This compares the
new and previous INT8 decoders; no new FP16-versus-INT8 quality claim is made.

After all validation hooks are removed, three preregistered latent inputs
receive one APPA/PAAP block each, with two timed calls per variant per block:

| Video index | Order | Public A (s) | Public P (s) | Lower latency |
| --- | --- | ---: | ---: | ---: |
| 0 | APPA | 6.499056 | 6.424951 | 1.140% |
| 5 | PAAP | 6.506916 | 6.429574 | 1.189% |
| 7 | APPA | 6.510321 | 6.435123 | 1.155% |

All three blocks are faster, with no recompilation, finite timed outputs
and exact A/P RGB comparisons. These holdouts validate the public integration;
the headline **6.440266 s** remains the primary six-block mean, rather than
selecting a smaller holdout or individual sample as the current score.

ComfyUI SDK `c1716a458d7978d37ec28608b7587887de5ee3f7` passes both the
17-frame 256×256 non-target input and 124-frame 672×672 target input, including
FP32 IMAGE input, actual encode→decode and wrapper/direct comparisons.
Actual `patcher.unpatch_model(cpu)` and `patcher.load(cuda, full_load=True)`
retain the contents of all 585 decoder tensors and 144 quantized weights.
Norm1 and FFN fusion flags are gated to **36→0→36**. All wrapper/direct and
before/after offload metrics have **RMSE/max_abs = 0**. The small smoke invokes
none of the new helpers; the target smoke records **280 block0, 280 final and
9800 residual** calls across its ten complete decodes. This random-input
smoke verifies integration separately from the eight-video quality screen.

The first acceptance attempt stopped before encoding video 1 because its
script assumed every input was 1344 pixels wide. Its saved anchor and video 0
had already passed. The separate corrected window freezes each native shape;
kernel code, quality gates and timing budget were unchanged. The public
acceptance archive preserves this partial failed attempt and its terminal
review instead of replacing it with the successful result.

## Measurement environment

RTX PRO 5000 72GB, SM120 (driver reports NVIDIA Graphics Device); Linux,
Python 3.12.14, PyTorch 2.11.0+cu130, CUDA 13.0, Triton 3.6.0,
comfy-kitchen 0.2.34. Input **768×1344×124**, encoder/decoder tile256,
encoder staged batch4, decoder batch4, seed20261003, SDPA auto, FP32 norms,
FP16 accumulation disabled, cuDNN benchmark enabled / limit5,
`max-autotune-no-cudagraphs`. Two warmups per variant precede 12 timed calls
per variant. Source HEAD is `7946aaabc69875ed4fd743bf5c91278e86410f9c` with
the explicitly hashed candidate files; the measurement is not claimed to
come from unchanged committed source.

Cold model loading/static quantization took 23.05 s. First warmup including
compilation took 18.14 s for A and 13.69 s for P; these costs are excluded
from the steady-state result and are not a separate cold-start benchmark.
Other GPU work remained running and was not rescheduled. The 250 ms
telemetry contains 963 samples: SM clock 2415–2497 MHz, temperature 56–70°C,
power 233.213–353.149 W. Protected cache manifests, hooks and model contents
were restored or unchanged after the run.

## Reproduce

Use the original model code and weights, the measured software stack and
`MINIMAX_H3_VAE_DECODER_VIT_FP32_NORM=1`. A candidate configuration check is:

```bash
OMP_NUM_THREADS=4 python bench_pyopt_vs_trt.py --pyopt-only \
  --decode-fusions --int8-decode \
  --model-code-dir "$H3_VAE_MODEL_CODE_DIR" --weights "$H3_VAE_WEIGHTS_PATH" \
  --height 768 --width 1344 --frames 124 \
  --encoder-tile 256 --decoder-tile 256 --staged-batch 4 --tile-batch 4 \
  --warmup 2 --runs 12 --seed 20261003 \
  --output results/norm1_candidate.json
```

This public command measures the candidate on synthetic input; it does not
replay the private retained-video paired experiment or establish its quality
result. For a paired replication on your own video, freeze one freshly
encoded FP16 `channels_last_3d` latent and feed the identical tensor to both
variants. Construct A by temporarily disabling only the new recipe selection:

```python
from unittest.mock import patch
from h3vae_runtime import H3VAEPyOptRuntime
from opt import int8_norm1_quant

# kwargs: identical model, device, dtype, tiles, compile flags and
# int8_encode=False, int8_decode=True, decode_fusions=True, fast_linear=False.
with patch.object(int8_norm1_quant, "supports_norm1_quant", return_value=False):
    baseline = H3VAEPyOptRuntime(**kwargs).eval()
candidate = H3VAEPyOptRuntime(**kwargs).eval()
```

Keep independent graph/cache ownership, warm both variants, remove validation
observers before timing and use the six block orders above. Require finite,
bitwise-equal full RGB and unchanged weights before accepting a speed result.
Benchmark JSON records the retained input/weights/code hashes without
distributing those assets.

## Evidence and remaining work

- [Independent authored normalization and natural RGB checks](benchmarks/decoder_norm1_final_authored_2026-10-08.json)
  (archive SHA-256 `e96ce4dead1fe517f95228d05cf461d6f1674e01792fa6babcce3d8737440e48`).
- [Complete decode timing, all samples and telemetry](benchmarks/decoder_norm1_full_timing_2026-10-08.json)
  (archive SHA-256 `ba065e7c6eb3c94f977ce47b5c09cada6d3d29ba692b80c559bab0097df0a8d1`).
- [Earlier block0/final-LayerNorm causal control](decoder_norm1_block0_2026-10-08.md).
- [Public integration, native videos, holdouts and Comfy acceptance](benchmarks/decoder_norm1_production_shapes_2026-10-08.json),
  including the earlier native-shape validation failure and the root adoption
  decision. The runner's raw pre-adoption flag remains unchanged; the later
  root terminal review records adoption of the fully validated implementation.

Public exports preserve numeric, boolean and SHA leaves per record while
removing private paths/media, tensors and generated third-party bodies.
They retain the original pre-adoption status of each completed window.

## Current Nsight profile and next targets

After production acceptance, Nsight Systems 2025.3.2 captured one complete
decode through the unmodified public candidate runtime, following two
unprofiled warmups. The separate private cache was copied from the accepted
cache; only its graph ownership was reset. The protected source cache,
139 frozen inputs and decoder tensor contents remained unchanged. All three
outputs were finite and bitwise equal to the saved RGB anchor.

The `H3VAE_NORM1_PROFILE` interval contains **23,662 kernels**, with no
straddling calls. Expected calls match for residual/norm1 quantization
(1715), block0 RMS (49), final LayerNorm (49), FFN-up (1764) and its output
quantizer (1764). Their presence confirms the production recipe is active.
The named interval is **6391.867 ms**, with **6313.353 ms** of kernel activity;
this single profiled decode is diagnostic evidence. It does not replace the
paired **6.440266 s** headline or establish another speed improvement.

| Remaining work | Kernel time | Share of summed kernel time |
| --- | ---: | ---: |
| Fused FFN-up / SwiGLU | 2241.965 ms | 35.51% |
| Two dominant CUTLASS INT8 visitor GEMM launch groups | 2175.133 ms | 34.45% |
| Flash attention | 1159.862 ms | 18.37% |
| FFN output quantization | 235.964 ms | 3.74% |
| QK normalization / RoPE | 125.021 ms | 1.98% |
| Residual / norm1 quantization | 99.608 ms | 1.58% |

FFN-up, the two GEMM groups and attention account for **88.34%** of the
summed kernel time. The remaining norm1 producer has a much smaller budget.
The next bounded experiment should therefore prioritize FFN-up and GEMM
schedules, preserving the existing INT32 accumulation and FP16 rounding.
The FFN-up launch reports **238 registers/thread** and **49,152 bytes** of
dynamic shared memory; these are useful starting points for a resource and
software-pipeline audit, but they do not establish an occupancy or stall
bottleneck. Any candidate still needs real-input local replay, paired full
decode timing and the existing video/Comfy acceptance before adoption.
Encoder and FP16 performance were not remeasured in this round.

The first NVTX-trigger attempt produced no trace and is preserved as a failed
capture. The separate successful window uses the documented
[CUDA profiler API capture trigger](https://docs.nvidia.com/nsight-systems/UserGuide/index.html).
[Public profile records, full symbols, launch geometry and capture identities](benchmarks/decoder_norm1_nsys_api_2026-10-08.json)
include both attempts and the independent terminal review; trace binaries
remain private. The 78.514 ms without this process's kernel activity inside
the named interval is not claimed as device idle time. NCU hardware counters
remain unavailable (`ERR_NVGPUCTRPERM`), so this profile makes no bandwidth,
occupancy or stall claim, and no full-generation ACA S4 claim.
