# Experimental INT8 encode / decode

Opt-in **mixed-precision** inference with real INT8 arithmetic on selected
encoder convolutions and decoder FFNs. Both options default to `false`.
This is not a fully quantized VAE and is not lossless.

## Scope and requirements

| Component | Quantized operations | Quantization | Unchanged operations |
| --- | --- | --- | --- |
| Encoder (`int8_encode`) | Eight 3×3×3 convolutions in prefix stages 0–1 | Static per-output-channel INT8 weights, dynamic per-tensor INT8 activations, INT32 accumulation | Input/output convolutions, downsampling, shortcuts, suffix, norms and other operations |
| Decoder (`int8_decode`) | 72 FFN Linear layers | Static per-output-row INT8 weights, dynamic per-row INT8 activations | Attention, norms and non-FFN operations |

The encoder uses a Triton implicit-GEMM convolution: it gathers convolution
windows directly without materializing an im2col matrix, and uses
`tl.dot(int8, int8, out_dtype=int32)`. The existing causal/spatial padding is
retained. GPU inspection confirmed integer Tensor Core instructions
(`mma.sync…s32.s8.s8.s32`), not an FP16 convolution with fake-quantized inputs.

Quantization scales remain FP32; quantized operators accept/return FP16,
and finalized RGB output is FP32. Original checkpoints are never rewritten.
No TensorRT engine, calibration dataset or offline engine build is required.
The original floating-point weights are not comprehensively removed, so
this implementation does not claim model-size or VRAM savings.

Requires NVIDIA CUDA SM80+ and FP16 runtime inputs. The decoder additionally
requires exactly `comfy-kitchen==0.2.34`. Only the hardware/software below has
been measured; CPU/ROCm are rejected and Windows is unvalidated. Unsupported
configurations fail explicitly without silently labeling an FP16 fallback
as INT8. `fast_linear` and `int8_decode` are mutually exclusive.

## Performance: four-way comparison

Measured on **NVIDIA RTX PRO 5000 72GB**, Linux, Python 3.12.14,
PyTorch 2.11.0+cu130, CUDA 13.0, Triton 3.6.0, comfy-kitchen 0.2.34.
Other GPU work remained running. This is a configuration-specific result,
not a cross-hardware speedup guarantee.

Input: sample 00015, **768×1344×124**, 24 fps. Encoder/decoder tile256,
encoder staged batch4, decode batch2, SDPA auto, both encoder and decoder
compiled; decoder compile mode `max-autotune-no-cudagraphs`.
cuDNN benchmark enabled, benchmark_limit5.

Two independent runtimes encode the **same FP16 pixel tensor** into E0
(default) and E1 (INT8) latents. D0 and D1 decode both latents to isolate
encoder and decoder effects. Public quality latents come from the compiled
paths; temporary uncompiled hooks are used only for diagnostics.

Each path receives two warmups and three timed calls. Encode order alternates
AB/BA; decode order rotates among four paths. CUDA Event timings include
per-call dynamic quantization and runtime output finalization. Loading,
static weight preparation, compilation, media I/O and quality metrics are
excluded.

| Path | Encode (s) | Decode (s) | Sum (s) | Change in sum vs default |
| --- | ---: | ---: | ---: | ---: |
| E0/D0: default | 11.946492 | 11.476686 | 23.423178 | — |
| E0/D1: INT8 decode only | 11.946492 | 8.277575 | 20.224067 | −13.66% |
| E1/D0: INT8 encode only | 12.537563 | 11.475950 | 24.013512 | +2.52% |
| E1/D1: both INT8 | 12.537563 | 8.278192 | 20.815755 | −11.13% |

The sum adds separately measured means; it is **not** a timed complete
video-generation pipeline. Decoder-only reduces decode latency by **27.87%**.
The INT8 encoder is **4.95% slower**, despite real integer execution.
It is included for experimentation, not recommended as an encode speedup.
For the best measured speed/quality trade-off, enable only `int8_decode`.

Raw CUDA Event samples (seconds):

| Operation | Run 1 | Run 2 | Run 3 |
| --- | ---: | ---: | ---: |
| E0 | 11.948858 | 11.939396 | 11.951222 |
| E1 | 12.537477 | 12.540232 | 12.534979 |
| D0(E0) | 11.486016 | 11.473120 | 11.470922 |
| D1(E0) | 8.282182 | 8.277327 | 8.273217 |
| D0(E1) | 11.480965 | 11.474015 | 11.472869 |
| D1(E1) | 8.284222 | 8.276388 | 8.273967 |

Process peak allocations with **both runtimes resident**: default encode
11,262,686,720 bytes; INT8 encode 11,421,322,752 bytes; decode
14,040,589,312 bytes. These are not isolated per-model VRAM measurements.
ComfyUI native and TensorRT were not rebenchmarked in this experiment;
do not combine these numbers with historical tables to infer new speedups.

## Quality method

PyAV decodes each source video to exact uint8 RGB /255 FP32 references.
Encoder inputs are separately normalized to [-1,1] and cast to FP16;
both encoders receive identical inputs. Their public FP32 latents are cast
to FP16 before each decoder. There is no resize, crop or frame dropping.

PSNR uses uncompressed FP32 RGB [0,1] before media encoding:
`PSNR = -10*log10(MSE)`, with all RGB pixels in each frame.
“Mean” is the arithmetic mean of frame PSNRs, not PSNR of the average MSE.
The benchmark also records global MSE/PSNR, per-frame metrics and counts
below 30 dB. Exactly 30 dB passes.

### Timed sample: 00015

| Path | Mean PSNR vs source | Minimum vs source | Mean vs default reconstruction | Minimum vs default |
| --- | ---: | ---: | ---: | ---: |
| Default | 34.852 | 33.694 | — | — |
| INT8 decode only | 34.710 | 33.569 | 49.248 | 48.529 |
| INT8 encode only | 34.198 | 33.222 | 42.319 | 40.872 |
| Both INT8 | 34.066 | 33.107 | 41.467 | 40.170 |

All entries are dB. All 124 frames passed 30 dB in all seven comparisons.
Source-reconstruction mean drops by 0.142 dB with decoder-only, 0.654 dB
with encoder-only, and 0.786 dB with both. Encoder latent relative RMSE
is **10.65%**: encoder quantization is not numerically near-lossless.
Default encode/decode control repeats had zero error on this sample.

These are video reconstruction tests, not original generation latents,
and 30 dB is a screening threshold—not proof of unchanged visual detail,
tile seams or temporal consistency. Model weights and source media are
not redistributed.

## Eight-video quality suite

All eight supplied videos were checked independently with compiled encoder
and decoder, the same tile/batch settings, and no resize/crop/frame drop.
This quality-only suite skips explicit warmups and timing; its results are
separate from the timed sample above. Each clip has 124 frames at 24 fps.

| Path | Mean vs source | Minimum vs source | Mean vs default | Minimum vs default | Frames below 30 dB |
| --- | ---: | ---: | ---: | ---: | ---: |
| Default | 35.110 | 32.711 | — | — | 0/992 |
| INT8 decode only | 34.963 | 32.618 | 49.585 | 48.542 | 0/992 |
| INT8 encode only | 34.486 | 32.338 | 42.589 | 39.478 | 0/992 |
| Both INT8 | 34.355 | 32.235 | 41.767 | 38.971 | 0/992 |

All PSNR values are dB; the zero counts apply to both source and default
comparisons where defined. Mean source-reconstruction PSNR falls by
**0.146 / 0.623 / 0.755 dB** for decoder-only / encoder-only / both.
Encoder latent relative RMSE ranges from **8.05% to 12.07%**.

An independent audit recomputed all 6,944 per-frame JSON/CSV comparison rows.
Default encode/decode repeats had zero error in every video. Each video checked
1,792 calls across eight INT8 encoder convolutions and 7,056 calls across 72
INT8 decoder linears for each encoder's latent; all inputs/outputs were finite.

Per-video mean source-reconstruction PSNR:

| Sample | H×W×frames | Default | INT8 decode | INT8 encode | Both | Both minimum |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 00007 | 768×1376×124 | 35.505 | 35.356 | 34.794 | 34.661 | 33.473 |
| 00008 | 768×1376×124 | 35.512 | 35.344 | 35.110 | 34.955 | 34.262 |
| 00009 | 768×1376×124 | 34.794 | 34.655 | 34.294 | 34.173 | 33.264 |
| 00010 | 768×1376×124 | 35.294 | 35.125 | 34.314 | 34.166 | 32.868 |
| 00013 | 768×1344×124 | 35.128 | 35.008 | 34.621 | 34.516 | 33.853 |
| 00014 | 768×1344×124 | 34.225 | 34.099 | 33.744 | 33.620 | 32.235 |
| 00015 | 768×1344×124 | 34.852 | 34.709 | 34.198 | 34.066 | 33.102 |
| 00016 | 768×1344×124 | 35.568 | 35.411 | 34.816 | 34.679 | 33.868 |

Samples 00013/00014 use 20 steps (seeds 42/43); 00015/00016 use 50 steps
(seeds 43/42). These labels identify supplied media, not benchmark parameters
that change VAE execution.

Public numeric evidence: [raw timing and configuration](benchmarks/int8_2026-09-22_timing.json)
and [eight-video aggregates and per-video results](benchmarks/int8_2026-09-22_quality.json).
The benchmark retains full per-frame JSON/CSV locally and generates the same
artifacts when reproduced. Public evidence excludes local paths and media.
After measurement, only initialization GPU-device validation and explanatory
text changed; quantization and convolution arithmetic stayed unchanged.

## ComfyUI compatibility validation

Independent-process checks use the real Comfy VAE wrapper and model patcher;
they do not submit jobs to a running ComfyUI server. With matching input
values **and layout**, wrapper encode/decode exactly matched direct runtime
calls. CPU offload followed by CUDA reload preserved outputs exactly. All
eight encoder and 72 decoder modules retained INT8 weights, FP32 scales and
the expected device in all three states.

Inputs covered FP16 IMAGE tensors with 17 frames at 256×256, and a standard
FP32 single-image input at 256×256. The 17-frame probe is a component-level
compatibility check, **not** a 17-frame roundtrip quality result: H3 encodes
that length to two latent frames, whose decode produces five video frames.
The eight-video quality suite uses supported 124-frame sequences.

The single-image check also decoded the actual wrapper-encoded latent before
and after offload: output was finite, 256×256×1, and exactly matched direct
runtime decode. This validates the sample's encode→decode connection, not
perceptual quality or a server-level workflow execution.

Keep input layout fixed when comparing results. A separate random-input
diagnostic found that making Comfy's channels-last input NCDHW-contiguous
changed encoder output despite identical pixel values: latent RMSE was
0.000652 for the default encoder and 0.029665 for the INT8 encoder.
INT8 is more layout-sensitive in this test. Matching Comfy's actual input
layout restored exact wrapper/direct equality; production layout behavior
was not changed to mask the difference. The video tables above are direct
runtime measurements, not server-level workflow quality guarantees.

## ComfyUI usage

Install the ordinary project requirements in ComfyUI's Python environment,
then the optional decoder dependency:

```bash
python -m pip install 'comfy-kitchen==0.2.34'
```

In `H3VAEPyOptLoader`, set `dtype=fp16`, `fast_linear=false` and choose
`int8_encode` / `int8_decode` independently. Existing workflows omitting
both options keep the original behavior. Standard `VAE Encode` and
`VAE Decode` nodes consume the same VAE output.

- [Both-options image roundtrip API example](../examples/minimal_h3vae_pyopt_int8_roundtrip_prompt.json):
  upload a 256×256 image, replace the image filename, and configure model paths.
- [Decoder-only API example](../examples/minimal_h3vae_pyopt_int8_prompt.json):
  zero-latent wiring demonstration, not a quality test.

The image example only demonstrates wiring; use videos for temporal checks.

## Reproduce

Use the matching MiniMax H3 model code and FP16 weights. Video benchmarks
also need `av` and `numpy`. No ComfyUI server is required.

```bash
python bench_int8_roundtrip.py \
  --video /path/to/sample.mp4 \
  --model-code-dir "$H3_VAE_MODEL_CODE_DIR" \
  --weights "$H3_VAE_WEIGHTS_PATH" \
  --decoder-tile 256 --encoder-tile 256 \
  --tile-batch 2 --encoder-staged-batch 4 \
  --warmup 2 --runs 3 --per-frame-csv \
  --output-dir results/int8_roundtrip
```

For all MP4s in a directory, replace `--video …` with
`--videos-dir /path/to/videos`. Add `--quality-only --warmup 0` to skip
steady-state timing and explicit warmups while still checking the compiled
runtime outputs. Use a fresh output directory; overwriting requires an
explicit option.

JSON records configuration, source/weight/code hashes, timing samples,
latent differences, module execution checks and every frame's PSNR.
`per_frame_metrics.csv` contains all seven comparisons.
`bench_int8_vae.py` remains available for decoder-only A/B.

Weight SHA256:
`7c1f131492e7eddacaac9069a61b81bdd39de5cc96561e677c5eab1cdce5e522`.

CPU CI checks importability, option guards, quantization contracts and
benchmark helpers. It does not establish GPU correctness or image quality;
see [testing boundaries](testing.md).
