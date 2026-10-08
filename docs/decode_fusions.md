# Opt-in decode fusions

The existing `H3VAEPyOptLoader` and `H3VAEPyOptRuntime` accept
`decode_fusions=true`. This is an **unreleased opt-in feature**; omitted/false
preserves v0.2.0 behavior, including its 72-linear INT8 decoder.
No checkpoint conversion, extra node, TensorRT engine or global `--fast` is needed.

## Profiles

| Setting | Floating-point fusions | INT8 mixed-precision fusions |
| --- | --- | --- |
| `dtype` | `fp16` | `fp16` |
| `decode_fusions` | `true` | `true` |
| `int8_decode` | `false` | `true` |
| Measured `tile_batch` | `8` | `4` |
| `int8_encode` / `fast_linear` | `false` / `false` | `false` / `false` |
| Decoder changes | 36 fused w1 GEMM + SwiGLU blocks | 72 FFN + 36 QKV + 36 output-projection INT8 linears |

Both retain FP16 encode, FP32 decoder normalization and FP32 RGB output.
FP16 GEMM uses FP32 accumulation; rounding/reduction order can still differ.
INT8 fuses SwiGLU/row quantization and residual/RMSNorm/row quantization,
with static INT8 weights and FP32 scales. Its prequantized GEMM uses a
**private comfy-kitchen 0.2.34 CUTLASS entry point**, intentionally pinned and
validated separately; a missing/rejected backend fails rather than falling back.
Attention computation, QK normalization and RoPE remain floating point.

Both fuse FP32 pixel conversion, multiplication, addition and clamp without
FMA contraction. This finalization alone preserves finite output bits at a
fixed decoder/batch; the complete profiles are **not** bitwise-equivalent to
v0.2.0. Original checkpoint tensors on disk are never modified. No model-size
or deployment-VRAM saving is claimed.

Requires NVIDIA CUDA SM80+, Triton, `compile_decoder=true`, and the default
`MINIMAX_H3_VAE_DECODER_VIT_FP32_NORM=1`. INT8 also requires exactly
`comfy-kitchen==0.2.34`. CPU, ROCm, BF16, uncompiled decode, `fast_linear`
and `int8_encode` combinations are rejected for these profiles. The batch
settings above are explicit user choices, not new defaults.

## SM120 INT8 norm1 follow-up (2026-10-08)

The INT8 profile now automatically selects the residual/norm1/QKV
quantization bundle for contiguous FP16 `[4,1797,2048]` states on the validated
SM120 / PyTorch 2.11.0 / CUDA 13.0 / Triton 3.6.0 stack. It owns the block0
RMS and final LayerNorm arithmetic and keeps the 144-linear quantization
scope, FP32 scales and existing FP16 rounding.

The primary six-block comparison measured **6.518516 → 6.440266 s (−1.20%)**
with unchanged peak allocated memory. Independent public-factory graphs
retained bitwise-equal natural RGB on 8 videos / 992 frames, including native
1344 and 1376 widths. Three holdouts were faster; both Comfy target/fallback
smokes passed actual CPU offload/CUDA reload with **36→0→36** fusion flags
and unchanged output/weight contents. Other shapes and supported stacks use
the preceding real INT8 path. Original INT8 loss remains. See
[implementation, timing and acceptance](decoder_norm1_optimization_2026-10-08.md).

## SM120 INT8 FFN-up follow-up (2026-10-03)

The INT8 fusion profile now automatically fuses FFN-up GEMM with SwiGLU on
SM120. It retains the 144-linear quantization scope, FP32 scales and existing
FP16 rounding. Other supported architectures retain the previous INT8 path.
A paired full-decode comparison measured **6.640226 → 6.508537 s (−1.98%)**;
an independent width measured **6.635767 → 6.505977 s (−1.96%)**.

Both variants used identical normalization schedules, with no run observer
active during timing. With this control, 8 videos / 992 frames retained
bitwise-equal RGB. Independent Inductor recompilations can select different
normalization reduction orders; this is not a cross-compilation bitwise
guarantee. Original INT8 loss remains. See the
[implementation, Nsight, correctness and Comfy evidence](decoder_int8_ffn_2026-10-03.md).

## Initial measurements (2026-09-24)

2026-09-24: RTX PRO 5000 72GB (user-confirmed model; driver reports NVIDIA
Graphics Device, 73415 MiB, CC 12.0), Linux 5.4.241, driver 580.82.07,
Python 3.12.14, PyTorch 2.11.0+cu130, CUDA 13.0, Triton 3.6.0,
comfy-kitchen 0.2.34. Shared GPU background work remained running.
Implemented against v0.2.0 runtime (tag `00d905f`) with subsequent docs-only
HEAD `bb603db`; this commit integrates the validated experiment kernels.

Two warmups, five CUDA Event full-decode measurements, seed 20260924,
decoder tile 256, SDPA auto, FP16 accumulation disabled. Loading, compilation,
media I/O, CPU copies and quality metrics excluded; dynamic INT8 activation
quantization and output finalization included. Decoder compile mode is
`max-autotune-no-cudagraphs`.

| H×W×frames | FP16 fusions, batch 8 | INT8 fusions, batch 4 |
| --- | ---: | ---: |
| 672×672×124 | 6.203846 s | Not measured for this exact profile |
| 768×1344×124 | 10.923379 s | 6.666817 s |

The final scheduling/finalization increment was paired against the preceding
research candidate, **not the release**: FP16 batch2 → batch8 decreased
11.275069 → 10.923379 s (3.12%); INT8 batch2 → batch4 decreased
6.710155 → 6.666817 s (0.646%). Each won all five paired runs. FP16 used
alternating AB/BA; INT8 used rotating/reversing multi-path ordering. These
percentages are not cumulative speedups versus ComfyUI, TRT or v0.2.0.

See [raw new timing samples and quality aggregates](benchmarks/decode_fusions_2026-09-24.json)
and [historical comparison CSV](benchmarks/decode_comparison_2026-09-24.csv).
ComfyUI historical values come from [the latest-fast report](h3_latest_fast_ab_2026-09-18.md),
TRT from [the same-tile report](h3_trt_256_same_tile_benchmark_2026-09-18.md),
and release values from [v0.2.0 README](https://github.com/fishelegs/ComfyUI-H3VAE-PyOpt/blob/v0.2.0/README.md).
Different dates, stacks, tile/batch plans and background load mean the CSV is
a chart-ready overview, not one controlled A/B. Historical values were not replaced.

## Quality and integration

Eight internal videos: four 768×1376×124 and four 768×1344×124, 24 fps,
992 frames total. No resize, crop, dropped frames or second video compression.
Same FP16 encoded latent within each paired comparison; encoder tile256,
staged batch4. PSNR is `-10*log10(MSE)` per uncompressed RGB [0,1] frame;
the reported mean averages frame PSNRs, not frame MSEs.

| Comparison | Mean frame PSNR | Minimum frame PSNR | Frames below 30 dB |
| --- | ---: | ---: | ---: |
| Latest FP16 / source | 35.109656 | 32.709368 | 0/992 |
| Latest INT8 / source | 34.939593 | 32.586896 | 0/992 |
| Latest FP16 / preceding FP16 candidate | 81.807203 | 80.455768 | 0/992 |
| Latest INT8 / preceding INT8 candidate | 62.176269 | 54.330909 | 0/992 |

All values are dB. The last two rows do **not** compare with v0.2.0 or compare
INT8 with FP16. Independent audits recomputed each suite's 2,976 per-frame
records. Shape/range/finite checks passed. These are reconstruction tests,
not generation-latent tests, subjective blind reviews or temporal-quality
guarantees; passing 30 dB is not proof of visual losslessness.

After integration, independent-process production-runtime checks passed for
both profiles at 124×672×672 with FP32 Comfy IMAGE input and actual encoded
latent. The Comfy wrapper and direct runtime matched exactly; CUDA → CPU →
CUDA offload/reload preserved outputs, INT8 weights and FP32 scales. The
encoder prefix/suffix and decoder stayed compiled. Comfy SDK revision:
`c1716a458d7978d37ec28608b7587887de5ee3f7`. No running server was modified.

Limitations: only the listed GPU/software and 124-frame workload are validated.
Windows and other NVIDIA models are unmeasured. A separate 22-frame probe
hit an existing encoder staged-batch limitation; it is not fixed here. A
multi-path experiment hit Dynamo's recompile limit during its later quality
stage; only its completed timing is used. Quality evidence instead comes from
independent processes without that fallback, without raising the cache limit.

## Reproduce current runtime

The [README](../README.md#直接测试) provides timing commands and minimal workflows.
For wrapper/offload validation, use a fresh output path:

```bash
python bench_decode_profiles_comfy.py --fp16 \
  --comfy-root "$COMFY_ROOT" \
  --model-code-dir "$H3_VAE_MODEL_CODE_DIR" --weights "$H3_VAE_WEIGHTS_PATH" \
  --frames 124 --height 672 --width 672 --tile-batch 8 \
  --encoder-staged-batch 4 --rgb-dtype fp32 --encoded-roundtrip \
  --output results/fp16_fused_comfy.json
```

For INT8 omit `--fp16`, use `--tile-batch 4` and a different output filename.
This random-input compatibility check does not replace real-video quality testing.
