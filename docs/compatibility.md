# Compatibility and Benchmark Evidence

This page records environments that have actually been measured for this
repository. It is evidence for those specific configurations, not a guarantee
that every GPU, operating system, driver, tile size, or model revision behaves
the same way.

## Maintainer-tested environment

| Area | Recorded environment / result |
| --- | --- |
| GPU | NVIDIA RTX PRO 5000, 72 GB class (73415 MiB reported during testing) |
| OS | Linux; distribution was not recorded in the benchmark note |
| PyOpt Python | 3.12.14 |
| PyOpt PyTorch | 2.11.0+cu130 |
| Workload | FP16, 768×1344×124 |
| Decoder / encoder tile | 256 / 256 |
| PyOpt batch settings | decoder tile batch 2; encoder staged batch 4 |
| PyOpt default | decode 11.437 s; encode 11.983 s; total 23.420 s |
| PyOpt optional `fast_linear` | decode 11.107 s; encode 11.989 s; total 23.096 s |
| Matched TensorRT comparison | TensorRT 11.2.1.2; total 26.237 s |

The strict 256/256 comparison used independent processes, 2 warmups, 7 CUDA
Event measurements, and seed `20260917`. The TensorRT path used Python
3.11.13 and PyTorch 2.8.0+cu128, so the result is a same-GPU engineering
comparison rather than a claim that the software stacks are identical.

See
[the full 256/256 benchmark report](h3_trt_256_same_tile_benchmark_2026-09-18.md)
for methodology, engine construction details, hashes, and caveats.

## Numerical / quality evidence

For a separate 768×1344×124 run on the same GPU class with Python 3.12.14 and
PyTorch 2.11.0+cu130, the optional PyOpt `fast_linear` decoder path was
compared against the default PyOpt decoder using the same random input,
weights, and tiles:

- pixel PSNR: approximately 61.17 dB
- pixel MAE: approximately 0.000616

These random-input metrics show that the optional path is not bitwise
equivalent. They do **not** prove that real video has no visible or temporal
quality change. See
[the fast-path A/B report](h3_latest_fast_ab_2026-09-18.md).

## Not yet established

This repository does not currently claim maintainer-verified compatibility or
performance numbers for other GPU models or for Windows. Those environments
should remain "not yet reported" until a reproducible result is submitted and
reviewed.

## SDPA backend fallback

The decoder defaults `MINIMAX_H3_TORCH_SDPA_BACKEND` to `auto`. PyTorch makes
the backend decision for each attention call, taking the GPU, dtype, tensor
shape, and installed build into account. A supported NVIDIA configuration can
therefore still use Flash SDPA, while unsupported configurations can fall back
to another enabled implementation instead of failing with `No available
kernel`.

An explicit environment value takes precedence. This is useful for controlled
benchmarks, but forcing `flash` disables the fallback and can fail when that
kernel is not available:

```bash
MINIMAX_H3_TORCH_SDPA_BACKEND=flash python main.py
```

On Windows PowerShell, restore the compatible default before starting ComfyUI
with:

```powershell
$env:MINIMAX_H3_TORCH_SDPA_BACKEND = "auto"
python main.py
```

Restart ComfyUI after changing the variable so model imports and compiled
graphs use the new selection policy. The active policy is included in the
`[H3VAE-PyOpt] runtime ready` log line.

## Contributing a result

Use the repository's **Benchmark report** issue form. A useful report includes:

- GPU model and VRAM
- OS
- Python, PyTorch, CUDA, and Triton versions
- ComfyUI revision and this repository's revision
- resolution, frames, dtype, and tile/batch settings
- warmup, run count, and seed where applicable
- decode and encode latency
- peak VRAM when available
- numerical or real-video quality evidence
- exact reproduction command

Validated community results can then be added to this page without turning
unverified anecdotes into compatibility claims.
