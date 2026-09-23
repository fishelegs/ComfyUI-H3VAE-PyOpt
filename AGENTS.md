# AGENTS.md

## Repository purpose

This repository provides an optimized PyTorch/Triton MiniMax H3 video VAE
runtime and ComfyUI custom-node integration.

The primary maintenance goal is to improve H3 VAE encode/decode performance
without hiding numerical, compatibility, or deployment trade-offs.

## Default behavior and experimental modes

Treat the validated FP16 path as the default and safest baseline.

- Default: `dtype=fp16`, `int8_encode=false`, `int8_decode=false`.
- `int8_decode=true` is an experimental speed/quality trade-off. On the
  currently measured RTX PRO 5000 setup it improves decoder latency, but the
  result is hardware- and workload-specific.
- `int8_encode=true` exists for research. The current measured implementation
  is slower than the FP16 encoder and must not be presented as a recommended
  speed optimization without new evidence.
- `fast_linear` and `int8_decode` are mutually exclusive.
- Never silently fall back to FP16 while reporting an INT8 path as active.
- Do not change the default precision path merely because an experimental mode
  is faster on one measured configuration.

Read `docs/experimental_int8.md` before changing INT8 behavior.

## Repository map

- `h3vae_nodes.py`: ComfyUI node surface and user-facing options.
- `h3vae_runtime.py`: model loading, tiling, compile setup, runtime options,
  and INT8 integration.
- `opt/`: optimized PyTorch/Triton kernels and integration helpers. Additional
  rules for this directory are in `opt/AGENTS.md`.
- `tests/`: CPU-safe contract and regression tests.
- `bench_*.py`: GPU/model benchmark and validation tools.
- `docs/testing.md`: CI versus GPU validation boundary.
- `docs/experimental_int8.md`: current INT8 design, measurements, and caveats.
- `docs/compatibility.md`: measured compatibility evidence.

## Lightweight verification

Run before submitting ordinary changes:

```bash
python -m compileall -q .
ruff check . --exclude "bench_*.py" --select E9,F63,F7,F82
python -m unittest discover -s tests -p 'test_*.py'
```

These checks are intentionally CPU-friendly. A green CI run does not prove
CUDA/Triton correctness, model compatibility, numerical equivalence, image
quality, tile behavior, VRAM behavior, or performance.

## GPU/runtime validation

Changes affecting any of the following require additional NVIDIA CUDA
validation:

- Triton kernels
- INT8 quantization or integration
- tiling or padding
- dtype conversions
- `torch.compile` boundaries
- attention/SDPA backend selection
- model load/offload behavior
- performance-sensitive runtime paths

For the FP16 runtime, use the representative command in `docs/testing.md`.

For INT8 changes, use the relevant tools:

```bash
python bench_int8_roundtrip.py ...
python bench_int8_vae.py ...
python bench_int8_comfy_smoke.py ...
```

Match the model code, weights, input shape, tile sizes, batch settings, warmup,
run count, seed, dtype, and software versions when comparing before/after
results.

## Performance and quality claims

Performance changes must report enough context to reproduce the result:

- GPU model and VRAM
- OS
- Python, PyTorch, CUDA, and Triton versions
- comfy-kitchen version when applicable
- ComfyUI revision/version
- repository commit/version
- input resolution and frame count
- dtype and tile/batch settings
- warmup, timed runs, and seed
- encode and decode latency separately
- peak VRAM when available
- correctness or quality evidence

Do not mix absolute timings from different tile configurations, PyTorch/CUDA
stacks, TensorRT engines, or benchmark paths to infer a new speedup.

Latency alone is not sufficient for a numerical optimization. Include an
appropriate error/quality measure such as RMSE, MAE, PSNR, or a real-video
visual/temporal check. Do not describe a change as quality-neutral unless the
evidence supports that exact claim.

## Correctness rules

- Preserve explicit failure for unsupported INT8 configurations.
- Preserve the distinction between component-level smoke tests and end-to-end
  video quality evidence.
- Input layout can affect output; comparisons must keep layout fixed.
- Tile changes can alter boundary behavior and numerical output.
- Do not rewrite or mutate original model checkpoints as part of an inference
  optimization.
- Do not add model weights, private media, credentials, local absolute paths,
  or proprietary artifacts to the repository.

## Project metadata and releases

Keep these values synchronized for every release:

- `__version__`
- `pyproject.toml` project version
- `CHANGELOG.md`
- Git tag / GitHub Release
- ComfyUI Registry version

Current release line: `0.2.x`.

Do not create a tag or publish to the Registry until the target commit has a
green CI run and release notes accurately describe experimental features and
known limitations.

## ComfyUI Registry

Registry identity is fixed:

- Publisher ID: `fishelegs`
- Node ID: `h3vae-pyopt`
- Display name: `ComfyUI-H3VAE-PyOpt`

Never commit Registry API keys. GitHub Actions must read the publishing key
only from the `REGISTRY_ACCESS_TOKEN` repository secret. See
`docs/registry.md`.

## Licensing and security

Repository-authored code is MIT-licensed. ComfyUI, PyTorch, Triton,
safetensors, optional comfy-kitchen support, MiniMax H3 / FL2VA model code,
and model weights remain separately licensed. See `LICENSE` and
`THIRD_PARTY_NOTICES.md`.

Do not copy third-party implementation code into this repository without
reviewing and preserving its applicable license notices.

For security-sensitive findings, follow `SECURITY.md` and avoid publishing
secrets or exploit details in a public issue.
