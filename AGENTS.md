# AGENTS.md

## Repository overview

This repository provides an optimized PyTorch/Triton MiniMax H3 video VAE
runtime and ComfyUI custom-node integration. The main goal is to accelerate
H3 VAE encode/decode while keeping benchmark methodology and numerical
trade-offs explicit.

## Lightweight verification

Run:

```bash
python -m compileall -q .
ruff check . --exclude "bench_*.py" --select E9,F63,F7,F82
python -m unittest discover -s tests -p 'test_*.py'
```

Do not treat CPU CI as proof that CUDA/Triton kernels, model loading, numerical
equivalence, or performance are correct. Read `docs/testing.md` before
changing GPU/runtime behavior.

## Benchmark

A representative PyOpt command is:

```bash
python bench_pyopt_vs_trt.py --pyopt-only \
  --model-code-dir "$H3_VAE_MODEL_CODE_DIR" \
  --weights "$H3_VAE_WEIGHTS_PATH" \
  --height 672 --width 672 --frames 124 \
  --decoder-tile 256 --encoder-tile 672 --tile-batch 2 --staged-batch 4 \
  --warmup 1 --runs 3 --seed 20260917 \
  --output results/pyopt_672x672x124.json
```

Performance changes must report GPU, software versions, input shape, tile/batch
configuration, encode/decode latency, and correctness/quality evidence.
Changing tile shape or numerical kernels can change output; do not present
latency improvements as quality-neutral without validation.

## Project metadata and releases

Keep these version values synchronized:

- `__version__`
- `pyproject.toml` project version
- `CHANGELOG.md`
- Git tag / Registry version when a release is published

Do not create a release tag until CI is green.

## ComfyUI Registry

The repository is structurally prepared for Registry packaging, but
`[tool.comfy].PublisherId` must be the exact publisher ID owned by the
maintainer. Never invent or commit a placeholder publisher ID. See
`docs/registry.md`.

## Licensing boundary

Repository-authored code is MIT-licensed. ComfyUI, PyTorch, Triton,
safetensors, optional comfy-kitchen support, MiniMax H3 / FL2VA model code,
and model weights remain separately licensed. See `LICENSE` and
`THIRD_PARTY_NOTICES.md`.
