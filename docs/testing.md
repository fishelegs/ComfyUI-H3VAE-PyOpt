# Testing and Validation

This project separates lightweight pull-request checks from GPU/model validation.

## Pull-request CI

The GitHub Actions workflow in `.github/workflows/ci.yml` runs on ordinary
GitHub-hosted CPU runners with Python 3.11 and 3.12.

It checks:

- Python source compilation with `compileall`
- critical Ruff failures: `E9`, `F63`, `F7`, and `F82`
- CPU-only regression tests under `tests/`

The CI job intentionally does **not** download MiniMax H3 model code or model
weights and does not require CUDA, Triton, TensorRT, or comfy-kitchen.

## What pull-request CI does not prove

A green pull-request CI run does not validate:

- CUDA or Triton kernel execution
- MiniMax H3 / FL2VA model loading
- encoder or decoder numerical equivalence against the reference model
- tile seam quality
- GPU memory behavior
- performance regressions
- TensorRT comparisons

Changes that affect GPU execution or numerical behavior require additional
manual validation in a supported NVIDIA CUDA environment.

## PyOpt GPU validation

For a representative 672×672×124 run:

```bash
python bench_pyopt_vs_trt.py --pyopt-only \
  --model-code-dir "$H3_VAE_MODEL_CODE_DIR" \
  --weights "$H3_VAE_WEIGHTS_PATH" \
  --height 672 --width 672 --frames 124 \
  --decoder-tile 256 --encoder-tile 672 --tile-batch 2 --staged-batch 4 \
  --warmup 1 --runs 3 --seed 20260917 \
  --output results/pyopt_672x672x124.json
```

For encoder numerical regression, use `bench_encoder_quality.py` with the same
model code, weights, input shape, and tile settings as the implementation being
compared.

## Performance report checklist

Performance-related changes should report enough information for another
maintainer or user to reproduce the result:

- GPU model
- VRAM
- operating system
- Python version
- PyTorch version
- CUDA version
- Triton version
- input resolution
- frame count
- decoder tile size
- encoder tile size
- tile batch / staged batch settings where relevant
- encode latency
- decode latency
- peak VRAM when available
- correctness or quality evidence appropriate to the change

For numerical or quality-sensitive optimizations, include the relevant error
metric or image/video quality comparison rather than reporting latency alone.
