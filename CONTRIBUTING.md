# Contributing

Thanks for helping improve ComfyUI-H3VAE-PyOpt.

## Before opening a pull request

Run the lightweight checks used by GitHub Actions:

```bash
python -m compileall -q .
ruff check . --exclude "bench_*.py" --select E9,F63,F7,F82
pytest -q tests
```

These checks are intentionally CPU-friendly. See [docs/testing.md](docs/testing.md)
for what they do and do not validate.

## GPU and performance changes

Changes that affect CUDA/Triton execution, tiling, compilation, numerical
behavior, or performance need additional GPU validation. A performance PR
should report:

- GPU model and VRAM
- operating system
- Python, PyTorch, CUDA, and Triton versions
- ComfyUI revision/version
- this repository's commit/version
- resolution and frame count
- dtype and tile/batch settings
- warmup/runs/seed where relevant
- decode latency
- encode latency
- peak VRAM when available
- correctness or quality evidence

Latency alone is not sufficient for a numerical optimization. Include an
appropriate comparison such as MAE/RMSE/PSNR and, when the change can affect
visible output, a real-video visual/temporal check.

For a representative PyOpt run, see [docs/testing.md](docs/testing.md). For a
community benchmark report, use the repository's **Benchmark report** issue
form.

## Licensing

Code authored for this repository is MIT-licensed. Do not copy third-party
source into the repository without preserving and reviewing its applicable
copyright and license notices. MiniMax H3 / FL2VA model code and model weights
are external assets and are not relicensed by this repository. See
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
