# Changelog

All notable changes to this project will be documented in this file.

The format is inspired by Keep a Changelog, and this project uses Semantic
Versioning for release tags.

## Unreleased

## 0.2.0 - 2026-09-22

### Added

- Independent opt-in experimental INT8 encoder and decoder modes: eight
  encoder convolutions and 72 decoder FFN linears use real INT8 arithmetic.
  Other operations and default behavior retain their existing precision.
- Four-way timing and video reconstruction-quality validation, including
  per-frame PSNR and explicit encoder/decoder trade-offs. The measured INT8
  encoder is slower; decoder acceleration does not imply encoder speedup.

### Changed

- Documented the validated FP16 floating-point path as the recommended default
  and the decoder-only INT8 mode as the practical experimental speed option,
  including focused performance and 992-frame PSNR tables.

### Fixed

- Default decoder SDPA selection to `auto`, allowing PyTorch to use Flash SDPA
  when supported and fall back to another available backend instead of raising
  `No available kernel` on unsupported Windows/GPU/input configurations.

## 0.1.0 - 2026-09-18

### Added

- ComfyUI `H3VAEPyOptLoader` integration for MiniMax H3 video VAE workflows.
- PyTorch/Triton decoder and encoder optimizations, including fused GPU kernels,
  `torch.compile`, tile batching, and staged encoder batching.
- Reproducible benchmark scripts for PyOpt, stock ComfyUI, and TensorRT
  comparison paths.
- Example ComfyUI prompt JSON and benchmark documentation.
- CPU-side regression tests for benchmark helpers and runtime tiling behavior.
- Explicit package version metadata.
- MIT licensing for repository-authored source code and third-party licensing
  boundary documentation.

### Notes

- MiniMax H3 / FL2VA model code and model weights are not included in this
  repository and remain subject to their own licenses.
- Current performance measurements are hardware-, software-, tile-, and
  workload-specific and should not be treated as universal guarantees.
- The experimental `fast_linear` path is optional and may change numerical
  results.
