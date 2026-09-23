# AGENTS.md

These instructions apply to files under `opt/` and supplement the repository
root `AGENTS.md`.

## Scope

`opt/` contains performance-sensitive PyTorch/Triton implementations and
experimental INT8 integrations. Changes here can affect numerical output,
hardware compatibility, compilation behavior, and GPU memory use even when
CPU tests remain green.

Prefer a small, measurable kernel or integration change over broad refactors.

## General kernel rules

- Preserve existing public call contracts unless the task explicitly changes
  them.
- Keep CPU import paths safe. CUDA/Triton compilation should remain lazy where
  the current module is designed that way.
- Unsupported hardware or dtype combinations must fail explicitly when a path
  cannot run correctly.
- Do not introduce a silent slower fallback and still label the result as the
  requested optimized mode.
- Avoid hidden synchronizations in timed hot paths. Validation that calls
  `.item()` or otherwise synchronizes should stay outside repeated kernel
  execution when practical.
- Preserve causal/spatial padding semantics and tile boundary behavior.
- Do not assume a kernel speedup is useful until full encode/decode timing and
  output quality are checked.

## INT8 encoder contract

For `encoder_int8.py` and `encoder_int8_integration.py`:

- The intended path is real INT8×INT8 accumulation into INT32, not fake
  quantization around FP16 convolution.
- Static weights are INT8; weight scales remain FP32.
- Activations are dynamically quantized as documented.
- Quantized operators accept/return FP16 at the integration boundary unless
  the documented contract is intentionally revised.
- No FP16 fallback may masquerade as successful INT8 execution.
- Maintain accumulator-bound validation when changing reduction dimensions.
- Preserve the existing padding/layout contract; input layout differences can
  produce materially different outputs.
- Current evidence shows INT8 encode is slower than FP16. Do not optimize the
  product defaults around `int8_encode` without new end-to-end evidence.

## INT8 decoder contract

For `kitchen_int8.py` and decoder INT8 integration:

- The validated implementation targets NVIDIA CUDA SM80+.
- The current decoder path depends on `comfy-kitchen==0.2.34`.
- Static quantized weights are prepared once; dynamic activation quantization
  remains part of the per-call path.
- Preserve the documented comfy-kitchen public API contract unless an upgrade
  is intentionally validated.
- `int8_decode` must remain mutually exclusive with `fast_linear` unless a
  new combined path is explicitly designed and measured.
- Do not add comfy-kitchen to the mandatory base dependency list solely for an
  opt-in experimental path unless the packaging policy is intentionally
  changed.

## Required CPU checks

At minimum run the full repository checks from the root `AGENTS.md`.

For INT8 work, pay particular attention to:

- `tests/test_encoder_int8.py`
- `tests/test_encoder_int8_integration.py`
- `tests/test_kitchen_int8.py`
- `tests/test_runtime_int8_options.py`
- `tests/test_bench_int8_roundtrip.py`
- `tests/test_bench_int8_vae.py`
- `tests/test_comfy_smoke_layout.py`
- `tests/test_int8_prompt.py`

CPU tests validate contracts and guards; they do not validate GPU kernel
correctness or quality.

## Required GPU evidence

Kernel or integration changes that affect INT8 arithmetic require a before/after
GPU run using the relevant benchmark tools:

```bash
python bench_int8_roundtrip.py ...
python bench_int8_vae.py ...
python bench_int8_comfy_smoke.py ...
```

For performance comparisons, keep the environment, model/code revision, input,
tile sizes, batch settings, warmups, run count, seed, and input layout fixed.

For numerical changes, report latent/output differences and quality evidence.
For video-facing changes, prefer the multi-video roundtrip methodology in
`docs/experimental_int8.md` rather than a single random tensor.

## Benchmark discipline

- Separate encode and decode timings.
- State whether compilation, loading, media I/O, and quantization preparation
  are included or excluded.
- Do not combine measurements from different environments into one claimed
  speedup.
- Do not compare different tile sizes as if they were identical execution
  plans.
- Preserve raw machine-readable evidence when publishing a new benchmark
  result, while removing local paths, credentials, private media, and other
  sensitive information.
