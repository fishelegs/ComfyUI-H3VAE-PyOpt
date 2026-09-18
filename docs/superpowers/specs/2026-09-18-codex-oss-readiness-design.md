# Codex for OSS Readiness Design

## Goal

Prepare `fishelegs/ComfyUI-H3VAE-PyOpt` as a maintainable, clearly licensed,
release-ready ComfyUI open-source project with reproducible tests, Registry
packaging, and measurable adoption evidence.

## Scope

Work is split into five sequential phases:

1. MIT licensing and third-party boundary documentation
2. Versioning and first formal release
3. CI and test hardening
4. ComfyUI Registry / Manager readiness
5. Adoption and ecosystem evidence

Each phase should be independently reviewable and should not require merging
unfinished work from later phases.

## Working Model

All changes are developed on `codex-oss-readiness` and reviewed before merging
to `main`. The default branch remains untouched until the final integration
decision.

## Phase 1: License

### Decision

The repository's own source code will use the standard MIT License.

### Rationale

The maintainer's goal is a permissive license that permits commercial use,
modification, redistribution, private use, and use in proprietary projects,
subject to retaining the MIT copyright and license notice.

The current repository structure treats ComfyUI, PyTorch, Triton, safetensors,
and MiniMax H3 / FL2VA code or model assets as separately distributed
components. The runtime imports or loads those components rather than
relicensing them.

A source audit of the repository root and `opt/` found no visible external
copyright headers, SPDX identifiers, or explicit "copied/adapted from" notices.
If a future file is identified as copied or substantially derived from
third-party source, that file must retain the applicable upstream notices and
may require a different licensing treatment.

### Files

Create:

- `LICENSE` — standard MIT text, copyright 2026 fishelegs.
- `THIRD_PARTY_NOTICES.md` — clarify that third-party software, model code,
  and model weights retain their own licenses.

Modify:

- `README.md` — add a concise License section and link to the third-party
  notices.

### Boundary

The MIT license covers source code authored for this repository. It does not
relicense:

- ComfyUI
- PyTorch
- Triton
- safetensors
- comfy-kitchen, when optionally used
- MiniMax H3 / FL2VA model code
- MiniMax H3 model weights or other separately downloaded assets

## Phase 2: Release

### Objective

Establish a conventional first release without changing runtime behavior.

### Deliverables

- Define project version metadata.
- Add `CHANGELOG.md` with an initial `0.1.0` entry.
- Prepare GitHub release notes describing:
  - optimized PyTorch/Triton H3 VAE runtime
  - ComfyUI integration
  - no TensorRT engine requirement for the PyOpt path
  - supported environment assumptions
  - known limitations
- Create tag/release `v0.1.0` only after Phase 3 CI passes.

## Phase 3: CI and Tests

### Objective

Make contributions reviewable without requiring a large model download or GPU
for every pull request.

### CI Layers

**CPU/lightweight PR CI**

- Python syntax/import checks where imports can be isolated safely
- unit tests for pure-Python configuration and helper behavior
- linting
- package metadata validation

**GPU/manual validation**

Keep model-dependent and GPU-dependent checks separate from ordinary PR CI:

- encoder/decoder numerical regression
- tile consistency
- quality regression
- benchmark scripts
- CUDA/Triton compatibility validation

GPU checks may be documented or run on a self-hosted runner later; ordinary
GitHub-hosted CI must not download multi-GB model assets.

### Testing Principles

- Preserve existing benchmark scripts.
- Add regression tests around behavior that can be tested without model
  weights.
- Avoid tests whose only assertion is that an implementation detail exists.
- Performance changes should document environment, shape, latency, and
  correctness deltas.

## Phase 4: ComfyUI Registry

### Objective

Make the project installable through the current ComfyUI Registry / Manager
workflow with minimal manual setup.

### Deliverables

- Add current Registry-compatible package metadata.
- Ensure dependency declarations reflect actual runtime requirements.
- Keep model weights and third-party model code out of the package.
- Prefer ComfyUI model discovery where possible.
- Keep environment variables as explicit overrides rather than the only normal
  installation path when feasible.
- Document Linux support and any Windows/Triton caveats accurately.

Registry publication is performed only after validating the current Registry
schema and package rules against the latest ComfyUI documentation.

## Phase 5: Adoption Evidence

### Objective

Turn performance work into evidence of ecosystem value and maintainability.

### Repository Presentation

Add or improve:

- concise GitHub repository description
- relevant GitHub topics
- badges for license, release, and CI
- English-facing project summary where useful
- reproducible benchmark instructions

### Benchmark Evidence

Standardize reports with:

- GPU model
- VRAM
- operating system
- Python version
- PyTorch version
- CUDA version
- Triton version
- resolution / frame count
- encode latency
- decode latency
- peak VRAM where available
- correctness or quality comparison where relevant

### Community Evidence

Add a benchmark/report issue template so users can submit comparable results.
Over time, summarize validated community results in a compatibility matrix.

## Non-Goals for This Readiness Pass

The following are valuable later but are not required before the first formal
release:

- generic acceleration support for non-H3 VAEs
- automatic VRAM/tile autotuning
- extensive Windows-specific optimization work
- mandatory GPU CI on every pull request
- major runtime refactors unrelated to OSS readiness

## Success Criteria

The readiness pass is complete when:

- GitHub recognizes the repository as MIT-licensed.
- Third-party licensing boundaries are explicit.
- A documented `v0.1.0` release exists.
- Pull requests have a lightweight automated CI gate.
- Existing GPU/model-dependent validation has a documented execution path.
- The repository satisfies current ComfyUI Registry packaging requirements.
- README and repository metadata clearly communicate the project's purpose,
  installation path, benchmark methodology, and maintenance status.
- Users have a structured way to contribute benchmark and compatibility data.

## Implementation Order

The implementation order is fixed unless a dependency forces a change:

1. LICENSE and notices
2. release/version scaffolding
3. CI/tests
4. ComfyUI Registry
5. adoption evidence

A release tag is intentionally delayed until CI/test work has passed.
