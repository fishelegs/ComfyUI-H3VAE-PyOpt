# ComfyUI Registry Readiness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the repository structurally ready for ComfyUI Registry publication while leaving the publisher identity unset until the maintainer owns a real Registry publisher ID.

**Architecture:** Add standard PEP 621 project metadata, package/archive exclusions, and publishing instructions. Do not invent `[tool.comfy].PublisherId`; final Registry publication is blocked only on creating/confirming a publisher identity and then adding that exact ID.

**Tech Stack:** `pyproject.toml`, ComfyUI Registry, Comfy CLI, Markdown.

**Spec:** `docs/superpowers/specs/2026-09-18-codex-oss-readiness-design.md`

## Global Constraints

- Work only on `codex-oss-readiness`.
- Project version remains `0.1.0`.
- Project license points to `LICENSE`.
- Registry metadata must not relicense MiniMax H3 / FL2VA assets.
- Do not guess or reserve a Comfy Registry publisher identity.
- Package dependencies mirror the current runtime dependency declaration: `torch>=2.8`, `triton`, and `safetensors`.
- Current validated platform is Linux with NVIDIA CUDA; do not claim Windows validation.

---

### Task 1: Add standard package metadata

**Files:**
- Create: `pyproject.toml`
- Create: `tests/test_project_metadata.py`

**Interfaces:**
- Produces: PEP 621 metadata with project name `comfyui-h3vae-pyopt`, version `0.1.0`, MIT license file, repository URLs, and runtime dependencies.

- [ ] Add a test using Python `tomllib` that asserts project name, version, license file, repository URL, and exact dependency set.
- [ ] Add `pyproject.toml` with the tested metadata.
- [ ] Add honest classifiers for Linux/NVIDIA CUDA and supported Python versions.
- [ ] Do not add `[tool.comfy]` until a real publisher ID is confirmed.
- [ ] Commit as `chore: add package metadata for registry`.

---

### Task 2: Control Registry archive contents

**Files:**
- Create: `.comfyignore`

**Interfaces:**
- Produces: a smaller published archive without benchmark reports, tests, local results, Git metadata, or Superpowers process docs.

- [ ] Exclude `.git/`, `.github/`, `tests/`, `results/`, caches, and `docs/superpowers/`.
- [ ] Keep README, LICENSE, THIRD_PARTY_NOTICES, runtime Python files, requirements, examples, and user-facing benchmark documentation available in the source repository.
- [ ] Commit as `chore: define Comfy Registry archive exclusions`.

---

### Task 3: Document final Registry publication step

**Files:**
- Create: `docs/registry.md`

**Interfaces:**
- Produces: exact maintainer steps for completing `[tool.comfy]` after a publisher exists.

- [ ] Document that Comfy Registry requires a publisher identity before publication.
- [ ] State that no public evidence of an existing `fishelegs` publisher was found during this readiness pass.
- [ ] Show the exact section to add only after the real ID is known:
```toml
[tool.comfy]
PublisherId = "<the exact publisher id owned by the maintainer>"
DisplayName = "ComfyUI-H3VAE-PyOpt"
Icon = ""
```
- [ ] Explain that the placeholder above is documentation only and must never be committed verbatim to `pyproject.toml`.
- [ ] Link to the official ComfyUI Registry publishing/specification documentation.
- [ ] Commit as `docs: document Comfy Registry publication`.

---

### Task 4: Verify structural readiness

- [ ] Parse `pyproject.toml` with Python `tomllib`.
- [ ] Verify version matches `__version__` and CHANGELOG.
- [ ] Verify `[tool.comfy]` is absent rather than containing a guessed identity.
- [ ] Verify `.comfyignore` does not exclude runtime code or licensing files.
- [ ] Record Registry identity as the only publication blocker.
