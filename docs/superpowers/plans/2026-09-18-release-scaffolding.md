# Release Scaffolding Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Establish explicit project versioning and release documentation for an eventual `v0.1.0` without publishing the release before CI exists and passes.

**Architecture:** Keep release metadata lightweight: expose `__version__ = "0.1.0"` from the custom node package, add a human-readable changelog, and prepare release notes as a repository document. Do not create a Git tag or GitHub Release in this phase.

**Tech Stack:** Python package metadata, Markdown, GitHub releases.

**Spec:** `docs/superpowers/specs/2026-09-18-codex-oss-readiness-design.md`

## Global Constraints

- Work only on `codex-oss-readiness`.
- Version is `0.1.0`; eventual Git tag is `v0.1.0`.
- Do not publish a GitHub Release until Phase 3 CI passes.
- Do not alter runtime behavior beyond exposing a version constant.
- Preserve current benchmark claims and caveats; release notes must not broaden them.

---

### Task 1: Add explicit package version metadata

**Files:**
- Modify: `__init__.py`
- Create: `tests/test_version_metadata.py`

**Interfaces:**
- Produces: `__version__: str == "0.1.0"`.

- [ ] Add a CPU-only test that parses `__init__.py` with `ast` and asserts exactly one literal `__version__ = "0.1.0"` assignment without importing ComfyUI.
- [ ] Add `__version__ = "0.1.0"` near the top-level module metadata.
- [ ] Verify the AST test passes in a plain Python environment.
- [ ] Commit as `chore: define version 0.1.0`.

---

### Task 2: Add changelog and release notes

**Files:**
- Create: `CHANGELOG.md`
- Create: `docs/releases/v0.1.0.md`

**Interfaces:**
- Consumes: existing README benchmark claims and known limitations.
- Produces: stable release history plus copy suitable for GitHub Release creation after CI.

- [ ] Create `CHANGELOG.md` with an `Unreleased` section and a `0.1.0 - 2026-09-18` section.
- [ ] List the initial ComfyUI loader, PyTorch/Triton optimizations, benchmark tooling, examples, tests, and MIT licensing.
- [ ] Create `docs/releases/v0.1.0.md` describing the initial public release, installation prerequisites, benchmark scope, and limitations.
- [ ] Explicitly state that MiniMax H3 model code/weights are external and separately licensed.
- [ ] Explicitly state that performance numbers are hardware/workload specific and not universal guarantees.
- [ ] Commit as `docs: prepare v0.1.0 release notes`.

---

### Task 3: Verify release scaffolding without publishing

**Files:**
- Verify: `__init__.py`
- Verify: `tests/test_version_metadata.py`
- Verify: `CHANGELOG.md`
- Verify: `docs/releases/v0.1.0.md`

- [ ] Re-read all four files from the branch.
- [ ] Confirm version strings are consistent: Python metadata `0.1.0`, changelog `0.1.0`, release filename/body `v0.1.0`.
- [ ] Confirm no tag named `v0.1.0` exists and no GitHub Release has been created.
- [ ] Defer tag/release creation until Phase 3 CI verification succeeds.
