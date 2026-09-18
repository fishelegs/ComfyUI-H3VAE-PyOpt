# Adoption Evidence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the repository's existing performance work into reproducible, contributor-friendly evidence of ecosystem value without fabricating adoption metrics.

**Architecture:** Add structured benchmark reporting, contributor/Codex guidance, a compatibility evidence page based only on already documented measurements, and README links/badges. Repository-global About/topics are updated only if the GitHub connector exposes a supported write action; otherwise record the recommended values for the maintainer.

**Tech Stack:** GitHub issue forms, Markdown, GitHub Actions badges, existing benchmark scripts.

**Spec:** `docs/superpowers/specs/2026-09-18-codex-oss-readiness-design.md`

## Global Constraints

- Work on `codex-oss-readiness` for file changes.
- Do not invent stars, downloads, users, GPUs, benchmark results, or compatibility claims.
- Preserve benchmark caveats: results are hardware/workload/configuration specific.
- Community benchmark submissions must capture environment and correctness evidence, not latency alone.
- Use existing README/docs measurements as the only initial compatibility evidence.
- Keep Codex/agent guidance concise and executable.

---

### Task 1: Add a structured benchmark report issue form

**Files:**
- Create: `.github/ISSUE_TEMPLATE/benchmark-report.yml`

**Interfaces:**
- Produces: structured community evidence fields for GPU/software/workload/performance/correctness.

- [ ] Add fields for GPU model, VRAM, OS, Python, PyTorch, CUDA, Triton, ComfyUI revision, project revision/version, resolution, frame count, tile settings, encode/decode latency, peak VRAM, correctness/quality evidence, reproduction command, and notes.
- [ ] Require core environment/workload/performance/correctness fields.
- [ ] Add a checkbox confirming the result is reproducible and does not include private model credentials.
- [ ] Commit as `docs: add benchmark report issue form`.

---

### Task 2: Add maintainer and agent contribution guidance

**Files:**
- Create: `CONTRIBUTING.md`
- Create: `AGENTS.md`

**Interfaces:**
- Produces: contributor workflow and Codex-compatible repository instructions.

- [ ] Document lightweight CI commands and GPU validation boundary in `CONTRIBUTING.md`.
- [ ] Require performance PRs to include environment, workload, latency, and correctness evidence.
- [ ] In `AGENTS.md`, document repository purpose, test/lint commands, benchmark command, licensing boundary, and performance-change reporting requirements.
- [ ] Keep instructions aligned with `docs/testing.md`.
- [ ] Commit as `docs: add contributor and agent guidance`.

---

### Task 3: Add compatibility/evidence page

**Files:**
- Create: `docs/compatibility.md`

**Interfaces:**
- Consumes: existing benchmark documents.
- Produces: a conservative compatibility/measurement table and a path for adding community results.

- [ ] Record the maintainer-tested RTX PRO 5000 72GB / Linux measurements already documented in README and benchmark reports.
- [ ] Separate "tested" from "unknown/not yet reported".
- [ ] Link to benchmark report issue template for community additions.
- [ ] State that a result is evidence for one environment, not a universal compatibility guarantee.
- [ ] Commit as `docs: add compatibility and benchmark evidence`.

---

### Task 4: Improve README adoption surfaces

**Files:**
- Modify: `README.md`

**Interfaces:**
- Consumes: CI workflow, LICENSE, CONTRIBUTING, compatibility page.
- Produces: visible license/CI badges and contribution/evidence links.

- [ ] Add MIT and CI badges near the title.
- [ ] Add a short "Contributing and compatibility" section linking to `CONTRIBUTING.md`, `docs/testing.md`, and `docs/compatibility.md`.
- [ ] Do not add a release badge until `v0.1.0` exists.
- [ ] Commit as `docs: surface CI and community evidence`.

---

### Task 5: Repository presentation metadata

**Repository-global metadata (not branchable):**

Recommended description:

`Optimized PyTorch/Triton MiniMax H3 video VAE for ComfyUI with reproducible benchmarks and no prebuilt TensorRT engine requirement.`

Recommended topics:

`comfyui`, `comfyui-custom-node`, `minimax`, `minimax-h3`, `video-generation`, `vae`, `pytorch`, `triton`, `torch-compile`, `cuda`, `gpu-optimization`

- [ ] If supported repository metadata write actions are available, update description/topics and re-read repository metadata.
- [ ] If unavailable, leave repository metadata unchanged and include these exact recommended values in the final handoff.

---

### Task 6: Final evidence verification

- [ ] Re-read all adoption files from GitHub.
- [ ] Verify no fabricated adoption metric appears.
- [ ] Verify the latest PR head CI succeeds on both Python 3.11 and 3.12.
- [ ] Update draft PR body with completed phases and remaining blockers.
- [ ] Do not merge, tag, publish a GitHub Release, or publish to Comfy Registry without the maintainer's final integration decision and Registry publisher identity.
