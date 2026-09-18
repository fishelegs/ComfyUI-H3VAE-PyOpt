# MIT Licensing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make ComfyUI-H3VAE-PyOpt unambiguously MIT-licensed while preserving clear licensing boundaries for separately distributed third-party software, model code, and model weights.

**Architecture:** Add a standard root MIT license, add a concise third-party notices document, and update the README to state exactly what the repository license covers. Do not alter runtime code or dependency behavior in this phase.

**Tech Stack:** Markdown, GitHub repository metadata, standard MIT License text.

**Spec:** `docs/superpowers/specs/2026-09-18-codex-oss-readiness-design.md`

## Global Constraints

- Repository-authored source code uses the standard MIT License.
- Copyright notice is `Copyright (c) 2026 fishelegs`.
- Third-party software, model code, and model weights keep their own licenses.
- Do not claim that MiniMax H3 / FL2VA code or model weights are MIT-licensed.
- Do not change runtime behavior in this phase.
- Work only on `codex-oss-readiness`; do not modify `main`.

---

### Task 1: Add the root MIT license and third-party notices

**Files:**
- Create: `LICENSE`
- Create: `THIRD_PARTY_NOTICES.md`

**Interfaces:**
- Consumes: the licensing decision in the design spec.
- Produces: canonical repository license text and the third-party licensing boundary referenced by README and future Registry metadata.

- [ ] **Step 1: Verify the files do not already exist**

Run:

```bash
test ! -e LICENSE
test ! -e THIRD_PARTY_NOTICES.md
```

Expected: both commands exit 0.

- [ ] **Step 2: Create the standard MIT license**

Create `LICENSE` with exactly:

```text
MIT License

Copyright (c) 2026 fishelegs

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

- [ ] **Step 3: Create third-party notices**

Create `THIRD_PARTY_NOTICES.md` with:

```markdown
# Third-Party Notices

The source code authored for this repository is licensed under the
[MIT License](LICENSE).

This project interoperates with or can optionally use third-party software,
model code, and model assets that are distributed separately under their own
licenses. The MIT License in this repository does not relicense those
third-party components.

## Third-party components

- **ComfyUI** — this project is a ComfyUI custom-node integration and imports
  ComfyUI APIs at runtime. ComfyUI is distributed separately under its own
  license.
- **PyTorch** — runtime tensor and compilation framework, distributed
  separately under its own license.
- **Triton** — used for custom GPU kernels, distributed separately under its
  own license.
- **safetensors** — used to load model weights, distributed separately under
  its own license.
- **comfy-kitchen** — optional dependency used only when the experimental
  `fast_linear` path is enabled; distributed separately under its own
  license.
- **MiniMax H3 / FL2VA model code** — loaded from a user-provided external
  model-code directory and not included in this repository.
- **MiniMax H3 model weights and other model assets** — downloaded or supplied
  separately and not included in this repository. Their use is governed by
  the applicable MiniMax model license and any terms accompanying the assets.

If source code from a third-party project is added to this repository in the
future, its original copyright and license notices must be preserved and this
document must be updated when required.
```

- [ ] **Step 4: Verify both documents contain the intended boundary**

Run:

```bash
grep -F "MIT License" LICENSE
grep -F "Copyright (c) 2026 fishelegs" LICENSE
grep -F "does not relicense those third-party components" THIRD_PARTY_NOTICES.md
grep -F "MiniMax H3 / FL2VA model code" THIRD_PARTY_NOTICES.md
```

Expected: all four commands print one matching line and exit 0.

- [ ] **Step 5: Commit the licensing files**

```bash
git add LICENSE THIRD_PARTY_NOTICES.md
git commit -m "docs: license repository under MIT"
```

Expected: one commit containing only the two new licensing files.

---

### Task 2: Document the MIT license in README

**Files:**
- Modify: `README.md`

**Interfaces:**
- Consumes: `LICENSE` and `THIRD_PARTY_NOTICES.md` from Task 1.
- Produces: user-facing licensing guidance and links used by prospective users, contributors, and Registry reviewers.

- [ ] **Step 1: Add a License section after the direct-testing section**

Append this section to `README.md`:

```markdown
## License

The source code authored for this repository is available under the
[MIT License](LICENSE).

ComfyUI, PyTorch, Triton, safetensors, optional comfy-kitchen support, and the
MiniMax H3 / FL2VA model code and model weights are separate third-party
components and retain their respective licenses. This repository does not
include or relicense MiniMax H3 model code or model weights.

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for the licensing
boundary and third-party component notes.
```

- [ ] **Step 2: Verify README wording**

Run:

```bash
grep -F "## License" README.md
grep -F "[MIT License](LICENSE)" README.md
grep -F "[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)" README.md
```

Expected: exactly one matching line for each command.

- [ ] **Step 3: Verify this phase did not touch runtime code**

Run:

```bash
git diff --name-only HEAD~1..HEAD
git status --short
```

Before committing the README, expected pending change list is only `README.md`. After commit, repository status should be clean.

- [ ] **Step 4: Commit the README change**

```bash
git add README.md
git commit -m "docs: document MIT licensing boundary"
```

Expected: one commit modifying only `README.md`.

---

### Task 3: Verify Phase 1 from the repository state

**Files:**
- Verify: `LICENSE`
- Verify: `THIRD_PARTY_NOTICES.md`
- Verify: `README.md`

**Interfaces:**
- Consumes: Tasks 1 and 2.
- Produces: evidence that Phase 1 is complete before Release work begins.

- [ ] **Step 1: Verify the branch contains all three licensing artifacts**

Run:

```bash
test -f LICENSE
test -f THIRD_PARTY_NOTICES.md
grep -qF "## License" README.md
```

Expected: all commands exit 0.

- [ ] **Step 2: Verify no runtime files changed in Phase 1**

Run:

```bash
git diff --name-only main...HEAD
```

Expected licensing-related implementation paths are:

```text
LICENSE
README.md
THIRD_PARTY_NOTICES.md
docs/superpowers/plans/2026-09-18-mit-licensing.md
docs/superpowers/specs/2026-09-18-codex-oss-readiness-design.md
```

No `.py`, requirements, benchmark, workflow, or example files should be changed by this phase.

- [ ] **Step 3: Confirm GitHub can detect the license after merge**

After the branch is eventually merged to `main`, query the repository metadata:

```bash
gh api repos/fishelegs/ComfyUI-H3VAE-PyOpt --jq '.license.spdx_id'
```

Expected after merge: `MIT`.

This last check is intentionally post-merge; before merge the default-branch metadata may still report no license.
