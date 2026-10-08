# Norm1 full-graph controls and cache replay — 2026-10-08

This follow-up separates compiler scheduling from the proposed norm1 fusion.
It does not publish a new full-decoder speedup. The adopted records remain
INT8 encode **8.458108 s**, INT8 decode **6.508537 s** and fused FP16 decode
**10.923379 s**.

The [preceding experiment](decoder_norm1_r1024_2026-10-08.md) established a
7.07% local-chain gain for fixed R1024/two-warp arithmetic and 35 exact layer
shadows, but independent natural full-video compilation failed the RGB screen.
Those checks did not establish equivalence to the original natural W16 graph.

## What the controls test

All variants use the identical saved FP16 latent and its original channels-last
stride, the same weights, tile256/batch4 and 124 frames at 768×1344. No resize,
crop, frame dropping or re-encoding is used. These are diagnostic executions,
with no latency measurement.

| Variant | Target residual/norm1 | Other compiled reductions | Purpose |
| --- | --- | --- | --- |
| A | Original captured W16 | Original | Must reproduce the retained full RGB hash exactly |
| B | Independent native W2 plus public CK quantization | Same original compiled object | Isolate changing target arithmetic |
| C | Authored fused W2 in the ordinary candidate module | Bound to A | Require B/C boundaries and full RGB exact |
| D | Captured A W16 plus public CK, in the same candidate graph | Candidate natural selections | Observe other scheduling changes |
| Dmatch | Same target and candidate graph as D | Bound to A | Require A/Dmatch boundaries and full RGB exact |

The first target call for each of 35 recurrent layers checks H/O/G/W input
identity and H/norm/Q/FP32-scale/QKV output identity. Full coverage separately
requires 49 calls per layer, 1,715 total. Input, checkpoint, quantized-weight
storage and model-tensor identities are guarded. All hooks are private
benchmark controls and are restored; they are not runtime changes.

## Cache copies need reconstructed compiled objects

The first A replay stopped before any candidate: its RGB SHA did not match the
retained baseline. CPU comparison gave global PSNR **56.657280 dB** and maximum
absolute delta **0.133492**. These values are informational; the control required
exact equality and was not waived.

A byte-identical cache directory copy retained serialized FX graph objects with
absolute filenames and autotuner save hooks into the original cache. The replay
changed five original `.best_config` files. Two launch tuples changed, including
an untouched norm from X2/R2048/W16 to X1/R2048/W16; three changes affected only
saved tuning-time fields. No live launcher trace was captured in this failed A,
so the RGB delta is not attributed to one changed operation.

All five mutated files were saved separately before checksum-verified restoration.
The original baseline and candidate manifests returned to their recorded values.
A separately registered cache-only repair removed FX/AOT serialized objects
from new private copies while retaining generated sources, tuned configurations
and Triton artifacts. It reconstructed private filename/save-hook ownership.

That repaired A **exactly reproduced the retained baseline RGB SHA**, and both
original cache manifests remained unchanged. The subsequent observer stopped at
a source-identity assertion: the diagnostic compared an AST source segment
without its trailing newline with Triton’s full JIT source. CPU reconstruction
proved their function bodies identical and the JIT text exactly equal to the
AST segment plus one newline. The failed report was preserved. A separate
validation-only correction uses the already archived full-source SHA; it changes
no arithmetic, configuration or output-quality gate.

## Earlier observer attempt

The corrected-source attempt reproduced A exactly again. Its pass-through
observer also reproduced the raw first-group input/output exactly and completed
independent native W16 H/norm checks for all **35 target layers**. H/norm/Q/FP32
scale/QKV boundary hashes and actual selected launchers were recorded.

| Live operation in A’s first group | Selected launch |
| --- | --- |
| Target norm1 `_8`, all 35 recurrent layers | X1 / R2048 / W16 / stage1 |
| Untouched norm `_4` | X2 / R2048 / W16 / stage1 |
| Final LayerNorm `_9` | X1 / R2048 / W8 / stage1 |

These are live selected launchers, rather than tuning candidates. Selected
register/shared-memory/spill data remain unknown.

B then started its first native W2 target call, but its Python observer read
`item['norm']` instead of the actual nested `item['boundaries']['norm']` and
raised `KeyError` before Q/S/Y capture. **No full B RGB or B/C/A/Dmatch comparison
exists.** C, D and Dmatch did not run. This is an instrumentation failure;
it is not a numerical or quality rejection of the candidate.

The original error was reproduced with the extracted observer and CPU tensors.
An in-memory two-lookup correction passes matching-input capture, pending-record
consumption, unmatched-layer pass-through and explicit mismatch failure checks.
The failed GPU runner/report remain immutable; the small correction is retained
as an unapplied private diff. It has not completed GPU validation.

Across the three attempts, three full A decodes completed, one B decode started
and aborted, and three tile replays completed (one additional observer replay
aborted). Every terminal attempt was preserved. The cache reconstruction and
source-identity correction were separately registered; the latter explicitly
amended the earlier attempt budget after a proven diagnostic text-representation
error. No kernel math, numerical threshold or input was adjusted.

At that point, full-graph attribution remained incomplete and no optimization
was adopted. The separately registered observer repair below completes A/B.

## Same-graph A/B result

A fresh diagnostic fixes the nested observer lookup, layer serialization and
stream-check ordering. CPU fixtures verify record capture/consumption, explicit
mismatch rejection and hook restoration. Kernel arithmetic and quality limits
remain unchanged. Execution commit `aac3fc2` differs from the archived design
commit only in four documentation/README files.

A again reproduces the retained full RGB exactly. B completes **49 calls per
layer × 35 layers = 1,715 calls**, replacing only the target norm1 reduction
with independent native W2 plus public CK quantization in the **same compiled
A object**. Its finalized CPU RGB is compared with A without re-encoding.

| Full 124-frame A/B metric | Observed | Existing primary limit |
| --- | ---: | ---: |
| Global RGB PSNR | **52.069021 dB** | ≥55 dB |
| Mean frame PSNR | **52.164503 dB** | ≥55 dB |
| Worst frame PSNR | 50.742132 dB | ≥50 dB |
| Maximum absolute RGB delta | 0.1355713 | ≤0.15 |
| Temporal delta RMSE | 0.00237026 | ≤0.003 |

The primary and stricter cross-schedule screens both fail. The source-relative
screen passes: mean source PSNR changes by +0.000885 dB, worst per-frame change
is −0.011441 dB, and no extra frames fall below 30 dB. That source result does
not waive the failed A/B RGB screen or establish perceptual equivalence.

The first group localizes propagation: H/O/G/W inputs at layer 1 are exact;
its FP16 norm differs at **832 values**, with maximum delta **0.0009765625**,
while H/Q/FP32 scale/QKV output remain exact. Q and QKV first differ at layer 2;
H and scale first differ at layer 3. Thus changing the target reduction alone
is sufficient to fail the RGB screen; the earlier independently compiled
worker was not needed to establish this result. The **7.07% W2 local gain is
not eligible for adoption**.

The candidate tile probe separately completes native W2 checks for all 35
layers: actual nondebug H/Q/FP32 scale/QKV are exact to that oracle; norm is
observed through the debug helper on a cloned H. It then stops at the untouched
source-set gate, before full C, D or Dmatch. Live candidate source-set details
were not persisted before that gate. This is neither full fused-graph equality
nor a W16 fusion result. Both retained cache manifests remain exact and hooks
are restored; the final all-model-tensor check was not reached.

This attempt completes two full decodes and three tile probes, with **no
timing**. The [separate redacted delta archive](benchmarks/decoder_norm1_observer_2026-10-08.json)
preserves its plan, CPU checks, raw terminal failure, root reviews and full
per-frame metrics. It references the earlier archive by SHA instead of copying
those attempts again. The W16 follow-up below validates its controlled full
graph, while natural-graph acceptance and performance remain unproven.

## W16 fusion integration

The W16 candidate keeps the original target arithmetic and the already measured
4.43% local-chain gain. Its private module changes only the norm warp count and
helper path from the W2 candidate; tuple carry, FFN warps and reuse of 144 INT8
weight storages remain unchanged. The registered validation uses A, C (the
fused candidate with untouched reductions bound to A), and N (the same candidate
with those bindings restored to its natural selections). C requires exact
A boundaries and full RGB; N must pass the primary, strict and source-relative
quality screens before any performance test.

The first attempt completes the retained A decode and two exact A tile anchors.
All **35 candidate W16 first-call shadows** pass actual nondebug H/Q/FP32 scale/
QKV comparisons to native W16; the norm comparison uses a debug helper on a
cloned H. The candidate tile then aborts before the final LayerNorm launch:
the observer requires a unique candidate launcher before its first ordinary
`run()`. The terminal record does not preserve whether that count was zero or
multiple. **C and N full decodes, quality screening and timing do not run.**
This is a host observer failure, not a numerical rejection of W16 fusion.

Installed PyTorch's `CachingAutotuner.run` precompiles an empty launcher set and
autotunes multiple launchers before selecting one. A separate validation-only
repair therefore checks candidate/A object separation before the ordinary run
and unique frozen selection afterwards. It retains the warmed A reference check,
source identities, stream and alias checks, kernel arithmetic and all quality
limits. The original failed runner, report and cache manifests are preserved.
The [W16 attempt archive](benchmarks/decoder_norm1_w16_2026-10-08.json) records
the first failure and root review; it does not claim a completed fused graph.

### Corrected A/C/N result

A separately registered lifecycle repair passes eight CPU cases extracted from
the actual observer, with no CUDA calls. Root independently replays those cases
and the exact binding class, verifies all 38 frozen input hashes and runs the
critical Ruff scope. The revised GPU run completes **three full decodes and
three tile probes**, with no timing or manual configuration changes.

A reproduces the retained RGB exactly. Candidate first-group checks pass for all
35 W16 targets. **C completes 49 calls per layer, 1,715 total, with all 35
first-group input and boundary checks exact to A, and full RGB bitwise equal.**
This establishes complete W16 fused-graph equivalence under the registered
unchanged-norm binding. The diagnostic binding is not a production feature.

N runs the **same enabled candidate graph**, after restoring its own untouched
reduction objects. Its 1,715 target calls complete. Natural tuner/launcher
objects are distinct from A, selected sources/configurations stay stable from
tile to full decode, and no A configuration is forced.

| Untouched reduction | A | Natural candidate tile and N |
| --- | --- | --- |
| First-block norm `_4` | X2 / R2048 / W16 / stage1 | X1 / R2048 / W16 / stage1 |
| Final LayerNorm, A `_9` / candidate `_8` | X1 / R2048 / W8 / stage1 | X1 / R2048 / W8 / stage1 |

The final LayerNorm source differs only in its generated identifier, verified
by an exact single-pair source/AST proof; its other compiler metadata, arguments,
aliases and stream checks remain guarded. In this run its first ordinary call
naturally reduces **eight prelaunchers to one**. This recorded count belongs to
the repaired run; the failed first attempt's count remains unknown. The final
R2048 selection also belongs to this run, rather than the preceding independent
worker's R1024 selection.

| Full 124-frame A/N metric | Observed | Primary limit | Strict limit |
| --- | ---: | ---: | ---: |
| Global RGB PSNR | **56.657280 dB** | ≥55 dB | ≥65 dB |
| Mean frame PSNR | **56.947183 dB** | ≥55 dB | ≥65 dB |
| Worst frame PSNR | 54.024690 dB | ≥50 dB | ≥60 dB |
| Maximum absolute RGB delta | 0.1334922 | ≤0.15 | ≤0.02 |
| Temporal delta RMSE | 0.00133206 | ≤0.003 | ≤0.00075 |

N passes the primary screen and source-relative screen, but **fails the
preregistered strict screen**. Source mean PSNR changes by +0.001186 dB, worst
per-frame source change is −0.004030 dB, and no additional frames fall below
30 dB. Root independently reloads the finalized CPU RGB and original saved
source, verifies their hashes, reproduces all three quality components exactly,
and separately verifies A/C tensor equality. The strict gate is not waived.

C/N first-group hashes already differ in Hpre and O at the **first recurrent
target**; G and W remain identical across all 35 targets. Those input changes
precede the recurrent fused arithmetic. The controlled C/N pair isolates the
untouched-reduction binding group; assigning the delta to `_4` alone still
needs a standalone oracle/capture. A static wrapper audit identifies `_4` as
block0's ordinary norm1: its embedding/token input is normalized to FP16 and
then immediately consumed by CK QKV linear. It has no preceding FFN residual;
the 35 fused inter-block targets start at block index 1. The next minimal
candidate is therefore a separate RMS-only FP32→FP16 helper with the original
X2/R2048/W16 arithmetic, continuing through the existing CK QKV path. It first
needs same-input norm and QKV equality; a residual helper with zero inputs is
not an equivalent contract. Its natural XBLOCK changed while the final
LayerNorm tuple did not, but this proposed helper has not run.

Model parameters/buffers and candidate quantized storage identities remain
unchanged; hooks are restored and both retained source-cache manifests match.
Including the preserved first failure, this W16 window uses four full decodes
and six tile-call attempts (five completed, one aborted). No full-stage timing,
eight-video acceptance or offload validation is run, and the **4.43% local gain
remains unadopted**. The [separate repaired-run archive](benchmarks/decoder_norm1_w16_launcher_2026-10-08.json)
retains the terminal rejection, CPU quality check, lifecycle proof, budget
amendment and reviews, referencing the first W16 archive by SHA.

## Encoder fusion feasibility

A separate CPU audit considered eliminating the encoder producer’s second
quantized-write pass by computing normalized activations inside convolution.
The required global-max/scale pass would remain. The current 3×3×3 consumer
would recompute the same activation approximately **25 times** on average at
the high-resolution C128→128 geometry, and approximately **49 times** with two
output-channel tiles. A straightforward fused loader is therefore deferred;
a candidate first needs quantized-patch reuse inside each CTA.

The historical profile attributes 1.032375 s to both producer passes together,
not just the potentially removable write pass. It cannot be used as a saving
forecast. Geometry records are available, but no raw live activation/statistics/
weight capture was found in the bounded experiment directories. The audit gives
a capture plan and requires exact q/FP32-scale/convolution-output checks before
any complete producer-plus-convolution local timing. No encoder candidate was
run or adopted in this follow-up.

## Evidence and limits

The [redacted evidence archive](benchmarks/decoder_norm1_causal_2026-10-08.json)
keeps each plan, CPU preparation, terminal attempt, root review, cache repair,
source audit, observer CPU reproduction and encoder feasibility record separate.
Numeric, boolean and standalone full-SHA leaves are preserved per record;
private paths/media, tensor payloads, generated third-party source and the
private diagnostic harness are excluded.

A clean tracked-source snapshot passes compileall, the required critical Ruff
scope and 105 repository tests (91 passed; 14 CUDA checks skipped). These checks
are separate from the private CPU observer replay and GPU causal controls.
Controlled W16 equivalence passes, while natural-graph acceptance remains
incomplete. Production runtime code is unchanged.

The earlier, separately compiled natural workers have cache-associated final
LayerNorm selections: baseline X1/R2048/W8 versus candidate X1/R1024/W8,
with the same source SHA. This describes the preceding experiment; the live A
selections captured in this follow-up are listed above. Together with the
untouched XBLOCK change, the earlier cache records show additional scheduling
differences, not proof that any one explains the earlier RGB loss.

Environment: RTX PRO 5000 72GB/SM120, Linux, Python3.12.14,
PyTorch2.11.0+cu130, CUDA13.0, Triton3.6.0, comfy-kitchen0.2.34,
FP32 norm enabled, FP16 accumulation disabled, SDPA auto. Baseline commit
`c40ccb0`. GPU executions hold the shared benchmark lock. No new Nsight or
NCU capture, full-stage timing, eight-video acceptance or offload validation
is claimed for these diagnostic runs. Production code and performance charts
remain unchanged.
