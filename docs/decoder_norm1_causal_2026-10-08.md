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

## Final controlled result

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

The norm1 full-graph attribution remains **incomplete**, and no optimization is
adopted. The immediate remaining check is the B/C and A/Dmatch control after
repairing the observer, before any full-stage performance claim.

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
are separate from the private CPU observer replay and the incomplete GPU
causal diagnostic. Production runtime code is unchanged.

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
