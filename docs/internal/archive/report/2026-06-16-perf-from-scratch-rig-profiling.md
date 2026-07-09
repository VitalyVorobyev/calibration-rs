# Performance profiling — from-scratch Scheimpflug rig calibration (2026-06-16)

## Context

Running the deferred end-to-end parity layer — from-scratch `RigHandeye`
calibration of the large-tilt (~−5°) Scheimpflug rig on the dense `puzzle_board`
`rtv3d_ref` dataset (~170–200 corners/view, 20 views, 6 cameras) — surfaced that
wall-clock was dominated by **dense linear-algebra hot paths, not the
algorithms**. The first two runs took 30–40 min and were killed. This report
records the profiling, the fixes landed in the same session, and the residual
work (Track P in the [backlog](../backlog.md#p--performance--profiling)).

The functional fix this work was validating — breaking the Scheimpflug
tilt/principal-point/distortion degeneracy in the per-camera init
(`optimize_scheimpflug_intrinsics_staged`) — is covered separately and is
unit-validated (synthetic τx=−0.10 recovered exactly from a cold `tilt=0`
start). This report is purely about the performance discoveries that running it
on real dense data exposed.

## Method

`cargo run --release --example rtv3d_ref_rig` (private). The example already
times each pipeline step (`detect`, `intrinsics init`, `intrinsics optimize`,
`rig init/optimize`, `hand-eye init/BA`). When a stage hung, the live process
was sampled with `sample <pid>` to attribute the cost to a function.

## Findings

### F1 — Homography DLT: nalgebra SVD non-convergence (FIXED)

`sample` of the hung process put **1521 / 1524 samples** in a single
`HomographySolver::dlt → nalgebra::linalg::svd::SVD::new → delimit_subproblem`
call. `delimit_subproblem` is the Golub-Kahan implicit-QR sweep; nalgebra's
`SVD::new` runs it with `max_niter = 0` (**unbounded**). On some real
detected-corner matrices the bidiagonal QR **fails to converge**, spinning for
minutes.

Crucially, the earlier `svd(true,true) → svd(false,true)` ("skip U") change was
**insufficient**: skipping U removes the U-side Givens accumulation but the
non-convergence is in the QR iteration itself, which runs regardless of
`compute_u`. A synthetic 225-correspondence test passed only because synthetic
data is well-conditioned; the real detected corners trigger the pathological
path.

**Fix:** solve `A h = 0` as the smallest-eigenvalue eigenvector of the 9×9
normal matrix `AᵀA` via `symmetric_eigen` (which always converges quickly on a
9×9), instead of `A.svd(...)`. With the existing Hartley normalization the
squared conditioning is harmless. This mirrors OpenCV's `findHomography`, which
accumulates a 9×9 `LtL` and eigen-solves it. `crates/vision-calibration-linear/
src/homography.rs`. Result: dense 225-correspondence DLT **1.9 ms** (was a
multi-minute hang); the per-camera linear init now completes in well under a
second per camera.

### F2 — Distortion fit: full SVD least-squares on dense data (FIXED)

`estimate_distortion_from_homographies` (`distortion_fit.rs:291`) solved the
overdetermined `A x = b` (A is `2N×k`, N up to ~3,800/camera, k ≤ 5) via
`a.svd(true,true).solve(&b)` — a full SVD of a ~7,600-row matrix, the same
pathology as F1 at ~19× the height. **Fix:** normal equations
`x = (AᵀA)⁻¹ Aᵀb` (k×k solve). The distortion init is refined downstream by
bundle adjustment, so the squared conditioning is harmless.

### F3 — Iterative init aborts the whole camera on one degenerate view (FIXED)

`estimate_intrinsics_iterative` collected per-view homographies with
`collect::<Result<_>>()?`, so a **single** degenerate view failed the entire
camera. Full-density detection exposed exactly one such view on camera 3 (the
decimated path had never exercised it). **Fix:** `valid_view_homographies` skips
singular/degenerate views and requires ≥ `MIN_VALID_VIEWS` (3) survivors,
keeping views and homographies index-aligned. One bad view in a dense capture no
longer sinks an otherwise well-observed camera. The new homography DLT also
returns `Error::Singular` (not a panic) on a non-finite normal matrix.

### F4 — ~18 latent `svd(true,true)` sites (DOCUMENTED → P1)

The same `svd(true,true)` idiom is used across the linear crate for both
null-space extraction (`camera_matrix.rs:82`, `pnp/dlt.rs:116`,
`epipolar/{fundamental,essential}.rs`) and least-squares (`handeye.rs:228,385`).
None are in the hot rig path today, but each is a latent hang on dense/large
inputs. Tracked as **P1-SVD-SWEEP** (centralize behind `math::null_space` /
`solve_lstsq`).

### F5 — Linear init result + joint-BA cost (PARTLY MEASURED → P2/P3)

After F1–F3, the **linear init for all 6 cameras runs in 9.17 ms** (was a
30+ min hang). Measured stages (`RTV3D_REF_MAXITERS=25`, 6 cameras, full ~190
corners/view):

| stage | time | note |
|-------|------|------|
| detect | ~13 s | puzzle_board detector, 2× upscale (not in scope here) |
| intrinsics init (Zhang, linear) | **9.17 ms** | was a multi-minute hang (F1/F2) |
| intrinsics optimize (staged, per-cam) | — | runs; 4/6 cameras converge, cams 3 & 4 diverge (**F6**) |
| rig init / optimize, hand-eye init / BA | **not reached** | the divergence guard (Phase 3) fires first |

The run now stops cleanly at the per-camera divergence guard:
`per-camera intrinsics diverged for 2 of 6 cameras (camera 3: 247.821 px,
camera 4: 52.245 px); refusing to poison the rig solve`. So the joint
`step_rig_optimize` / `step_handeye_optimize` were **not reached** and their cost
remains unmeasured — but they run on ~45k residuals (6 cameras × 20 views × ~190
corners × 2) with the single-threaded `tiny-solver` LM and an autodiff
(dual-number) Jacobian recomputed every iteration (up to 32 damping retries each
re-evaluating all residuals). Per-iteration cost scales linearly with corner
count, while extrinsics/hand-eye do not need full corner density. Candidate work:
**P2** (principled corner budget for the joint BA), **P3** (backend cost —
analytic/cached/parallel Jacobians). Measuring the BA stages is blocked on F6.

### F6 — From-scratch per-camera Scheimpflug init diverges on cameras 3 & 4 (OPEN → P6)

With the init unblocked, the per-camera staged Scheimpflug solve
(`optimize_scheimpflug_intrinsics_staged`, synthetic-validated to exact recovery)
converges for 4 of 6 real cameras but **diverges on cameras 3 (~248 px) and 4
(~52 px)** — the same two cameras that ran away (`fx→0`) in the original
pre-staging report. Contributing factors: camera 3's iterative linear init falls
back to the distortion-free iteration-0 estimate (F3 best-effort loop), giving the
staged sweep a poorer seed; camera 4 has the fewest detections (3355). The Phase 3
guard correctly rejects both rather than poisoning the rig. Resolving this is a
**robustness** task, not a performance one: better linear seeds for ill-conditioned
cameras, a wider/again-data-driven tilt-sweep, or the gated Euclidean L2 prior
(degeneracy "Rung 4"). Tracked as **P6** (or folded into the degeneracy work).

## Root cause (systemic)

1. **`nalgebra::svd(true,true)` is used pervasively** for null-space and
   least-squares. Its unbounded Golub-Kahan QR can fail to converge on real
   dense matrices, and even when it converges it accumulates U across every row.
   Null-space problems should use `AᵀA` + symmetric eigen (or `svd(false,true)`
   where convergence is safe); least-squares should use normal equations / QR.
2. **The `tiny-solver` backend** recomputes a dual-number Jacobian over every
   residual each LM iteration, re-evaluates all residuals on each damping retry,
   and is single-threaded — so joint-BA cost is linear in corner count with no
   parallelism.
3. **No data-density control for the joint BA** — the pipeline runs every stage
   at full corner density even though the rig/hand-eye stages converge on a
   fraction of the corners.

## Landed this session

- **F1** homography null-space via 9×9 `AᵀA` symmetric eigen, replacing
  `A.svd(...)` (`homography.rs`; `vision-geometry` still pending). NaN-safe:
  returns `Error::Singular` on a non-finite normal matrix instead of panicking.
- **F2** distortion fit via ridge-regularized normal equations + non-finite-point
  guard (`distortion_fit.rs`) — fast and cannot hang on a degenerate/NaN design.
- **F3** view-tolerant, best-effort iterative init (`iterative_intrinsics.rs`):
  skips degenerate views (min 3), and an iteration that collapses keeps the last
  good estimate rather than failing the camera.
- Net: linear init **30+ min hang → 9.17 ms**; the pipeline now fails fast and
  precisely (Phase 3 guard) on the genuinely hard cameras instead of hanging.

## Next (Track P)

P1 SVD sweep (finish the ~18 remaining sites), P2 joint-BA corner budget, P3
tiny-solver backend cost, P4 criterion guards, P5 per-stage timing
instrumentation, **P6** from-scratch convergence for cameras 3 & 4 (robustness;
unblocks measuring the joint-BA stages). See
[backlog](../backlog.md#p--performance--profiling).
