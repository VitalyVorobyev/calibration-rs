# ADR 0015: Multiple-View Geometry Crates and Scope Ceiling

- Status: Accepted
- Date: 2026-06-14

## Context

Two-view and multiple-view geometry (fundamental/essential matrices, homography,
triangulation, camera-matrix decomposition, robust pose recovery) had grown
inside `vision-calibration-linear`. PR #28 proposed splitting it into two
dedicated crates on a long-lived `mvg` branch. That branch then diverged ~136
commits behind `main` — it predates `vision-calibration-detect`,
`vision-calibration-dataset`, `vision-calibration-bench`, and the Tauri app, and
it carried a `vision-calibration-linear` refactor (moving solvers *out* of
linear) that now conflicts with the many `main` callers added since. Merging the
branch is high-risk; the **crate source itself** is clean and well-scoped.

The MVG track (Track C in `docs/ROADMAP.md`) was postponed behind the Tauri
diagnose viewer, which has since matured. This ADR also exists to cap the track's
ceiling explicitly, as the roadmap promised.

## Decision

1. **Land two new crates, ported fresh and additively** from the `mvg` branch
   source (a source port, not a branch merge) onto current `main`:
   - **`vision-geometry`** — deterministic, allocation-light geometric solvers:
     `math` (Hartley normalization, polynomial/SVD helpers), `epipolar`
     (fundamental, essential, decomposition), `homography`, `triangulation`,
     `camera_matrix`. Depends only on `vision-calibration-core` + `nalgebra` +
     `anyhow`.
   - **`vision-mvg`** — pipelines and estimation over the deterministic solvers:
     `pose_recovery`, robust estimation, `cheirality`, `degeneracy`,
     `triangulation`, `residuals`, `homography`, optional nonlinear `refine`
     (behind a `refine` feature → `tiny-solver`). Depends on `vision-geometry` +
     core.

2. **Do not refactor `vision-calibration-linear` in this slice.** The geometry
   crates are purely additive; `linear` keeps its existing epipolar/homography/
   triangulation code. This guarantees zero regression to the validated
   calibration paths. De-duplicating `linear` onto `vision-geometry` (having
   `linear` depend on `vision-geometry`) is an explicit **follow-up**, not part
   of landing the crates.

3. **`publish = false` for now.** Both crates land internally first (their public
   API is not yet stable). Promoting them to the crates.io publish set and the
   release version-lockstep (see `CLAUDE.md`) is a deliberate later step.

4. **Python bindings deferred** (consistent with Track A5): no PyO3 wrappers in
   this slice; revisit after the Rust API stabilises.

## Scope ceiling (explicit, per the roadmap)

- **Dense stereo matcher: pure-Rust in the library; OpenCV is benchmark-only**
  (amended 2026-06-21 — see [Amendments](#amendments); the original ceiling
  forbade an in-house matcher and mandated wrapping `opencv-rust` SGBM). Track C5
  ships a **pure-Rust** dense matcher in `vision-mvg`. `opencv-rust` SGBM appears
  **only** in the unpublished `vision-calibration-bench` crate as a
  quality-baseline reference for that matcher — **never** in a published crate,
  and never as the shipped implementation.
- **No full structure-from-motion.** No incremental SfM, no global pose-graph
  optimisation, no loop closure. `vision-mvg` targets geometry over already-
  calibrated rigs (N-view triangulation, BA with frozen intrinsics,
  rectification), not unconstrained reconstruction.
- `vision-geometry` stays *deterministic solvers only* (no robust loops, no
  domain policy); robust estimation and pipelines live in `vision-mvg`.

## Consequences

- Unblocks Track C: C2 (N-view triangulation + nonlinear refinement) and C3
  (bundle adjustment, frozen intrinsics) build on these crates next.
- Temporary duplication between `vision-calibration-linear` and `vision-geometry`
  until the de-dup follow-up; both are tested independently, so behaviour is
  pinned on each side.
- Landing is low-risk and reversible (additive new crates, `publish = false`),
  appropriate for an autonomous batch; the riskier branch merge is avoided
  entirely.

## Amendments

### 2026-06-21 — dense matcher is pure-Rust; OpenCV is benchmark-only

The original ceiling (above) said the workspace would **not** ship a hand-rolled
dense matcher and that dense matching, if added, would wrap `opencv-rust` SGBM
behind a feature flag. That is reversed.

**Decision (user, 2026-06-21):** the production stack is Rust-native, so a
heavy C++ OpenCV dependency must not live in any *published* crate. OpenCV's
value here is as a **solid algorithmic baseline to benchmark against**, not as
the shipped implementation. Track C5 therefore:

- ships a **pure-Rust** dense stereo matcher in `vision-mvg`;
- uses `opencv-rust` SGBM **only** inside the unpublished
  (`publish = false`) `vision-calibration-bench` crate, behind an off-by-default
  feature, as the quality yardstick;
- validates dense reconstruction on **existing calibration-target data**: a
  C4-rectified stereo pair has a known target-plane depth (from the
  calibration), giving ground truth without any new capture.

**Sequencing:** the benchmark harness lands first — the `DenseMatcher` trait,
the synthetic/rectified ground-truth fixtures + error metrics, and the OpenCV
SGBM baseline — so the Rust matcher has a yardstick the moment it is written.

This keeps the `vision-mvg` ceiling otherwise intact (no SfM / pose-graph / loop
closure); it only moves the dense-matcher line from "wrap OpenCV" to "Rust in
the library, OpenCV in the bench."
