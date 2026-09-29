# ADR 0015: Multiple-View Geometry Crates and Scope Ceiling

- Status: Accepted
- Date: 2026-06-14

## Context

Two-view and multiple-view geometry (fundamental/essential matrices, homography,
triangulation, camera-matrix decomposition, robust pose recovery) had grown
inside `vision-calibration-linear`. Splitting it into dedicated crates keeps the
calibration solvers focused and lets the geometry be used on its own. This ADR
also caps the scope of that work explicitly.

## Decision

1. **Two crates own multiple-view geometry:**
   - **`vision-geometry`** — deterministic, allocation-light geometric solvers:
     `math` (Hartley normalization, polynomial/SVD helpers), `epipolar`
     (fundamental, essential, decomposition), `homography`, `triangulation`,
     `camera_matrix`. Depends only on `vision-calibration-core` + `nalgebra` +
     `anyhow`.
   - **`vision-mvg`** — pipelines and estimation over the deterministic solvers:
     `pose_recovery`, robust estimation, `cheirality`, `degeneracy`,
     `triangulation`, `residuals`, `homography`, bundle adjustment,
     rectification, dense stereo, and optional nonlinear `refine` (behind a
     `refine` feature -> `tiny-solver`). Depends on `vision-geometry` + core.

2. **`vision-calibration-linear` consumes `vision-geometry`** for its
   homography/epipolar/camera-matrix/triangulation solvers rather than keeping
   parallel copies (see [ADR 0006](0006-layered-crate-architecture.md)).

3. **Python bindings** do not wrap these crates directly; they are reached
   through the facade.

## Scope ceiling

- **Dense stereo matcher: pure-Rust in the library.** `vision-mvg::dense`
  ships block matching (`match_block`) plus optional SGM. The production stack
  is Rust-native, so no C++ dependency lives in any published crate; OpenCV
  SGBM is not a dependency of the workspace. Dense reconstruction is validated
  on existing calibration-target data: a rectified stereo pair has a known
  target-plane depth from the calibration, giving ground truth without new
  capture.
- **No full structure-from-motion.** No incremental SfM, no global pose-graph
  optimisation, no loop closure. `vision-mvg` targets geometry over already-
  calibrated rigs (N-view triangulation, BA with frozen intrinsics,
  rectification), not unconstrained reconstruction.
- `vision-geometry` stays *deterministic solvers only* (no robust loops, no
  domain policy); robust estimation and pipelines live in `vision-mvg`.

## Consequences

- Each side of the `linear`/`geometry` boundary is tested independently, so
  solver behaviour is pinned where it lives.
- The ceiling keeps `vision-mvg` a geometry library over calibrated rigs, not a
  reconstruction framework.
