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

- **No in-house dense stereo matcher.** Dense matching, if ever added, wraps an
  external library (`opencv-rust` SGBM) behind a feature flag (Track C5). The
  workspace does not ship a hand-rolled dense matcher.
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
