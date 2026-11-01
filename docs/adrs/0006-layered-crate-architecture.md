# ADR 0006: Layered Crate Architecture

## Context

Calibration spans math primitives, closed-form initialization, non-linear
refinement, multiple-view geometry, dataset and detector I/O, and workflow
orchestration. Mixed in one crate, every consumer pulls in the solver stack
and every change ripples everywhere.

## Decision

The workspace is a strict layered DAG. AGENTS.md §1 is the authoritative
list of allowed dependency edges; this ADR records the shape and the
reasons.

```
vision-calibration (facade: re-exports only)       vision-calibration-py
    |                                                   |
    +-- vision-calibration-pipeline (sessions, problem types, dataset_runner)
    |       +-- vision-calibration-dataset  (manifest, validator, sniffer)
    |       +-- vision-calibration-detect   (detectors, detection cache)
    |       +-- vision-calibration-optim    (non-linear refinement)
    |       +-- vision-calibration-linear   (closed-form solvers)
    |       +-- vision-geometry             (deterministic two-view solvers)
    +-- vision-mvg (N-view pipelines over vision-geometry)
    +-- vision-calibration-core (types, camera models, RANSAC, linalg)
```

- **`core` has no workspace dependencies.** It holds primitives shared by
  everything (normalization, null space, polynomial solvers). Solvers built
  on them live above it even when widely used: `dlt_homography` and the
  epipolar and triangulation solvers live in `vision-geometry`, whose
  consumers all depend on it anyway.
- **`linear` and `optim` are peers.** Initialization and refinement never
  depend on each other; the pipeline composes them.
- **Geometry is two crates.** `vision-geometry` holds deterministic solvers
  only: no robust loops, no domain policy. `vision-mvg` holds the pipelines
  over them: robust estimation, pose recovery, N-view triangulation, bundle
  adjustment (behind `refine`), rectification and dense stereo. `linear`
  uses `vision-geometry` instead of keeping parallel copies.
- **`detect` exchanges plain arrays**, so detector crates may use a
  different nalgebra than the solver stack. Its boundary is typed: a closed
  `DetectorConfig` enum (one variant per detector) reading an 8-bit
  `GrayImageView`, so neither `image` nor `serde_json` is in its API.
- **The facade is the compatibility boundary.** It re-exports and holds no
  logic. The pipeline does not depend on `vision-mvg`; the facade
  re-exports it directly. Python binds the facade.
  - It re-exports chosen items, never a lower crate wholesale: the
    two-view solvers sit at the root of `mvg`, and `linear` and `geometry`
    are used directly.
  - `nalgebra` is the one public dependency, re-exported as
    `vision_calibration::nalgebra`; its major version moves only with a
    breaking release of ours.
  - What the workspace's own crates need from each other but users do not
    (optim's `optimize_*` and their options, core's parameter masks) is
    `#[doc(hidden)]`: callable, undocumented, outside the semver promise.
    Test-data generators sit behind the `synthetic` feature.

### Scope ceiling of `vision-mvg`

`vision-mvg` is geometry over already-calibrated rigs. It does not grow into
structure-from-motion: no incremental SfM, pose graphs or loop closure.
Dense matching is pure Rust (block matching and SGM); no published crate
depends on C++ (OpenCV).

## Consequences

- A consumer can depend on `core` or `linear` alone without the optimizer.
- Lower crates may evolve faster than the facade; their hidden items may
  change in any release.
- A new solver backend touches `optim` only.
