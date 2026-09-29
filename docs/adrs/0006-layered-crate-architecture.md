# ADR 0006: Layered Crate Architecture

- Status: Accepted
- Date: 2026-03-07 (retroactive)

## Context

Camera calibration involves distinct algorithmic layers: math primitives, closed-form initialization, iterative refinement, and workflow orchestration. Mixing these layers leads to tangled dependencies and makes it hard to use parts of the library independently.

## Decision

Organize the workspace as a strict layered DAG:

```
vision-calibration (facade)
    |
    +-- vision-calibration-pipeline (session workflows)
    |       |
    +-------+-- vision-calibration-optim (non-linear refinement)
    |       |
    +-------+-- vision-calibration-linear (closed-form solvers)
    |       |
    +-------+-- vision-mvg (N-view MVG: bundle adjust, rectification, dense stereo)
                |
                +-- vision-geometry (two-view solvers)
                    |
                    +-- vision-calibration-core (types, models, RANSAC)
```

Rules:
- **core** has no workspace dependencies. Minimal external deps.
- **linear** and **optim** depend on core but NOT on each other.
- **vision-geometry** (deterministic two-view solvers: `homography`, `epipolar`, `camera_matrix`, `triangulation`) depends only on core.
- **vision-mvg** (pipelines over `vision-geometry`: robust estimation, pose recovery, bundle adjustment, rectification, dense stereo) depends on core + `vision-geometry`.
- **linear** and **pipeline** also depend on `vision-geometry`; **optim** has no (non-dev) edge to `geometry`, `linear`, or `mvg`.
- **pipeline** depends on core, linear, optim, and geometry.
- **facade** re-exports from pipeline, `vision-geometry`, and `vision-mvg`.
- **vision-calibration-py** depends only on the facade crate.

`dlt_homography` (and the rest of `vision-geometry`) stays in `vision-geometry`, not `vision-calibration-core::linalg`. `core::linalg` holds the *primitive* math shared by both crates (`normalize_points_2d`/`_3d`, `null_space`, the polynomial solvers); the higher-level DLT/epipolar/triangulation *solvers* have one consumer family (`linear`, `pipeline`, `mvg`, the facade), all of which already depend on `geometry`, so moving them into `core` would bloat the zero-workspace-dependency foundation crate for no benefit.

## Consequences

- Users can depend on just `vision-calibration-core` for types, or just `vision-calibration-linear` for solvers, without pulling in optimization or pipeline machinery.
- The facade crate is the stability boundary: lower crates may evolve faster.
- Adding a new solver layer (e.g., a different optimizer backend) doesn't affect linear or core.
