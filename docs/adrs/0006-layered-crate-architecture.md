# ADR 0006: Layered Crate Architecture

- Status: Accepted
- Date: 2026-03-07 (retroactive)

## Context

Camera calibration involves distinct algorithmic layers: math primitives, closed-form initialization, iterative refinement, dataset/detector I/O, and workflow orchestration. Mixing these layers leads to tangled dependencies and makes it hard to use parts of the library independently.

## Decision

Organize the workspace as a strict layered DAG:

```
vision-calibration (facade: re-exports only)         vision-calibration-py (PyO3)
    |                                                      |
    +-- vision-calibration-pipeline (sessions, problem types, dataset_runner)
    |       |
    |       +-- vision-calibration-dataset (DatasetSpec manifest, validator, sniffer)
    |       +-- vision-calibration-detect  (Detector trait + detection cache)
    |       +-- vision-calibration-optim   (non-linear refinement)
    |       +-- vision-calibration-linear  (closed-form solvers)
    |       +-- vision-geometry            (two-view solvers)
    |
    +-- vision-mvg (N-view MVG: pose recovery, triangulation, bundle adjust,
    |               rectification, dense stereo)
    |       +-- vision-geometry
    |
    +-- vision-calibration-core (types, models, RANSAC, linalg) -- bottom layer

vision-calibration-bench, vision-calibration-examples-private: unpublished consumers
```

Rules:
- **core** has no workspace dependencies. Minimal external deps.
- **linear**, **optim** and **vision-geometry** depend on core. **linear** and **optim** are peers: neither depends on the other. **linear** may also use `vision-geometry`.
- **vision-geometry** (deterministic two-view solvers: `homography`, `epipolar`, `camera_matrix`, `triangulation`) is the foundation for the geometry stack.
- **vision-mvg** (pipelines over `vision-geometry`: robust estimation, pose recovery, N-view triangulation, bundle adjustment behind the `refine` feature, rectification, dense stereo) depends on core + `vision-geometry`.
- **dataset** (manifest, validator, folder sniffer) and **detect** (chessboard / ChArUco / puzzleboard / ring-grid detectors behind the sealed `Detector` trait, plus the detection cache) are the manifest/detector layer feeding the pipeline's `dataset_runner`. `detect` stays nalgebra-free at its boundary (plain arrays), so detector crates may use a different nalgebra than the solver stack.
- **pipeline** depends on core, linear, optim, geometry, dataset and detect. It does not depend on `vision-mvg`.
- **facade** re-exports from pipeline, `vision-geometry` and `vision-mvg`; it contains no logic and is the compatibility boundary.
- **vision-calibration-py** depends on the facade (preferred) plus binding tooling.
- **vision-calibration-bench** and **vision-calibration-examples-private** are unpublished consumers of the facade and pipeline; `bench` also reaches core/optim/dataset directly, confined to itself.

`dlt_homography` (and the rest of `vision-geometry`) stays in `vision-geometry`, not `vision-calibration-core::linalg`. `core::linalg` holds the *primitive* math shared by both crates (`normalize_points_2d`/`_3d`, `null_space`, the polynomial solvers); the higher-level DLT/epipolar/triangulation *solvers* have one consumer family (`linear`, `pipeline`, `mvg`, the facade), all of which already depend on `geometry`, so moving them into `core` would bloat the zero-workspace-dependency foundation crate for no benefit.

## Consequences

- Users can depend on just `vision-calibration-core` for types, or just `vision-calibration-linear` for solvers, without pulling in optimization or pipeline machinery.
- The facade crate is the stability boundary: lower crates may evolve faster.
- Adding a new solver layer (e.g., a different optimizer backend) doesn't affect linear or core.
- Detector crates can use their own nalgebra version because `detect` exchanges plain arrays.
