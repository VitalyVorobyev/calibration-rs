# ADR 0006: Layered Crate Architecture

- Status: Accepted (amended 2026-07-04)
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
                |
                +-- vision-calibration-core (types, models, RANSAC)
```

Rules:
- **core** has no workspace dependencies. Minimal external deps.
- **linear** and **optim** depend on core but NOT on each other.
- **pipeline** depends on core, linear, and optim.
- **facade** re-exports from pipeline (and transitively from all layers).
- **vision-calibration-py** depends only on the facade crate.

## Consequences

- Users can depend on just `vision-calibration-core` for types, or just `vision-calibration-linear` for solvers, without pulling in optimization or pipeline machinery.
- The facade crate is the stability boundary: lower crates may evolve faster.
- Adding a new solver layer (e.g., a different optimizer backend) doesn't affect linear or core.

## Amendments

### 2026-07-04 — `vision-geometry` / `vision-mvg` join the DAG; peer rule intact

[ADR 0015](0015-mvg-ceiling.md) (2026-06-14) added two crates the diagram
above predates, and PR #72 (2026-06-21, the C1-FOLLOWUP solver dedup) wired
them into the calibration chain. The original diagram is otherwise still
accurate; this amendment records the crates and edges it omits, verified
in-tree at HEAD (2026-07-04):

- **`vision-geometry`** — deterministic two-view solvers (`homography`,
  `epipolar`, `camera_matrix`, `triangulation`). Depends only on
  `vision-calibration-core` (`crates/vision-geometry/Cargo.toml`
  `[dependencies]`).
- **`vision-mvg`** — pipelines over `vision-geometry` (robust estimation,
  pose recovery, bundle adjustment, rectification, dense stereo). Depends on
  `vision-calibration-core` + `vision-geometry`
  (`crates/vision-mvg/Cargo.toml` `[dependencies]`).
- Verified dependency edges (each crate's `[dependencies]` table read
  directly):
  - `vision-calibration-linear` → `vision-geometry`
    (`crates/vision-calibration-linear/Cargo.toml`).
  - `vision-calibration-pipeline` → `vision-geometry`
    (`crates/vision-calibration-pipeline/Cargo.toml`).
  - `vision-calibration` (facade) → `vision-geometry` **and** →
    `vision-mvg` (`crates/vision-calibration/Cargo.toml`).
- **The linear/optim peer rule above still holds.**
  `vision-calibration-optim`'s `[dependencies]` list only
  `vision-calibration-core` plus solver/serde deps (`tiny-solver`, `faer`,
  `faer-ext`, `serde`, `schemars`) — no edge to `geometry`, `linear`, or
  `mvg`.
- **Decision: `dlt_homography` (and the rest of `vision-geometry`) stays in
  `vision-geometry`, not `vision-calibration-core::linalg`.** `core::linalg`
  already holds the *primitive* math shared by both crates
  (`normalize_points_2d`/`_3d`, `null_space`, the polynomial solvers —
  landed via C1-FOLLOWUP, 2026-06-17). The higher-level DLT/epipolar/
  triangulation *solvers* have exactly one consumer family (`linear`,
  `pipeline`, `mvg`, the facade), all of which already depend on
  `geometry`; moving them into `core` would bloat the
  zero-workspace-dependency foundation crate for no consumer that doesn't
  already pull in `geometry`.
- The "temporary duplication between `vision-calibration-linear` and
  `vision-geometry`" that ADR 0015 flagged as a Consequence is resolved: PR
  #72 deleted `linear`'s parallel `homography`/`epipolar`/`camera_matrix`/
  `triangulation` modules in favor of the `vision-geometry` dependency
  above (net −1811 LoC). See `docs/backlog.md` Q7-SOLVER-DEDUP
  (resolved-as-already-done, 2026-07-04).
