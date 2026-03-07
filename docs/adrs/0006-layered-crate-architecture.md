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
