# ADR 0008: Backend-Agnostic Optimization IR

- Status: Accepted
- Date: 2026-03-07 (retroactive)

## Context

Non-linear least-squares solvers (Ceres, g2o, tiny-solver, etc.) have incompatible APIs. Coupling problem definitions to a specific solver makes it hard to switch or benchmark backends.

## Decision

Define optimization problems in a solver-independent intermediate representation (IR), then compile to specific backends:

- `ProblemIR`: collection of `ParamBlock` and `ResidualBlock` entries.
- `ParamBlock`: named variable with dimension, manifold kind, fixed mask, and bounds.
- `ResidualBlock`: connects parameter blocks to a `FactorKind` with optional robust loss.
- `FactorKind`: enum of supported residual functions (e.g., `ReprojPointPinhole4Dist5`).
- `ManifoldKind`: parameter geometry (Euclidean, SE3, SO3, S2).

Backend pattern:
1. Problem builder constructs `ProblemIR` + initial values.
2. Backend `compile()` translates IR to solver-specific structures.
3. Backend `solve()` runs optimization, returns `BackendSolution`.

Factor functions are generic over `T: RealField` for autodiff compatibility.

## Consequences

- New backends require only a `compile` + `solve` implementation.
- New factor types require adding a `FactorKind` variant and implementing the generic residual function.
- IR serves as documentation of the optimization problem structure.
- Trade-off: the `FactorKind` enum grows with each new factor type (acceptable for a focused library).
