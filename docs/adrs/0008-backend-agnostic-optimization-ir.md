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

Factor functions are generic over `T: RealField` for autodiff compatibility. The IR is pure data (no derivative concepts in `ProblemIR` or `OptimBackend`); the kernels are autodiff-capable rather than autodiff-dependent. Two backends compile it — tiny-solver and factrs — under one shared Levenberg–Marquardt loop (ADR 0025).

### Backend contract

A backend (`OptimBackend::solve(ir, initial, opts) -> BackendSolution`,
private to `vision-calibration-optim`, registered in the single dispatch
`backend::solve` on `BackendSolveOptions::backend`) must:

- validate the IR, then allocate one parameter per `ParamBlock`, initialized
  from `initial` (keyed by block **name**);
- apply the block's manifold (`Euclidean`, `SE3` 7→6, `SO3` 4→3, `S2` 3→2),
  its fixed mask, and its bounds;
- evaluate every `ResidualBlock` through its `FactorKind`'s generic kernel,
  with the block's `RobustLoss` (`Huber`, `Cauchy`, `Arctan`) applied;
- drive the shared LM loop through a `LinearizationEngine` (ADR 0025)
  rather than its own optimizer;
- return the optimized values keyed by block name, plus a `SolveReport`
  whose `final_cost` is `½ Σ ρ(‖r‖²)`.

A new backend is accepted when every factor kind linearizes like the
existing ones (`backend/parity_tests.rs`) and it reproduces the optim
problem tests from the same initial values.

## Consequences

- New backends require only a `compile` + `solve` implementation.
- New factor types require adding a `FactorKind` variant and implementing the generic residual function.
- IR serves as documentation of the optimization problem structure.
- Trade-off: the `FactorKind` enum grows with each new factor type (acceptable for a focused library).
