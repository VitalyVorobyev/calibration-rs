# ADR 0008: Backend-Agnostic Optimization IR

- Status: Accepted
- Date: 2026-03-07 (retroactive)
- Note (2026-07-04, supersedes the 2026-07-02 note): Track O (apex-solver) is
  closed won't-do. Precisely: this IR is backend-neutral (pure data, no
  derivative concepts in `ProblemIR` or `OptimBackend`), and the factor
  kernels are autodiff-*capable* rather than autodiff-*dependent* — a
  dual-number adapter over the `T: RealField` kernels could feed a
  hand-Jacobian backend like apex-solver 1.3. The closure is a value call, not
  an impossibility: apex's missing S2 manifold / robust losses / documented
  conventions plus its duplicated solver core (LM + sparse Cholesky ≈ our
  LM + faer) make bridging not worth it (backlog Track O has the revive
  triggers). tiny-solver remains the sole backend. The backend-agnostic IR
  shape is kept: it still isolates problem definitions from the solver API.

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
