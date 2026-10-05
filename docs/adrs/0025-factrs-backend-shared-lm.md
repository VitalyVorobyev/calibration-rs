# ADR 0025: A Second Backend (factrs) Under One Levenberg–Marquardt Loop

- Status: Accepted
- Date: 2026-10-05

## Context

tiny-solver was the only backend (ADR 0008). Its autodiff uses heap-backed
dynamic-size dual numbers, and P3-BACKEND-COST suspected the Jacobian build
dominates solve time. factrs 0.3 uses the same nalgebra 0.34 / faer 0.23
stack, so our generic kernels (`T: RealField`) run on it unchanged; its
forward-mode duals are static-size. A measured comparison needs a second
backend that is first-class: selectable from Rust, JSON, Python and the app.

factrs' own optimizers are unsuitable: its LM `expect()`s on rank-deficient
systems, discards values on a failed step, has no bounds, no iteration count
and an unclamped `λ·diag(JᵀJ)` damping. Two LM loops would also make a
backend comparison measure LM flavours rather than engines.

## Decision

**One LM loop, two linearization engines.** `backend/lm.rs` holds the only
Levenberg–Marquardt loop (Jacobi column scaling, clamped Marquardt diagonal,
Ceres gain ratio on the robust cost `Σρ`, damping retries, relative-step and
decrease stops). A backend implements the private `LinearizationEngine`
trait: `cost` (`Σρ`), `linearize` (loss-corrected `r̃`, `J̃`), `retract`
(with bounds) and `norm`. tiny-solver implements it with its `Problem`
primitives (Triggs loss correction); factrs with its `Graph` (IRLS: `√ρ′`
scaling). `SolveReport.final_cost` is `½Σρ` for both.

**Selection.** Public `SolverBackend { TinySolver (default), Factrs }` in
`BackendSolveOptions::backend` and `SolverConfig.backend` (serde
`snake_case`, `#[serde(default)]`). `SolverConfig::backend_options()` is the
single lowering from config to solver options; step functions override only
iteration budgets on top of it.

**factrs mapping** (`backend/factrs_backend/`):

- *Block fusion* — factrs residuals take at most six variables; our widest
  factor (laser + rig hand-eye + robot delta) has nine blocks. A camera's
  `(intrinsics, distortion, sensor)` fuse into `VectorVar<N>`,
  `N ∈ {4,5,6,7,9,11,12,13,14,15}`; a plane's `(normal, distance)` into
  `PlaneVar` (S² × ℝ). The widest factor then has six variables. Compile
  rejects an IR that combines a camera's blocks differently in two factors.
- *SE3* → factrs `SE3` (right perturbation, rotation-first tangent, `xyzw`) —
  the same convention as tiny-solver's `SE3Manifold`, so the two Jacobians
  agree at the linearization point.
- *PlaneVar* — factrs reaches a variable's manifold only through
  `x.compose(exp(δ))`; `exp(δ)` carries the step and `compose` applies the
  shared S² retraction (`backend/s2.rs`, also used by `UnitVector3Manifold`).
- *Fixed components* keep their variable; their Jacobian columns are zeroed,
  which decouples them so the step leaves them exactly unchanged. Partially
  fixed SE3/S² blocks are rejected, as in tiny-solver.
- *Bounds* are clamped after each step (`max(lower).min(upper)`), only on
  vector variables.
- *Residuals* — one generic struct per arity × factor family (`ReprojPoint`
  chains, laser chains, the tangent prior) with
  `Differ = ForwardProp<Const<DIN>>`; `N` and `DIN` are literals per
  camera-dispatch row, computed from the kernels' `DIM` in a non-generic
  context. The camera dispatch table is shared by both backends.
- *Losses* — `IrRobustCost` maps every `RobustLoss` (`loss = ½ρ`,
  `weight = ρ′`) from `RobustLoss::rho` / `rho_prime`, the single loss
  definition (bit-identical to tiny-solver's losses).
- factrs features stay off (`serde`/typetag, `rerun`, `left`, `f32`).

## Consequences

- Both backends reach the same minimizer: every factor kind × camera model ×
  chain linearizes identically at a point (residual 1e-12, Jacobian 1e-9),
  solves agree to 1e-6 under L2/Huber/Cauchy/Arctan, and the facade matrix
  tests run on both.
- About 70 residual monomorphizations with dual width up to 42: the optim
  crate's debug artifact grows from ~18 MB to ~140 MB and its build time
  rises accordingly (measured in the PR).
- `D4-NALGEBRA-035` is now blocked on tiny-solver **and** factrs.
- A new factor kind needs a tiny-solver factor, a factrs residual and a
  parity case.
