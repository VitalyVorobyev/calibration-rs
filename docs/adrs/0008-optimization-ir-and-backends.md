# ADR 0008: Optimization IR and Solver Backends

## Context

Non-linear least-squares solvers have incompatible APIs, and a problem
written against one cannot be benchmarked on another. Within a problem, a
residual depends on three independent axes: the **camera model**
(projection × distortion × sensor), the **pose chain** that carries a target
point into the camera (single pose, rig, hand-eye, hand-eye with robot-pose
correction, the laser chains) and the **residual family** (point
reprojection, laser point-to-plane, laser line distance). Encoding the
camera model and chain in factor *names* multiplies variants with every new
model and makes new camera models unaffordable.

## Decision

### A solver-independent IR

Problems are built as a `ProblemIR`, pure data with no derivative concepts:

- `ParamBlock` — a named variable with dimension, `ManifoldKind`
  (Euclidean, SE3, SO3, S2), fixed mask and bounds.
- `ResidualBlock` — references parameter blocks, a `FactorKind` and an
  optional `RobustLoss` (Huber, Cauchy, Arctan).

### One factor variant per residual family; model and chain as data

```rust
pub enum FactorKind {
    ReprojPoint       { model: CameraModelDesc, chain: ReprojChain, pw, uv, w },
    LaserPointToPlane { model: CameraModelDesc, chain: LaserChain, laser_pixel, w },
    LaserLineDistance { model: CameraModelDesc, chain: LaserChain, laser_pixel, w },
    Se3TangentPrior   { sqrt_info },
}
```

`CameraModelDesc { projection, distortion, sensor }` names each stage's kind;
every kind reports its parameter `dim()`. A factor's parameter layout is
derived from the descriptor and chain (`FactorKind::param_layout`), never
hand-listed, and `ProblemIR::validate` checks blocks against it. Fix masks
stay on parameter blocks.

### Monomorphized kernels

Residual kernels are generic over `T: RealField` so dual-number autodiff
gives exact Jacobians; the camera model therefore cannot be a `dyn` object.
Each descriptor kind maps to a zero-sized kernel type
(`factors/camera_kernels.rs`), and one shared residual per family composes
them. A backend matches the descriptor **once per factor** through the
shared `dispatch_camera_model!` table and monomorphizes; the chain is a
small data match inside the residual.

Adding a camera model is: a descriptor variant with its `dim()`, a kernel
type, one dispatch row, and the parameter plumbing in core and the configs.
No new factor variants, validation arms or per-chain code.

### One Levenberg–Marquardt loop, pluggable linearization engines

`backend/lm.rs` holds the only LM loop (Jacobi column scaling, clamped
Marquardt diagonal, gain ratio on the robust cost `Σρ`, damping retries,
step and decrease stops) and assembles the normal equations once per solve
(`backend/normal_equations.rs`). A backend implements the private
`LinearizationEngine` (`cost`, `linearize`, `retract` with bounds, `norm`)
and is selected by `SolverBackend` (`TinySolver` default, `Factrs`) in
`SolverConfig.backend`.

A backend must allocate one parameter per block keyed by name, apply its
manifold, fixed mask and bounds, evaluate every residual through its
factor's generic kernel with its loss, drive the shared loop, and report
`final_cost = ½ Σ ρ(‖r‖²)`. It is accepted when every factor kind
linearizes like the existing backend (`backend/parity_tests.rs`) and it
reproduces the optim problem tests from the same initial values.

The factrs backend adapts the IR to factrs' limits: a camera's
intrinsics, distortion and sensor blocks fuse into one vector variable, and
a plane's normal and distance into one S²×ℝ variable, so the widest factor
fits factrs' six-variable limit; fixed components are held by zeroed
Jacobian columns; bounds are clamped after each step. Losses come from the
single `RobustLoss::rho` / `rho_prime` definition on both backends.

## Consequences

- Both backends reach the same minimizer: factors agree to 1e-12 (residual)
  and 1e-9 (Jacobian), solves to 1e-6.
- The default backend changes only on a measured comparison and the
  maintainer's decision.
- A new factor kind needs a kernel, a factor on each backend and a parity
  case. The `FactorKind` enum grows per residual family, not per model.
- nalgebra and faer versions are pinned jointly by both backends; an
  upgrade moves both.
- About 70 residual monomorphizations make the optim crate's debug build
  large and slower to compile.

## Alternatives rejected

- **Factor variants per model and chain**, hand-written or macro-generated —
  the variants, validation and backend arms multiply with every model.
- **`dyn CameraProject` in factors** — `f64`-only, so no autodiff.
- **Matching the model enum per residual evaluation** — puts a growing
  branch inside the autodiff hot loop.
- **Const-generic dimensions** — fight variable-size distortion blocks.
- **Each backend's own optimizer** — factrs' LM panics on rank-deficient
  systems and has no bounds, and two loops would make a backend comparison
  measure LM flavours instead of engines.
