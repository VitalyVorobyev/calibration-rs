# ADR 0027: A Public Optimization Extension

## Context

A user who needs a residual the library lacks (a measured distance, a CAD
constraint, their own sensor) or a pipeline the workflows do not cover has
to fork: the optimization IR, the factors and the backends are private, and
`ProblemType` is sealed (ADR 0007). Opening the IR as it is would not do:
the built-in kernels rely on invariants that only the optim crate's own
builders establish (a block's dimension matching the camera model, an SE3
block holding a unit quaternion), so a hand-built IR panics inside the
solve, and every internal type would become semver-bound.

## Decision

### A public problem builder in front of the private IR

`vision_calibration_optim::problem` (on the facade as
`vision_calibration::optim::problem`) holds a `Problem` builder. It owns its
blocks and residuals and lowers to the private `ProblemIR` when solved; the
IR, `FactorKind` and the backends stay private.

- **Blocks.** `add_block(name, initial, Manifold)` returns a `BlockId`.
  `Manifold` is `Euclidean`, `Se3` (stored `[qx, qy, qz, qw, tx, ty, tz]`,
  ADR 0009) or `UnitVector` (S², stored `[x, y, z]`). A block is fixed whole,
  or, if Euclidean, per component; Euclidean components take bounds. Names
  are unique; each coordinate has a label (given, or `name.i`, an SE3 block's
  `name.rx … name.tz`, a unit vector's `name.nx … name.nz`).
- **Every input is checked when added.** A residual's block count and
  dimensions against what it declares, and its value at the initial blocks
  (length and finiteness), are checked in `add_residual`; an error names
  the residual and block. A problem that was built is one the solve can
  run.

### User residuals: a generic `eval`, bridged once

```rust
pub trait Residual: Send + Sync + 'static {
    fn dim(&self) -> usize;               // residual length
    fn block_dims(&self) -> Vec<usize>;   // ambient dimension per block
    fn eval<T: RealField>(&self, blocks: &[DVector<T>]) -> DVector<T>;
}
```

`eval` is generic so dual-number autodiff gives exact Jacobians, as for the
built-in kernels (ADR 0008); finite differences are not offered. The trait
is not object-safe and need not be: an internal generic adapter turns any
`R: Residual` into the object-safe factor the backend stores, behind a
private `FactorKind::Custom(Arc<dyn CustomFactor>)`. Helpers convert block
storage generically (`se3::<T>(&block) -> Isometry3<T>`, unit vectors). A
residual must not panic; a non-finite value at a step is a rejected step,
as for any factor.

`add_residual(residual, &[BlockId], RobustLoss, Noise)` attaches it.

### The noise model makes the covariance work

The covariance (ADR 0026) pools residuals into families and counts priors
as known information. Each user residual says which it is:

- `Noise::Estimated(family)`: its σ is estimated from the scatter of the
  residuals of the family named `family`, which is pooled like the
  built-in families (reprojection, laser plane, laser line), never with
  them;
- `Noise::Known(σ)`: a prior; its information is known, with residual
  standard deviation σ (1 for a whitened residual).

The estimator's family key widens from `ResidualKind` to a family identity
that includes user names; the view-scatter scaling is unchanged.

### Built-in residuals through typed constructors

The library's kernels join a user problem only through constructors that
establish their invariants: `add_camera(&CameraParams, fix)` lays out the
camera's blocks from its `CameraModelSpec` and labels them with the names of
`camera_parameters`, and `add_reprojection(&camera, pose, point, pixel,
sqrt_info, loss)` adds the reprojection residual through one SE3 pose. More
chains (rig, hand-eye, laser) are added as constructors when a need is
shown; the IR's chain enums stay private.

### One solve, with the covariance

`solve(&Problem, &SolveOptions) -> Result<Solution, Error>` runs the shared
LM loop. `Solution` holds each block's value by name, the `SolveReport`,
each family's noise level and the `Uncertainty` (`ParameterCovariance` over
the free coordinates' labels, or the typed reason it is unavailable), the
same types the exports carry, so `projection_uncertainty` and the analysis
functions read a user camera's covariance. `SolveOptions` is the public
form of the options the problem types' `SolverConfig` lowers to.

### Backends

User residuals run on tiny-solver, whose factors take any `RealField`
scalar. factrs fixes every variable's dimension at compile time; a problem
with a user residual and `SolverBackend::Factrs` is an `InvalidInput`
before any work. Built-in constructors run on both.

### `ProblemType` stays sealed

A workflow is composed from public pieces: the closed-form initializers,
re-exported on the facade as `vision_calibration::init` (Zhang, PnP,
homography, hand-eye, laser plane), then a `Problem`, `solve` and the
analysis functions. The session framework, its steps and the export
contract stay the library's own.

## Consequences

- New public, semver-bound surface: `Problem`, `BlockId`, `Manifold`,
  `Residual`, `Noise`, `SolveOptions`, `Solution` and the `init` re-exports;
  the enums and option structs are `#[non_exhaustive]`.
- `FactorKind` gains `Custom`; its `PartialEq` compares that arm by pointer.
- A user residual is monomorphized in the user's crate, not ours.
- Python stays Rust-only here (recorded in the binding parity check): a
  Python residual would be a per-evaluation callback that cannot take dual
  numbers.
- The book gains "Your own residual" and "Compose your own pipeline".

## Alternatives rejected

- **Publishing `ProblemIR` and `FactorKind`.** Hand-built IR panics in the
  built-in kernels, and every internal change becomes breaking.
- **`dyn Residual` over `f64` with finite differences.** Inexact Jacobians
  and a slower solve, against the library's autodiff rule.
- **Unsealing `ProblemType`.** Steps, state and the export contract would
  become API to hold a pipeline the public pieces already compose.
- **User residuals on factrs.** It needs each variable's size as a type;
  a dynamic adapter would rebuild its variable system.
- **Residuals in Python.** No dual numbers through a callback, and a call
  per evaluation.
