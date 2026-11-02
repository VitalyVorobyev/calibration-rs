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

- **Blocks.** `add_block(name, initial, Manifold)` returns a `BlockId`
  (`add_pose` takes an `Iso3`). `Manifold` is `Euclidean`, `Se3` (stored
  `[qx, qy, qz, qw, tx, ty, tz]`, ADR 0009) or `UnitVector` (S², stored
  `[x, y, z]`). A block is fixed whole, or, if Euclidean, per component;
  Euclidean components take bounds. Names are unique (`cam<i>` is kept for
  cameras); each coordinate is labelled `name.<coordinate>`, the coordinate
  given by `set_labels` or `i`, an SE3 block's `rx … tz`, a unit vector's
  `nx … nz`. A `BlockId` carries its problem, so one from another problem
  is an error, not a different block.
- **Every input is checked when added.** A residual's block count and
  dimensions against what it declares, and its value at the initial blocks
  (length and finiteness), are checked in `add_residual`; an error names
  the residual and block. A problem that was built is one the solve can
  run. Lowering names IR blocks by position, so no name a user picks takes
  on the meaning the IR reads into names (a per-view `pose/<i>`).

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
private `FactorKind::Custom(CustomResidual)` that holds an
`Arc<dyn CustomFactor>` and the noise model. `se3::<T>(&block) ->
Isometry3<T>` reads a pose block at any scalar type; a unit vector is its
stored `[x, y, z]`. A residual must not panic; a non-finite value at a
step is a rejected step, as for any factor, and so is a value of the wrong
length (the adapter turns it into `NaN`s, since the backend sizes its rows
by `dim`).

`add_residual(residual, &[BlockId], RobustLoss, Noise)` attaches it.

### The noise model makes the covariance work

The covariance (ADR 0026) pools residuals into families and counts priors
as known information. Each user residual says which it is:

- `Noise::Estimated(family)`: its σ is estimated from the scatter of the
  residuals of the family named `family`, which is pooled like the
  built-in families (reprojection, laser plane, laser line), never with
  them;
- `Noise::Known(σ)`: known information, like a prior. The adapter divides
  the residual by σ, so the solve weighs it by its precision, and the
  covariance counts it as a prior of unit residual variance. (Scaling only
  the covariance would leave the solve weighing, say, a 0.2 mm range in
  metres against pixels, where it barely moves anything.)

An estimated family enters the solve unweighted, in its own unit: its σ is
unknown until the solve ends. The sandwich covariance holds for the
weights the solve used, so it stays right; a user who wants a family to
count divides its residual by an expected σ.

`ResidualKind` gains `Custom(String)` (losing `Copy`), and the estimator
keys families by `&ResidualKind`, so a user family is pooled on its own and
reported in `ResidualNoise` like a built-in one. The view-scatter scaling
(ADR 0026) is not applied to a user problem: it reads views from IR block
names, which lowering does not produce, so a user problem's covariance
treats residuals as independent. Grouping blocks into views can come with a
need.

A free block that no residual reads has no columns in tiny-solver's
Jacobian and would drop out of the covariance unnoticed; `solve` reports
it as `RankDeficient` instead.

### Built-in residuals through typed constructors

The library's kernels join a user problem only through constructors that
establish their invariants: `add_camera(&CameraParams, &CameraFix)` lays out the
camera's blocks from its `CameraModelSpec` and labels them as the
workflows' results do (`fx`, or `cam<i>.fx` with several cameras), so
`projection_uncertainty` reads them; `add_reprojection(camera,
camera_se3_target, point, pixel, loss)` adds the reprojection residual
through one SE3 pose, checking that the point is in the projection's domain
at the initial values (the projection itself is total, so a point behind a
pinhole camera has a finite, meaningless residual). Stated per-point
covariances (whitened reprojection) and more chains (rig, hand-eye, laser)
are added as constructors when a need is shown; the IR's chain enums stay
private.

### One solve, with the covariance

`solve(&Problem, &SolveOptions) -> Result<Solution, Error>` runs the shared
LM loop. `Solution` holds each block's value (`value`, `pose`, `camera` by
id), the `SolveReport`, each family's noise level and the `Uncertainty`
(`ParameterCovariance` over the free coordinates' labels, or the typed
reason it is unavailable), the same types the exports carry, so
`projection_uncertainty` and the analysis functions read a user camera's
covariance. `SolveOptions` holds the backend and the iteration budget; the
stopping thresholds are the workflows'.

### Backends

User residuals run on tiny-solver, whose factors take any `RealField`
scalar. factrs fixes every variable's dimension at compile time; a problem
with a user residual and `SolverBackend::Factrs` is an `InvalidInput`
before any work. Built-in constructors run on both.

### `ProblemType` stays sealed

A workflow is composed from public pieces: the closed-form initializers,
re-exported on the facade as `vision_calibration::init` (Zhang with
distortion, pose from a homography, PnP, hand-eye, a plane through points;
the homography DLT is already `mvg::dlt_homography`), then a `Problem`,
`solve` and the analysis functions. The session framework, its steps and the export
contract stay the library's own.

## Consequences

- New public, semver-bound surface: `Problem`, `BlockId`, `CameraId`,
  `Manifold`, `Residual`, `Noise`, `se3`, `SolveOptions`, `Solution`,
  `SolveReport` on the facade, and the `init` re-exports; the enums and
  option structs are `#[non_exhaustive]`.
- `FactorKind` gains `Custom`; its `PartialEq` compares that arm by pointer.
- `ResidualKind` gains `Custom(String)` and is no longer `Copy`: a break of
  the public API and of the export schemas (a new variant).
- A user residual is monomorphized in the user's crate, not ours.
- Python stays Rust-only here (recorded in the binding parity check): a
  Python residual would be a per-evaluation callback that cannot take dual
  numbers.
- The book gains "Your own residual" and "Compose your own pipeline".
- [ADR 0028](0028-measurements-from-a-calibration.md) extends the builder
  for measurements: a calibration with its covariance (`add_calibrated_rig`),
  point-block and rig-chain reprojection, whitened reprojection.

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
