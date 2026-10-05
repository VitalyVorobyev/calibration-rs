# Solver Backends

Every non-linear solve in calibration-rs runs one Levenberg–Marquardt (LM)
loop. A **backend** supplies what the loop needs at the current parameters:
the residuals and their Jacobian with the robust loss folded in, the robust
objective, and the retraction that applies a step to each parameter
manifold. Two backends are available:

- **tiny-solver** (the default): forward-mode autodiff with dynamic-size
  dual numbers; a robust loss enters through the Triggs correction.
- **factrs**: forward-mode autodiff with static-size dual numbers; a robust
  loss enters through iterative reweighting.

Both evaluate the same residual kernels, retract SE(3) and S² the same way,
and minimize the same objective, so they reach the same minimizer. They
differ in how fast they build the Jacobian and, on robust problems, in the
path they take to the minimum.

## Selecting a Backend

The backend is the `backend` field of the solver settings of every
workflow.

Rust:

```rust
use vision_calibration::optim::SolverBackend;
use vision_calibration::planar_intrinsics::PlanarIntrinsicsConfig;

let mut config = PlanarIntrinsicsConfig::default();
config.solver.backend = SolverBackend::Factrs;
```

JSON config:

```json
{ "solver": { "max_iters": 50, "robust_loss": "None", "backend": "factrs" } }
```

Python:

```python
config = vc.PlanarCalibrationConfig(solver=vc.SolverConfig(backend="factrs"))
```

In the desktop app, the backend is a field of the solver group of each
config form. Workflows with several solve stages (for example the joint rig,
hand-eye and laser workflow) have one solver group per stage, each with its
own backend.

Code that calls an `optimize_*` function directly passes the backend in
`BackendSolveOptions`:

```rust
let opts = BackendSolveOptions {
    backend: SolverBackend::Factrs,
    ..BackendSolveOptions::default()
};
```

`SolverConfig::backend_options()` gives the options a solver config stands
for.

## The Levenberg–Marquardt Loop

The objective is the robust cost

$$F(\mathbf{x}) = \frac{1}{2} \sum_i \rho_i\left(\lVert \mathbf{r}_i(\mathbf{x}) \rVert^2\right),$$

where $\rho_i$ is the residual block's loss ($\rho(s) = s$ without one; see
[Robust Loss Functions](robust_loss.md)). Each iteration:

1. **Linearizes** at $\mathbf{x}$: the backend returns a residual
   $\tilde{\mathbf{r}}$ and Jacobian $\tilde{J}$ with the loss folded in, so
   that the model $\lVert \tilde{\mathbf{r}} + \tilde{J}\boldsymbol{\delta}
   \rVert^2$ has the same first-order change as $2F$.
2. **Scales** the Jacobian's columns by $1 / (1 + \lVert \tilde{J}_c \rVert)$,
   computed once at the first iteration, so that parameters of different
   magnitude (focal lengths in pixels, distortion coefficients, rotations)
   are damped comparably.
3. **Solves** the damped normal equations
   $(\tilde{J}^T\tilde{J} + \lambda D)\,\boldsymbol{\delta} = -\tilde{J}^T
   \tilde{\mathbf{r}}$ with sparse Cholesky (default) or sparse QR, where
   $D$ is the diagonal of $\tilde{J}^T\tilde{J}$ clamped to
   $[10^{-6}, 10^{32}]$.
4. **Applies** the step through each block's retraction (vector addition,
   $\mathbf{x} \cdot \exp(\boldsymbol{\delta})$ on SE(3), the exponential
   map on S²), then clamps bounded parameters to their bounds.
5. **Accepts or rejects** by the gain ratio of the actual to the predicted
   decrease of $F$. An accepted step scales $\lambda$ by
   $\max(1/3,\, 1 - (2\varrho - 1)^3)$; a rejected one doubles $\lambda$ and
   retries, up to 32 times.

The loop stops when $F$ falls below `min_error`, when its absolute or
relative decrease falls below `min_abs_decrease` / `min_rel_decrease`, when
a step can no longer change the parameters
($\lVert\boldsymbol{\delta}\rVert \le 10^{-8}(\lVert\mathbf{x}\rVert +
10^{-8})$), when no step is accepted after 32 retries, or after `max_iters`
iterations. Parameters held fixed never move, and the same input gives
bit-identical results run to run.

## How the Backends Differ

| | tiny-solver | factrs |
|---|---|---|
| Dual numbers | dynamic size (heap) | static size: the factor's tangent dimension |
| Robust loss | Triggs correction: second-order exact in $\rho$ | iterative reweighting: block scaled by $\sqrt{\rho'}$ |
| Variables | one per parameter block | a camera's intrinsics, distortion and sensor fuse into one vector; a laser plane's normal and distance into one plane variable |
| Fixed parameters | removed from the system | kept, with their Jacobian columns zeroed |

Without a robust loss the two linearizations are the same, and so is every
step up to rounding. With a robust loss they take different steps towards
the same minimum, so iteration counts can differ.

factrs supports what the calibration problems use. It rejects an IR it
cannot map, with an `InvalidInput` error: a partially fixed SE(3) or S²
block, bounds on a non-vector block, or one camera's parameter blocks
combined differently in two factors.

## Solver Options

```rust
pub struct BackendSolveOptions {
    pub backend: SolverBackend,                  // default: TinySolver
    pub max_iters: usize,                        // default: 100
    pub verbosity: usize,                        // 0 = silent
    pub linear_solver: Option<LinearSolverKind>, // default: Some(SparseCholesky)
    pub min_abs_decrease: Option<f64>,           // default: Some(1e-5)
    pub min_rel_decrease: Option<f64>,           // default: Some(1e-5)
    pub min_error: Option<f64>,                  // default: Some(1e-10)
}

pub enum LinearSolverKind {
    SparseCholesky,  // Default: fast for well-conditioned problems
    SparseQR,        // More robust for ill-conditioned problems
}
```

- **SparseCholesky** factors the normal equations directly. It is fast but
  can fail when $J^TJ$ is poorly conditioned; the loop then raises the
  damping and retries.
- **SparseQR** factors $J$ itself. It is slower and more robust near
  singular directions.

## Solve Report

```rust
pub struct SolveReport {
    pub final_cost: f64,
    pub num_iters: usize,
}
```

`final_cost` is the robust objective $F$ at the solution, the same
definition for both backends, so it is comparable between them.
`num_iters` counts the LM iterations.

## Typical Convergence

For a well-initialized planar intrinsics problem:

- **Final cost**: $\sim 10^{-2}$–$10^0$ (sub-pixel residuals)
- **Iterations**: 5–50 (depends on problem size and initial quality)
- **Termination**: usually the relative decrease falling below
  `min_rel_decrease`

## Error Handling

A solve fails with an error for:

- a missing or wrongly sized initial value for a parameter block;
- an invalid IR (for example a non-positive robust-loss scale);
- an IR the selected backend cannot map (see above);
- a non-finite initial cost.
