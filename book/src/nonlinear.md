# Non-linear Optimisation

Problem definitions and solvers used for refinement.

- The `NllsProblem` trait, parameter packing, and Jacobian computation.
- Backends: Levenberg–Marquardt (current), design for future Dogleg / trust region.
- Robust kernels and loss functions; when to prefer Huber vs. Cauchy.
- Parameterizations for rotations (quaternions vs. Lie algebra) and gauge freedoms.
- Convergence diagnostics, stopping criteria, and numerical stability tips.
- Benchmarks and how to add new backends.

## Solver options

`SolveOptions` controls termination criteria across NLLS backends:

- `max_iters`: hard cap on iterations.
- `ftol`: relative cost reduction tolerance.
- `gtol`: orthogonality/gradient tolerance.
- `xtol`: relative parameter update tolerance.
- `verbose`: backend-specific logging.

Example:

```rust
use calib_optim::SolveOptions;

let opts = SolveOptions {
    max_iters: 100,
    xtol: 1.0e-10,
    ..Default::default()
};
```

## NLLS problem interface

`NllsProblem` separates unweighted residuals from robust IRLS scaling. Implement
`residuals_unweighted` and `jacobian_unweighted`, and optionally override
`robust_row_scales` to apply per-row `sqrt(w)` scaling without differentiating
the weights.

```rust
use calib_core::Real;
use calib_optim::NllsProblem;
use nalgebra::{DMatrix, DVector};

struct MyProblem;

impl NllsProblem for MyProblem {
    fn num_params(&self) -> usize { 2 }
    fn num_residuals(&self) -> usize { 2 }

    fn residuals_unweighted(&self, x: &DVector<Real>) -> DVector<Real> {
        DVector::from_vec(vec![x[0], x[1]])
    }

    fn jacobian_unweighted(&self, _x: &DVector<Real>) -> DMatrix<Real> {
        DMatrix::identity(2, 2)
    }
}
```

## Jacobians

The planar intrinsics problem computes unweighted Jacobians via per-view
forward-mode autodiff using `num-dual`. Each view differentiates only the
local parameter block (intrinsics + that view’s pose) and scatters the result
into the global Jacobian. Robust IRLS weights are applied as row scales after
Jacobian evaluation (no weight derivatives).

> TODO: add example of custom problem implementation and profiling checklist.
