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

> TODO: add example of custom problem implementation and profiling checklist.
