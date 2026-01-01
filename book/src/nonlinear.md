# Non-linear Optimisation

Problem definitions and solvers used for refinement.

- The `NllsProblem` trait, parameter packing, and Jacobian computation.
- Backends: Levenberg–Marquardt (current), design for future Dogleg / trust region.
- Robust kernels and loss functions; when to prefer Huber vs. Cauchy.
- Parameterizations for rotations (quaternions vs. Lie algebra) and gauge freedoms.
- Convergence diagnostics, stopping criteria, and numerical stability tips.
- Benchmarks and how to add new backends.

> TODO: add example of custom problem implementation and profiling checklist.
