# vision-calibration-optim

Non-linear least-squares refinement for camera calibration: the optimization
engine behind the [`vision-calibration`](https://crates.io/crates/vision-calibration)
sessions.

Each calibration problem (planar intrinsics, rig extrinsics, hand-eye, laser
plane, joint rig + hand-eye + laser) is lowered to a backend-agnostic
intermediate representation: parameter blocks on their manifolds (SE(3)
poses, the unit sphere for a plane normal), residual factors generic over
`RealField` for automatic differentiation, and robust losses (`Huber`,
`Cauchy`, `Arctan`). One Levenberg–Marquardt loop solves it over two
linearization engines, tiny-solver (default) and factrs, with sparse Cholesky
or QR linear solvers.

## Public surface

The documented items are the vocabulary and results the sessions hand to
callers: `RobustLoss`, `SolverBackend`, `HandeyeMode`, each problem's
`*Estimate` and `*Params` with its `SolveReport`, the laser and hand-eye input
types, `LaserPlane`, and projection uncertainty.

The solve entry points (`optimize_*`) and their option structs are
workspace-internal: public so the pipeline crate can call them, but hidden
from the docs and outside the semver promise. Calibrate through the step
functions of `vision-calibration`.

## See Also

- [vision-calibration](https://crates.io/crates/vision-calibration): the calibration workflows
- [Book: Non-linear Optimization](https://vitalyvorobyev.github.io/calibration-rs/nlls_overview.html), [robust losses](https://vitalyvorobyev.github.io/calibration-rs/robust_loss.html), [solver backends](https://vitalyvorobyev.github.io/calibration-rs/solver_backends.html)
