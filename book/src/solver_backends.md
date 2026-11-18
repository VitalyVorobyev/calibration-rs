# Solver Backends

Every non-linear solve in calibration-rs runs one Levenberg–Marquardt (LM)
loop. A **backend** supplies what the loop needs at the current parameters:
the residuals and their Jacobian with the robust loss folded in, the robust
objective, and the retraction that applies a step to each parameter
manifold. Two backends are available:

- **tiny-solver** (the default): forward-mode autodiff with stack dual
  numbers, four tangent directions per pass; a robust loss enters through
  the Triggs correction.
- **factrs**: forward-mode autodiff with static-size dual numbers; a robust
  loss enters through iterative reweighting.

Both evaluate the same residual kernels, retract S² the same way, and
minimize the same objective, so they reach the same minimizer. Both apply an
SE(3) step on the right; factrs maps it through the exponential map, while
tiny-solver applies the rotation and the translation parts separately; the
two agree to first order. They differ in how fast they build the Jacobian and, on robust problems, in the
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
{ "solver": { "max_iters": 50, "verbosity": 0, "backend": "factrs" } }
```

Python:

```python
config = vc.PlanarIntrinsicsConfig(
    solver=vc.SolverConfig(backend="factrs", max_iters=120),
)
```

A `SolverConfig` replaces the whole solver group, so pass the workflow's
`max_iters` with it: `vc.SolverConfig()` alone has `max_iters=50`, while
planar intrinsics defaults to 120.

In the desktop app, the backend is a field of the solver group of each
config form. The joint rig, hand-eye and laser workflow embeds two other
configs and has a solver group for each of them as well as its own, each with
its own backend.

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
   \tilde{\mathbf{r}}$ with sparse Cholesky, where
   $D$ is the diagonal of $\tilde{J}^T\tilde{J}$ clamped to
   $[10^{-6}, 10^{32}]$. The sparsity pattern of $\tilde{J}^T\tilde{J}$ and the
   symbolic factorization are computed once per solve; each iteration fills
   the values of the lower triangle only.
4. **Applies** the step through each block's retraction (vector addition,
   $\mathbf{x} \cdot \exp(\boldsymbol{\delta})$ on SE(3), the exponential
   map on S²), then clamps bounded parameters to their bounds.
5. **Accepts or rejects** by the gain ratio of the actual to the predicted
   decrease of $F$. An accepted step scales $\lambda$ by
   $\max(1/3,\, 1 - (2\varrho - 1)^3)$; a rejected one doubles $\lambda$ and
   retries, up to 32 times.

The loop stops at the first of:

- the cost $2F$ falls below $10^{-10}$;
- an accepted step decreases $2F$ by less than $10^{-5}$, in absolute terms
  or relative to its previous value;
- a step can no longer change the parameters
  ($\lVert\boldsymbol{\delta}\rVert \le 10^{-8}(\lVert\mathbf{x}\rVert +
  10^{-8})$);
- no step is accepted after 32 retries, which is a minimum to working
  precision;
- `max_iters` iterations, the one threshold the solver settings set.

Parameters held fixed never move, and the same input gives bit-identical
results run to run.

## How the Backends Differ

| | tiny-solver | factrs |
|---|---|---|
| Dual numbers | four directions, on the stack; a block takes one pass per four tangent coordinates | static size: the factor's tangent dimension, one pass |
| Robust loss | Triggs correction: second-order exact in $\rho$ | iterative reweighting: block scaled by $\sqrt{\rho'}$ |
| Variables | one per parameter block | a camera's intrinsics, distortion and sensor fuse into one vector; a laser plane's normal and distance into one plane variable |
| Fixed parameters | removed from the system | kept, with their Jacobian columns zeroed |

Without a robust loss the two linearizations are the same, and so is every
step up to rounding. With a robust loss they take different steps towards
the same minimum, so iteration counts can differ.

factrs supports what the calibration problems use. It rejects an IR it
cannot map, with an `InvalidInput` error: a residual of your own (see
[Your Own Residual](custom_residual.md)), which runs on tiny-solver only,
bounds on a non-vector block, or one camera's parameter blocks combined
differently in two factors. Both backends reject a partially fixed SE(3),
SO(3) or S² block.

## Performance and Quality

Measured on a synthetic benchmark of the seven workflows, each with a
pinhole and a fisheye (Kannala-Brandt) camera, and planar intrinsics and rig
extrinsics also with a tilted (Scheimpflug) sensor: 264 scenes across three
sizes (8 to 36 views), two pixel-noise levels and 5 % outliers under Huber
or Cauchy losses, timed as the median of three runs on an Apple M4 Pro
(8 performance + 4 efficiency cores). Times are the median non-linear solve
per scene, all stages and the covariance included; the ratio is the median
of the per-scene ratios.

| Problem | tiny-solver | factrs | factrs / tiny-solver | final-cost agreement |
|---|---:|---:|---:|---:|
| Planar intrinsics | 19 ms | 39 ms | 2.14× | ≤ 4e-07 |
| Planar intrinsics (Scheimpflug) | 105 ms | 523 ms | 5.31× | ≤ 4e-07 |
| Planar intrinsics (fisheye) | 20 ms | 50 ms | 2.56× | ≤ 2e-08 |
| Single-camera hand-eye | 324 ms | 1390 ms | 3.92× | ≤ 2e-08 |
| Single-camera hand-eye (fisheye) | 304 ms | 1716 ms | 5.25× | ≤ 3e-08 |
| Laserline device | 30 ms | 100 ms | 3.34× | ≤ 4e-08 |
| Laserline device (fisheye) | 31 ms | 136 ms | 4.60× | ≤ 2e-06 |
| Rig extrinsics | 119 ms | 216 ms | 1.84× | ≤ 5e-08 |
| Rig extrinsics (Scheimpflug) | 292 ms | 983 ms | 3.35× | ≤ 7e-07 |
| Rig extrinsics (fisheye) | 107 ms | 360 ms | 3.17× | ≤ 2e-09 |
| Rig hand-eye | 791 ms | 3164 ms | 3.67× | ≤ 8e-08 |
| Rig hand-eye (fisheye) | 672 ms | 3448 ms | 4.52× | ≤ 2e-08 |
| Rig laserline | 25 ms | 93 ms | 3.77× | — |
| Rig laserline (fisheye) | 23 ms | 76 ms | 3.17× | — |
| Rig hand-eye + laser | 1727 ms | 7546 ms | 4.37× | ≤ 5e-07 |
| Rig hand-eye + laser (fisheye) | 1966 ms | 6837 ms | 3.42× | ≤ 2e-08 |

**Quality is the same.** The backends end at the same final cost (the
largest relative difference is 1e-6, the median 3e-10), take the same
number of iterations in 246 of 252 scenes, and recover the same parameters.

**tiny-solver is faster.** It evaluates residual blocks in parallel (2.2–3.5×
over one thread); factrs evaluates them on one thread. On a single thread
(measured on the planar intrinsics, laserline device and rig hand-eye
scenes) the two are within 10 % on planar intrinsics with a pinhole or
fisheye camera, and tiny-solver is 1.2–2.6× faster on the others.

**Which to use.** tiny-solver, the default, is the faster backend. factrs
is an independent second linearization of the same problem: it reaches the
same minimum by a different path, which makes it a cross-check of a result.

## Solver Options

Each solve stage's `SolverConfig` sets the `backend`, the iteration budget
`max_iters` and the `verbosity`; the robust loss belongs to the stage's
optimize group, not the solver (see [Robust Loss Functions](robust_loss.md)). The stopping rules are
those of [the loop above](#the-levenbergmarquardt-loop).

The damped normal equations are always solved by sparse Cholesky. When the
factorization fails (the damped $J^TJ$ is not numerically positive
definite), the loop treats it as a rejected step: it raises the damping and
retries.

## Solve Report

```rust,ignore
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
- **Termination**: usually the relative decrease falling below $10^{-5}$

## Error Handling

A solve fails with an error for:

- a missing or wrongly sized initial value for a parameter block;
- an invalid IR (for example a non-positive robust-loss scale);
- an IR the selected backend cannot map (see above);
- a non-finite initial cost.
