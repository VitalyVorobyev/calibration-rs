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

The loop stops at the first of the rules below; the solve report names it
as its `termination`:

| Rule | `termination` |
|---|---|
| the solution is within $0.01\sigma$ of the minimum (below) | `sigma_step` |
| an accepted step moved the solution by less than $0.001\sigma$ while it was still further than $0.01\sigma$ from the minimum: progress has stalled | `stalled_step` |
| a step can no longer change the parameters ($\lVert\boldsymbol{\delta}\rVert \le 10^{-8}(\lVert\mathbf{x}\rVert + 10^{-8})$) | `relative_step` |
| no step is accepted after 32 retries: a minimum to working precision, or a parameter held on its bound | `no_accepted_step` |
| after an accepted step, the cost $2F$ is below $10^{-10}$ | `min_error` |
| `max_iters` iterations ran without meeting a rule; the one threshold the solver settings set | `max_iters` |
| every parameter is fixed, so no iteration runs | `all_parameters_fixed` |

The result is reported with a covariance that holds at the minimum, so
the main rule measures how far the solution still is from it, in that
covariance's units. With $m$ residuals, $n$ free parameters and
$\hat\sigma^2 = 2F/(m-n)$, the covariance is about
$\hat\sigma^2 (\tilde{J}^T\tilde{J})^{-1}$, and the undamped step to the
minimum of the local model, $\boldsymbol{\delta}_N =
-(\tilde{J}^T\tilde{J})^{-1}\tilde{J}^T\tilde{\mathbf{r}}$, has the
Mahalanobis length

$$d^2 = \frac{\tilde{\mathbf{r}}^T\tilde{J}\,(\tilde{J}^T\tilde{J})^{-1}\tilde{J}^T\tilde{\mathbf{r}}}{\hat\sigma^2},$$

the distance to the minimum in standard deviations, which bounds every
parameter's distance in units of its own $\sigma$. On the first iteration,
and whenever the last accepted step moved the solution by at most one
$\sigma$, the loop also solves for $\boldsymbol{\delta}_N$; when
$d \le 0.01$ it takes that step if it lowers the cost and stops. Under a
robust loss the linearization's curvature can exceed the objective's
(where $\rho'' < 0$ both backends weight by $\rho'$ alone), so the step
covers part of the way; its gain ratio $\varrho = 2 - a/b$ (curvatures of
objective and model) tells how much, and the loop stops only when the
distance the step leaves, $\lvert\varrho - 1\rvert\, d/\sqrt{2 - \varrho}$,
is within $0.01$ too, iterating on otherwise. The rule needs redundancy
($m > n$) and a nonzero cost; without them the other rules end the solve.
A coordinate held on its bound is held out of $\boldsymbol{\delta}_N$, and
with one held the step is taken whenever it lowers the cost, so a solve
against a bound converges. A `stalled_step` happens where a weakly
determined parameter lies along a long, curved valley, and a solve there may
also run to `max_iters`: the result then states its covariance unavailable
rather than claiming a minimum it has not reached.

A rule on the decrease of the last step would measure the step taken,
not the distance left: a fraction of the cost corresponds to more
standard deviations the more residuals there are, and a step shortened by
the damping reads as nearer the minimum than it is. A rule met on the last
allowed iteration is convergence, not `max_iters`.
Every reason but `max_iters` is convergence
(`SolveTermination::is_converged`), and only a converged solve has a
covariance; one stopped by its budget reports its uncertainty unavailable
as not converged. A cancelled solve, or one whose initial cost is not
finite, returns an error rather than a report.

Parameters held fixed never move, and the same input gives bit-identical
results run to run.

## How the Backends Differ

| | tiny-solver | factrs |
|---|---|---|
| Dual numbers | four directions, on the stack; a block takes one pass per four tangent coordinates | static size: the factor's tangent dimension, one pass |
| Robust loss | Triggs correction where $\rho'' > 0$ (second-order exact there); where $\rho'' \le 0$, as Ceres, the block scaled by $\sqrt{\rho'}$ | iterative reweighting: block scaled by $\sqrt{\rho'}$ |
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
    pub termination: Option<SolveTermination>,
}
```

`final_cost` is the robust objective $F$ at the solution, the same
definition for both backends, so it is comparable between them.
`num_iters` counts the LM iterations. `termination` is the rule that ended
the solve (the table above), in JSON its snake-case name. It is `None` when
unknown: a report saved before the reason was recorded, or a result that no
solve produced (a tilted camera taken from its best-scoring initial
guess). Each step result that runs a solve carries it too, as
`termination`, or `per_cam_terminations` for a stage that solves each
camera on its own; in Python it is a string or `None`:

```rust
# use vision_calibration::optim::{SolveReport, SolveTermination};
# fn check(report: &SolveReport) {
match report.termination {
    Some(t) if t.is_converged() => println!("converged: {t:?}"),
    Some(_) => println!("stopped by its iteration budget"),
    None => println!("not recorded"),
}
# }
```

## Typical Convergence

For a well-initialized planar intrinsics problem:

- **Final cost**: $\sim 10^{-2}$–$10^0$ (sub-pixel residuals)
- **Iterations**: 5–50 (depends on problem size and initial quality)
- **Termination**: `sigma_step`, within $0.01\sigma$ of the minimum

## Error Handling

A solve fails with an error for:

- a missing or wrongly sized initial value for a parameter block;
- an invalid IR (for example a non-positive robust-loss scale);
- an IR the selected backend cannot map (see above);
- a non-finite initial cost;
- a residual or Jacobian that is not finite where the solver linearizes,
  at the start, after a step or at the solution: a residual that is not
  differentiable there, which no step could be computed from.
