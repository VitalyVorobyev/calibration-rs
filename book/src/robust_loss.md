# Robust Loss Functions

Standard least squares minimizes the sum of squared residuals $\sum r_i^2$. This objective is highly sensitive to outliers: a single point with a large residual can dominate the entire cost function and corrupt the solution. **Robust loss functions** (M-estimators) reduce the influence of large residuals, making optimization tolerant to outliers in the data.

## Problem Setup

In non-linear least squares, calibration-rs minimizes

$$F(\theta) = \frac{1}{2} \sum_{i=1}^{N} \rho\left(s_i(\theta)\right), \qquad s_i = \lVert \mathbf{r}_i(\theta) \rVert^2,$$

where $\mathbf{r}_i$ is a residual block (a reprojection's 2D pixel error, say) and $\rho$ is the loss applied to its squared norm. Without a robust loss $\rho(s) = s$, which gives the standard $\frac{1}{2} \sum \lVert \mathbf{r}_i \rVert^2$. This is the Ceres convention: $\rho(0) = 0$ and $\rho'(0) = 1$, so every loss agrees with least squares for small residuals. `RobustLoss::rho` evaluates it.

## Available Loss Functions

`RobustLoss` has four variants: `None` (plain squared loss) and three robust loss functions, each parameterized by a scale $c > 0$ that controls the transition from quadratic (inlier) to robust (outlier) behavior.

### Huber Loss

$$\rho(s) = \begin{cases} s & \text{if } s \leq c^2 \\ 2c\sqrt{s} - c^2 & \text{if } s > c^2 \end{cases}$$

**Properties**:
- Quadratic in $\lVert \mathbf{r} \rVert$ for small residuals, linear for large ones
- Continuous first derivative
- **Influence**: bounded — an outlier contributes a constant-magnitude gradient, not a growing one

**When to use**: The default robust loss. Good general-purpose choice when you expect a moderate number of outliers.

### Cauchy Loss

$$\rho(s) = c^2 \ln\left(1 + \frac{s}{c^2}\right)$$

**Properties**:
- Grows logarithmically for large residuals (slower than linear)
- Smooth everywhere
- **Influence**: $\rho'(s) = 1 / (1 + s/c^2)$ decreases to zero for large residuals, effectively down-weighting far outliers

**When to use**: When outliers are far from the bulk of the data and should have near-zero influence.

### Arctan Loss

$$\rho(s) = c \arctan\left(\frac{s}{c}\right)$$

Here the scale is in squared-residual units: the transition is near $\lVert \mathbf{r} \rVert \approx \sqrt{c}$.

**Properties**:
- Bounded: $\rho(s) \to \frac{\pi}{2} c$ as $s \to \infty$
- **Influence**: $\rho'(s) = 1 / (1 + s^2/c^2)$ approaches zero for large residuals (redescending)

**When to use**: When very strong outlier rejection is needed. More aggressive than Cauchy but can make convergence harder.

## Comparison

| Loss | Large-$r$ growth | Outlier influence | Convergence |
|------|-------------------|-------------------|-------------|
| None ($\rho(s) = s$) | Quadratic | Unbounded | Best |
| Huber | Linear | Bounded (constant) | Good |
| Cauchy | Logarithmic | Decreasing | Moderate |
| Arctan | Bounded | Approaching zero | Can be tricky |

## Choosing the Scale Parameter $c$

The scale $c$ sets the boundary between "inlier" and "outlier" behavior:

- **Too small**: Treats good data as outliers, reducing effective sample size
- **Too large**: Outliers still dominate (approaches standard least squares)
- **Rule of thumb**: For Huber and Cauchy, set $c$ to the expected residual magnitude for good data points; for reprojection residuals $c = 1\text{-}3$ pixels is typical. Arctan's scale is in squared units, so the equivalent is $c = 1\text{-}9$.

## Usage in calibration-rs

Robust losses are selected with the `RobustLoss` enum:

```rust,ignore
pub enum RobustLoss {
    None,
    Huber { scale: f64 },
    Cauchy { scale: f64 },
    Arctan { scale: f64 },
}
```

The `solver` group holds engine settings only (`max_iters`, `verbosity`,
`backend`). Each optimize stage carries the robust loss of the residual family
it solves: `reprojection_loss` for target corners, `laser_loss` for laser
residuals. There is one loss per stage, so a convex Huber in the per-camera
intrinsics stage and a redescending Cauchy in the final stage is expressible:

```rust
use vision_calibration::optim::RobustLoss;
# use vision_calibration::planar_intrinsics::PlanarIntrinsicsProblem;
# use vision_calibration::session::CalibrationSession;
# fn main() -> anyhow::Result<()> {
# let mut session = CalibrationSession::<PlanarIntrinsicsProblem>::new();

session.update_config(|c| {
    c.optimize.reprojection_loss = RobustLoss::Huber { scale: 2.0 };
})?;
# Ok(())
# }
```

To make a whole problem robust, set the loss in every stage that solves the
corners:

| Problem | Stages that take a `reprojection_loss` |
|---|---|
| planar intrinsics | `optimize` |
| single-camera hand-eye | `intrinsics_optimize`, `handeye_optimize` |
| rig extrinsics | `intrinsics_optimize`, `rig` |
| rig hand-eye | `intrinsics_optimize`, `rig`, `handeye_optimize` |
| laserline device | `optimize` (`reprojection_loss` and `laser_loss`) |
| rig laserline device | `optimize` (`laser_loss`; the per-camera stages come from the upstream calibration) |
| rig hand-eye laserline | `joint_optimize` (`reprojection_loss` and `laser_loss`), plus the hand-eye and laserline-init groups it embeds |

The default is `None` everywhere, except the laser stages, which default to
Huber with scale 1.0. The laserline device's initial camera-only tilt solve
(the cold start of a Scheimpflug sensor) is plain least squares.

The backend applies the loss function during residual evaluation, modifying both the cost and the Jacobian.

## How the Solver Applies a Loss

The Levenberg–Marquardt loop always measures progress with $F$ itself. To build each step's linear model, a backend folds the loss into the residual block and its Jacobian:

- **tiny-solver** uses the Triggs correction, which matches $\rho$ to second order.
- **factrs** uses iterative reweighting: the block and its Jacobian are scaled by $\sqrt{\rho'(s_i)}$.

Both models have the same first-order change as $F$, so both converge to the same minimum; see [Solver Backends](solver_backends.md).

## Interaction with RANSAC

RANSAC and robust losses address outliers at different stages:

- **RANSAC** (linear initialization): Binary inlier/outlier classification. Used during model fitting to reject gross outliers before any optimization.
- **Robust losses** (non-linear refinement): Soft down-weighting. Used during optimization to reduce the influence of moderate outliers that passed RANSAC.

The two approaches are complementary: RANSAC handles gross outliers during initialization, while robust losses handle smaller outliers during refinement.
