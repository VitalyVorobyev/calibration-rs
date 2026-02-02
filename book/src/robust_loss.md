# Robust Loss Functions

Standard least squares minimizes the sum of squared residuals $\sum r_i^2$. This objective is highly sensitive to outliers: a single point with a large residual can dominate the entire cost function and corrupt the solution. **Robust loss functions** (M-estimators) reduce the influence of large residuals, making optimization tolerant to outliers in the data.

## Problem Setup

In non-linear least squares, we minimize:

$$\min_\theta \sum_{i=1}^{N} \rho(r_i(\theta))$$

where $\rho$ is the loss function applied to each residual $r_i$. The standard (non-robust) case uses $\rho(r) = \frac{1}{2} r^2$.

## Available Loss Functions

calibration-rs provides three robust loss functions, each parameterized by a scale $c > 0$ that controls the transition from quadratic (inlier) to robust (outlier) behavior.

### Huber Loss

$$\rho(r) = \begin{cases} \frac{1}{2} r^2 & \text{if } |r| \leq c \\ c \left( |r| - \frac{c}{2} \right) & \text{if } |r| > c \end{cases}$$

**Properties**:
- Quadratic for small residuals, linear for large residuals
- Continuous first derivative
- **Influence function**: bounded at $\pm c$ — outliers contribute constant gradient, not growing

**When to use**: The default robust loss. Good general-purpose choice when you expect a moderate number of outliers.

### Cauchy Loss

$$\rho(r) = \frac{c^2}{2} \ln\left(1 + \frac{r^2}{c^2}\right)$$

**Properties**:
- Grows logarithmically for large residuals (slower than linear)
- Smooth everywhere
- **Influence function**: $\psi(r) = r / (1 + r^2/c^2)$ — decreases to zero for large $|r|$, effectively down-weighting far outliers

**When to use**: When outliers are far from the bulk of the data and should have near-zero influence.

### Arctan Loss

$$\rho(r) = c^2 \arctan\left(\frac{r^2}{c^2}\right)$$

**Properties**:
- Bounded: $\rho(r) \to \frac{\pi}{2} c^2$ as $|r| \to \infty$
- **Influence function** approaches zero for large residuals (redescending)

**When to use**: When very strong outlier rejection is needed. More aggressive than Cauchy but can make convergence harder.

## Comparison

| Loss | Large-$r$ growth | Outlier influence | Convergence |
|------|-------------------|-------------------|-------------|
| Quadratic ($r^2/2$) | Quadratic | Unbounded | Best |
| Huber | Linear | Bounded (constant) | Good |
| Cauchy | Logarithmic | Decreasing | Moderate |
| Arctan | Bounded | Approaching zero | Can be tricky |

## Choosing the Scale Parameter $c$

The scale $c$ sets the boundary between "inlier" and "outlier" behavior:

- **Too small**: Treats good data as outliers, reducing effective sample size
- **Too large**: Outliers still dominate (approaches standard least squares)
- **Rule of thumb**: Set $c$ to the expected residual magnitude for good data points. For reprojection residuals, $c = 1\text{-}3$ pixels is typical.

## Usage in calibration-rs

Robust losses are specified per-residual block in the optimization IR:

```rust
pub enum RobustLoss {
    None,
    Huber { scale: f64 },
    Cauchy { scale: f64 },
    Arctan { scale: f64 },
}
```

Each problem type exposes the loss function as a configuration option:

```rust
session.update_config(|c| {
    c.robust_loss = RobustLoss::Huber { scale: 2.0 };
})?;
```

The backend applies the loss function during residual evaluation, modifying both the cost and the Jacobian.

## Iteratively Reweighted Least Squares (IRLS)

Under the hood, robust loss functions are typically implemented via IRLS: each residual is weighted by $w_i = \rho'(r_i) / r_i$, and the weighted least-squares problem is solved iteratively. The Levenberg-Marquardt backend handles this automatically.

## Interaction with RANSAC

RANSAC and robust losses address outliers at different stages:

- **RANSAC** (linear initialization): Binary inlier/outlier classification. Used during model fitting to reject gross outliers before any optimization.
- **Robust losses** (non-linear refinement): Soft down-weighting. Used during optimization to reduce the influence of moderate outliers that passed RANSAC.

The two approaches are complementary: RANSAC handles gross outliers during initialization, while robust losses handle smaller outliers during refinement.
