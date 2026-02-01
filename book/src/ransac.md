# RANSAC

RANSAC (Random Sample Consensus) is a robust estimation method that fits a model to data containing outliers. Unlike least-squares methods that use all data points and can be corrupted by even a single outlier, RANSAC repeatedly samples minimal subsets, fits a model to each, and selects the model with the most support (inliers).

## Algorithm

Given $N$ data points and a model requiring $m$ points to fit:

1. Repeat for up to $k_{\max}$ iterations:
   1. **Sample** $m$ random data points
   2. **Fit** a model to the $m$-point sample (skip if degenerate)
   3. **Score**: count inliers — points with residual $< \epsilon$
   4. If this model has more inliers than the current best, update the best
   5. Optionally **refit** the model on all inliers for a tighter fit
   6. **Update** the iteration bound dynamically based on the current inlier ratio

2. Return the best model and its inlier set

## Dynamic Iteration Bound

After finding a model with inlier ratio $w = n_{\text{inlier}} / N$, the number of iterations needed to find an all-inlier sample with probability $p$ is:

$$k = \frac{\log(1 - p)}{\log(1 - w^m)}$$

where $m$ is the minimum sample size. As more inliers are found, $w$ increases and $k$ decreases, allowing early termination. The iteration count is clamped to $[k_{\text{current}}, k_{\max}]$ — it can only decrease.

**Example**: For $m = 4$ (homography), 50% inliers, and 99% confidence: $k = \frac{\log(0.01)}{\log(1 - 0.5^4)} = \frac{-4.6}{-0.065} \approx 72$ iterations.

## The `Estimator` Trait

calibration-rs provides a generic RANSAC engine that works with any model implementing the `Estimator` trait:

```rust
pub trait Estimator {
    type Datum;
    type Model;

    const MIN_SAMPLES: usize;

    fn fit(data: &[Self::Datum], sample_indices: &[usize])
        -> Option<Self::Model>;

    fn residual(model: &Self::Model, datum: &Self::Datum) -> f64;

    fn is_degenerate(data: &[Self::Datum], sample_indices: &[usize])
        -> bool { false }

    fn refit(data: &[Self::Datum], inliers: &[usize])
        -> Option<Self::Model> {
        Self::fit(data, inliers)
    }
}
```

| Method | Purpose |
|--------|---------|
| `MIN_SAMPLES` | Minimum points for a model (e.g., 4 for homography) |
| `fit` | Fit model from selected points |
| `residual` | Distance from a point to the model |
| `is_degenerate` | Reject degenerate samples (e.g., collinear points for homography) |
| `refit` | Optional: refit model on full inlier set (may differ from `fit`) |

## Configuration

```rust
pub struct RansacOptions {
    pub max_iters: usize,      // Maximum iterations
    pub thresh: f64,           // Inlier distance threshold
    pub min_inliers: usize,    // Minimum consensus set size
    pub confidence: f64,       // Desired probability (0-1) for dynamic bound
    pub seed: u64,             // RNG seed for reproducibility
    pub refit_on_inliers: bool, // Refit model on full inlier set
}
```

**Choosing the threshold $\epsilon$**: The threshold should reflect the expected noise level. For pixel-space residuals, $\epsilon = 2\text{-}5$ pixels is typical. For normalized-coordinate residuals, scale accordingly by the focal length.

## Result

```rust
pub struct RansacResult<M> {
    pub success: bool,
    pub model: Option<M>,
    pub inliers: Vec<usize>,
    pub inlier_rms: f64,
    pub iters: usize,
}
```

## Instantiated Models

RANSAC is used with several models in calibration-rs:

| Model | `MIN_SAMPLES` | Residual | Usage |
|-------|--------------|----------|-------|
| Homography | 4 | Reprojection error (pixels) | `dlt_homography_ransac` |
| Fundamental matrix | 8 | Epipolar distance | `fundamental_8point_ransac` |
| PnP (P3P) | 3 | Reprojection error (pixels) | `pnp_ransac` |

## Deterministic Seeding

All RANSAC calls in calibration-rs use a deterministic seed for the random number generator. This ensures:

- **Reproducible results** across runs
- **Deterministic tests** that do not flake
- **Debuggable behavior** — the same input always produces the same output

The seed can be configured via `RansacOptions::seed`.

## Best-Model Selection

When multiple models achieve the same inlier count, RANSAC selects the model with the **lowest inlier RMS** (root mean square of inlier residuals). This breaks ties in favor of more accurate models.
