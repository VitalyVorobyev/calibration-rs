# Per-Point Pixel Covariance

By default every observed image point is trusted alike, and the noise level
is estimated from the residuals in pixels. When you know more about each
point's noise, say a detector that reports how well it located each corner,
state it as a covariance per point. The solve then weighs each point by
what it states: a point with a large covariance counts less, and an
elongated one is trusted more across its long axis than along it.

## Stating covariances

A covariance is the symmetric 2 × 2 matrix `[[xx, xy], [xy, yy]]` in px²,
one per point of a view. It must be finite and positive definite (`xx > 0`,
`yy > 0`, `xy² < xx·yy`); anything else is rejected when the view is built
or read.

```rust
use vision_calibration::core::{CorrespondenceView, PointCovariance};

// σ = 0.3 px along u, 0.1 px along v.
let cov = PointCovariance::new(0.09, 0.0, 0.01)?;
let view = CorrespondenceView::new_with_covariances(points_3d, points_2d, vec![cov; n])?;
```

```python
cov = vc.PointCovariance(0.09, 0.0, 0.01)           # or vc.PointCovariance.isotropic(0.2)
obs = vc.Observation(points_3d=p3, points_2d=p2, cov_px=[cov] * len(p2))
```

In JSON a view carries them as `cov_px`, one object per point:

```json
{"points_3d": [[0.0, 0.0, 0.0]], "points_2d": [[412.3, 228.9]],
 "cov_px": [{"xx": 0.09, "xy": 0.0, "yy": 0.01}]}
```

**All or none.** Within one calibration input either every view states
covariances or none does, across every camera of a rig. Residuals in pixels
and whitened residuals are not one noise population, so an input that mixes
them is rejected (`MixedCovariance`) before any work. There is no setting
to switch the weighting: the solve uses what the observations carry.

A uniform weight `w` on every point is the covariance `I / w`; covariances
that are all the same multiple of the identity give the same estimate and
covariance as stating none.

## What changes in the solve

Each reprojection residual is *whitened*: the pixel error `e` becomes `L e`,
with `L` the upper-triangular square root of `Σ⁻¹`. The whitened residual has
unit covariance when the stated covariance is right, so it is dimensionless.
Two consequences:

- **Noise level.** The result's reprojection noise is reported as `sigma0`
  (`σ̂₀`) instead of `sigma`: the Gaussian-equivalent standard deviation of
  the whitened residuals, the square root of the a-posteriori variance
  factor. It is about 1 when the stated covariances describe the noise; 2
  means the actual noise is twice what they state, 0.5 half. It reads a few
  percent below 1 even when the covariances are exact, since the fit absorbs
  some of the residuals' freedom (0.976 for 68 parameters over 1620
  residuals). The parameter covariance is scaled by `σ̂₀²`, so it is right
  even when the covariances are right only up to a common factor.
- **Robust loss scale.** A loss's scale applies to the whitened residual,
  in units of each point's own σ: `Huber { scale: 2.0 }` starts to
  down-weight a point at twice its stated σ, whatever that is in pixels.

The reported reprojection errors stay in pixels: the mean and per-camera
errors, the histograms, and each feature's `error_px`. Beside `error_px`,
each target feature carries `error_whitened`, `√(eᵀ Σ⁻¹ e)`, its error in
units of its own σ.

On synthetic planar and two-camera scenes whose points state covariances of
0.1 to 1 px with up to 4:1 anisotropy in any orientation, and whose noise
is drawn from them, `σ̂₀` was 0.98 ± 0.03, the predicted σ of every
parameter matched the Monte-Carlo scatter within 4 %, and 95 % of the
normalized errors fell within 2σ. Weighing the same points alike instead
gave errors 2.5 to 3 times larger.

## Reading the noise

A result's `noise` entry for reprojection is either:

| JSON key | Rust | Python | Meaning |
|---|---|---|---|
| `sigma` | `NoiseLevel::Sigma(s)` | `whitened == False` | σ̂ in pixels per axis |
| `sigma0` | `NoiseLevel::Sigma0(s)` | `whitened == True` | σ̂₀, dimensionless |

The laser residuals keep their own units (pixels for the line distance,
metres for the point-to-plane distance) and are never whitened.

The [quality report](quality.md) uses the distinction: with whitened
residuals the variance-factor check tests `σ̂₀²` against 1 and needs no
`expected_feature_noise_px`, and the view-outlier check averages
`error_whitened / σ̂₀` per view.
