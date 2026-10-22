# Judging a Result

A calibration can finish without error and still not be fit to use: the
views may leave a parameter undetermined, cover one corner of the image, or
include a misdetected board. `assess` reads any calibration result and says
so, as a verdict — `pass`, `warn` or `fail` — and the checks behind it. It
builds on the result's [uncertainty](uncertainty.md) and residuals; it does
not re-run the calibration.

```rust
use vision_calibration::analysis::{CheckStatus, QualityThresholds, assess};

let export = session.export()?;
let mut thresholds = QualityThresholds::default();
thresholds.expected_feature_noise_px = Some(0.2); // the detector's σ per axis
let report = assess(&export, &[[1280, 1024]], &thresholds)?;
println!("verdict: {:?}", report.verdict);
for check in &report.checks {
    if check.status != CheckStatus::Pass {
        println!("{:?} {:?}: {}", check.kind, check.status, check.message);
    }
}
```

```python
report = vc.assess(
    result,
    vc.QualityThresholds(expected_feature_noise_px=0.2),
    image_sizes=[(1280, 1024)],
)
print(report.verdict)
for check in report.checks:
    if check.status != "pass":
        print(check.kind, check.camera, check.status, check.message)
```

`image_sizes` gives each camera's `[width, height]` in pixels, in camera
order; pass an empty list (Python: `None`) when the result's image manifest
records them. The thresholds' defaults suit a perspective camera; every one
can be changed, and in JSON they reject unknown fields.

## The checks

| Check | Per | Fails when | Warns when |
|---|---|---|---|
| `observability` | result | the data does not determine some parameters (`rank_deficient`), or there is no redundancy | a parameter ended on a bound |
| `convergence` | result | the final solve stopped on its iteration budget | — |
| `variance_factor` | noise level | σ̂²/σ₀² (σ̂₀² with stated point covariances) is above its χ² range | it is below the range |
| `projection_sigma` | camera | — | σ somewhere in the image exceeds 1 px |
| `feature_coverage` | camera | features reached under 30 % of the image | under 50 % |
| `view_outliers` | camera | — | a view's mean residual exceeds 3 σ̂ |
| `parameter_correlation` | camera | — | two parameters correlate above 0.99 |

Each check carries its status, the measured value with its unit and the
threshold it was compared to (`measurement`), what it names (`subjects`:
parameter labels, parameter pairs such as `fx/cx`, or views such as
`view 3`) and a one-sentence `message`.

- **Observability** reads the uncertainty's status. A `rank_deficient`
  result names the parameters the views do not separate, such as a sensor
  tilt and the principal point; add views that differ in the way those
  parameters need, or fix one of them.
- **Convergence** fails a solve that ran out of iterations: its numbers are
  not a minimum. Raise `solver.max_iters`.
- **Variance factor.** σ̂ is the noise level the residuals imply (see
  [noise](uncertainty_reading.md#what-is-reported)); σ₀ is the noise you expect from
  the detector, `expected_feature_noise_px`. There is no universal value, so
  the check is not applicable until you state one. Then σ̂²/σ₀² is tested
  against the χ² distribution at the residuals' degrees of freedom, two-sided
  at `variance_test_alpha` (0.01). Above the range, the residuals exceed what
  the detector explains: a wrong model, a board that is not flat, or bad
  detections. Below it, σ₀ was overstated. Because σ̂ is a median-based
  estimate, which scatters more than a sample variance, the test uses 37 % of
  the degrees of freedom, the estimate's efficiency under Gaussian noise.
  When the observations state a covariance per point, the residuals are
  whitened and their level is σ̂₀, which is 1 when the stated covariances
  hold: the check then tests σ̂₀² against 1 and needs no
  `expected_feature_noise_px` ([Per-Point Pixel
  Covariance](uncertainty_point_covariance.md)).
- **Projection σ** is the largest value of the camera's
  [projection-uncertainty map](uncertainty_projection.md), over
  the grid and its corners. It exceeds a pixel where the views did not reach
  and the distortion is extrapolated. The map is conservative for uses that
  re-estimate the camera's pose, so this check only warns.
- **Feature coverage** splits the image into 8 × 6 cells and counts those
  holding at least one observed feature. Views bunched in one part of the
  image leave the rest to extrapolation; add views with the board near the
  empty cells, corners first.
- **View outliers** averages `|r| / σ̂` per view and camera (with stated
  point covariances, each feature's whitened error over σ̂₀). Over Gaussian
  inliers that mean is about 1.25; a view above 3 has residuals well over
  twice the typical ones, as from a misdetection, motion blur or a wrong
  robot pose. Inspect or drop the named views.
- **Parameter correlation** looks at the camera's own parameters
  (intrinsics, distortion, tilt). Above 0.99 the data determines a
  combination of the pair, not each. `fx` and `fy` are left out: the board's
  distance scales both, so they correlate almost fully in every planar
  calibration while their ratio is well determined.

## Not applicable is not a pass

A check whose inputs the result does not carry is `not_applicable`, with the
reason in its message, and does not count toward the verdict:

- no image size for the camera: the projection and coverage checks;
- no stated σ₀ and no stated point covariances: the variance factor;
- no covariance (the uncertainty is unavailable): the checks that read it;
- a camera whose parameters the final solve held fixed — the rig and rig
  hand-eye workflows by default — has no covariance to project or
  correlate.

The report's `verdict` is the worst status of the checks that were
assessed. A report in which no check could be assessed is `warn`, since then
nothing vouches for the result.

## Image sizes

A camera's image size comes from the result's image manifest when its frames
are regions of a larger image (all of the same size), and otherwise from
`image_sizes`. Sizes that disagree, regions of different sizes, or a list
with the wrong number of cameras are an error rather than a guess.

## In JSON

The report serializes as

```json
{
  "verdict": "warn",
  "checks": [
    {"kind": "observability", "status": "pass", "message": "..."},
    {"kind": "feature_coverage", "camera": 0, "status": "warn",
     "measurement": {"value": 0.44, "threshold": 0.5, "unit": "fraction"},
     "message": "features reached 21 of 48 cells (8 × 6) of the image, 44 %"}
  ]
}
```

and the thresholds as

```json
{
  "expected_feature_noise_px": null,
  "variance_test_alpha": 0.01,
  "max_projection_sigma_px": 1.0,
  "coverage_warn_below": 0.5,
  "coverage_fail_below": 0.3,
  "max_view_residual": 3.0,
  "max_parameter_correlation": 0.99
}
```

where an absent field takes its default.
