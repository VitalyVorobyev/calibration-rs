# Bundle Adjustment

`bundle_adjust` jointly refines the **capture poses** of a rig and the 3D **points** it observed, minimizing the reprojection error of the stated pixel noise. It is the polish of an already initialized reconstruction (for example the output of [pose recovery](mvg_pose.md) and [triangulation](mvg_pose.md)), not a structure-from-motion pipeline: there is no feature matching, pose-graph optimization or loop closure.

The cameras come from a `CalibratedRig`, so every camera model works (a fisheye beside a pinhole, distortion, tilted sensors) with no separate undistortion, and the result carries the joint covariance of what it refined, the calibration's uncertainty included. It runs on the same problem builder as [your own residuals](custom_residual.md); bundle adjustment is always available, with no cargo feature.

## Usage

```rust
use vision_calibration::mvg::bundle_adjust::{BundleAdjustOptions, Scale, bundle_adjust};
use vision_calibration::mvg::triangulation::RigObservation;

// rig:           CalibratedRig, the cameras and where each sits in the rig
// rig_se3_world: &[Iso3], the initial pose of the rig at each capture (world → rig)
// points:        &[Pt3], the initial world points
// tracks:        Vec<Vec<RigObservation>>, tracks[i] observes points[i]
let tracks = vec![vec![RigObservation::new(0, 0, pixel, noise) /* , ... */]];

let opts = BundleAdjustOptions::new(Scale::Rig);
let adjusted = bundle_adjust(&rig, &rig_se3_world, &points, &tracks, &opts)?;

println!("RMS {:.3} px", adjusted.reprojection_rms_px);
let cov = adjusted.point_covariance(0); // Option<Mat3>, m²
let pose = adjusted.pose_covariance(1); // Option<Matrix6>, capture 1
```

```python
import numpy as np
from vision_calibration import mvg

# rig_se3_world: (K, 4, 4) initial capture poses; points: (N, 3) initial points;
# tracks[i] is a list of mvg.RigObservation of points[i].
adjusted = mvg.bundle_adjust(result, rig_se3_world, points, tracks, scale="rig")
# or fix the scale by a stated distance between points 0 and 1: 0.250 m, σ 0.1 mm
adjusted = mvg.bundle_adjust(
    result, rig_se3_world, points, tracks, scale=("distance", (0, 1), 0.250, 1e-4)
)
adjusted.reprojection_rms_px
adjusted.point_covariance(0)    # (3, 3) in m², or None
adjusted.pose_covariance(1)     # (6, 6), capture 1 (capture 0 is the gauge: None)
```

`calibration="consider"` (the default) or `"prior"`, `loss=vc.robust_huber(...)` and `max_iters` are the remaining options; the refined cameras are in `adjusted.cameras` and `adjusted.cam_se3_rig`.

The observations are the ones [`triangulate`](mvg_pose.md) takes: `RigObservation::new(capture, camera, pixel, noise)`, grouped into tracks, with the noise of each pixel stated as a `PointCovariance`. A pixel is weighted by its covariance, and the stated noise is also what the result's covariance is built from. The result carries the refined `rig_se3_world` and `points`, the `rig` (the given one, or the refined one under a prior), the `uncertainty`, the solver's `report` and the RMS reprojection distance in pixels (`√2 σ` for isotropic noise of `σ` per axis).

Everything is checked before any solve, and a mistake is an `MvgError::InvalidInput` naming the track, point or capture: points and tracks that do not pair up, a track with fewer than two observations, an index out of range, a non-finite start, fewer than two captures (or one capture seen by one camera, where there is nothing to adjust), and a gauge the data cannot fix.

## The Gauge

Reprojection error cannot see the world frame or the scale of a reconstruction: moving or rescaling every capture and point together changes no pixel. So both are **stated**:

- **Frame.** Capture 0 is fixed at the pose given: the world frame is capture 0's rig frame as given. It must have observations. Every other capture and every point is refined relative to it, and capture 0 has no covariance.
- **Scale.** `Scale` has no default, because a silent scale would put a wrong metric answer in every result.
  - `Scale::Rig` takes it from the rig's extrinsics: the baseline between two cameras that see points in the same capture. It needs a rig of two or more cameras whose centres differ, and some capture seen by two of them.
  - `Scale::Distance { points: [a, b], distance, sigma }` takes it from a stated distance between two points (metres), as a residual of its own weighed by `sigma`. This is the way for one moving camera, which sees only up to scale.

A scale the data cannot fix, such as `Scale::Rig` with one camera or with two cameras at one centre, is an error that names the alternative.

The covariance is in this gauge. Under `Scale::Rig` the uncertainty of the baseline is part of it; under `Scale::Distance` the stated `sigma` is.

## The Calibration: Consider or Prior

`BundleAdjustOptions::calibration` chooses how the rig's uncertainty enters:

- **`Use::Consider`** (the default). The cameras and extrinsics stay at their values, and their covariance is propagated into the capture poses' and points': the result is the one a fixed camera gives, with the covariance of one that is not exact. Every point and capture shares the calibration, so they are correlated (`point_covariance(i)` is a block of a dense joint covariance). A rig without a covariance still adjusts, and its uncertainty says why it has none.
- **`Use::Prior`**. The cameras and extrinsics are refined too, held by a Gaussian prior with the calibration's covariance, so the new observations sharpen them. It needs a rig with a covariance. The result's `rig` is the refined rig, with its covariance after the adjustment (labels of the cameras and extrinsics only, as a calibration export has them), and can be passed to `triangulate` or to another adjustment.

## Noise and Robust Losses

`BundleAdjustOptions::loss` puts a robust loss on every reprojection (`RobustLoss::Huber`, `Cauchy`, `Arctan`). Because the residuals are whitened by their stated noise, the loss scale is in σ units: `Huber { scale: 2.0 }` is quadratic up to two σ. A loss changes the estimate and not the covariance, which takes every residual at its stated variance.

On a stereo rig with 4 captures of 30 points and 5 % of the observations 25 to 60 px off, the points' RMS error was 1.0 cm on clean data, 32 cm with no loss, 1.6 cm with Huber and 1.2 cm with Cauchy.

## Reading σ

`adjusted.uncertainty` is the joint covariance of the free parameters, with the labels `capture{k}.rx … tz` for the capture poses (a right-perturbation tangent, `T·Exp(δ)`: rotation in radians, translation in metres in the pose's own frame), `p{i}.x/.y/.z` for the points, and under `Use::Prior` the cameras' own. `point_covariance(i)` and `pose_covariance(k)` slice it. A capture or point that no connected observation constrains makes it unavailable, with the unobservable parameters named; so does a solve that stops on its iteration budget (`max_iters`, 100 by default).

The covariance is dense, so its cost grows with the cube of the parameter count: for thousands of points, adjust a window of captures.

### How it was checked

Over repeated draws of the pixel noise, with a calibration of negligible covariance, the scatter of the points and of capture 1's pose against the reported σ was within 3 % over 300 trials, for all three coordinates of the points and all six components of the pose.

With a real calibration in the loop, each trial calibrated a stereo rig from freshly noisy board views (eight views spread over the image, tilted up to 0.5 rad) and adjusted three captures of forty points. Over 500 such calibrations:

| | scatter / stated |
|---|---|
| `Use::Consider`, point depth | 0.95 |
| `Use::Consider`, capture 1 translation (x, y, z) | 0.98, 1.06, 0.96 |
| `Use::Prior`, refined `fx` (cameras 0, 1) | 0.95, 0.95 |
| `Use::Prior`, refined rig, NEES per parameter (22) | 1.00 |

A second set of calibrations gave 0.94, (0.96, 0.92, 0.97), (0.94, 0.96) and 0.99. The NEES (the error weighed by the whole covariance) checks every direction of the refined rig at once, not only each parameter's own σ. Under `Use::Prior` each camera's σ(`fx`) after the adjustment was no larger than the calibration's (2.8 px before, 2.7 px after, on average); the scene sharpens the distortion and camera 1's pose in the rig more, by a quarter to a third of their σ.

### A Weak Calibration as a Prior

A prior takes the calibration's covariance as a Gaussian, which is exact to first order. A calibration from board views bunched in the middle of the image and tilted by at most 0.2 rad hardly tells a camera's principal point from its rotation in the rig (σ(cx) ≈ 5 px, σ of camera 1's yaw ≈ 9 mrad). Their large errors cancel in the combinations the views do pin down, and through second-order terms they still reach them: along two such combinations the calibration's error was 2.5 to 3 times its stated σ, although every parameter's own σ was right. The refined cameras inherit it: their σ comes out 10 to 25 % optimistic. At a tenth of the pixel noise, or with views that fill the image and tilt further, the calibration's error matches its covariance and so does the refined rig's. The calibration's uncertainty predicts this (`second_order`), and the quality report's `covariance_linearity` check warns, naming the combination ([Judging a Result](quality.md)). Before refining a calibration with `Use::Prior`, give it views that cover the image.

## Relation to Calibration

Calibration workflows such as the rig extrinsics refine intrinsics, distortion and poses together, against a known target. `bundle_adjust` is the complement for scenes without a known target: the points are unknowns, and the cameras are given, with their uncertainty.
