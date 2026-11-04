# Pose Recovery and Triangulation

Given matched pixels in two calibrated views, `recover_relative_pose` finds the rotation and translation direction between the cameras and triangulates the points.

## Recovering the Relative Pose

Input is the rig, the camera of each view and a list of `PixelMatch`, pairs of **pixels** with their stated noise: `pt1` in rig camera `cameras[0]` at the first capture, `pt2` in rig camera `cameras[1]` at the second (`[0, 0]` for one camera that moved). Each pixel is unprojected through its camera's model into a bearing, so a distorted lens or a fisheye needs no undistortion first. The function runs the 5-point algorithm on several evenly spaced subsets of the correspondences, scores each candidate essential matrix by its covariance-weighted Sampson error (below) and by cheirality over **all** correspondences, keeps the pose for which the most points lie ahead along both bearings, and triangulates:

```rust
use vision_calibration::mvg::pose_recovery::recover_relative_pose;
use vision_calibration::mvg::types::PixelMatch;

let matches: Vec<PixelMatch> = /* (pixel and its covariance in the first view, the same in the second) */;
let rel = recover_relative_pose(&rig, [0, 0], &matches)?;
// rel.second_se3_first : pose of the first camera's frame in the second's, unit translation
// rel.essential        : b2ᵀ E b1 = 0 on the bearings
// rel.points           : triangulated points with reprojection and parallax diagnostics
// rel.scene            : parallax, pure-rotation and planarity diagnostics
```

In Python the same call takes the calibration result itself, and the matches are `mvg.PixelMatch` values:

```python
import vision_calibration as vc
from vision_calibration import mvg

noise = mvg.reprojection_noise(result, camera=0)   # the calibration's own level
matches = [mvg.PixelMatch(p1, noise, p2, noise) for p1, p2 in pixel_pairs]
rel = mvg.recover_relative_pose(result, (0, 1), matches)
rel.second_se3_first   # 4x4 numpy matrix, unit translation
rel.points             # list of TriangulatedPoint
rel.scene              # SceneDiagnostics
```

### Stating the Pixel Noise

Every pixel of a `PixelMatch` carries its covariance (a `PointCovariance`, px²). The noise is stated, never guessed, and there are three ways to state it:

```rust
use vision_calibration::core::PointCovariance;

// Per observation: a detector's own 2×2 covariance, elongated or correlated.
let cov = PointCovariance::new(0.25, 0.08, 0.09)?;
// One σ for every pixel, in pixels.
let iso = PointCovariance::isotropic(0.5)?;
// The calibration's level: the reprojection σ̂ it measured on its own data.
let level = rig.reprojection_noise(0)?;   // camera 0 of the rig
let m = PixelMatch::new(pt1, level, pt2, iso);
```

`reprojection_noise` is an error when the calibration has no covariance, no reprojection level for that camera, or a level that is the dimensionless $\hat\sigma_0$ of residuals whitened by stated covariances; state the noise instead.

The functions work on bearings, so each pixel covariance $\Sigma_k$ is mapped onto its bearing through the camera model's projection Jacobian: $m_k = J_k \Sigma_k J_k^\top$ with $J_k = \partial b / \partial\,\text{pixel}$ (computed by central differences of the model's forward projection, so every distortion and fisheye model is served). For a bearing pair $(b_1, b_2)$ and an essential matrix $E$, the **covariance-weighted Sampson error** is the epipolar constraint over its first-order standard deviation:

$$
e = \frac{|b_2^\top E\, b_1|}{\sqrt{g_1^\top m_1 g_1 + g_2^\top m_2 g_2}},\qquad
g_1 = (I - b_1 b_1^\top) E^\top b_2,\quad g_2 = (I - b_2 b_2^\top) E\, b_1 .
$$

It is dimensionless, in σ, and does not change with the scale of $E$.

At least 5 matches are required. The translation is recovered **up to scale**: two bare views contain no metric anchor, so `rel.second_se3_first` has a unit-norm translation. Fix the scale with a known baseline (for a calibrated rig, the extrinsics already carry it) or a known 3D distance.

Each entry of `rel.points` is a `TriangulatedPoint`:

| Field | Meaning |
|---|---|
| `point` | the 3D position, in the first camera's frame at unit baseline |
| `reprojection_error` | RMS distance in pixels between each observation and the point projected through that camera's model; `None` when the point does not project in some camera |
| `parallax_deg` | the largest angle between two observing rays; small angles mean poor triangulation geometry |
| `in_front` | whether the point lies ahead along every observing bearing |

### Degenerate Geometry

The 5-point estimator needs a genuinely three-dimensional scene. A near-planar cloud, a pure rotation (no baseline) or nearly parallel rays make the essential matrix poorly determined. `rel.scene` (a `SceneDiagnostics`) reports the median parallax, pure-rotation and planar flags and a baseline ratio (the unit baseline over the median depth, measured along the first camera's bearing); for a planar scene the homography model (`mvg::homography::decompose_homography`) is the right one instead.

## Outliers: the Robust Variants

Real matches contain outliers. The `mvg::robust` wrappers put a RANSAC loop around the minimal solvers:

```rust
use vision_calibration_core::RansacOptions;
use vision_calibration::mvg::robust::{estimate_essential, recover_relative_pose_robust};

let opts = RansacOptions {
    min_inliers: 12,
    ..RansacOptions::new(3.0)   // weighted Sampson error, in σ of the stated noise
};
let robust = recover_relative_pose_robust(&rig, [0, 0], &matches, &opts)?;
// robust.second_se3_first, robust.essential, robust.inliers (indices), robust.inlier_rms
```

```python
ransac = mvg.RansacOptions(thresh=3.0, min_inliers=12)    # thresh in σ of the stated noise
robust = mvg.recover_relative_pose_robust(result, (0, 1), matches, ransac)
estimate = mvg.estimate_essential(result, (0, 1), matches, ransac)
robust.inliers, robust.inlier_rms
```

**The threshold is in σ**: the residual is the weighted Sampson error above, so `3.0` keeps about 99.7 % of the inliers when the noise is Gaussian and as stated, independent of the focal length and of the camera model. The stated noise is the scale: understating the pixel noise by a factor of two doubles every residual and rejects good matches. There is no default threshold: `RansacOptions::new(thresh)` requires one, and a threshold that is not finite and positive is an error. The run is deterministic for a given `seed` field.

`RansacOptions` is defined in `vision-calibration-core`, which the facade does not re-export; add that crate to your dependencies to name it.

`estimate_essential` returns the essential matrix with its inlier indices and the RMS of the weighted Sampson error over the inliers (in σ, about 1 when the stated noise describes the pixels); `estimate_homography` does the same for a homography (4-point DLT with symmetric transfer error, on plain image points). In Python, `mvg.estimate_homography(correspondences, mvg.RansacOptions(thresh=...))` takes an `(N, 2, 2)` array (or a list of `((x1, y1), (x2, y2))`) and returns the `homography` `(3, 3)`, `inliers` and `inlier_rms`; its threshold is in pixels. `recover_relative_pose_robust` selects the inliers first and disambiguates the four decompositions of the essential matrix with cheirality on the inliers only, so outliers cannot corrupt the positive-depth vote.

## Triangulating Points from N Views

`mvg::triangulation::triangulate` takes the rig, the pose of the rig at each capture (`rig_se3_world`, taken as exact) and, for each point, a track of observations: the pixel, its noise, the rig camera that saw it and the capture. It returns a `Triangulation`: a `TriangulatedPoint` per track in the world frame, with the same diagnostics, and the joint covariance of the points.

```rust
use vision_calibration::mvg::triangulation::{RigObservation, triangulate};
use vision_calibration::control::RunControl;
use vision_calibration::core::{Iso3, PointCovariance};

// One stereo capture: the rig frame is the world frame. Each pixel is
// stated to 0.3 px.
let noise = PointCovariance::isotropic(0.3)?;
let track = vec![
    RigObservation::new(0, 0, pixel_in_camera_0, noise),
    RigObservation::new(0, 1, pixel_in_camera_1, noise),
];
let found = triangulate(&rig, &[Iso3::identity()], &[track], &RunControl::default())?;
println!("{:?} parallax {:.2}°", found.points[0].point, found.points[0].parallax_deg);
if let Some(cov) = found.point_covariance(0) {
    // The 3×3 covariance in m²; the depth's σ along the world z axis.
    println!("σ_z = {:.2} mm", cov[(2, 2)].sqrt() * 1e3);
}
```

```python
import numpy as np

noise = vc.PointCovariance.isotropic(0.3)   # stated per pixel
track = [
    mvg.RigObservation(0, 0, pixel_in_camera_0, noise),
    mvg.RigObservation(0, 1, pixel_in_camera_1, noise),
]
found = mvg.triangulate(result, np.eye(4), [track])   # rig_se3_world: (4, 4) or (K, 4, 4)
found.points[0].point, found.points[0].parallax_deg
cov = found.point_covariance(0)                       # (3, 3) in m², or None
```

The last argument is a `RunControl` ([Sessions](session.md) describes it): its observers receive each iteration of the point refinement and its token stops the call with `MvgError::Cancelled`. `PoseOptions::control` and `BundleAdjustOptions::control` do the same for `estimate_pose` and `bundle_adjust`, and `StereoRectification::triangulate` takes a control as `triangulate` does. In Python these functions take `progress=`, a callable given each report; raising from it, or Ctrl-C, stops the call.

Camera $i$ at capture $k$ sees the world through `rig.cam_se3_rig()[i] * rig_se3_world[k]`, so a moving rig, several cameras and a single moving camera are all the same call. The call is all-or-nothing: a track with fewer than two observations, an index out of range or a zero-parallax system is an error that names the track.

**The estimate.** The linear algorithm of [Linear Triangulation](triangulation.md) on bearings starts each point; all the points are then refined together on the reprojection error, each pixel weighted by its stated covariance (a general `PointCovariance`, an isotropic σ, or `rig.reprojection_noise(camera)` for the calibration's own noise level). The noise is stated because a point seen twice leaves one degree of freedom, too few to estimate a noise level from.

**The covariance** has two sources. The first is the observations' noise. The second is the calibration's: the rig's cameras and extrinsics enter as *consider* parameters, held at their values while their covariance is propagated into the points'. Every point of a call shares the calibration's error, so the points are correlated: `cross_covariance(i, j)` gives the 3×3 block between two points, and a distance or an angle between points needs it. The points are solved in one problem and the covariance is dense over all coordinates, so the cost grows as the cube of the number of points; for thousands of points, call per group of points that need a joint covariance.

A rig exported without a covariance still gives refined points, and the uncertainty says why it has none rather than reporting the observations' share alone.

**Why the calibration's share matters.** Take a stereo pair calibrated from eight views at 0.3 px noise, and ten points 0.6 to 0.8 m ahead measured with 0.3 px noise. Over 60 repeated calibrations the depth of a point scatters by 6.7 mm (root mean square of the reported σ: 6.7 mm, scatter over reported 0.95). The σ from the observations' noise alone is 2.1 mm, and the scatter is 3.2 times it: a camera taken as exact would report error bars three times too small.

## Laser Points with σ

A laser line seen by a calibrated camera, with its plane known, gives a 3D point per line pixel: the viewing ray meets the plane. `mvg::laser::triangulate_laser` takes the rig, which carries each camera's laser plane (an export of a laser workflow gives such a rig from `calibrated_rig()`), and the laser pixels with their stated noise:

```rust
use vision_calibration::core::{PointCovariance, Pt2};
use vision_calibration::mvg::laser::{LaserObservation, triangulate_laser};
use vision_calibration::mvg::metrology::{PointSet, distance};

// The laser-peak detector's noise, 0.1 px.
let noise = PointCovariance::isotropic(0.1)?;
let observations: Vec<LaserObservation> = laser_pixels
    .iter()
    .map(|&pixel: &Pt2| LaserObservation::new(0, pixel, noise))
    .collect();
let found = triangulate_laser(&rig, &observations)?;
println!("{:?} seen at {:.0}°", found.points[0].point, found.points[0].incidence_deg);

// The joint covariance feeds the measurements.
let set = PointSet::from_laser_triangulation(&found)?;
let length = distance(&set, 0, observations.len() - 1)?;
println!("{:.3} m ± {:.3} mm", length.value, length.sigma * 1e3);
```

```python
observations = [mvg.LaserObservation(0, px, noise) for px in laser_pixels]
found = mvg.triangulate_laser(result, observations)   # a laser calibration's result
points = mvg.PointSet.from_laser_triangulation(found)
length = mvg.distance(points, 0, len(observations) - 1)
print(f"{length.value:.3f} m ± {length.sigma * 1e3:.3f} mm")
```

The points are in the rig frame (the single camera's own frame for a laserline device). The pixel goes through its camera's model, so distortion, fisheye (beyond 90° from the axis) and tilted sensors need no separate undistortion; a ray parallel to the plane, or meeting it behind the camera, is an error that names the observation. `incidence_deg` is the angle between the ray and the plane: depth is ill-determined as it nears zero.

**The covariance** has two sources, as for triangulation: the pixels' noise, and the calibration's, here the camera's parameters, the laser plane and the camera's pose in the rig, *with their correlations* as the calibration estimated them together. Each pixel fixes its point through $\pi(p;\theta) = u$ and $n^\top p + d = 0$, so the point moves with the pixel, the camera, the plane and the pose by the implicit-function derivatives of those two equations, computed exactly from the camera model. Every point of a call shares the calibration, so the points are correlated and `cross_covariance(i, j)` gives the block between two. The covariance is conditional on what the calibration held fixed (a plane or camera it did not estimate counts as exact), and the pixel noise is the detector's, stated by you. The cost grows as the square of the number of points times the number of calibration parameters, with a dense covariance over the coordinates.

A rig whose covariance lists none of an observing camera's intrinsics, distortion or sensor parameters (a laser plane fitted with the cameras held fixed and their covariance left out) gives points whose uncertainty says it is unavailable, rather than a σ from the pixels alone. A rig without a covariance does the same. A rig laserline device result whose upstream rig calibration carried its covariance (as `RigHandeyeExport::to_upstream_calibration` and the dataset runner pass it) lists the cameras and their extrinsics with the planes, so its points carry the whole calibration.

**Why the calibration's share matters.** Take a laserline device calibrated from six views at 0.3 px noise, and eight points of the plane 0.4 to 0.6 m ahead measured with 0.1 px noise. Over 300 repeated calibrations the depth of a point scatters by 1.02 times the reported σ (4.7 mm, root mean square), and the length between the first and last point by 0.99 times the σ that `metrology` derives from the joint covariance. The σ from the pixels alone is 0.017 mm: a plane and camera taken as exact would report error bars hundreds of times too small.

The same holds for a two-camera rig on a robot arm: a rig hand-eye calibration from ten stations at 0.3 px, then each camera's laser plane fitted against it from six laser views, then eight points of each camera's plane measured with 0.1 px noise. Over 300 repeated calibrations the depth of a point scatters by 0.99 times the reported σ (0.45 mm). The board poses the planes are fitted against come through the hand-eye chain, so most of that σ is the hand-eye's: the planes' own fit alone, with the rig calibration taken as exact, would report 0.025 mm, 18 times too small.
