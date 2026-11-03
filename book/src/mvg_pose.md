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

**The threshold is in σ**: the residual is the weighted Sampson error above, so `3.0` keeps about 99.7 % of the inliers when the noise is Gaussian and as stated, independent of the focal length and of the camera model. The stated noise is the scale: understating the pixel noise by a factor of two doubles every residual and rejects good matches. There is no default threshold: `RansacOptions::new(thresh)` requires one, and a threshold that is not finite and positive is an error. The run is deterministic for a given `seed` field.

`RansacOptions` is defined in `vision-calibration-core`, which the facade does not re-export; add that crate to your dependencies to name it.

`estimate_essential` returns the essential matrix with its inlier indices and the RMS of the weighted Sampson error over the inliers (in σ, about 1 when the stated noise describes the pixels); `estimate_homography` does the same for a homography (4-point DLT with symmetric transfer error, on plain image points). `recover_relative_pose_robust` selects the inliers first and disambiguates the four decompositions of the essential matrix with cheirality on the inliers only, so outliers cannot corrupt the positive-depth vote.

## Triangulating Points from N Views

`mvg::triangulation::triangulate` takes the rig, the pose of the rig at each capture (`rig_se3_world`) and, for each point, a track of observations: the pixel, the rig camera that saw it and the capture. It returns a `TriangulatedPoint` per track in the world frame, with the same diagnostics:

```rust
use vision_calibration::mvg::triangulation::{RigObservation, triangulate};
use vision_calibration::core::{Iso3, Pt2};

// One stereo capture: the rig frame is the world frame.
let track = vec![
    RigObservation::new(0, 0, pixel_in_camera_0),
    RigObservation::new(0, 1, pixel_in_camera_1),
];
let found = triangulate(&rig, &[Iso3::identity()], &[track])?;
println!("{:?} parallax {:.2}°", found[0].point, found[0].parallax_deg);
```

Camera $i$ at capture $k$ sees the world through `rig.cam_se3_rig()[i] * rig_se3_world[k]`, so a moving rig, several cameras and a single moving camera are all the same call. The call is all-or-nothing: a track with fewer than two observations, an index out of range or a zero-parallax system is an error that names the track. The estimate is the linear algorithm of [Linear Triangulation](triangulation.md) on bearings, with no reprojection polish; its failure modes are described there.
