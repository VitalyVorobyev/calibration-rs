# Pose Recovery and Triangulation

Given matched pixels in two calibrated views, `recover_relative_pose` finds the rotation and translation direction between the cameras and triangulates the points.

## Recovering the Relative Pose

Input is the rig, the camera of each view and a list of `Correspondence2D`, pairs of **pixels**: `pt1` in rig camera `cameras[0]` at the first capture, `pt2` in rig camera `cameras[1]` at the second (`[0, 0]` for one camera that moved). Each pixel is unprojected through its camera's model into a bearing, so a distorted lens or a fisheye needs no undistortion first. The function runs the 5-point algorithm on several evenly spaced subsets of the correspondences, scores each candidate essential matrix by its angular epipolar error and by cheirality over **all** correspondences, keeps the pose for which the most points lie ahead along both bearings, and triangulates:

```rust
use vision_calibration::mvg::pose_recovery::recover_relative_pose;
use vision_calibration::mvg::types::Correspondence2D;

let matches: Vec<Correspondence2D> = /* (pixel in the first view, pixel in the second) */;
let rel = recover_relative_pose(&rig, [0, 0], &matches)?;
// rel.second_se3_first : pose of the first camera's frame in the second's, unit translation
// rel.essential        : b2ᵀ E b1 = 0 on the bearings
// rel.points           : triangulated points with reprojection and parallax diagnostics
// rel.scene            : parallax, pure-rotation and planarity diagnostics
```

At least 5 correspondences are required. The translation is recovered **up to scale**: two bare views contain no metric anchor, so `rel.second_se3_first` has a unit-norm translation. Fix the scale with a known baseline (for a calibrated rig, the extrinsics already carry it) or a known 3D distance.

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
    ..RansacOptions::new(1.0e-3)   // angular epipolar error, in radians
};
let robust = recover_relative_pose_robust(&rig, [0, 0], &matches, &opts)?;
// robust.second_se3_first, robust.essential, robust.inliers (indices), robust.inlier_rms
```

**The threshold is in radians**: the residual is the first-order angle by which the two bearings would have to move to satisfy the epipolar constraint. A pixel threshold $t$ is about $t / f$ radians for a focal length $f$ in pixels near the image centre: `1.0e-3` is about 0.8 px for a focal length of 800 px. A pixel-scale value such as 2.0 is far too loose for residuals on this scale and would accept every match. There is no default threshold: `RansacOptions::new(thresh)` requires one, and a threshold that is not finite and positive is an error. The run is deterministic for a given `seed` field.

`RansacOptions` is defined in `vision-calibration-core`, which the facade does not re-export; add that crate to your dependencies to name it.

`estimate_essential` returns the essential matrix with its inlier indices and the RMS angular error over the inliers (in radians); `estimate_homography` does the same for a homography (4-point DLT with symmetric transfer error, on plain image points). `recover_relative_pose_robust` selects the inliers first and disambiguates the four decompositions of the essential matrix with cheirality on the inliers only, so outliers cannot corrupt the positive-depth vote.

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
