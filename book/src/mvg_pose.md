# Pose Recovery and Triangulation

Given matched points in two calibrated views, `recover_relative_pose` finds the rotation and translation direction between the cameras and triangulates the points.

## Recovering the Relative Pose

Input is a list of `Correspondence2D` in **normalized** coordinates (pixels after applying $K^{-1}$ and, for a distorted lens, removing the distortion). The function runs the 5-point algorithm on several evenly spaced subsets of the correspondences, scores each candidate essential matrix by Sampson distance and cheirality over **all** correspondences, keeps the pose for which the most points lie in front of both cameras, and triangulates:

```rust
use vision_calibration::mvg::pose_recovery::recover_relative_pose;
use vision_calibration::mvg::types::Correspondence2D;

let corrs: Vec<Correspondence2D> = /* (normalized point in cam0, normalized point in cam1) */;
let rel = recover_relative_pose(&corrs)?;
// rel.r      : rotation cam0 → cam1
// rel.t      : unit translation direction
// rel.points : triangulated 3D points with reprojection and parallax diagnostics
```

At least 5 correspondences are required. The translation is recovered **up to scale**: two bare views contain no metric anchor, so `rel.t` has unit length. Fix the scale with a known baseline (for a calibrated rig, the extrinsics already carry it) or a known 3D distance.

Each entry of `rel.points` is a `TriangulatedPoint`:

| Field | Meaning |
|---|---|
| `point` | the 3D position, in the first camera's frame at unit baseline |
| `reprojection_error` | RMS reprojection error across the views used |
| `parallax_deg` | the angle between the observing rays; small angles mean poor triangulation geometry |
| `in_front` | whether the point has positive depth in all cameras |

### Degenerate Geometry

The 5-point estimator needs a genuinely three-dimensional scene. A near-planar cloud, a pure rotation (no baseline) or nearly parallel rays make the essential matrix poorly determined. `mvg::degeneracy` detects these cases from the correspondences and a candidate pose (`analyze_scene` returns the median parallax, pure-rotation and planar flags and a baseline ratio); for a planar scene the homography model (`mvg::homography::decompose_homography`) is the right one instead.

## Outliers: the Robust Variants

Real matches contain outliers. The `mvg::robust` wrappers put a RANSAC loop around the minimal solvers:

```rust
use vision_calibration_core::RansacOptions;
use vision_calibration::mvg::robust::{estimate_essential, recover_relative_pose_robust};

let opts = RansacOptions {
    min_inliers: 12,
    ..RansacOptions::new(1.0e-3)   // Sampson distance, in normalized units
};
let robust = recover_relative_pose_robust(&corrs, &opts)?;
// robust.r, robust.t, robust.essential, robust.inliers (indices), robust.inlier_rms
```

**The threshold is in the units of the residual**, and the residual lives on the normalized image plane: `1.0e-3` is about 0.8 px for a focal length of 800 px. A pixel-scale value such as 2.0 is far too loose for residuals on this scale and would accept every match. There is no default threshold: `RansacOptions::new(thresh)` requires one, and a threshold that is not finite and positive is an error. The run is deterministic for a given `seed` field.

`RansacOptions` is defined in `vision-calibration-core`, which the facade does not re-export; add that crate to your dependencies to name it.

`estimate_essential` returns the essential matrix with its inlier indices and the RMS Sampson distance over the inliers; `estimate_homography` does the same for a homography (4-point DLT with symmetric transfer error). `recover_relative_pose_robust` selects the inliers first and disambiguates the four decompositions of the essential matrix with cheirality on the inliers only, so outliers cannot corrupt the positive-depth vote.

## Triangulating a Point from N Views

`mvg::triangulation::triangulate_nview` takes one 3x4 camera matrix per view and the matching normalized point in each, and returns a `TriangulatedPoint` with the same diagnostics (the reprojection error and the all-views cheirality included):

```rust
use vision_calibration::mvg::Mat34;
use vision_calibration::mvg::triangulation::triangulate_nview;
use vision_calibration::core::Pt2;

let tri = triangulate_nview(&cameras, &points)?; // cameras: &[Mat34], points: &[Pt2]
println!("{:?} parallax {:.2}°", tri.point, tri.parallax_deg);
```

The camera matrices and the points are in the same coordinates (normalized, or pixels with matrices that include $K$). Two-view variants (`triangulate_two_view`, `triangulate_point_two_view`) serve the common stereo case. The linear algorithm and its failure modes are described in [Linear Triangulation](triangulation.md).
