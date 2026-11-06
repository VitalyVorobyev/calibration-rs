# Multiple-View Geometry

Calibration tells you *what each camera is*: intrinsics, distortion, pose. **Multiple-view geometry (MVG)** is what you do *with* calibrated cameras: recover the relative pose between two views, triangulate 3D structure, refine both with bundle adjustment, rectify a stereo pair and match it densely. This part is the tour of that surface, reachable through the facade as `vision_calibration::mvg`.

It is aimed at someone who already has calibrated cameras (for example from [Stereo Rig with Real Data](stereo_real_data.md)) and wants to reconstruct or rectify. The mathematics behind the building blocks is in the theory parts: [Epipolar Geometry](epipolar.md), [Linear Triangulation](triangulation.md), [Homography Estimation](homography.md) and [RANSAC](ransac.md).

## Conventions

The calibrated functions take a **`CalibratedRig`**: the cameras, where each sits in the rig (`cam_se3_rig`, the reference camera's the identity) and the uncertainty of their parameters. A single camera is a rig of one. Observations are **pixels**; two conventions matter:

- **Pixels in, bearings inside.** Each pixel is unprojected through its own camera's model into a *bearing*, a unit ray, and the closed-form steps (the 5-point essential matrix, cheirality, the linear triangulation) work on bearings. Distortion, fisheye and tilted sensors need no separate undistortion, and a fisheye point more than 90° from the optical axis, which has no normalized coordinates, is an ordinary bearing.
- **Stated noise.** The calibrated two-view functions take `PixelMatch`es: two pixels with their covariances (`PointCovariance::new`, `isotropic`, or `rig.reprojection_noise(camera)`). Their residual is the epipolar constraint over its first-order σ, so RANSAC thresholds are in σ, not radians or pixels.

Poses are named `<frame>_se3_<frame>` and map the right frame into the left: `second_se3_first` takes a point from the first camera's frame into the second's, $p_2 = R\,p_1 + t$, and a world-to-camera pose is `cam_se3_world`.

## The Surface

In dependency order:

```text
mvg::pose_recovery   recover_relative_pose  ← 5-point + cheirality (also triangulates)
mvg::triangulation   triangulate            ← N-view DLT on bearings, refined, with covariance
mvg::laser           triangulate_laser      ← laser pixels on a known plane, with covariance
mvg::types           Captures               ← the frame: rig, world, or the robot's gripper or base
mvg::pose            estimate_pose          ← object pose from known points, RANSAC in σ, with covariance
mvg::bundle_adjust   bundle_adjust          ← capture poses + points together, with covariance
mvg::metrology       distance, fit_plane, ...  ← measurements with σ from the points' joint covariance
mvg::rectification   rectify                ← rectified pair from a calibration, Q, points with σ
mvg::dense           match_block            ← block matching and semi-global matching
mvg::robust          *_robust               ← RANSAC wrappers for outlier-laden data
```

Further modules: `mvg::homography` (homography decomposition and transfer) and `mvg::residuals` (Sampson, epipolar and symmetric transfer errors). The low-level solvers underneath (8-point and 5-point essential matrix, DLT homography, camera-matrix decomposition, linear triangulation) are at the root of `vision_calibration::mvg`.

The chapters of this part follow the pipeline:

1. [Pose Recovery and Triangulation](mvg_pose.md): relative pose from correspondences, with RANSAC for outliers.
2. [Object Pose with Uncertainty](mvg_object_pose.md): the pose of a known object from its pixels in one or several cameras, with RANSAC in σ and the calibration's share in the covariance.
3. [Bundle Adjustment](mvg_bundle_adjust.md): refining poses and structure together, with a stated gauge and the calibration's uncertainty.
4. [Measurements with Uncertainty](mvg_metrology.md): distances, angles and plane fits over triangulated points, each with the σ from the points' joint covariance.
5. [Stereo from a Calibration](mvg_rectification.md): rows that line up for any camera model, the disparity-to-depth matrix and the uncertainty of the points.
6. [Dense Stereo](mvg_dense.md): per-pixel disparity by block matching or semi-global matching.

## The Companion Example

The `mvg_two_view` example builds a synthetic calibrated stereo pair with known ground truth and runs pose recovery, bundle adjustment and rectification:

```bash
cargo run -p vision-calibration --example mvg_two_view
```

```text
scene: 36 correspondences visible in both views
1. pose recovery : rotation Δ = 2.43e-15, |t·t_gt| = 1.000000, 36 points triangulated
3. bundle adjust : RMS 0.0000 px over 2 captures, 36 points; worst point error 1.75e-7 m
4. rectification : baseline = 0.610, max row disagreement = 1.14e-13 px
```

## From a Calibration to MVG Inputs

Every export with cameras gives its calibration as a `CalibratedRig`, with the cameras' parameters, their poses in the rig and the uncertainty of the estimated parameters (labelled as in the export: `fx` for one camera, `cam1.k1` and `cam1.cam_se3_rig.rx` in a rig):

```rust
use vision_calibration::common::CalibratedCameras;

let rig = export.calibrated_rig()?;
// rig.cameras(), rig.cam_se3_rig(), rig.uncertainty(), rig.model(i)
```

`cam_se3_rig[i]` is the pose of the rig frame in camera `i` (`T_Ci_R`). The relative pose of two cameras of a rig is **composed**:

```rust
let cam1_se3_cam0 = rig.cam_se3_rig()[1] * rig.cam_se3_rig()[0].inverse();
```

A single `cam_se3_rig[i]` equals the relative pose only in the special case where camera 0 is the reference frame (its pose is the identity). The reference camera need not be camera 0.

Pose recovery, triangulation, bundle adjustment and rectification all take the pixels as detected, through the rig's camera models.

## In Python

`vision_calibration.mvg` measures with a calibration. Each function takes the calibration **result** (what a `run_*` function or a session returns, or an export loaded as a dict) and builds the rig from it, so there is no rig object to construct. Poses are `4x4` numpy matrices with the same naming as above, and every σ and covariance is a numpy array:

```python
import numpy as np
import vision_calibration as vc
from vision_calibration import mvg

noise = mvg.reprojection_noise(result, camera=0)
tracks = [[mvg.RigObservation(0, 0, px0, noise), mvg.RigObservation(0, 1, px1, noise)]
          for px0, px1 in pixel_pairs]
found = mvg.triangulate(result, np.eye(4), tracks)
length = mvg.distance(mvg.PointSet.from_triangulation(found), 0, 1)
print(f"{length.value:.4f} m ± {length.sigma:.4f} m")
```

The chapters of this part show the Python call beside the Rust one. An input the caller can fix (an index out of range, too few matches, a calibration without a covariance where one is needed) raises `ValueError`; a degenerate geometry or a failed numerical step raises `RuntimeError`.

The closed-form building blocks (the 5-point and 8-point solvers, the DLT homography and camera matrix, homography decomposition and transfer, the epipolar residuals) are Rust only: Python measures through the calibrated functions above, and numpy and OpenCV cover raw linear algebra.

## In the Desktop App

The app's [Epipolar](app_inspect.md#epipolar) and [Depth](app_inspect.md#depth) workspaces run the same code on a calibrated rig export: epipolar lines between two cameras, and rectification plus dense matching with disparity, depth and point-cloud views.
