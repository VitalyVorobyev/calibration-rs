# Multiple-View Geometry

Calibration tells you *what each camera is*: intrinsics, distortion, pose. **Multiple-view geometry (MVG)** is what you do *with* calibrated cameras: recover the relative pose between two views, triangulate 3D structure, refine both with bundle adjustment, rectify a stereo pair and match it densely. This part is the tour of that surface, reachable through the facade as `vision_calibration::mvg`.

It is aimed at someone who already has calibrated cameras (for example from [Stereo Rig with Real Data](stereo_real_data.md)) and wants to reconstruct or rectify. The mathematics behind the building blocks is in the theory parts: [Epipolar Geometry](epipolar.md), [Linear Triangulation](triangulation.md), [Homography Estimation](homography.md) and [RANSAC](ransac.md).

## Conventions

Everything in `mvg` operates on **calibrated** observations, and two coordinate conventions matter:

- **Normalized image coordinates** are pixels after applying $K^{-1}$. Relative-pose recovery and triangulation work here, so the geometry is independent of the specific intrinsics.
- **Pixel coordinates** are what bundle adjustment and rectification consume (they fold $K$ back in).

Poses follow the project convention: `T_C1_C0` (`cam1_se3_cam0`) maps a point from camera 0's frame into camera 1's frame, $p_{c1} = R\,p_{c0} + t$. Bundle adjustment poses are `T_C_W`: world to camera.

## The Surface

In dependency order:

```text
mvg::pose_recovery   recover_relative_pose  ← 5-point + cheirality (also triangulates)
mvg::triangulation   triangulate_nview      ← N-view DLT with diagnostics
mvg::bundle_adjust   bundle_adjust          ← joint poses + structure (feature `refine`)
mvg::rectification   rectify_stereo_pair    ← rectifying homographies (Scheimpflug-aware)
mvg::dense           match_block            ← block matching and semi-global matching
mvg::robust          *_robust               ← RANSAC wrappers for outlier-laden data
```

Further modules: `mvg::degeneracy` (pure rotation, planar scene and poor baseline detection), `mvg::homography` (homography decomposition and transfer) and `mvg::residuals` (Sampson, epipolar and symmetric transfer errors). The low-level solvers underneath (8-point and 5-point essential matrix, DLT homography, camera-matrix decomposition, linear triangulation) are in `vision_calibration::geometry`.

The chapters of this part follow the pipeline:

1. [Pose Recovery and Triangulation](mvg_pose.md): relative pose from correspondences, with RANSAC for outliers.
2. [Bundle Adjustment](mvg_bundle_adjust.md): refining poses and structure together.
3. [Stereo Rectification](mvg_rectification.md): rows that line up, for pinhole and Scheimpflug cameras.
4. [Dense Stereo](mvg_dense.md): per-pixel disparity by block matching or semi-global matching.

## The Companion Example

The `mvg_two_view` example builds a synthetic calibrated stereo pair with known ground truth and runs pose recovery, bundle adjustment and rectification:

```bash
cargo run -p vision-calibration --example mvg_two_view --features refine
```

```text
scene: 36 correspondences visible in both views
1. pose recovery : rotation Δ = 2.43e-15, |t·t_gt| = 1.000000, 36 points triangulated
3. bundle adjust : RMS 18.1423 → 0.0000 px over 2 cameras, 36 points
4. rectification : baseline = 0.610, max row disagreement = 1.14e-13 px
```

The `refine` cargo feature enables bundle adjustment (it pulls in the `tiny-solver` backend); everything else is available by default.

## From a Calibration to MVG Inputs

A rig calibration provides what rectification and triangulation need. From a `RigExtrinsicsExport`, `cameras[i]` (a `CameraParams`: intrinsics and, for a tilted camera, its sensor) builds each camera's intrinsic matrix, and the relative pose is **composed** from the two cameras' extrinsics:

```rust
// cam_se3_rig[i] is T_Ci_R (camera i ← rig), so:
let cam1_se3_cam0 = export.cam_se3_rig[right] * export.cam_se3_rig[left].inverse();
```

A single `cam_se3_rig[i]` equals the relative pose only in the special case where the left camera is the reference frame (`cam_se3_rig[left]` is the identity). Compose explicitly for any other pair or `reference_camera_idx`.

Pixels from the real cameras must be **undistorted** first for the stages that expect it (rectification); the MVG tools work on the ideal pinhole model and leave lens distortion to the calibrated camera.

## In the Desktop App

The app's [Epipolar](app_inspect.md#epipolar) and [Depth](app_inspect.md#depth) workspaces run the same code on a calibrated rig export: epipolar lines between two cameras, and rectification plus dense matching with disparity, depth and point-cloud views.
