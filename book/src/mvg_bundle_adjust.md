# Bundle Adjustment

`bundle_adjust` jointly refines all camera **poses** and the 3D **structure** to minimize reprojection error, holding the per-camera intrinsics frozen. It is the final polish of an already initialized reconstruction (for example the output of [pose recovery](mvg_pose.md)), not a structure-from-motion pipeline: there is no feature matching, pose-graph optimization or loop closure.

It lives behind the `refine` feature, which pulls in the `tiny-solver` Levenberg-Marquardt backend:

```toml
[dependencies]
vision-calibration = { version = "0.10", features = ["refine"] }
```

## Usage

```rust
use vision_calibration::mvg::bundle_adjust::{
    BundleAdjustmentOptions, BundleObservation, bundle_adjust,
};

// intrinsics:  &[Mat3], one K per camera (frozen)
// observations: pixel measurements, each naming the camera and the point
// init_poses:  &[Iso3], T_C_W per camera
// init_points: &[Pt3], one world point each
let observations = vec![BundleObservation::new(0, 0, pixel_cam0_pt0) /* , ... */];

let res = bundle_adjust(
    &intrinsics,
    &observations,
    &init_poses,
    &init_points,
    &BundleAdjustmentOptions::default(),
)?;
println!("RMS {:.3} → {:.3} px", res.initial_rms, res.final_rms);
```

`BundleObservation::new(cam, point, pixel)` records that camera `cam` saw world point `point` at the pixel `pixel`; the indices refer to the `intrinsics`/`init_poses` slices and the `init_points` slice. Observations are **pixel** coordinates: the frozen $K$ of each camera projects the camera-frame ray to pixels. Poses are `T_C_W`, mapping a world point into the camera frame.

The result carries the refined `poses` and `points` (cameras and points without observations come back unchanged), the RMS reprojection error before and after (`initial_rms`, `final_rms`) and `per_camera_rms`. In the companion example an initial reprojection error of about 18 px collapses to numerical zero on noise-free data.

## Errors

`bundle_adjust` returns an `MvgError` when the number of intrinsics and poses disagree (`CountMismatch`), when there are no observations (`InsufficientData`), when an observation names a camera or point out of range (`InvalidInput`), or when the optimizer returns no solution (`NotConverged`). Points should be observed in at least two views for their depth to be well constrained; under-observed points are admitted (the solver's damping leaves them near their initial value) so partially observed data does not fail outright.

## The Gauge

A reconstruction from images alone has a seven-degree-of-freedom similarity gauge: a world rotation, translation and scale cannot be seen by reprojection error.

- `BundleAdjustmentOptions::fix_first_camera` (on by default) holds the lowest-index camera that has observations fixed, which removes the six rigid degrees of freedom. Disable it only when an external prior already pins the world frame; otherwise the reconstruction drifts along those directions, which the solver's damping merely regularizes.
- The remaining **global scale** cannot be observed while the structure is free: scaling every point depth and every free-camera translation leaves all projections unchanged. Anchor the scale externally (a known baseline, a known 3D distance) for metric output. In practice the damping keeps it near the initial reconstruction.

`max_iterations` (default 100) bounds the Levenberg-Marquardt iterations.

## Relation to Calibration

Calibration workflows such as the rig extrinsics refine intrinsics, distortion and poses together, against a known target. `bundle_adjust` is the complement for scenes without a known target: the points are unknowns, the intrinsics are given.
