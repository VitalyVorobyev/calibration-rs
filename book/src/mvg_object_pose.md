# Object Pose with Uncertainty

`mvg::pose::estimate_pose` gives the pose of an object whose points are known in its own frame, from the pixels where one or several cameras of a calibrated rig saw them. The result is in the **rig frame** (`rig_se3_object`) with its 6×6 covariance, the calibration's share included. It is the calibrated counterpart of the [PnP solvers](pnp.md), which take a pinhole $K$ and return a bare pose.

## Usage

```rust
use vision_calibration::mvg::pose::{PoseObservation, PoseOptions, estimate_pose};

// rig:     CalibratedRig
// board:   the object's points in its own frame, metres
// pixels:  where camera 0 saw each of them
let noise = PointCovariance::isotropic(0.3)?; // stated per pixel
let observations: Vec<PoseObservation> = board
    .iter()
    .zip(&pixels)
    .map(|(p, px)| PoseObservation::new(0, *p, *px, noise))
    .collect();

let pose = estimate_pose(&rig, &observations, &PoseOptions::default())?;
println!("{}", pose.rig_se3_object);
if let Some(cov) = pose.pose_covariance() {
    // 6x6 over [rx ry rz tx ty tz], rad and m, right-perturbation tangent
}
```

```python
import vision_calibration as vc
from vision_calibration import mvg

noise = vc.PointCovariance.isotropic(0.3)   # stated per pixel
observations = [
    mvg.PoseObservation(camera=0, object=p, pixel=px, noise=noise)
    for p, px in zip(board, pixels)
]
pose = mvg.estimate_pose(result, observations)
pose.rig_se3_object        # 4x4 numpy matrix
pose.pose_covariance()     # 6x6 over [rx ry rz tx ty tz], rad and m; None if unavailable

# RANSAC in σ and a robust loss for outliers:
robust = mvg.estimate_pose(
    result, observations, ransac=mvg.RansacOptions(thresh=3.0), loss=vc.robust_huber(2.0)
)
```

An observation names the rig camera that saw the point (`camera`), so a rig of several cameras that see the same object contributes all of them to one pose. `pose.inliers` indexes the observations the pose used (all of them without RANSAC), and `pose.reprojection_rms_px` is the RMS pixel error over those.

## Steps

1. **Start.** The camera with the most observations (the lowest index on a tie) unprojects its pixels through its own model into bearings; those ahead of the camera (z > 0) become normalized coordinates $(x/z, y/z)$ and EPnP starts the pose on them (a planar object, such as a calibration board, is started from its plane homography). The camera-frame pose is composed into the rig frame with `cam_se3_rig`.
2. **Refine.** The pose is refined on the reprojection error of **every** inlier of **every** camera (a generalized PnP), each pixel weighted by its stated noise, with the rig's calibration as *consider* parameters: they stay at their values and their covariance is carried into the pose's. Observations the start could not use, such as a fisheye point beyond 90° from the axis, enter here.
3. **Covariance.** The pose covariance has two sources, the pixels' noise and the calibration's uncertainty. A rig without a covariance still gives the pose; the uncertainty then says it is missing and `pose_covariance()` is `None`.

## Stated Noise, RANSAC in σ

Every observation carries its pixel covariance (`PointCovariance::new`, `isotropic`, or `rig.reprojection_noise(camera)`), as in [triangulation](mvg_pose.md). The noise is stated, not estimated.

`PoseOptions::ransac` runs RANSAC on the initializing camera: P3P on three points, the fourth choosing among its candidates, and as residual the reprojection distance through the camera model **in σ** of the stated noise (the Mahalanobis distance). A threshold of `3.0` keeps about 99.7 % of the inliers whatever the focal length or camera model, and means the same for every camera of the rig. The inliers are then taken from the start pose over all cameras, the pose is refined on them, the inliers are re-selected by the refined pose and refined again until the set settles (at most three rounds). `PoseOptions::loss` adds a robust loss to the refinement.

Errors come before any solve: a camera index out of range, a non-finite point or pixel, and fewer than four observations in the initializing camera are `InvalidInput`; collinear points are a `Geometry` error; RANSAC without a consensus is `NoConsensus`.

## What Was Measured

On synthetic rigs with pixels drawn from the stated covariance (anisotropic and correlated):

- **Ground truth.** Without noise the pose is exact to $10^{-12}$ rad and m for pinhole, Brown-Conrady and Kannala-Brandt fisheye cameras, planar and spatial objects, seen by either camera or both.
- **RANSAC.** With 30 % gross outliers (40 to 140 px) on a two-camera rig, RANSAC at 3σ accepted none of the 24 outliers and lost none of the 46 inliers; without it the translation is off by more than 5 mm.
- **Pixel noise.** Over 600 draws (a ratio from $n$ draws has a relative error of $1/\sqrt{2n}$, 2.9 %), the scatter over the stated σ of each tangent component, rotation then translation, is 1.02, 0.99, 1.01, 1.02, 0.98, 1.01 with one camera and 1.02, 1.06, 1.03, 1.02, 1.01, 0.99 with two.
- **Two cameras.** For an object seen by both cameras of the rig, the σ of the translation is 2.0 times smaller than with the better camera alone and the σ of the rotation 1.4 times smaller.
- **The calibration's share.** Over 300 stereo calibrations from freshly noisy board views spread over the image, with a held-out board's pose estimated from one camera each time, the scatter over the stated σ is 1.07, 0.99, 0.99, 0.96, 1.04, 0.93 with the calibration considered. The σ from the pixels alone understates the translation by a factor of 3.4 to 26 and the rotation by 1.8 to 2.3: the pose of an object measured with a calibrated rig is limited by the calibration, not by the pixels.
