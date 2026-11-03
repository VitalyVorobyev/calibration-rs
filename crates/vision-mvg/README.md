# vision-mvg

Multiple-view geometry tools for calibrated cameras.

Most users want [`vision-calibration`](https://crates.io/crates/vision-calibration) instead, which re-exports this crate as
`vision_calibration::mvg` next to the calibration that produces its cameras.

This crate builds on `vision-geometry` and adds the multi-view pipelines over
already-calibrated cameras.

## Capabilities

- Relative pose recovery with cheirality disambiguation, and robust essential
  and homography estimation
- N-view triangulation of pixel tracks from a calibration, refined on stated pixel noise, with diagnostics and the points' covariance (the calibration's share included)
- Bundle adjustment of capture poses and points from a calibration, with a stated gauge (capture 0 and a scale), robust losses and the joint covariance, the calibration as consider parameters or under a prior
- Stereo from a calibration: rectification of two rig cameras of any model (raw pixels in), the disparity-to-depth matrix, points with the covariance of their disparity, and measurements that carry the calibration's uncertainty
- Dense stereo: block matching and semi-global matching

## Coordinate Conventions

- Calibrated APIs take a `CalibratedRig` (the cameras and where each sits in
  the rig) and **pixels**. Each pixel is unprojected through its camera's own
  model into a unit bearing, so every camera model works, including a fisheye
  beyond 90 degrees from its axis.
- Poses are named `<frame>_se3_<frame>` and map the right frame into the left:
  `second_se3_first` takes first-camera coordinates to second-camera
  coordinates.

## Usage

```rust
use nalgebra::Rotation3;
use vision_calibration_core::{
    BrownConrady5, CalibratedRig, FxFyCxCy, PointCovariance, Pt3, Uncertainty, Vec3,
    make_pinhole_camera, pinhole_camera_params,
};
use vision_mvg::{PixelMatch, recover_relative_pose};

// One calibrated camera, moved between two captures.
let k = FxFyCxCy { fx: 800.0, fy: 800.0, cx: 640.0, cy: 480.0 };
let camera = pinhole_camera_params(&make_pinhole_camera(k, BrownConrady5::default()));
let rig = CalibratedRig::single(camera, Uncertainty::default())?;

// Matched pixels (at least 5), each with its stated noise: here projected
// from a synthetic scene and detected to about half a pixel.
let noise = PointCovariance::isotropic(0.5)?;
let model = rig.model(0).unwrap();
let r = Rotation3::from_euler_angles(0.0, -0.1, 0.0).into_inner();
let t = Vec3::new(-0.2, 0.0, 0.0);
let matches: Vec<PixelMatch> = (0..8)
    .map(|i| {
        let a = f64::from(i);
        let p = Pt3::new((0.7 * a).sin(), 0.5 * (1.3 * a).cos(), 2.0 + 0.1 * a);
        PixelMatch::new(
            model.project_point_c(&p.coords).unwrap(),
            noise,
            model.project_point_c(&(r * p.coords + t)).unwrap(),
            noise,
        )
    })
    .collect();

let pose = recover_relative_pose(&rig, [0, 0], &matches)?;
println!("second_se3_first = {:?}", pose.second_se3_first);
println!("scene: {:?}", pose.scene);
# Ok::<(), vision_mvg::MvgError>(())
```

The pixel noise is stated, never assumed: every `PixelMatch` carries the
covariance of each of its pixels, and the calibrated functions weigh the
epipolar constraint by it. Build a covariance per observation with
`PointCovariance::new(xx, xy, yy)`, from one σ with
`PointCovariance::isotropic(sigma_px)`, or take the calibration's own level
with `rig.reprojection_noise(camera)?`. RANSAC thresholds and `inlier_rms` of
the essential-matrix functions are in σ of that noise (3.0 keeps about 99.7 %
of the inliers), whatever the focal length or camera model.

## See Also

- `vision-geometry` for low-level deterministic solvers
- `vision-calibration-linear` for calibration-specific initialization
- `vision-calibration` for the high-level facade API
- [Book: Multiple-View Geometry](https://vitalyvorobyev.github.io/calibration-rs/mvg.html)
