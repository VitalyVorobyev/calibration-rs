# Reading the Covariance

The covariance is part of every result. This chapter shows how to read it and what each part of it contains.

### Rust

```rust
use vision_calibration::core::Uncertainty;

let export = session.export()?;
match &export.uncertainty {
    Uncertainty::Available(cov) => {
        let fx = export.params.intrinsics().fx;
        println!("fx = {fx:.1} ± {:.2} px", cov.std_dev("fx").unwrap());
        println!("corr(fx, cx) = {:.2}", cov.correlation("fx", "cx").unwrap());
        for noise in &cov.noise {
            println!("{:?} noise σ ≈ {:.3}", noise.kind, noise.sigma);
        }
    }
    Uncertainty::Unavailable { reason } => println!("no covariance: {reason:?}"),
}
```

### Python

```python
result = vc.run_planar_intrinsics(dataset, vc.PlanarIntrinsicsConfig())
u = result.uncertainty
if u.available:
    print(f"fx = {result.camera.intrinsics.fx:.1f} ± {u.std_dev('fx'):.2f} px")
    print(f"corr(fx, cx) = {u.correlation('fx', 'cx'):.2f}")
else:
    print("no covariance:", u.reason, u.reason_labels)
```

### JSON

In JSON the field is `{"status": "available", "labels": [...], "covariance":
[[...]], ...}` or `{"status": "unavailable", "reason": {"kind": "...", ...}}`.

## What Is Reported

- **`labels` and `covariance`**: the shared parameters and their covariance
  matrix, one row per label:
  - intrinsics `fx, fy, cx, cy`;
  - distortion by coefficient name (`k1, k2, k3, p1, p2` for Brown-Conrady,
    `lambda` for the division model, and so on);
  - the Scheimpflug tilt `tilt_x, tilt_y`, in radians;
  - the poses a result reports once, under their export names, with six
    labels each (`<pose>.rx` … `<pose>.tz`, see `views` below):
    `cam_se3_rig` per camera, the hand-eye pose (`gripper_se3_rig` or
    `rig_se3_base`, `gripper_se3_camera` or `camera_se3_base` for a single
    camera) and the target pose (`base_se3_target` or `gripper_se3_target`);
  - a laser plane `n · p + d = 0` in its camera's frame: `plane.nx`,
    `plane.ny`, `plane.nz` and `plane.d` (metres). The normal is a unit
    vector, so its three components have a covariance of rank 2.

  In a result with several cameras, each camera's labels carry its index:
  `cam1.fx`, `cam1.k1`, `cam1.cam_se3_rig.tx`. `std_dev(label)` is the square
  root of a diagonal entry, and `correlation(a, b)` the normalized
  off-diagonal one.
- **`views`**: the parameters that belong to one view, each view's block
  separately: the board pose `camera_se3_target` for intrinsics results,
  `rig_se3_target` for rig results, and the robot-pose correction
  `robot_delta` for hand-eye results that refine the robot poses. A pose's
  six labels are `rx, ry, rz` (radians) and `tx, ty, tz` (metres) of the
  perturbation `T · Exp(δ)` on the right. Both are expressed in the pose's
  source frame, which is the target frame for `camera_se3_target`. The
  correlation of a view's parameters with the shared ones is not reported.
- **`fixed`**: the parameters the solve held fixed, such as `k3` with the
  default configuration, or `cam0.cam_se3_rig`, the reference camera that
  defines the rig frame. A fixed pose is named once, without its six
  coordinates. The covariance is *conditional* on them: it does not include
  their uncertainty, so fixing a parameter that the data cannot pin down
  makes the others look better determined than they are. Holding the
  intrinsics in a rig's final solve (`rig.refine_intrinsics: false`), for
  example, leaves their uncertainty out of the extrinsics' σ; by default the
  rig workflows refine them there.
- **`noise`**: for each kind of residual, the Gaussian-equivalent standard
  deviation of one scalar residual. For reprojection that is the corner noise
  per pixel axis; for the laser, the line distance in pixels or the
  point-to-plane distance in metres. It is estimated from the median absolute
  residual, so a few outliers do not inflate it. The rig laserline solves
  each camera's plane on its own, with the rig calibration held fixed, so it
  reports each camera's noise levels apart (`camera` names the camera) and
  planes of different cameras are uncorrelated.
- **`num_residuals`, `dof`**: the scalar residuals and the redundancy they
  leave after the estimated parameters.
