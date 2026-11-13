# Reading the Covariance

The covariance is part of every result. This chapter shows how to read it and what each part of it contains.

### Rust

```rust,no_run
use vision_calibration::core::{NoiseLevel, Uncertainty};
# use vision_calibration::planar_intrinsics::PlanarIntrinsicsProblem;
# use vision_calibration::session::CalibrationSession;
# fn main() -> anyhow::Result<()> {
# let mut session = CalibrationSession::<PlanarIntrinsicsProblem>::new();

let export = session.export()?;
match &export.uncertainty {
    Uncertainty::Available(cov) => {
        let fx = export.camera.intrinsics.fx;
        println!("fx = {fx:.1} ± {:.2} px", cov.std_dev("fx").unwrap());
        println!("corr(fx, cx) = {:.2}", cov.correlation("fx", "cx").unwrap());
        for noise in &cov.noise {
            match noise.level {
                NoiseLevel::Sigma(s) => println!("{:?} noise σ ≈ {s:.3}", noise.kind),
                NoiseLevel::Sigma0(s) => println!("{:?} noise σ̂₀ ≈ {s:.3}", noise.kind),
            }
        }
    }
    Uncertainty::Unavailable { reason } => println!("no covariance: {reason:?}"),
}
# Ok(())
# }
```

### Python

```python
result = vc.planar_intrinsics.run_calibration(dataset, vc.PlanarIntrinsicsConfig())
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
    `rig_se3_base`, `gripper_se3_cam` or `cam_se3_base` for a single
    camera) and the target pose (`base_se3_target` or `gripper_se3_target`);
  - a laser plane `n · p + d = 0` in its camera's frame: `plane.nx`,
    `plane.ny`, `plane.nz` and `plane.d` (metres). The normal is a unit
    vector, so its three components have a covariance of rank 2.

  In a result with several cameras, each camera's labels carry its index:
  `cam1.fx`, `cam1.k1`, `cam1.cam_se3_rig.tx`. `std_dev(label)` is the square
  root of a diagonal entry, and `correlation(a, b)` the normalized
  off-diagonal one.
- **`views`**: the parameters that belong to one view, each view's block
  separately: the board pose `cam_se3_target` for intrinsics results,
  `rig_se3_target` for rig results, and the robot-pose correction
  `robot_delta` for hand-eye results that refine the robot poses. A pose's
  six labels are `rx, ry, rz` (radians) and `tx, ty, tz` (metres) of the
  perturbation `T · Exp(δ)` on the right. Both are expressed in the pose's
  source frame, which is the target frame for `cam_se3_target`. The
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
  residual, so a few outliers do not inflate it. In JSON the level is
  `sigma`, in the residual's unit, or, when the observations stated a
  covariance per point, `sigma0`: the dimensionless `σ̂₀` of the whitened
  residuals, about 1 when the stated covariances hold
  ([Per-Point Pixel Covariance](uncertainty_point_covariance.md)). Rust
  types the two as `NoiseLevel::Sigma` and `NoiseLevel::Sigma0`; Python's
  `ResidualNoise.whitened` says which `sigma` holds. The rig laserline solves
  each camera's plane on its own, with the rig calibration held fixed, so it
  reports each camera's noise levels apart (`camera` names the camera). When
  the upstream calibration carries its covariance, the planes' covariance
  carries its share and the planes of different cameras are correlated
  through it; without one, it is conditional on the upstream and the planes
  are uncorrelated.
- **`num_residuals`, `dof`**: the scalar residuals and the redundancy they
  leave after the estimated parameters.
- **`view_scatter`**: how many times the shared parameters vary more
  across views than the residual noise predicts, when significantly more;
  the covariance is already scaled by it
  ([Parameter Uncertainty](uncertainty.md)).
- **`second_order`**: how far the error of the shared parameters is
  predicted to depart from the covariance once its second-order term is
  counted. The covariance is the error's first-order term, linear in the
  noise. Weak views pin some combinations of parameters down only through
  larger errors of others that cancel in them (centred, mildly tilted boards
  leave a camera's rotation in a rig and its principal point to trade off
  this way); there the second-order term, quadratic in the noise, can exceed
  the first, while every parameter's own σ still holds.
  `max_scatter_ratio` is the predicted scatter of the worst combination over
  the σ the covariance states for it, `nees_ratio` the predicted mean of
  `eᵀ Σ⁻¹ e` per parameter (both 1 when the first order describes the
  error), and `combination` names the combination as `Σ wᵢ pᵢ / σᵢ`, its
  largest weights first. The prediction comes from the residuals' curvature
  along the covariance's own errors, at the result's noise level, so it
  grows with the noise variance. [Judging a Result](quality.md) warns when
  the ratio is above 1.2. It costs `s (s + 1)` residual evaluations for `s`
  shared parameters, so it is left out (`null`) above 64 of them, and for
  noise-free data. A rig laserline device result whose covariance joins
  its planes to its upstream calibration's leaves it out too: the cameras
  enter that solve as given, so their residuals' curvature is not there to
  measure; the upstream export's own `second_order` describes them.
