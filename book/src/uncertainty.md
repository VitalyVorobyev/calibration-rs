# Parameter Uncertainty

A calibration result says not only what the parameters are but how well the
data determines them. The intrinsics, hand-eye and rig results carry an
`uncertainty`: the covariance of the estimated parameters at the solution,
the noise level the residuals imply, and, when no covariance can be given,
the reason.

Reprojection error alone does not answer this. A camera calibrated from a
handful of similar views can reproject its own corners to 0.1 px and still have
a focal length that is off by a percent, with the principal point
traded off against it. The covariance shows that directly: a large `σ` of `fx`,
and a strong `fx`–`cx` correlation.

## Reading it

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

```python
result = vc.run_planar_intrinsics(dataset, vc.PlanarCalibrationConfig())
u = result.uncertainty
if u.available:
    print(f"fx = {result.camera.intrinsics.fx:.1} ± {u.std_dev('fx'):.2} px")
    print(f"corr(fx, cx) = {u.correlation('fx', 'cx'):.2}")
else:
    print("no covariance:", u.reason, u.reason_labels)
```

In JSON the field is `{"status": "available", "labels": [...], "covariance":
[[...]], ...}` or `{"status": "unavailable", "reason": {"kind": "...", ...}}`.

## What is reported

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
    camera) and the target pose (`base_se3_target` or `gripper_se3_target`).

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
  makes the others look better determined than they are. The rig and rig
  hand-eye workflows, for example, calibrate each camera's intrinsics first
  and by default hold them fixed in the final solve, so their extrinsics'
  σ leaves out the intrinsics' uncertainty.
- **`noise`**: for each kind of residual, the Gaussian-equivalent standard
  deviation of one scalar residual. For reprojection that is the corner noise
  per pixel axis. It is estimated from the median absolute residual, so a few
  outliers do not inflate it.
- **`num_residuals`, `dof`**: the scalar residuals and the redundancy they
  leave after the estimated parameters.

## What a σ means

The covariance is first order at the solution: it describes how the estimate
would scatter if the calibration were repeated with new noise of the same
kind, for the same views. The noise level is taken from the residuals
themselves, so nothing has to be known in advance. Two consequences follow:

- `σ` measures the **noise-driven** error. A systematic error, such as a wrong
  board cell size, a board that is not flat, or the wrong distortion model,
  leaves `σ` small and the estimate biased. Watch the per-feature residuals for
  structure.
- `σ` shrinks with more data roughly as `1/√views` and grows quickly when the
  views are similar. The most effective way to reduce `σ` of `fx` and `cx` is
  board tilt: views at varied angles separate focal length from distance.

With a robust loss (`Huber`, `Cauchy`, `Arctan`), the covariance is that of
the robust estimate actually computed, including the bounded pull of the
residuals the loss down-weights. It does not ignore the outliers, and it does
not treat them as Gaussian noise either. Without a loss it is the classical
`σ̂² (JᵀJ)⁻¹`.

These σ were checked against Monte-Carlo scatter on synthetic scenes (single
cameras, a stereo rig, an eye-in-hand camera), a few hundred noise
realizations each. The predicted σ matched the empirical scatter within 10 %
on clean data, and within 20 % with 5 % of the corners displaced by
10–30 px under Huber or Cauchy.

## When it is unavailable

| `reason` | Meaning | What to do |
|---|---|---|
| `not_converged` | The solve stopped on its iteration budget | Raise `solver.max_iters` |
| `rank_deficient` | The data does not determine the parameters listed in `unobservable` (alone or in combination) | Add views that differ in the way the listed parameters need, e.g. tilted boards for `fx`, `cx`; or fix one of them |
| `at_bound` | A parameter ended on a bound (`labels`) | Widen or remove the bound, or fix the parameter |
| `no_redundancy` | No more observations than parameters | Add views |
| `not_computed` | The result was produced or stored without uncertainty | — |

`rank_deficient` is a statement about the data, not a failure of the
calibration: the solve still returns its best parameters. A Scheimpflug
camera whose views do not separate the sensor tilt from a principal-point
shift, for example, reports `tilt_y`, `cx` and the board translations as
unobservable together.

## Scope

The intrinsics, hand-eye and rig workflows report uncertainty, from their
final solve; the laser workflows report `not_computed`. Robot poses are taken
as exact unless the hand-eye solve refines them. Then the priors'
standard deviations (`robot_rot_sigma`, `robot_trans_sigma`) are taken as
the robot's true pose accuracy, and the covariance is that of the estimate
computed with those priors against the corner residuals. The uncertainty of a projected point (how well a
pixel's ray is determined across the image) follows from this covariance but
is not reported.
