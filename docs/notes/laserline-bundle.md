# Laserline bundle — proof pack

Family: single-camera + laser-plane device from planar-target views plus
laser-stripe pixels (`LaserlineDeviceProblem`; init in
`vision-calibration-linear::laserline` + the shared Zhang path, refinement in
`vision-calibration-optim::problems::laserline_bundle`). The rig laser axes
(`RigLaserlineDeviceProblem`, `RigHandeyeLaserlineProblem`) reuse the same
laser cost with a composed pose (§Rig axes).

## Model

The camera is the planar/Scheimpflug families' composable camera (ADR 0005),
`pixel = K( sensor( distort( project(X_c) ) ) )`, `X_c = T_C_T · X_t` — see
[planar-intrinsics.md](planar-intrinsics.md) and
[scheimpflug-intrinsics.md](scheimpflug-intrinsics.md). Its model is
`config.camera.model` (ADRs 0005, 0013): any distortion kind, an identity or
a Scheimpflug sensor; an untilted camera has no sensor block. The fixed
parameters are `config.camera.fix`, named per slot; the default refines a
Scheimpflug tilt, and a known mount tilt is the manual-init `sensor` seed
held by naming it in `camera.fix.sensor`.

The **laser plane** is stored in the camera frame as
`LaserPlane { normal: Unit<Vector3>, distance }`
(`vision-calibration-optim::params::laser_plane`):

```
n̂ · p + d = 0      (p in camera coordinates)
```

In optimization it is two param blocks: `plane_normal` on the **S²**
manifold and `plane_distance` a Euclidean scalar, so the unit-norm
constraint is carried by the manifold rather than penalized.

Laser observations enter as per-view **stripe pixels**: `LaserlineMeta`
carries `laser_pixels` (optionally per-pixel `laser_weights`) alongside the
target correspondences. On real data they come from a subpixel line
extractor injected through the `LaserPixelExtractor` trait (ADR 0016); the
published crates ship no extractor, only the geometry that consumes it.

## Initialization

1. **Intrinsics / distortion / poses** — the per-camera bootstrap shared
   with the rigs (`planar_family::bootstrap_camera`): Zhang's method embedded
   into the model's distortion, or for a Scheimpflug sensor the linear tilt
   sweep (a `sensor` seed replaces it). Without a tilt seed, a Scheimpflug
   camera is then refined alone through the staged tilt multi-start
   (`planar_family::optimize_camera`) before the plane fit.
2. **Laser plane** — `LaserlinePlaneSolver::from_views`: each laser pixel is
   back-projected through the camera model (undistorted) to a camera-frame
   ray, intersected with the target plane (`z = 0` in the known per-view
   pose), and a plane is fit to the pooled points by covariance
   eigendecomposition (smallest eigenvector = `n̂`, `d = −n̂ · centroid`).
   The fit rejects rank-1 (collinear) clouds, requires ≥ 2 views (one
   stripe alone is collinear), and records `initial_plane_rmse`.

`step_init_with_seed` (ADR 0011) lets any of intrinsics / distortion / sensor
tilt / poses / plane be supplied instead of estimated.

## Cost

Joint bundle adjustment (`optimize_laserline`, IR per ADR 0008) over
the camera's parameters not held by `camera.fix` (the model's defaults:
`k3` for Brown-Conrady; everything but `k1`, `k2` / `lambda` under a
Scheimpflug sensor, whose tilt is free), all per-view poses (SE(3); `fix_poses` empty by default), and the laser plane
(S² normal + scalar distance). Two residual families sum into one problem:

```
E = Σ_i Σ_j ρ_c‖ π(K,d,sensor,T_i·X_j) − u_ij ‖²   (target corners, 2D)
  + Σ_i Σ_k ρ_l · r(K,d,sensor,T_i, n̂,d ; q_ik)²    (laser stripe, 1D)
```

`ρ_c` / `ρ_l` are `calib_loss` / `laser_loss` (default Huber 1 px each); a
loss scale is in its residual's unit. Per-family weights `calib_weight` /
`laser_weight` default to 1. The laser residual `r` has two forms
(`LaserlineResidualType`, `factors::laserline`):

- **`LineDistNormalized`** (default): intersect the laser and target planes
  into a 3D line, project it to the `z = 1` normalized plane (where it stays
  straight), undistort the pixel to that plane, take the perpendicular
  distance and scale by `√(fx·fy)`. Units: **pixels**, comparable to the
  reprojection floor.
- **`PointToPlane`**: undistort → ray → intersect target plane → 3D point →
  signed distance `n̂·p + d` to the laser plane. Units: **metres**.

The plane appears **only** in the laser residuals, and the poses are pinned
overwhelmingly by the corners, so the two halves are nearly block-separable:
the plane is fit from the stripes conditioned on the poses.

### Rig axes

`RigLaserlineDeviceProblem` and `RigHandeyeLaserlineProblem` reuse the same
laser cores through `LaserChain::RigHandeye` / `RigHandeyeRobotDelta`
(`optim::ir::types`), which compose `cam_se3_target` from
`(cam_se3_rig, hand-eye, target_ref, robot pose[, robot Δ])` instead of a
free pose. Unit tests in `factors::laserline`
(`rig_handeye_residuals_match_single_camera_via_composition`,
`rig_handeye_robot_delta_laser_residual_matches_corrected_robot_pose`) pin
the rig residual to the single-pose residual at the composed pose. Entry
points: `optim::problems::{laserline_rig_bundle,
rig_handeye_laserline_bundle}`, pipeline `rig_laserline_device` /
`rig_handeye_laserline`. The laser plane adds no gauge beyond the S² sign
below; the rig and hand-eye gauges are in their own packs.

## Identifiability and degeneracies

Each stripe is the plane ∩ board line — **collinear** points — so one board
pose constrains only one in-plane direction. Recovery needs the per-view
stripes to **span the plane**:

- **Stripe non-collinearity (the core requirement).** ≥ 2 board poses whose
  stripes differ in direction and/or offset. Board roll about the optical
  axis rotates the stripe *within* the board, pitch/yaw reposition it; roll
  is the cheap, robust source because it does not drive the stripe off the
  board. The linear solver rejects a rank-1 pooled cloud outright.
- **Distance is the soft DOF.** `n̂` is pinned by stripe geometry; `d` is
  constrained by how much the metric stripe positions move across
  **depth-diverse** poses. Shallow tilt or one board depth leaves `d` weakly
  observable (high variance, not a wrong minimum).
- **Degenerate geometries.**
  - *Stripes all parallel* (rotation only about an axis parallel to the
    intersection line, or pure depth translation): the in-plane direction
    orthogonal to them is unobservable.
  - *Laser plane through the camera centre* (`d → 0`): the plane images to
    one fixed line for every pose, so stripes carry no depth information;
    calibration can still succeed off the known poses, but the deployed
    triangulation is ill-posed. Small `|d|` is the same pathology, softened.
  - *Laser plane parallel to a board pose*: the intersection line is at
    infinity — no stripe in that view (guarded: large residual).
  - *Ray parallel to / behind the target plane*: guarded with a large
    residual so a stray pixel cannot pull the solve.
- **View count.** The pipeline requires ≥ 3 views, each with ≥ 4 corners
  and ≥ 1 laser pixel; the plane needs ≥ 2 non-collinear stripes.

The intrinsics inherit the planar family's degeneracies.

**Metric scale in the joint rig + hand-eye + laser BA.** On the six-camera
`scheimpflug_rig` dataset, the camera-and-rig-only stage leaves a shallow
direction trading overall rig scale against tilt, principal point and pose:
its extrinsics come out ~10 % small (hexagon edges 88.9 ± 0.4 mm) with the
shape intact. The joint BA, with the converged tilts frozen and the laser
point-to-plane term added, pins that direction: 98.21 ± 0.41 mm against the
reference calibration's 98.13 ± 1.10 mm (0.08 %). Scale comparisons must use
the joint-BA extrinsics. The board's 5.2 mm cell is confirmed: a cell-size
error would give a fixed ratio, not a stage-dependent one.

## Gauge

- **No global gauge.** The plane lives in the camera frame, the board fixes
  each view's target frame, and every pose is measured by its own corners,
  so no pose is held fixed (`fix_poses` empty).
- **S² sign gauge.** `(n̂, d)` and `(−n̂, −d)` name the same plane. The signed
  distance only flips sign, so its square is invariant and the solve may land
  on either sheet. Comparisons sign-align to a reference first (the matrix
  test aligns the normal to GT).

## Noise sensitivity

For iid uniform pixel noise `[−a, a]` per axis and `N ≫ #params`, the target
reprojection approaches the planar floor (`≈ 0.765·a`) and the line-distance
laser residual settles near `0.5·a`. `n̂` is recovered to a small fraction of
a degree; the stand-off `d` degrades fastest as tilt / depth spread shrink.
Matrix test (6 views, `fx = fy = 900`, working distance 0.50–0.62 m):

| noise `a` | normal | distance | laser resid | reproj |
|---:|---:|---:|---:|---:|
| 0.00 px | 0.000° | 0.000 mm | 0.000 px | 0.000 px |
| 0.15 px | ≤ 0.045° | ≤ 0.51 mm | ≈ 0.073 px | ≈ 0.113 px |
| 0.30 px | ≤ 0.084° | ≤ 1.10 mm | ≈ 0.159 px | ≈ 0.226 px |

The reprojection column tracks `0.765·a` (0.115 / 0.230): the corners hit
their noise floor and the plane is the only extra unknown.

**Real-data anchor.** On the private `scheimpflug_rig` / `scheimpflug_rig_ref` Scheimpflug laser
datasets the point-to-plane laser residual is **0.018–0.035 mm** at the
working distance — plane consistency on real hardware (a fit-quality metric,
not GT recovery).

## Evidence

- **Matrix test**: `crates/vision-calibration/tests/laserline_device_matrix.rs`
  — `laserline_device_recovers_plane_across_grid_and_noise` (2 plane
  orientations × 2 working distances × noise `{0, 0.15, 0.30}` px, default
  config through `run_calibration` on both backends, `UniformPixelNoise`
  seed 7); asserts normal, distance, laser and reprojection floors per cell.
- **Property tests** (same file): `point_to_plane_residual_is_se3_invariant`
  (`n̂·p + d` is unchanged by any SE(3) re-framing of point + plane) and
  `plane_sign_flip_is_a_gauge`; 5000 deterministic cases each.
- **Pipeline / optim tests**: `vision-calibration-pipeline/tests/laserline_device.rs`
  (pinhole + Scheimpflug convergence, JSON round-trip);
  `optim::problems::laserline_bundle` unit tests (S² normal block,
  per-feature residuals zero on perfect data, residuals and the projected
  line under strong distortion, pose-count guard); the `factors::laserline`
  rig-chain tests above.
- **Acceptance gate**: `calib-bench accept` on `scheimpflug_rig` (rig hand-eye, then
  the joint rig + hand-eye + laser solve when built with the `laser`
  feature); the gate is per-camera reprojection (≤ 2.5 px) — the laser
  residual is recorded, not gated.
- **Regression suite**: `synth_laserline`, `synth_rig_handeye_laserline`.
- **Related**: ADR 0016 (laser-frame manifest, injected
  `LaserPixelExtractor`), ADR 0012 (per-feature laser residuals on export).
