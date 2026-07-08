# Laserline bundle — proof pack

Family: single-camera + laser-plane device from planar-target views plus
laser-stripe pixels (`LaserlineDeviceProblem`; init in
`vision-calibration-linear::laserline` + the shared Zhang path, refinement in
`vision-calibration-optim::problems::laserline_bundle`). The same laser cost
is reused, with a composed pose, by the rig laser axes
(`RigLaserlineDeviceProblem`, `RigHandeyeLaserlineProblem`) — cited in
§Rig axes, not re-derived here (their rig/hand-eye parts have their own packs).

## Model

Same composable camera as the planar/Scheimpflug families (ADR 0005),
`pixel = K( sensor( distort( project(X_c) ) ) )`, `X_c = T_C_T · X_t`
(pinhole projection, Brown–Conrady 5 distortion, an optional Scheimpflug
`HomographySensor`, `FxFyCxCySkew`) — see `docs/notes/planar-intrinsics.md`
for the projection/distortion equations and
`docs/notes/scheimpflug-intrinsics.md` for the sensor stage. The sensor is a
hardware property, fixed by default (`fix_sensor: true`); it is taken from
`config.init.sensor_init`, never from a manual-init seed (ADR 0011).

The new object is the **laser plane**, stored in the camera frame as
`LaserPlane { normal: Unit<Vector3>, distance }` with plane equation

```
n̂ · p + d = 0      (p in camera coordinates)
```

(`vision-calibration-optim::params::laser_plane`). In optimization it is two
param blocks: `plane_normal` on the **S²** manifold (dim 3) and
`plane_distance` a Euclidean scalar (dim 1) — so the unit-norm constraint is
carried by the manifold rather than penalized.

Laser observations enter as per-view **stripe pixels**: `LaserlineMeta`
carries `laser_pixels` (optionally per-pixel `laser_weights`) alongside the
target correspondences. On real data those pixels come from a subpixel line
extractor injected through the open `LaserPixelExtractor` trait (ADR 0021) —
the published crates ship no extractor, only the geometry that consumes its
output.

## Initialization

Two stages, both closed-form:

1. **Intrinsics / distortion / poses** — the shared Zhang path
   (`estimate_intrinsics_iterative`: Hartley-normalized DLT homographies →
   image of the absolute conic → `K`, per-view PnP poses, iterative
   `(k1, k2[, p1, p2])` fit), identical to the planar family.
2. **Laser plane** — `linear::laserline::LaserlinePlaneSolver::from_views`:
   each laser pixel is undistorted and back-projected to a camera-frame ray,
   intersected with the target plane (`z = 0` in the *known* per-view pose) to
   a 3D point, then a plane is fit to the pooled points by covariance
   eigendecomposition (smallest eigenvector = `n̂`, `d = −n̂ · centroid`). The
   fit rejects rank-1 (collinear) point clouds and records an RMSE
   (`initial_plane_rmse`). It requires **≥ 2** views because one stripe alone
   is collinear (see §Identifiability).

`step_init_with_seed` (ADR 0011) lets any of intrinsics / distortion / poses /
plane be supplied instead of estimated.

## Cost

Joint bundle adjustment (`optimize_laserline`, IR per ADR 0008) over
intrinsics, distortion (`k3` fixed by default), the sensor (if freed), all
per-view poses (SE(3), `fix_poses: [0]` by default), and the laser plane
(S² normal + scalar distance). Two residual families sum into one problem:

```
E = Σ_i Σ_j ρ_c‖ π(K,d,sensor,T_i·X_j) − u_ij ‖²   (target corners, 2D)
  + Σ_i Σ_k ρ_l · r(K,d,sensor,T_i, n̂,d ; q_ik)²    (laser stripe, 1D)
```

`ρ_c` / `ρ_l` are the configured robust losses (defaults Huber 1.0 px and
Huber 0.01, per-family weights). The laser residual `r` has two forms
(`LaserlineResidualType`, `factors::laserline`):

- **`LineDistNormalized`** (default): intersect the laser and target planes
  into a 3D line, project it to the `z = 1` normalized plane, undistort the
  pixel, take the perpendicular 2D point-to-line distance, and scale by
  `√(fx·fy)`. Units: **pixels** — directly comparable to the reprojection
  floor, which is why it is the default.
- **`PointToPlane`**: undistort → ray → intersect target plane → 3D point →
  signed distance `n̂·p + d` to the laser plane. Units: **metres**.

The plane appears **only** in the laser residuals; corners never touch it. So
the plane is fit from the stripes *conditioned on the poses*, and the poses
are pinned overwhelmingly by the corners — the two halves are nearly
block-separable. Factors are autodiff-generic (ADR 0008); the S² retraction
supplies the normal Jacobian.

### Rig axes

`RigLaserlineDeviceProblem` and `RigHandeyeLaserlineProblem` reuse the *same*
laser cores through `LaserChain::RigHandEye{,RobotDelta}`
(`optim::ir::types::LaserChain`), which composes `cam_se3_target` from
`(cam_se3_rig, hand-eye, target_ref, robot pose[, robot Δ])` instead of taking
a single free pose. A unit test pins the rig residual equal to the single-pose
residual evaluated at the explicitly composed pose, in both hand-eye modes and
both residual types (`factors::laserline` tests). Refinement/residual entry
points: `optim::problems::{laserline_rig_bundle, rig_handeye_laserline_bundle}`,
pipeline `rig_laserline_device` / `rig_handeye_laserline`. The rig extrinsics
and hand-eye gauges are covered by their own packs; the laser plane adds none
beyond the S² sign gauge below.

## Identifiability and degeneracies

The plane is seen only through its stripes, each of which is the plane ∩ board
line — **collinear** points. One board pose therefore constrains only one
in-plane direction; the plane is rank-deficient along the stripe. Recovery
needs the per-view stripes to **span the plane's 2D**:

- **Stripe non-collinearity (the core requirement).** ≥ 2 board poses whose
  stripes differ in direction and/or offset. Board **orientation** diversity
  (roll about the optical axis rotates the stripe *within* the board; pitch/yaw
  reposition it) is the cheap, robust source — it perturbs the stripe without
  driving it off the board. The linear solver rejects a rank-1 pooled cloud
  outright.
- **Distance is the soft DOF.** `n̂` is well pinned by stripe geometry, but the
  stand-off `d` is constrained by how much the metric stripe positions move
  across **depth-diverse** poses. Shallow plane tilt or board depths clustered
  at one distance leaves `d` weakly observable (high variance, not a wrong
  minimum) — the matrix test mixes tilt with a small depth ramp for this
  reason.
- **Degenerate geometries.**
  - *Stripes all parallel* (e.g. board rotated only about an axis parallel to
    the intersection line, or pure depth translation with no tilt diversity):
    stripes stay collinear/parallel → the in-plane direction orthogonal to
    them is unobservable.
  - *Laser plane through the camera centre* (`d → 0`): the plane projects to a
    single fixed image line for every pose, so the stripe carries **no** depth
    information of its own; calibration can still succeed off the known poses,
    but the *deployed* triangulation is ill-posed and `d` is maximally
    ill-conditioned. Small `|d|` is the same pathology, softened.
  - *Laser plane parallel to a board pose*: the intersection line goes to
    infinity — no stripe in that view (guarded: the line-direction cross
    product vanishes → large residual / empty stripe).
  - *Ray parallel to / behind the target plane*: guarded with a large residual
    so a stray pixel cannot pull the solve.
- **View count.** The pipeline requires ≥ 3 views, each with ≥ 4 corner points
  (homography) and ≥ 1 laser pixel; the plane needs ≥ 2 non-collinear stripes.

The intrinsics inherit the planar family's degeneracies (fronto-parallel
focal↔distance trade-off, weak principal point under weak distortion, `k3`
collinearity → `fix_k3: true`) — see `docs/notes/planar-intrinsics.md`.

## Gauge

- **No global gauge.** The plane lives in the physical camera frame, which is
  the fixed reference; the board fixes each view's target frame; `K`, `d`,
  sensor are shared; every `T_i` is an independent free block. `fix_poses: [0]`
  in the default config is a stability/speed convention, **not** a gauge
  necessity — noise-free data recovers exactly with pose 0 held (matrix test),
  because the noise-free init is already exact. (Contrast rig extrinsics, where
  a reference camera genuinely pins the rig frame.)
- **S² sign gauge.** `(n̂, d)` and `(−n̂, −d)` name the same plane (the S²
  double cover). The signed distance only flips sign; its magnitude — what the
  residual squares — is invariant, so the solve is free to land on either
  sheet. Downstream comparisons sign-align to a reference (the matrix test
  aligns the recovered normal to GT before measuring the angle/distance).

## Noise sensitivity

For iid pixel noise with per-axis amplitude `a` (uniform `[−a, a]`) and
`N ≫ #params`, the target reprojection approaches the planar floor
(`≈ 0.765·a` mean error-norm) and the laser line-distance residual settles at
`≈ 0.5·a` px. `n̂` is recovered to a small fraction of a degree; the stand-off
`d` — the soft DOF — degrades fastest as tilt / depth-spread shrink. Measured
(matrix test, 6 views, `fx = fy = 900`, working distance 0.50–0.62 m):

| noise `a` | normal | distance | laser resid | reproj |
|---:|---:|---:|---:|---:|
| 0.00 px | 0.000° | 0.000 mm | 0.000 px | 0.000 px |
| 0.15 px | ≤ 0.045° | ≤ 0.51 mm | ≈ 0.073 px | ≈ 0.113 px |
| 0.30 px | ≤ 0.084° | ≤ 1.10 mm | ≈ 0.159 px | ≈ 0.226 px |

Exact data reproduces GT to solver tolerance. The reprojection column tracks
`0.765·a` (0.115 / 0.230), confirming the corners hit their noise floor and
the plane is the only extra unknown.

**Real-data anchor.** On the private `rtv3d`/`rtv3d_ref` Scheimpflug laser
datasets, the point-to-plane laser residual measures **0.018–0.035 mm** at the
working distance — the plane-consistency achieved on real hardware (a
fit-quality metric, not GT recovery), consistent with the sub-millimetre
synthetic distance bounds above.

## Evidence

- **Matrix test**: `crates/vision-calibration/tests/laserline_device_matrix.rs`
  — GT grid (2 plane orientations × 2 working distances) × noise
  `{0, 0.15, 0.30}` px, standard `run_calibration` (`step_init →
  step_optimize`) pipeline; asserts normal (deg), distance (mm), and the
  laser/reprojection floors per cell. Deterministic noise
  (`UniformPixelNoise`, seed 7); runtime ~2.3 s.
- **Property tests** (same file): `point_to_plane_residual_is_se3_invariant`
  (`n̂·p + d` is unchanged by any SE(3) re-framing of point + plane — the
  invariance the point-to-plane residual rests on) and
  `plane_sign_flip_is_a_gauge` (the S² double-cover). 5000 deterministic cases
  each.
- **Pipeline / optim tests**:
  `vision-calibration-pipeline/tests/laserline_device.rs` (pinhole +
  Scheimpflug convergence, JSON roundtrip);
  `optim::problems::laserline_bundle` unit tests (S² manifold for the normal,
  per-feature residuals zero on perfect data, pose-count guards);
  `optim::factors::laserline` (rig-chain residual = single-pose composition).
- **Acceptance gates**: `calib-bench accept` on the `rtv3d` laser entries
  (laser residual gated at the mm scale above); Q2 committed Fit records add
  drift gates.
- **Related**: ADR 0021 (laser-frame manifest, injected `LaserPixelExtractor`,
  per-view laser images), ADR 0012 (per-feature laser residuals on export),
  `docs/notes/planar-intrinsics.md` (shared projection/distortion + Zhang
  init), `docs/notes/scheimpflug-intrinsics.md` (shared sensor stage).
