# Rig extrinsics — proof pack

Family: multi-camera rig extrinsics from shared planar-target views
(`RigExtrinsicsProblem`; per-camera intrinsics init in
`vision-calibration-linear`, linear rig init in
`vision-calibration-linear::extrinsics`, joint refinement in
`vision-calibration-optim::problems::rig_extrinsics` /
`rig_extrinsics_scheimpflug`). The pipeline lives in
`vision-calibration-pipeline/src/rig_extrinsics/` with shared sensor-axis
helpers in `rig_family` (ADR 0013).

## Model

A rigidly-coupled set of `C` cameras observes a planar target from `V`
stations. Each camera keeps its own composable camera (ADR 0005); the
extrinsics tie them to one **rig frame** `R`. Per observation `(cam i,
view v, board point X_t)`:

```
pixel = K_i( sensor_i( distort_i( project( T_Ci_R · T_R_Tv · X_t ) ) ) )
```

- `T_Ci_R` (`cam_se3_rig`, ADR 0009): the camera-from-rig extrinsic,
  constant across views — the unknowns of interest.
- `T_R_Tv` (`rig_se3_target`): the rig-from-target pose of station `v`,
  one SE(3) per view.
- `project` / `distort` / `K`: the planar family's sub-models (see
  [planar-intrinsics.md](planar-intrinsics.md)).
- `sensor_i`: `IdentitySensor` for an untilted camera; the OpenCV-style
  `HomographySensor(tilt_x, tilt_y)` for a Scheimpflug camera (ADR 0013),
  chosen per camera by `camera` / `camera_overrides`. Everything below holds
  for both; only the sensor stage and its named fix list differ.

The per-observation chain is a product of two SE(3)s (`ReprojChain::TwoSe3`
in the optim IR). `RigHandeye` replaces the free per-view `T_R_Tv` with a
robot-kinematics chain; that axis is covered in [hand-eye.md](hand-eye.md).

## Initialization

Two closed-form stages (`rig_extrinsics/steps.rs`,
`step_intrinsics_init_all` → `step_rig_init`):

1. **Per-camera intrinsics.** Each camera is calibrated in isolation by
   iterative Zhang (`rig_family::bootstrap_rig_intrinsics` →
   `vision-calibration-linear`), yielding `K_i, d_i` and a per-view
   `T_Ci_Tv` (`camera_se3_target`). Scheimpflug cameras use the tilt-aware
   linear init (`scheimpflug_init`). A per-camera reprojection guard
   (`guard_percam_reproj_errors`, 50 px ceiling) fails fast on a diverged
   camera before its poses can poison the rig fit.
2. **Rig extrinsics.** `estimate_extrinsics_from_cam_target_poses` pins the
   rig frame to the reference camera `r` (`cam_to_rig[r] = I`); for every
   view where camera `i` and `r` both see the target,
   `T_R_Ci = T_Cr_Tv · (T_Ci_Tv)⁻¹`, averaged over co-observed views
   (arithmetic mean of translations, hemisphere-corrected quaternion mean of
   rotations — initialization-grade). Each view's rig pose follows as
   `T_R_Tv = T_R_Ci · T_Ci_Tv`, averaged over the cameras that saw it.

## Cost

Non-linear refinement (`optimize_rig_extrinsics`) minimizes the total
squared reprojection error over the whole rig:

```
E = Σ_i Σ_v Σ_j ρ( ‖ π_i( T_Ci_R · T_R_Tv · X_j ) − u_ivj ‖² )
```

Parameter blocks (`build_rig_extrinsics_ir`): per-camera intrinsics `K_i`,
distortion `d_i`, extrinsic `T_Ci_R` (SE(3)), per-view rig pose `T_R_Tv`
(SE(3)), and — for Scheimpflug rigs — per-camera `ScheimpflugParams`.
Intrinsics and distortion are **fixed by default**
(`refine_intrinsics_in_rig_ba: false` → `CameraFix::all`, see
`rig_family::rig_ba_fix`), so the joint solve moves only the geometry;
enabling refinement re-opens `K_i, d_i` under each camera's `fix` lists. `ρ` is the
configured robust loss (default none). Factors are autodiff-generic
(ADR 0008).

## Identifiability and degeneracies

- **Minimum data** (`validate_input`): ≥ 2 cameras and ≥ 3 views; each
  camera needs ≥ 3 views with observations for its own Zhang init
  (`views_to_planar_dataset`).
- **Co-visibility.** The relative pose `T_Ci_Cr` is only observable from
  views where camera `i` and the reference `r` *both* see the target. The
  linear init requires a **direct** reference↔camera edge (it does not chain
  `i–k–r`); without one, `estimate_extrinsics_from_cam_target_poses` fails
  with `"no overlapping views between camera i and reference r"`. A
  disconnected graph is a hard failure, not a silent bad fit.
- **Single shared view.** One co-observed view gives a full-rank relative
  pose but no averaging: all noise from that station flows straight into
  the extrinsic. Extrinsic error drops roughly as `1/√(shared views)`.
- **All boards parallel.** If every station shares one plane normal, each
  camera's Zhang init is rank-deficient (the planar plane-orientation
  degeneracy), and a pure-translation station set leaves the relative
  rotation weakly constrained. The matrix test's stations mix pitch and yaw.
- **Weak baseline.** A short inter-camera baseline weakly constrains the
  translation of `T_Ci_Cr` (see §Noise sensitivity): the minimum is in the
  right place, but its variance inflates.
- **Per-camera intrinsics coupling.** With intrinsics fixed in the rig BA
  (default), any residual focal/distortion error from stage 1 is absorbed
  into the extrinsics and rig poses rather than corrected — hence the
  per-camera guard.

## Gauge

- **Rig-frame gauge (6 DOF).** Left-multiplying every `T_Ci_R` and every
  `T_R_Tv` by a common `H` leaves all observations unchanged. The code
  removes it by pinning the **reference camera** (`rig.reference_camera_idx`)
  to identity in the linear init and holding that block fixed in BA
  (`fix_extrinsics[r] = true`, `step_rig_optimize`). The rig frame *is* the
  reference camera frame. Changing the reference re-expresses every
  extrinsic but preserves the relative inter-camera poses and the
  reprojection error (relative pose to `< 5e-4` rad / `< 5e-4` m,
  reprojection to `< 5e-3` px).
- **Rig poses.** All per-view rig poses are free; the reference-camera fix
  is the only gauge removal. Also fixing view 0's pose over-constrains the
  fit (wide-baseline matrix cell: same relative pose, mean reprojection
  0.134 px vs 0.124 px free).

## Noise sensitivity

For iid uniform pixel noise `[−a, a]` per axis and `N ≫ #params`, the
post-fit mean reprojection approaches the `≈ 0.765·a` floor and extrinsic
errors scale linearly in `a` through `(JᵀJ)⁻¹`. The inter-camera
**translation** is the most sensitive DOF, and its sensitivity depends on
rig geometry. Matrix test (two-camera rig, 8 diverse stations at 0.6–0.8 m,
per-camera independent noise streams):

| geometry | metric | `a = 0` | `a = 0.15 px` | `a = 0.30 px` |
|---|---|---:|---:|---:|
| narrow (6 cm, ~parallel) | Δrot (rad) | 2e-14 | 2.7e-3 | 5.3e-3 |
| narrow | Δtrans (m) | 1e-13 | 4.1e-3 | 8.2e-3 |
| narrow | mean reproj (px) | 8e-13 | 0.116 | 0.232 |
| wide (25 cm, ~8° converged) | Δrot (rad) | 3e-14 | 3.2e-3 | 6.4e-3 |
| wide | Δtrans (m) | 2e-13 | 1.1e-2 | 2.2e-2 |
| wide | mean reproj (px) | 1e-12 | 0.144 | 0.286 |

- **Rotation tracks noise almost geometry-independently** (≈ 2e-2 rad/px),
  but **translation is worse on the wide, converged pair** (≈ 7e-2 m/px vs
  ≈ 2.7e-2 m/px narrow): the convergence angle couples the rotation error
  into a longer moment arm. Narrow vs wide is a trade — wide gives stiffer
  downstream triangulation but a noisier extrinsic translation from this
  planar-target fit.
- The **shared** extrinsic is stiffer than the planar family's independent
  per-view poses, so the wide-pair mean (0.286 px at `a = 0.3`) sits above
  the `0.765·a ≈ 0.23 px` floor: per-camera noise cannot be absorbed into
  per-view freedom.

## Evidence

- **Matrix test**: `crates/vision-calibration/tests/rig_extrinsics_matrix.rs`
  — `rig_extrinsics_recovers_gt_across_grid_and_noise` (narrow / wide
  baseline × noise `{0, 0.15, 0.30} px` through `step_intrinsics_init_all →
  step_intrinsics_optimize_all → step_rig_init → step_rig_optimize`, both
  backends); `rig_extrinsics_reference_camera_gauge_invariance`;
  `rig_extrinsics_rejects_disconnected_covisibility_graph`.
- **Unit tests**: `rig_optimize_keeps_reprojection_error_reasonable`
  (`rig_extrinsics/steps.rs`; exact-data rig converges to `< 1e-3` px, state
  and export stats agree); `extrinsics_from_cam_target_poses_two_cameras`,
  `extrinsics_allow_missing_ref_camera_in_view`
  (`vision-calibration-linear::extrinsics`, `< 1e-10` on exact poses).
- **Acceptance gates**: `calib-bench accept` — `stereo_rig` (≤ 0.4 px per
  camera) with a committed Fit baseline.
- **Regression suite**: `stereo_rig`, `synth_rig` (intrinsics refined in
  the rig BA).
- **Related**: ADR 0013 (`rig_family` sensor axis; pinhole ↔ Scheimpflug),
  ADR 0009 (`frame_se3_frame` / SE(3) storage).
