# Rig extrinsics — proof pack

Family: multi-camera rig extrinsics from shared planar-target views
(`RigExtrinsicsProblem`; per-camera intrinsics init in
`vision-calibration-linear`, linear rig init in
`vision-calibration-linear::extrinsics`, joint refinement in
`vision-calibration-optim::problems::rig_extrinsics` /
`rig_extrinsics_scheimpflug`). The pipeline lives in
`vision-calibration-pipeline/src/rig_extrinsics/` and its shared
sensor-axis helpers in `crate::rig_family` (ADR 0013).

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
- `project` / `distort` / `K`: pinhole + Brown–Conrady 5 + `FxFyCxCySkew`,
  exactly the planar family's sub-models (see
  `docs/notes/planar-intrinsics.md`).
- `sensor_i`: `IdentitySensor` for `SensorMode::Pinhole`; the OpenCV-style
  `HomographySensor(tilt_x, tilt_y)` for `SensorMode::Scheimpflug` (ADR
  0013), one `ScheimpflugParams` per camera. Everything below is identical
  across the two flavours except the sensor stage and its
  `ScheimpflugFixMask`.

The composite per-observation chain is a product of two SE(3)s
(`ReprojChain::TwoSe3` in the optim IR): the shared extrinsic and the
per-view rig pose. `RigHandeye` extends this exact model by replacing the
free per-view `T_R_Tv` with a robot-kinematics-constrained chain
(`T_R_Tv = handeye · T_base_gripper,v · target`); the extra hand-eye axis
is covered separately in `docs/notes/hand-eye.md`.

## Initialization

Two stages, both closed-form (`rig_extrinsics/steps.rs`,
`step_intrinsics_init_all` → `step_rig_init`):

1. **Per-camera intrinsics.** Each camera is calibrated in isolation by
   Zhang's method with iterative distortion (`rig_family::bootstrap_rig_intrinsics`
   → `vision-calibration-linear`), yielding `K_i, d_i` and a per-view
   `T_Ci_Tv` (`camera_se3_target`) from each camera's own DLT homographies.
   Scheimpflug cameras use the tilt-aware linear init
   (`scheimpflug_init`). A per-camera reprojection guard
   (`guard_percam_reproj_errors`, 50 px ceiling) fails fast on a diverged
   camera before its garbage poses can poison the rig fit.
2. **Rig extrinsics.** `estimate_extrinsics_from_cam_target_poses`
   (`vision-calibration-linear::extrinsics`) turns the per-camera poses
   into rig transforms. With the rig frame pinned to the reference camera
   `r` (`cam_to_rig[r] = I`), for every view where camera `i` and `r` both
   see the target,
   `T_R_Ci = T_Cr_Tv · (T_Ci_Tv)⁻¹`, and these per-view estimates are
   averaged over co-observed views (arithmetic mean of translations,
   hemisphere-corrected quaternion mean of rotations — an
   initialization-grade SE(3) average). Each view's rig pose then follows
   as `T_R_Tv = T_R_Ci · T_Ci_Tv`, averaged over the cameras that saw it.

## Cost

Non-linear refinement (`optimize_rig_extrinsics`) minimizes the total
squared reprojection error over the whole rig:

```
E = Σ_i Σ_v Σ_j ρ( ‖ π_i( T_Ci_R · T_R_Tv · X_j ) − u_ivj ‖² )
```

Parameter blocks (optim IR, `build_rig_extrinsics_ir`): per-camera
intrinsics `K_i`, per-camera distortion `d_i`, per-camera extrinsic
`T_Ci_R` (SE(3) manifold), per-view rig pose `T_R_Tv` (SE(3) manifold),
and — for Scheimpflug rigs — per-camera `ScheimpflugParams`. Intrinsics
and distortion are **fixed by default** (`refine_intrinsics_in_rig_ba:
false`, `CameraFixMask::all_fixed`), so the joint solve moves only the
geometry; enabling refinement re-opens `K_i, d_i` under the usual
`fix_k3`/`fix_tangential` masks. `ρ` is the configured robust loss
(default none; Huber for detector tails). Factors are autodiff-generic
(ADR 0008).

## Identifiability and degeneracies

- **Minimum data** (`validate_input`): `≥ 2` cameras and `≥ 3` views; each
  camera needs `≥ 3` views with observations for its own Zhang init
  (`views_to_planar_dataset`).
- **Co-visibility / connectivity.** The relative pose `T_Ci_Cr` is only
  observable from views where camera `i` and the reference `r` *both* see
  the target. The linear init computes each non-reference extrinsic purely
  from such shared views; if a camera never co-observes with the reference,
  `estimate_extrinsics_from_cam_target_poses` returns
  `"no overlapping views between camera i and reference r"`. In graph
  terms the cameras must form a connected co-visibility graph *with the
  reference in the same component* — the current linear init is stricter
  still, requiring a **direct** reference↔camera edge (it does not chain
  `i–k–r`). A disconnected graph is a hard failure, not a silent bad fit
  (matrix test, `rig_extrinsics_rejects_disconnected_covisibility_graph`).
- **Single shared view.** One co-observed view suffices for a rank-full
  relative pose, but there is then no averaging: all pixel + pose noise
  from that one station flows straight into the extrinsic. Extrinsic
  variance drops roughly as `1/√(shared views)`; a healthy rig wants the
  target co-visible across many diverse stations.
- **All boards parallel.** If every station shares one plane normal, each
  camera's own Zhang init is rank-deficient (the planar family's
  plane-orientation degeneracy — see `docs/notes/planar-intrinsics.md`),
  so the intrinsics feeding the rig fit are ill-posed. Independently,
  pose diversity is what lets the SE(3) average of `T_R_Ci` resolve the
  rotation cleanly; a pure-translation station set leaves the relative
  rotation weakly constrained. The matrix test's stations mix pitch and
  yaw for exactly this reason.
- **Weak baseline.** A short inter-camera baseline weakly constrains the
  translation DOF of `T_Ci_Cr` (see §noise sensitivity); the minimum is in
  the right place, but its variance inflates.
- **Per-camera intrinsics coupling.** With intrinsics fixed in the rig BA
  (default), any residual focal/distortion error from stage 1 is absorbed
  into the extrinsics and rig poses rather than corrected — a reason the
  per-camera guard and a good stage-1 fit matter.

## Gauge

- **Rig-frame gauge (6 DOF).** The whole parameter set has one global
  freedom: left-multiplying every `T_Ci_R` and every `T_R_Tv` by a common
  `H` leaves all observations unchanged (the map
  `(extrinsics, rig poses) ↦ {T_Ci_Tv}` has `H` in its kernel). The code
  removes it by pinning the **reference camera** to identity and holding
  that block fixed in BA: `cam_to_rig[reference_camera_idx] = I` in the
  linear init, and `fix_extrinsics[r] = true` in
  `RigExtrinsicsSolveOptions` (`rig_extrinsics/steps.rs`). The rig frame
  therefore *is* the reference camera frame. Shifting the reference to a
  different camera re-expresses every extrinsic (left-multiply by the old
  reference-to-new-reference pose) but preserves the observable relative
  inter-camera poses and the reprojection error — verified by
  `rig_extrinsics_reference_camera_gauge_invariance` (relative pose agrees
  to `< 5e-4` rad / `< 5e-4` m, reprojection error to `< 5e-3` px across a
  reference-camera swap).
- **First rig pose (removed, ADR 0024 D2).** `RigExtrinsicsConfig` used to
  carry a `fix_first_rig_pose` knob (default `true`) that additionally fixed
  view 0's `T_R_T0`. Because the reference-camera fix already removes the
  full 6-DOF gauge, this was *not* required for gauge — it was an extra
  constraint that locked station 0's target pose to its linear-init value
  instead of jointly refining it. Empirically it was benign-to-mildly-
  pessimizing: disabling it on the wide-baseline matrix cell left the
  recovered relative pose unchanged and slightly *lowered* the mean
  reprojection error (0.124 vs 0.134 px), consistent with a redundant
  constraint rather than a gauge necessity. On exact data it was harmless
  (the fixed value equals the ground-truth pose). ADR 0024 (R3, 2026-07)
  deleted the knob and the constraint outright — view 0's rig-from-target
  pose is now always free in the rig BA; the reference-camera fix above is
  the only gauge-removal mechanism.

## Noise sensitivity

For iid pixel noise with per-axis amplitude `a` and `N ≫ #params`
residuals, the post-fit mean reprojection approaches the noise floor
(uniform `[−a, a]` per axis ⇒ expected error-norm mean `≈ 0.765·a`) and the
extrinsic-parameter errors scale linearly in `a` through `(JᵀJ)⁻¹`. The
inter-camera **translation** is the most sensitive DOF, and its sensitivity
depends on rig geometry. Empirically (matrix test, two-camera rig, 8
diverse stations at 0.6–0.8 m, per-camera independent noise streams):

| geometry | metric | `a = 0` | `a = 0.15 px` | `a = 0.30 px` |
|---|---|---:|---:|---:|
| narrow (6 cm, ~parallel) | Δrot (rad) | 2e-14 | 2.7e-3 | 5.3e-3 |
| narrow | Δtrans (m) | 1e-13 | 4.1e-3 | 8.2e-3 |
| narrow | mean reproj (px) | 8e-13 | 0.116 | 0.232 |
| wide (25 cm, ~8° converged) | Δrot (rad) | 3e-14 | 3.2e-3 | 6.4e-3 |
| wide | Δtrans (m) | 2e-13 | 1.1e-2 | 2.2e-2 |
| wide | mean reproj (px) | 1e-12 | 0.144 | 0.286 |

Both rows scale cleanly with `a`. Two observations worth calling out:

- **Rotation tracks noise almost geometry-independently** (≈ 2e-2 rad/px),
  but **translation is worse on the wide, strongly-converged pair** (≈ 7e-2
  m/px vs ≈ 2.7e-2 m/px narrow): the larger convergence angle couples the
  ~6e-3 rad rotation error into a longer translation moment arm, so ~2 cm
  of translation error at 0.3 px on the 25 cm baseline is ~3× the narrow
  pair. The relative merit of narrow vs wide is thus a trade — wide gives
  stiffer triangulation for downstream stereo but a noisier extrinsic
  translation from this planar-target fit.
- The rig's **shared** extrinsic is a stiffer constraint than the planar
  family's independent per-view poses, so the wide-pair post-fit mean
  (0.286 px at 0.3 px) sits slightly *above* the `0.765·a ≈ 0.23 px` floor —
  the model cannot absorb per-camera noise into per-view freedom.
- Exact data reproduces ground truth to solver tolerance (`< 1e-12` px,
  `< 1e-13` m).

## Evidence

- **Matrix test**:
  `crates/vision-calibration/tests/rig_extrinsics_matrix.rs` — GT grid
  (narrow/wide 2-camera baseline) × noise `{0, 0.15, 0.30} px` through the
  standard `step_intrinsics_init_all → step_intrinsics_optimize_all →
  step_rig_init → step_rig_optimize` pipeline; asserts inter-camera
  rotation/translation recovery and the reprojection floor per cell, plus
  the reference-camera gauge-invariance and disconnected-graph degeneracy
  property tests above. Runtime ~2 s.
- **Pipeline unit tests**:
  `rig_extrinsics/steps.rs::rig_optimize_keeps_reprojection_error_reasonable`
  (exact-data rig converges to `< 1e-3` px and the state/export stats
  agree), `extrinsics.rs` linear-init recovery (`< 1e-10` on exact poses,
  including a missing-reference-view case).
- **Acceptance gates**: `calib-bench accept` — the `stereo_left`/
  `stereo_right` planar entries gate the per-camera intrinsics that feed
  the rig; Q2 committed Fit records add drift gates.
- **Related**: ADR 0013 (`rig_family` sensor-axis refactor; pinhole ↔
  Scheimpflug unification), ADR 0009 (`frame_se3_frame` / SE(3) storage),
  `docs/notes/planar-intrinsics.md` (the per-camera projection/distortion
  sub-models), `docs/notes/hand-eye.md` (the hand-eye axis that extends the
  per-view rig pose).
