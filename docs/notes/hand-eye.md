# Hand-eye — proof pack

Family: robot-mounted-camera calibration (`SingleCamHandeyeProblem`; the
hand-eye axis of `RigHandeyeProblem`). Linear `AX = XB` initialization in
`vision-calibration-linear::handeye` (Tsai–Lenz, all pairs); pipeline in
`vision-calibration-pipeline::single_cam_handeye`; joint bundle adjustment in
`vision-calibration-optim::problems::handeye` (`optimize_handeye`).

## Model

Each view carries a measured robot pose `T_B_G` (`base_se3_gripper`, ADR 0009)
and a planar-target observation. The camera stage (pinhole + Brown–Conrady) is
exactly the planar family's — see `docs/notes/planar-intrinsics.md`. Hand-eye
adds the rigid unknown `X` and a single fixed target pose `Y`, wired two ways
(`HandEyeMode`, `handeye_observer_se3_target`):

- **EyeInHand** — camera on the gripper. `X = gripper_se3_camera` (`T_G_C`),
  `Y = base_se3_target` (`T_B_T`); per view
  `T_C_T_i = (T_B_G_i · X)^{-1} · Y = X^{-1} · T_G_B_i · Y`.
- **EyeToHand** — camera fixed in the scene, target on the gripper.
  `X = camera_se3_base` (`T_C_B`), `Y = gripper_se3_target` (`T_G_T`); per view
  `T_C_T_i = X · T_B_G_i · Y`.

`T_C_T_i` then projects the board through the shared camera. For a rig the
observer is the rig frame, not the camera, so `RigHandeyeProblem` composes
`cam_se3_rig` on top of the same chain (`vision-calibration-pipeline::rig_handeye`;
its extrinsics axis is the rig-extrinsics pack's job — not re-derived here).

Robot poses reach the pipeline as `Iso3`. The row-major-4×4 loader
(`vision-calibration-pipeline::dataset_runner::poses`) re-orthonormalizes each
rotation with an SVD polar decomposition (`nearest_rotation`) instead of
nalgebra's identity-seeded iterative `Rotation3::from_matrix`, which silently
mis-converges on **exact 180° rotations** (common in real robot stations) and
returned the wrong axis. The linear solver's own matrix→quaternion conversions
use `from_matrix_unchecked` (non-iterative) and are unaffected. Guarded by the
regression test `matrix4x4_rotation_roundtrips_exact_180_degree`.

## Initialization (Tsai–Lenz, all pairs)

For every view pair `(i, j)` the pipeline forms two relative motions: the
gripper motion `A = T_B_G_i^{-1} · T_B_G_j` and the observation motion
`B = S_i^{-1} · S_j`, where the stream `S` is `target_se3_camera` (EyeInHand)
or `camera_se3_target` (EyeToHand) — both routes call `tsai_lenz_allpairs`
(`estimate_handeye_dlt` / `estimate_gripper_se3_target_dlt`). Each pair is a
constraint `A · X = X · B`, split into

```
R_A R_X = R_X R_B                 (rotation)
(R_A − I) t_X = R_X t_B − t_A     (translation)
```

Rotation is solved in quaternions: each pair contributes
`[L(q_A) − R(q_B)] q_X = 0` (left/right multiplication matrices); stacking all
pairs gives a `4N×4` system whose smallest right-singular vector is `q_X`,
extracted via the `AᵀA` symmetric-eigen null space (`math::null_space`) rather
than nalgebra's dense SVD, which can hang on tall real matrices. Translation
stacks the `(R_A − I)` blocks against `R_X t_B − t_A` (`3N×3`) and solves ridge
normal equations (`math::ridge_lstsq`, `λ = 1e-12`). `R_X` is projected back to
SO(3) (`math::project_to_so3`).

Pair conditioning is guarded before the solve (`build_all_pairs` /
`is_good_pair`): pairs whose smaller rotation angle is below
`min_motion_angle_deg` (default **5°**) are dropped, and — when
`reject_axis_parallel` is on (the pipeline default) — pairs whose rotation axes
are near-parallel (`|α̂ × β̂| < 1e-3`) are dropped as ill-conditioned.

## Cost

`optimize_handeye` (`build_handeye_ir`) minimizes total robust reprojection

```
E = Σ_i Σ_j ρ( ‖ π( camera, T_C_T_i(X, Y, T_B_G_i) · X_j ) − u_ij ‖² )
```

over: per-camera intrinsics + distortion (`k3` fixed by default), the hand-eye
`X` and the fixed target `Y` (both SE(3)), and `cam_se3_rig` (fixed for
single-camera via `fix_extrinsics`). Optionally (default **on** for
single-cam) each robot pose gets a per-view se(3) correction
`T_B_G_i ↦ exp(δ_i) · T_B_G_i` (left-multiply) with a zero-mean anisotropic
prior (rotation σ = 0.5°, translation σ = 1 mm) and `δ_0 ≡ 0`. Factors are
autodiff-generic (ADR 0008); `ρ` defaults to none (Huber for detector tails).

## Identifiability and degeneracies

- **Minimum motion.** `R_A R_X = R_X R_B` from one motion only forces `R_X` to
  carry `axis(B)` onto `axis(A)` — a one-parameter family (free spin about that
  axis) remains. Two motions with **non-parallel** rotation axes pin `R_X`
  fully. The pipeline requires ≥ 3 views (≥ 2 independent motions).
- **Translation along the rotation axis is unobservable per motion.**
  `(R_A − I)` is rank-deficient: for a rotation of angle θ about axis `n`, its
  null space is `span(n)`. So one motion never sees the component of `t_X` along
  its own rotation axis; only stacking ≥ 2 non-parallel-axis motions makes the
  `(R_A − I)` stack full-rank-3 and `t_X` observable.
- **Pure translations.** With no rotation, `q_A = q_B = 1 ⇒ L − R = 0` and
  `R_A − I = 0`: the pair carries *zero* hand-eye information. Rejected by the
  min-angle gate.
- **Parallel axes / common screw axis.** If all motions share a rotation axis
  `n`, the `(R_A − I)` stack stays rank-2 (`t_X` along `n` unobservable) and
  `R_X` is fixed only up to a spin about `n`. This is exactly the near-parallel
  case the `axis_parallel_eps` guard removes — and why the matrix test's robot
  stations mix roll, pitch, and yaw.
- **Small rotations.** `(R_A − I) ≈ [θ n]_×` is ill-conditioned as `θ → 0`,
  amplifying translation-noise into `t_X`; the 5° gate trades usable pairs for
  conditioning.
- **Intrinsics coupling.** Hand-eye is estimated *after* intrinsics, so a
  focal-length bias skews every per-view `T_C_T_i` and hence the hand-eye
  translation *scale*; the joint BA re-refines intrinsics to absorb this.

## Gauge

With raw robot poses (refinement off) the joint reprojection cost has a
**base-frame gauge**: rebasing `T_B_G_i ↦ G · T_B_G_i` and `Y ↦ G · Y` leaves
every `T_C_T_i` unchanged (`(G T_B_G X)^{-1}(G Y) = X^{-1} T_B_G^{-1} Y`), so
`X` is invariant and only `Y` absorbs `G`. Otherwise there is no free gauge: the
board fixes the target frame metrically, the robot poses are measurements, and
`K` fixes the camera frame, so `X` and `Y` are fully determined once the motion
requirements above are met.

Robot-pose refinement adds the per-view correction field `{δ_i}`. Its
**common mode is a gauge**: a constant correction is re-absorbable into a fixed
transform — the target `Y` in EyeInHand (`exp(-δ)` sits adjacent to `Y`), the
hand-eye `X` in EyeToHand. `build_handeye_ir` fixes `δ_0 ≡ 0` to remove it, and
the zero-mean priors regularize the remaining near-null directions. Note the
priors, written in the base frame via a *left*-multiplicative correction, are
not `Ad_G`-invariant, so enabling refinement mildly breaks the base-frame gauge
above (measured: rebasing shifts `X` by ~5e-4 rad / 5e-4 m with refinement on,
vs ~1e-14 with it off — the gauge test runs with refinement off to assert the
exact invariant).

## Noise sensitivity

For iid pixel noise with per-axis amplitude `a` (uniform `[−a, a]`, expected
error-norm mean ≈ `0.765·a`) the post-fit mean reprojection approaches that
floor, and hand-eye errors scale ≈ linearly in `a` through `(JᵀJ)⁻¹`.
Translation is the most sensitive DOF (it is the ratio of pose residuals to the
rank-deficient `(R_A − I)` blocks). Empirically (matrix test below):

| noise `a` | mean reproj | hand-eye Δrot | hand-eye Δt |
|---|---:|---:|---:|
| 0.00 px | ~0 (< 1e-4) | ~1e-14 rad | ~1e-13 m |
| 0.15 px | 0.112 px | ≤ 7e-4 rad | ≤ 0.5 mm |
| 0.30 px | 0.224 px | ≤ 1.4e-3 rad | ≤ 0.9 mm |

(EyeInHand, 9 robot stations, 63-point board, `fx ∈ {800, 1200}`.) The reproj
column tracks `0.765·a` (0.115, 0.230). EyeToHand shows the translation
sensitivity of a *distant, derived* pose: at `a = 0.20 px` the direct DLT output
`gripper_se3_target` recovers to ~0.1 mm, while `camera_se3_base` (target ~1.2 m
away) drifts to ~8 mm.

## Evidence

- **Matrix test**:
  `crates/vision-calibration/tests/single_cam_handeye_matrix.rs` —
  `single_cam_handeye_recovers_gt_across_grid_and_noise` (GT grid: 2 focal
  regimes × 2 hand-eye transforms, noise `{0, 0.15, 0.30}` px, standard
  `step_intrinsics_init → step_intrinsics_optimize → step_handeye_init →
  step_handeye_optimize` pipeline); `single_cam_handeye_eye_to_hand_recovers_gt`
  (the EyeToHand convention); `single_cam_handeye_base_frame_gauge_invariance`
  (the base-frame gauge, refinement off).
- **Robot-pose ingestion**: `nearest_rotation` (SVD polar decomposition) and
  `matrix4x4_rotation_roundtrips_exact_180_degree` in
  `vision-calibration-pipeline::dataset_runner::poses`.
- **Linear init unit test**: `handeye_dlt_recovers_ground_truth`
  (`vision-calibration-linear::handeye`).
- **Related**: `docs/notes/planar-intrinsics.md` (shared camera / projection
  equations and pixel-noise-floor argument); `RigHandeyeProblem` for the rig
  axis (same Tsai–Lenz DLT with the rig as observer); ADR 0008 (autodiff IR),
  ADR 0009 (`frame_se3_frame` conventions).
