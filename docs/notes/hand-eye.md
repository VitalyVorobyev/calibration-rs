# Hand-eye — proof pack

Family: robot-mounted-camera calibration (`SingleCamHandeyeProblem`; the
hand-eye axis of `RigHandeyeProblem`). Linear `AX = XB` initialization in
`vision-calibration-linear::handeye` (Tsai–Lenz, Park–Martin, Daniilidis,
Shah); pipeline in
`vision-calibration-pipeline::single_cam_handeye`; joint bundle adjustment in
`vision-calibration-optim::problems::handeye` (`optimize_handeye`).

## Model

Each view carries a measured robot pose `T_B_G` (`base_se3_gripper`, ADR 0009)
and a planar-target observation. The camera stage (pinhole + Brown–Conrady) is
the planar family's — see [planar-intrinsics.md](planar-intrinsics.md).
Hand-eye adds the rigid unknown `X` and a single fixed target pose `Y`, wired
two ways (`HandeyeMode`, `handeye_observer_se3_target`):

- **EyeInHand** — camera on the gripper. `X = gripper_se3_cam` (`T_G_C`),
  `Y = base_se3_target` (`T_B_T`); per view
  `T_C_T_i = (T_B_G_i · X)^{-1} · Y = X^{-1} · T_G_B_i · Y`.
- **EyeToHand** — camera fixed in the scene, target on the gripper.
  `X = cam_se3_base` (`T_C_B`), `Y = gripper_se3_target` (`T_G_T`); per view
  `T_C_T_i = X · T_B_G_i · Y`.

`T_C_T_i` then projects the board through the shared camera. For a rig the
observer is the rig frame, so `RigHandeyeProblem` composes `cam_se3_rig` on
top of the same chain (`vision-calibration-pipeline::rig_handeye`; its
extrinsics axis is covered in [rig-extrinsics.md](rig-extrinsics.md)).

**Robot-pose ingestion.** Robot poses reach the pipeline as `Iso3`. The
row-major-4×4 loader (`vision-calibration-pipeline::dataset_runner::poses`)
re-orthonormalizes each rotation by SVD polar decomposition
(`nearest_rotation`). nalgebra's identity-seeded iterative
`Rotation3::from_matrix` is avoided because it mis-converges on **exact 180°
rotations** (common at real robot stations) and returns the wrong axis. The
linear solver's own matrix→quaternion conversions use the non-iterative
`from_matrix_unchecked`.

## Initialization (closed form, four methods)

Both modes are the robot-world equation `A_i X = Z B_i` with `A_i = T_B_G_i`
(`linear::handeye`, `estimate_eye_in_hand` / `estimate_eye_to_hand`, both
taking `cam_se3_target` per view):

| mode | `B_i` | `X` | `Z` |
|---|---|---|---|
| EyeInHand | `T_T_C_i` | `gripper_se3_cam` | `base_se3_target` |
| EyeToHand | `T_C_T_i` | `gripper_se3_target` | `base_se3_cam` (returned inverted, `cam_se3_base`) |

For the rig the observer is the rig frame (`rig_se3_target`), and EyeToHand's
`cam_se3_base` is the rig's `T_R_B`. `HandeyeMethod` selects the solver
(`handeye_init.method`, per-step `HandeyeInitOptions::method`; default
`tsai_lenz`, unchanged results).

**Motion pairs** (Tsai–Lenz, Park–Martin, Daniilidis). For every view pair
`(i, j)`: `A = A_i^{-1} A_j`, `B = B_i^{-1} B_j`, rotations projected to
SO(3) (`math::project_to_so3`); then `A X = X B`, i.e.

```
R_A R_X = R_X R_B                 (rotation)
(R_A − I) t_X = R_X t_B − t_A     (translation)
```

- **Tsai–Lenz** (the default; strictly the separable quaternion form of
  Tsai–Lenz's two-stage solution): each pair contributes
  `[L(q_A) − R(q_B)] q_X = 0`; the `4N×4` stack's null vector (`AᵀA`
  symmetric eigen, `linalg::null_space`) is `q_X`. Translation by ridge
  normal equations (`math::ridge_lstsq`, `λ = 1e-12`).
- **Park–Martin**: `α = log R_A = R_X log R_B = R_X β`; `R_X` is the
  rotation nearest to `Σ α βᵀ` (orthogonal Procrustes with the det fix — the
  `(MᵀM)^{-1/2}Mᵀ` of the paper, made proper). Weighs a pair by its angle²
  (Tsai–Lenz by `sin²(θ/2)`). Same translation solve.
- **Daniilidis**: dual quaternions `q̌ = q + ε ½ t q`; each pair gives the
  `6×8` block `[[a−b, [a+b]×, 0, 0], [a'−b', [a'+b']×, a−b, [a+b]×]]` on
  `(q_X, q'_X)` (vector parts; scalar parts assumed equal — the screw
  congruence, dropped under noise). The two smallest eigenvectors of the
  `8×8` normal matrix span the solution; `λ₁v₁ + λ₂v₂` is fixed by
  `|q| = 1` and `q·q' = 0` — the zero directions `√μ₂ e₁ ± √(−μ₁) e₂` of the
  2×2 form, the larger-`|q|` root taken; if noise makes the form definite,
  its smallest-|μ| eigenvector, then `q'` orthogonalized. Translations are
  divided by their RMS first so the dual rows weigh like the real ones.
- `Z` is closed through the first view (`Z = A_0 X B_0^{-1}`), as before —
  the default's results do not move.

**Shah** (absolute poses, no pairs): `R_A R_X = R_Z R_B` is linear in
`(vec R_X, vec R_Z)` — `[I ⊗ R_A, −R_Bᵀ ⊗ I]`, a `9N×18` stack whose null
vector is scaled to `det R_X = 1` and both halves projected to SO(3); then
`[R_A, −I](t_X; t_Z) = R_Z t_B − t_A` by least squares. `Z` is estimated over
all views, so it seeds the bundle adjustment's target (EyeInHand) or base
(EyeToHand) pose directly; a seeded `X` still closes `Z` through view 0.

**Quaternion sign.** `q_A q_X = q_X q_B` needs `q_A`, `q_B` with the same
sign of their (equal) scalar parts. nalgebra's matrix→quaternion picks the
sign by branch, which for rotations past 120° depends on the axis; every
quaternion is now taken with `w ≥ 0` (`quat_of`). Before, such a pair entered
Tsai–Lenz as `q_A q_X = −q_X q_B`: harmless when the other pairs outnumber it
(the true `q_X` stays an eigenvector), but three views with 137° and 172°
pairs returned a rotation 180° off
(`every_method_recovers_motions_past_120_degrees` fails without the fix).
Data whose pairs all rotate less than 120° gives bit-identical results.

**Near 180°.** `w ≥ 0` fails within noise of a half turn, where both scalar
parts are at noise level (or exactly zero, and nalgebra's branch decides).
The methods differ in what a mismatched pair does:

- **Tsai–Lenz**: at exactly 180° the true `q_X` is an eigenvector of the
  mismatched pair's block `2(I + L(q_A)R(q_B))` (eigenvalue 4); the pair raises
  the smallest eigenvalue without moving its eigenvector. Unaffected while
  the consistent pairs outnumber the mismatched ones.
- **Park–Martin**: a pair whose `β = log R_B` comes out flipped turns
  `Σ α βᵀ = R_X S` into `R_X (S − 2π² b bᵀ)`; the polar factor is still
  `R_X` while that stays positive definite. Unaffected likewise.
- **Shah**: absolute poses, no pairs, no quaternions. Unaffected.
- **Daniilidis**: its six rows per pair drop the scalar equation and lose that
  invariance. Two mismatched pairs among 91 (six exact 180° wrist flips on
  eight stations) put it 1.03° and 5.0 mm off, noise-free. Fix: its signs
  are matched through the Tsai–Lenz rotation, `q_B → −q_B` when
  `q_A · (q_X0 q_B q_X0⁻¹) < 0` (`sign_matched_quats`); this is the `w ≥ 0`
  choice whenever `q_X0` is within 90° of the truth and the pair is away from
  180°, so other data gives bit-identical results.

Measured (`near_half_turn_motions_keep_every_method_accurate`: 14 views,
six partners at 170°–180°, versus the same scene at 85°–90°; 40 draws, camera
0.2° / 1 mm, robot 0.04° / 0.2 mm per axis; worst error of `X`):

| method | mode | quarter turns | half turns, before | half turns, after |
|---|---|---|---|---|
| Tsai–Lenz | eye-in-hand | 0.230° 1.31 mm | 0.222° 1.03 mm | same |
| Park–Martin | eye-in-hand | 0.244° 1.32 mm | 0.224° 1.03 mm | same |
| Daniilidis | eye-in-hand | 0.265° 1.34 mm | **1.922° 11.10 mm** | 0.236° 0.95 mm |
| Shah | eye-in-hand | 0.230° 1.32 mm | 0.215° 0.89 mm | same |
| Tsai–Lenz | eye-to-hand | 0.278° 6.36 mm | 0.242° 4.82 mm | same |
| Park–Martin | eye-to-hand | 0.286° 6.36 mm | 0.251° 4.82 mm | same |
| Daniilidis | eye-to-hand | 0.580° 6.36 mm | **2.209° 13.95 mm** | 0.400° 4.95 mm |
| Shah | eye-to-hand | 0.278° 6.47 mm | 0.233° 4.80 mm | same |

The min-angle filter needs no upper bound: no method loses accuracy on pairs
near 180°, which carry the most rotation.

**Conditioning.** Pairs whose smaller rotation angle (of `A` or `B`) is below
`min_motion_angle_deg` (default **5°**) are dropped. There is no per-pair
`A`/`B` axis test — `A` and `B` share their axis whenever `R_X` is the
identity or rotates about that axis, and such pairs are informative. The pair
set must contain robot rotation axes spanning ≥ 2 directions
(`robot_axes_span_two_directions`, `|â₁ × â₂| ≥ 1e-3`), else
`NoValidMotionPairs` — for every method, Shah's included: with all relative
robot rotations about one axis `n`, `(Rot(n, φ) R_X, R_A0 Rot(n, φ) R_A0ᵀ R_Z)`
solves `AX = ZB` for every `φ`. Non-finite poses are `InvalidInput`.

## Method comparison under rotation-poor motion

`handeye_init_comparison.rs` (`handeye_methods_comparison_table`, ignored;
run in release with `--ignored --nocapture`, ~4 min): eye-in-hand, known
intrinsics (fixed), random X (≤ ~30°, 10 cm), ten stations viewing the board
from directions spread `±{2, 5, 15, 45}°` per axis at ~0.6 m, optionally four
translation-only stations; noise low (0.1 px, exact robot) or high (0.5 px,
robot ±0.05° / ±0.3 mm per axis); `min_motion_angle_deg = 1°`; 100 trials per
cell, both backends, robot poses fixed (plus ±2°/±5° high with them refined,
25 trials). Basin: the refinement from the method's init ends within 1 σ of
the refinement from the truth (σ the reference's own hand-eye std).

Findings (median init errors in the book chapter's table):

- **Basin: 100 % everywhere**, both backends, refined or fixed robot poses.
  Worst final offset 0.82 σ (±15°, low noise) — the LM stopping tolerance,
  identical across methods: two refinements of the same problem stop ~1e-5
  of the cost apart, which at low noise is a large fraction of a small σ. The
  init method does not decide convergence on this model, even at ±2° / high
  noise (init off by 1.3–2.5°, 3–4 cm).
- **Accuracy**: Tsai–Lenz ≡ Park–Martin (to 0.001°, 0.01 mm). Daniilidis
  halves the rotation error at ±2° (0.24° vs 0.45° low, 1.30° vs 2.37°
  high) but has the larger translation / target error at ±2° high (39 vs
  35 mm, 36 vs 27 mm) and is slightly worse from ±15° on (0.33° vs 0.22°).
  Shah's X = Tsai's; its Z (over all views) wins under noise with rich
  motion (0.39 vs 0.59 mm at ±45° high).
- **Translation-only stations** help every method 5–30 %.
- **Default filter**: at ±2° and 5°, 91/100 captures are `NoValidMotionPairs`
  for every method.
- **Decision**: default stays `tsai_lenz` — no method dominates and the basin
  is method-independent. Measured, not changed.
- Side finding: the hand-eye BA with robot-pose refinement costs ~350 ms for
  10 views × 63 points (two weighted solves plus covariance), ~20× the plain
  solve; the table runs with fixed robot poses for that reason.

## Cost

`optimize_handeye` (`build_handeye_ir`) minimizes total robust reprojection

```
E = Σ_i Σ_j ρ( ‖ π( camera, T_C_T_i(X, Y, T_B_G_i) · X_j ) − u_ij ‖² )
```

over per-camera intrinsics + distortion (`k3` fixed by default), the
hand-eye `X` and the fixed target `Y` (both SE(3)), and `cam_se3_rig`
(fixed for a single camera via `fix_extrinsics`). With robot-pose
refinement (`RobotPoseConfig::Refine`, default **on**) each robot pose gets
a per-view se(3) correction `T_B_G_i ↦ exp(δ_i) · T_B_G_i` (left-multiply)
with a zero-mean anisotropic prior (rotation σ = 0.5°, translation
σ = 1 mm) and `δ_0 ≡ 0`. Factors are autodiff-generic (ADR 0008); `ρ`
defaults to none (Huber for detector tails).

## Identifiability and degeneracies

- **Minimum motion.** `R_A R_X = R_X R_B` from one motion only forces `R_X` to
  carry `axis(B)` onto `axis(A)` — a one-parameter family (free spin about that
  axis) remains. Two motions with **non-parallel** rotation axes pin `R_X`
  fully. The pipeline requires ≥ 3 views (≥ 2 independent motions).
- **Translation along the rotation axis is unobservable per motion.**
  For a rotation of angle θ about axis `n`, the null space of `(R_A − I)` is
  `span(n)`. Only ≥ 2 non-parallel-axis motions make the `(R_A − I)` stack
  rank 3 and `t_X` observable.
- **Pure translations.** With no rotation, `q_A = q_B = 1 ⇒ L − R = 0` and
  `R_A − I = 0`: the pair carries *zero* hand-eye information. Rejected by the
  min-angle gate.
- **Parallel axes / common screw axis.** If all motions share a rotation axis
  `n`, the `(R_A − I)` stack stays rank 2 (`t_X` along `n` unobservable) and
  `R_X` is fixed only up to a spin about `n`. The two-direction set check
  rejects this; the matrix test's robot stations mix roll, pitch and yaw.
- **Small rotations.** `(R_A − I) ≈ [θ n]_×` is ill-conditioned as `θ → 0`,
  amplifying translation noise into `t_X`; the 5° gate trades usable pairs for
  conditioning.
- **Intrinsics coupling.** Hand-eye is initialized *after* intrinsics, so a
  focal-length bias skews every per-view `T_C_T_i` and hence the hand-eye
  translation *scale*; the joint BA re-refines intrinsics to absorb this.

## Gauge

With raw robot poses (refinement off) the cost has a **base-frame gauge**:
rebasing `T_B_G_i ↦ G · T_B_G_i` and `Y ↦ G · Y` leaves every `T_C_T_i`
unchanged (`(G T_B_G X)^{-1}(G Y) = X^{-1} T_B_G^{-1} Y`), so `X` is invariant
and only `Y` absorbs `G`. There is no other free gauge: the board fixes the
target frame metrically, the robot poses are measurements, and `K` fixes the
camera frame, so `X` and `Y` are determined once the motion requirements
above hold.

Robot-pose refinement adds the correction field `{δ_i}`, whose **common mode
is a gauge**: a constant correction is re-absorbable into a fixed transform —
`Y` in EyeInHand (`exp(-δ)` sits adjacent to `Y`), `X` in EyeToHand.
`build_handeye_ir` fixes `δ_0 ≡ 0` to remove it, and the zero-mean priors
regularize the remaining near-null directions. The priors, written in the
base frame via a *left*-multiplicative correction, are not `Ad_G`-invariant,
so refinement mildly breaks the base-frame gauge (rebasing shifts `X` by
~5e-4 rad / 5e-4 m with refinement on, ~1e-14 off); the gauge test runs with
refinement off to assert the exact invariant.

## Noise sensitivity

For iid uniform pixel noise `[−a, a]` per axis (expected error-norm mean
≈ `0.765·a`) the post-fit mean reprojection approaches that floor, and
hand-eye errors scale ≈ linearly in `a` through `(JᵀJ)⁻¹`. Translation is
the most sensitive DOF (it is the ratio of pose residuals to the
rank-deficient `(R_A − I)` blocks). Empirically (matrix test below):

| noise `a` | mean reproj | hand-eye Δrot | hand-eye Δt |
|---|---:|---:|---:|
| 0.00 px | ~0 (< 1e-4) | ~1e-14 rad | ~1e-13 m |
| 0.15 px | 0.112 px | ≤ 7e-4 rad | ≤ 0.5 mm |
| 0.30 px | 0.224 px | ≤ 1.4e-3 rad | ≤ 0.9 mm |

(EyeInHand, 9 robot stations, 63-point board, `fx ∈ {800, 1200}`.) The reproj
column tracks `0.765·a` (0.115, 0.230). EyeToHand shows the translation
sensitivity of a *distant, derived* pose: at `a = 0.20 px` the directly
estimated `gripper_se3_target` recovers to ~0.1 mm, while `cam_se3_base`
(target ~1.2 m away) drifts to ~8 mm.

## Evidence

- **Matrix test**:
  `crates/vision-calibration/tests/single_cam_handeye_matrix.rs` —
  `single_cam_handeye_recovers_gt_across_grid_and_noise` (2 focal regimes ×
  2 hand-eye transforms × noise `{0, 0.15, 0.30}` px, standard
  `step_intrinsics_init → step_intrinsics_optimize → step_handeye_init →
  step_handeye_optimize` pipeline on both backends);
  `single_cam_handeye_eye_to_hand_recovers_gt`;
  `single_cam_handeye_base_frame_gauge_invariance` (refinement off).
- **Robot-pose ingestion**: `matrix4x4_rotation_roundtrips_exact_180_degree`
  (`dataset_runner::poses`).
- **Linear init** (`vision-calibration-linear::handeye`, every method):
  `every_method_recovers_exact_eye_in_hand` / `_eye_to_hand`,
  `_an_identity_rotation_mount`, `_a_mount_rotated_about_a_motion_axis`,
  `_motions_past_120_degrees` (the sign fix), `_half_turn_motions` and
  `near_half_turn_motions_keep_every_method_accurate` (near 180°);
  `every_method_rejects_motions_about_a_single_axis`,
  `_translation_only_motion`; `invalid_input_is_a_typed_error`;
  `rotation_poor_noisy_motion_stays_finite`.
- **Method comparison**: `crates/vision-calibration/tests/handeye_init_comparison.rs`
  — `handeye_methods_gate` (every run, both backends) and the ignored
  `handeye_methods_comparison_table` (the table above).
- **Acceptance gates**: `calib-bench accept` — `kuka_1` (single-camera,
  ≤ 0.25 px) and `ds8` (rig hand-eye, ≤ 0.45 px), with committed Fit
  baselines.
- **Regression suite**: `kuka_1`, `ds8`, `synth_handeye`.
- **Related**: ADR 0008 (autodiff IR), ADR 0009 (`frame_se3_frame`
  conventions).
