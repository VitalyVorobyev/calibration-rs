# Scheimpflug intrinsics — proof-pack stub

Family: single-camera intrinsics from planar-target views on a tilted
sensor (`ScheimpflugIntrinsicsProblem`; init in
`vision-calibration-linear::scheimpflug_init`, refinement in
`vision-calibration-optim::problems::scheimpflug_intrinsics`). This is a
**stub** — the full proof pack (matrix test, identifiability/gauge
analysis, noise sensitivity) is Q8; this note exists to close the loop on
the basin-study evidence ADR 0022 promised (Q6), run 2026-07-04 against
HEAD `q6-basin-study` (cut from `main` at `a974a48`).

## Model

Composable camera (ADR 0005), Scheimpflug specialization: `pixel =
K(sensor(distortion(projection(dir))))`, i.e. pinhole projection, then
Brown-Conrady radial/tangential distortion on normalized coordinates
(same equations as the planar family — see
`docs/notes/planar-intrinsics.md`), then an OpenCV-compatible
`ScheimpflugParams { tilt_x, tilt_y }` homography sensor stage
(`HomographySensor`, `vision_calibration_core::models::sensor`) that maps
the tilted sensor plane back to the frontal normalized plane before the
pinhole intrinsics `K` are applied. `tilt_x`/`tilt_y` couple with focal
length and radial distortion — the classic tilt↔focal↔distortion
degeneracy that motivates ADR 0022's seeded route (from-scratch init
lands in the wrong basin; see ADR 0022's Context section for the
measured evidence).

## The seeded route (ADR 0022/0023)

`step_init_with_seed` takes a spec-derived coarse prior — `fx = fy =
focal_mm·1000/pixel_pitch_um`, principal point at the resolution center,
and mount tilt in radians (`device_seed::scheimpflug_seed`, ADR 0023) —
and refines it with bundle adjustment. The trusted-tilt solve is
two-phase: **Phase A** sweeps `k1` (or the model's leading radial term,
ADR 0022's 2026-07-04 note) start points `{0, −0.20, −0.40}` with
intrinsics and tilt fixed, each preceded by a pose-only adaptation;
**Phase B** is a bounded joint refine with tilt held to `seed ± 0.10
rad`, focal to `[0.75, 1.5]×`, principal point free. Hard acceptance gate:
mean reprojection ≤ 0.5 px per camera (`calib-bench accept`, S4).

## The basin study (Q6)

`calib-bench basin` (`crates/vision-calibration-bench/src/basin.rs`)
quantifies how much spec error this route tolerates. For every registered
`scheimpflug_intrinsics` entry it detects the camera's views once, derives
the ADR 0023 seed once, then perturbs the seed over three independent
per-axis sweeps (never a cross-product — 24 solves/camera) and re-runs
the seeded route per cell, gating each on the entry's
`accept.max_per_cam_mean_px`:

- **focal multiplier** (`fx`, `fy` scaled together): ×{0.50, 0.70, 0.85,
  0.95, 1.05, 1.15, 1.30, 1.50, 2.00}
- **tilt offset** (added to both `tilt_x`, `tilt_y`): {−4°, −2°, −1°,
  −0.5°, +0.5°, +1°, +2°, +4°}
- **principal-point offset** (added to both `cx`, `cy`): {−100, −50, −20,
  +20, +50, +100} px
- plus the unperturbed baseline cell

Run: `cargo run --release -p vision-calibration-bench --bin calib-bench
--features "tier-b laser" -- basin`. Total wall time for all 12 registered
cameras (`rtv3d_ref_cam0..5`, `rtv3d_ringgrid_cam0..5`): **~1.5 minutes**
(well under the ~30-45 min budget) — each entry's 24-cell sweep reuses one
detection pass, so the cost is dominated by 24 bundle-adjustment solves on
a 20-22-view dataset, not detection.

### Measured tables (2026-07-04)

**`rtv3d_ref`** (6 cameras, gate ≤ 0.5 px/camera):

| cell | pass-rate |
|---|---:|
| baseline (unperturbed spec seed) | 6/6 |
| focal ×0.50 | 0/6 |
| focal ×0.70 | 4/6 |
| focal ×0.85 | 6/6 |
| focal ×0.95 | 6/6 |
| focal ×1.05 | 6/6 |
| focal ×1.15 | 6/6 |
| focal ×1.30 | 6/6 |
| focal ×1.50 | 1/6 |
| focal ×2.00 | 0/6 |
| tilt −4.00° .. +4.00° (all 8 offsets) | 6/6 |
| pp −100 .. +100 px (all 6 offsets) | 6/6 |

- focal basin: ×[0.85, 1.30] all-pass
- tilt basin: [−4.00°, +4.00°] all-pass (the full sweep — no failure observed)
- principal-point basin: [−100, +100] px all-pass (the full sweep)

**`rtv3d_ringgrid`** (6 cameras, gate ≤ 1.0 px/camera):

| cell | pass-rate |
|---|---:|
| baseline (unperturbed spec seed) | 6/6 |
| focal ×0.50 | 0/6 |
| focal ×0.70 | 5/6 |
| focal ×0.85 | 6/6 |
| focal ×0.95 | 6/6 |
| focal ×1.05 | 6/6 |
| focal ×1.15 | 6/6 |
| focal ×1.30 | 6/6 |
| focal ×1.50 | 3/6 |
| focal ×2.00 | 0/6 |
| tilt −4.00° .. +4.00° (all 8 offsets) | 6/6 |
| pp −100 .. +100 px (all 6 offsets) | 6/6 |

- focal basin: ×[0.85, 1.30] all-pass
- tilt basin: [−4.00°, +4.00°] all-pass (the full sweep)
- principal-point basin: [−100, +100] px all-pass (the full sweep)

(Full per-cell tables, rendered by the tool itself, are reproducible with
the command above; these are the same numbers, condensed here for the
tilt/pp axes since every tested offset passed on all cameras.)

### Conclusion vs. the decision rule

The decision rule (ADR 0022/0023, backlog Q6) is: the basin must
comfortably contain realistic spec error — **focal ±5 %, tilt ±2°** —
on every camera. Both families clear it with wide margin:

- **Focal**: measured all-pass basin ×[0.85, 1.30] ⊃ the required
  ×[0.95, 1.05] — **2-3× the required margin** in relative terms (15 %
  below / 30 % above vs. the required 5 %).
- **Tilt**: measured all-pass basin covers the *entire* tested sweep,
  ±4° ⊃ the required ±2° — **2× the required margin**, and the true
  basin may be wider still (the sweep did not find a tilt-axis failure
  at any tested offset).
- **Principal point** (not part of the decision rule, informational):
  all-pass over the entire ±100 px sweep on both families.

**RESULT: PASS.** The seeded route's basin comfortably contains
realistic spec error on both private datasets; this closes the Q6 loop
and is the quantitative evidence ADR 0022 forward-referenced. No
reopening of the Phase A sweep design is warranted by this run.

## Evidence

- **Acceptance gates**: `calib-bench accept` — `rtv3d_ref_cam0..5` gated
  at ≤ 0.5 px/camera, `rtv3d_ringgrid_cam0..5` at ≤ 0.7 px/camera, both
  against the ADR 0023 device-spec seed.
- **Basin study**: `calib-bench basin` (this note).
- **Related**: ADR 0022 (seeded init is the supported default — the
  from-scratch fragility this decision steps around), ADR 0023
  (`DeviceSpec` schema and seed derivation), `docs/notes/planar-intrinsics.md`
  (the distortion/projection equations shared with this family).
- **Follow-up (Q8)**: the full proof pack — synthetic-GT matrix test,
  identifiability/gauge write-up (tilt↔focal↔distortion, `k1`'s spurious
  zero-basin), and noise-sensitivity numbers.
