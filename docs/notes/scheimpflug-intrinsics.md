# Scheimpflug intrinsics

Family: single-camera intrinsics from planar-target views on a tilted
sensor (`ScheimpflugIntrinsicsProblem`; init in
`vision-calibration-linear::scheimpflug_init`, refinement in
`vision-calibration-optim::problems::scheimpflug_intrinsics`, staging in
`vision-calibration-pipeline::scheimpflug_intrinsics::steps`). The pack is
partial: the open items are listed at the end.

## Model

`pixel = K(sensor(distortion(projection(dir))))` (ADR 0005): pinhole
projection, a distortion model on normalized coordinates (Brown–Conrady by
default; equations in `planar-intrinsics.md`), then the OpenCV-compatible
`ScheimpflugParams { tilt_x, tilt_y }` homography (`HomographySensor`,
`vision_calibration_core::models::sensor`), then `K`.

## Identifiability

Tilt, focal length and radial distortion trade off: the cost has a long,
flat valley along their combination. Zhang's pinhole init ignores tilt and
underestimates the focal length on a tilted sensor, and the solve then
settles in a wrong tilt/focal basin. This is why the supported route is
seeded (ADR 0011).

A second trap remains once tilt is seeded: starting from `k1 = 0`, the
fixed-tilt sub-problem has a spurious minimum near `k1 = 0`. The per-view
poses adapt to absorb the radial distortion, and the gradient pushes `k1`
back to zero; more iterations do not escape it, a multi-start over `k1`
does.

## The seeded solve

`step_init_with_seed` takes a coarse prior — `fx = fy =
focal_mm·1000/pixel_pitch_um`, the principal point at the image centre, and
the mount tilt (`device_seed::scheimpflug_seed`) — with distortion and
poses initialized automatically. A seeded tilt makes `step_optimize` trust
that basin instead of running the cold-start tilt sweep:

- **Phase A** sweeps the leading radial term (`k1`; `lambda` for
  `Division1`; a no-op for no distortion) over `{0, −0.20, −0.40}` with
  intrinsics and tilt fixed, so the degenerate high-tilt basin is out of
  reach. Each start is preceded by a pose-only adaptation to that `k1`.
- **Phase B** refines the best start jointly under Huber, with tilt bounded
  to `seed ± 0.10 rad`, focal length to `[0.75, 1.5]×` the seed and the
  principal point free.
- The first pose is not fixed: planar intrinsics has no pose gauge, and the
  homography-recovered seed pose is biased under strong distortion.

The fixed parameters are `CameraFix` lists named in each model's own
coefficients (`CameraFix::resolve` maps them onto the packed layout); a slot
left `None` takes `CameraModelSpec::default_fix`, which under a Scheimpflug
sensor holds every coefficient but the low-order radial ones (`k1`, `k2`;
Division1's `lambda`). The staging fix lists are named the same way.

Unseeded `step_init` remains, documented as experimental with a warning in
the init log; it shares the cold-start staged solver with the rig path.

## Convergence basin

`calib-bench basin` (`crates/vision-calibration-bench/src/basin.rs`)
measures how much spec error the seeded route tolerates. For each
registered `scheimpflug_intrinsics` entry it detects once, derives the
device-spec seed once, then perturbs the seed along three independent axes
and re-runs the seeded route per cell against the entry's acceptance gate:

- focal multiplier (`fx`, `fy` together): ×{0.50, 0.70, 0.85, 0.95, 1.05,
  1.15, 1.30, 1.50, 2.00}
- tilt offset (both axes): {±0.5°, ±1°, ±2°, ±4°}
- principal-point offset (both axes): {±20, ±50, ±100} px

Run with `cargo run --release -p vision-calibration-bench --bin calib-bench
--features "tier-b laser" -- basin` (about 1.5 minutes for 12 cameras).

Cameras passing their gate, per cell (6 cameras each):

| cell | `scheimpflug_rig_ref` | `scheimpflug_rig_ringgrid` |
|---|---:|---:|
| unperturbed seed | 6 | 6 |
| focal ×0.50 | 0 | 0 |
| focal ×0.70 | 4 | 5 |
| focal ×0.85 … ×1.30 | 6 | 6 |
| focal ×1.50 | 1 | 3 |
| focal ×2.00 | 0 | 0 |
| every tilt offset (±4°) | 6 | 6 |
| every principal-point offset (±100 px) | 6 | 6 |

The all-pass basin is ×[0.85, 1.30] in focal length and at least ±4° in
tilt, against a realistic spec error of ±5 % and ±2°: two to three times
the needed margin. A principal-point seed off by up to 100 px does not
matter.

## Rig cameras

The rig workflows calibrate each tilted camera from its own views first
(`planar_family::{bootstrap_camera, optimize_camera}`: the linear stage,
then the staged tilt multi-start), and the rig bundle adjustment refines
the intrinsics and the tilt (`rig.refine_sensor`, `handeye_optimize.
refine_sensor` and the laser joint stage's default fix, all on since
UQ-HELD-TILT). The per-camera stage still has to land near the camera's
own minimum: the rig solve is local and does not leave a wrong tilt /
principal-point basin.

**Why the rig solve refines the tilt** (UQ-HELD-TILT). A camera's own
views pin the tilt poorly; the rig's shared board poses pin it far better.
Held at the per-camera estimate, the tilt's error goes into `fx` and the
principal point, and the covariance, conditional on the held tilt, does
not show it. Measured by Monte-Carlo (`rig_extrinsics` from scratch, the
held tilt against the refined one):

| scene | held: `fx` bias / scatter / stated σ | refined: `fx` bias / scatter / stated σ | refined: tilt scatter / σ |
|---|---|---|---|
| 4 cameras, `k1 ≈ −0.4`, −5° tilt, 6 views ≤ 0.1 rad, σ = 0.2 px (60 draws, `pipeline/tests/rig_tilted_recovery.rs`; TinySolver, Factrs within 6 px held and 0.05 px refined) | −17 to −26 / 27–32 / 3.4–3.6 px | ≤ 0.3 / 7.2–7.8 / 6.9–7.6 px | 0.048–0.053° / 0.049–0.050° |
| 2 cameras, `k1 ≈ −0.04`, 8 views ≤ 0.45 rad, σ = 0.1 px (100 draws, facade `rig_extrinsics_scheimpflug.rs`) | `cx`: 0 / 1.5–2.6 / 0.10–0.14 px | `cx`: 0 / 1.5–2.6 / 1.5–2.3 px | 0.07–0.08° / 0.07–0.08° |

On the first scene holding the tilt is a point-estimate problem (the
per-camera tilts are biased by about +0.5° and scatter 0.5–0.9°, some
cameras land in a wrong basin) as well as a missing-variance one; on the
second it is missing variance only. Refining fixes both; the alternative,
carrying the per-camera tilt covariance as consider parameters, would fix
the σ but not the bias, and has no case where it would apply (a tilt the
rig views cannot determine, the camera's own views cannot either). The
Monte-Carlo gate `rig_covariance_with_tilts_matches_monte_carlo` (400
draws per backend on the second scene) measures scatter / stated σ of
0.94–1.02 for `fx, fy, cx, cy, tilt_x, tilt_y` of both cameras on both
backends.

On the private six-camera ChArUco rig (`scheimpflug_rig`, from scratch
through `vcal run`), which pins the tilt / `cy` / `fx` valley very weakly,
refining moves the cameras along it (up to about 25° of tilt and 25 % of
`fx`) for a mean reprojection error about 8 % lower; the held solution
itself moves by up to 25° of tilt and 5 % of `fx` when only the tilt seed
changes, against a stated `fx` σ under 0.5 %. Neither is trustworthy
there; the refined covariance is the less overconfident.

Two rules make it land there, both shared with the single-camera
workflows:

- **The linear stage holds `p1`, `p2` at zero on a tilted camera**
  (`IntrinsicsInitConfig::distortion_fit_opts`). The Scheimpflug linear
  init zeroes the radial terms of its seed but keeps fitted tangential
  ones, and the default fix list holds `p1`, `p2` in every later solve, so
  a fitted value stays frozen. A tangential term is a decentering, nearly
  a tilt plus a principal-point shift, so the frozen value moves the
  solution along the tilt / principal-point valley.
- **The camera's own solve frees every target pose**
  (`planar_family::camera_solve_opts`). Planar intrinsics has no pose
  gauge; a pose held at its linear-init value (recovered with the radial
  terms at zero) holds the solve off the minimum.

Evidence, on the two-camera synthetic rig of the solver benchmark
(`rig_extrinsics/scheimpflug/*/clean`; camera 1 tilted by (0.06, −0.04)
rad, `k1 = −0.04`, seeded with its mount tilt about 0.6° off; small scene,
σ = 0.1 px), camera 1's own solve:

| per-camera stage | principal point error | tilt error | objective | RMS |
|---|---:|---:|---:|---:|
| fitted `p1`, `p2` frozen, view 0 held | (−24.8, −8.7) px | (−0.52, +1.41)° | 22.41 | 0.298 px |
| `p1`, `p2` zero, view 0 held | (−31.1, −26.9) px | (−1.60, +2.08)° | 12.56 | 0.223 px |
| fitted `p1`, `p2` frozen, poses free | (+59.3, −10.9) px | (−0.76, −4.10)° | 4.980 | 0.141 px |
| `p1`, `p2` zero, poses free | (−1.8, −0.9) px | (−0.07, +0.13)° | 4.983 | 0.141 px |
| started at the truth, poses free | (−1.8, −0.9) px | (−0.07, +0.13)° | 4.983 | 0.141 px |

The linear stage had fitted `p2 = −0.003`. With it frozen the free solve
reaches the noise floor in a wrong basin, 60 px and 4° off: noise-free,
that basin's objective is 0.045 (RMS 0.013 px), so at σ = 0.1 px on eight
views the data barely tells it from the truth (4.980 against 4.983). On 36
views it is 0.8 worse than the truth and still a local minimum. With the
tangential terms at zero, the solve from the linear init and the solve
from the truth reach the same minimum.

The tilt is observable: at the minimum the covariance gives
σ(cx) = 2.4 px, σ(cy) = 1.5 px, σ(tilt_x) = 0.09°, σ(tilt_y) = 0.16° on
the small scene, with ρ(cy, tilt_x) = +0.995 and ρ(cx, tilt_y) = −0.997:
the tilt and principal point are one valley that the radial terms close.
Noise-free, the solve from the truth stays there (objective 0). On the
large scene (36 views) the standard deviations are about three times
smaller.

After the fix every `rig_extrinsics/scheimpflug/*/clean` scene reaches
the noise floor (RMS 0.141 px at σ = 0.1 px, 0.68–0.71 px at σ = 0.5 px)
with every principal-point and tilt error within 1.5σ: 0.4–2.1 px and
0.03–0.15° at σ = 0.1 px, 5.4–17.7 px and 0.35–1.19° at σ = 0.5 px. The
contaminated scenes (5 % outliers, Huber or Cauchy) stay 20–45 px and
2–3° off: the staged solve's last stage, which frees the principal point,
solves without the robust loss and is accepted on the mean reprojection
error, outliers included, so it is rejected and the principal point keeps
its linear-init value.

### From scratch: the tilted recovery

With nothing seeded, the staged solve on a camera's own views can still
settle in a wrong tilt / principal-point basin. After a full auto-init the
rig intrinsics stage of every rig problem type (`rig_extrinsics`,
`rig_handeye`) therefore solves each tilted camera through
`rig_family::tilted`: a wide multi-start over linear tilt seeds and, for a
Brown-Conrady camera still above 0.5 px, a re-solve from a nominal seed of
the well-fitted cameras (then against their rig). Seeded cameras skip it.
The stage runs before any hand-eye or rig coupling, so the recovery does
not depend on robot poses; until LEAN-RIG-TILTED only `rig_handeye` ran it.

Measured on `rig_extrinsics` from scratch (Brown-Conrady + Scheimpflug,
`k3, p1, p2` fixed, Huber 1 px, rig intrinsics and tilt held), the
recovery off and on:

| scene | mean reprojection, off → on | worst camera, off → on | runtime (release), off → on |
|---|---:|---:|---:|
| `scheimpflug_rig_ref` (6 cameras) | 0.632 → 0.304 px | 2.55 → 0.35 px | 3.3 → 3.5 s |
| `scheimpflug_rig` (6 cameras, ChArUco) | 1.025 → 0.837 px | 1.20 → 0.94 px | 2.4 → 13.9 s |
| `scheimpflug_rig_ringgrid` (6 cameras) | 0.928 → 0.592 px | 1.38 → 0.81 px | 1.7 → 6.5 s |
| synthetic 4 cameras, `k1 ≈ −0.4`, −5° tilt, 6 views ≤ 0.1 rad, σ = 0.2 px | 0.340 → 0.264 px | 0.564 → 0.266 px | 0.7 → 1.5 s |
| synthetic 6 cameras, same lens, 12 views ≤ 0.35 rad, σ = 0.1 px (8 seeds) | unchanged (≤ 1e-4 px) | unchanged | +20 % |
| the same with one camera at σ = 1 px (8 seeds) | unchanged (≤ 1e-4 px) | unchanged | +140 % |

Without it, one camera of `scheimpflug_rig_ref` ends at fx 1061 px and
tilt_x −3.4° against its five neighbours' 1143–1159 px and −5.0 to −5.3°;
with it, at 1152 px and −5.2°. On `scheimpflug_rig` five cameras settle in
tilts of either sign (+14.7° to −13.7°, fx 1780–2070 px) without it and
all at +13.1° to +14.8° (fx 2041–2099 px) with it. On the synthetic
4-camera scene the wrong basin is 290 px and 14° off the truth (both
backends); the recovery lands every camera within 3.3 px and 0.3°
(`tests/rig_tilted_recovery.rs`). `rig_handeye` gives the same per-camera
stage on every scene, and its final means are 0.302, 0.792 and 0.571 px
on the three real rigs. The cost is the multi-start's extra staged solves:
none when the first candidate fits below 0.45 px, ten when no candidate
does.

## Evidence

- Acceptance: `calib-bench accept` gates every `scheimpflug_rig_ref` and
  `scheimpflug_rig_ringgrid` camera on the device-spec seed.
- Basin: the table above.
- Synthetic guard:
  `seeded_coarse_prior_converges_on_strong_tilt_distortion`
  (`crates/vision-calibration/tests/scheimpflug_intrinsics.rs`) seeds a focal
  length about 9 % low on strong-tilt, strong-distortion data and converges
  to ground truth.

## Open

The synthetic-GT matrix test, a written identifiability and gauge analysis
beyond the summary above, and noise-sensitivity numbers.
