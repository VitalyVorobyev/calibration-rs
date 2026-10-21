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

The fix mask `DistortionFixMask { k1, k2, k3, p1, p2 }` is translated by
name onto each model's packed layout (`fix_mask_indices`): Rational8's
`k4..k6` follow `k3`, ThinPrism9's `s1..s4` are fixed when both `p1` and
`p2` are, and Division1's `lambda` follows `k1`. The staging masks use the
same translation, so every model keeps the defaults (`fix_k3`, tangential
fixed).

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

| cell | `rtv3d_ref` | `rtv3d_ringgrid` |
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

## Evidence

- Acceptance: `calib-bench accept` gates every `rtv3d_ref` and
  `rtv3d_ringgrid` camera on the device-spec seed.
- Basin: the table above.
- Synthetic guard:
  `seeded_coarse_prior_converges_on_strong_tilt_distortion`
  (`crates/vision-calibration/tests/scheimpflug_intrinsics.rs`) seeds a focal
  length about 9 % low on strong-tilt, strong-distortion data and converges
  to ground truth.

## Open

The synthetic-GT matrix test, a written identifiability and gauge analysis
beyond the summary above, and noise-sensitivity numbers.
