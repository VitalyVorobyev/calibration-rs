# BENCH-W2C Compact Puzzle Diagnostics

## Scope

Implemented benchmark schema v3 for compact default records and optional full
residual sidecars. Added Markdown report rendering, deterministic hand-eye
diagnostic sweeps, puzzleboard detector dispatch, typed hand-eye registry
overrides, optional `vision-metrology` laser extraction from tag `v0.1.0`, and
local private registry wiring for `130x130_puzzle`.

The implementation intentionally does not add seeded fallback for the puzzle
rig. The unseeded run is treated as the benchmark result.

## Files Changed

- `crates/vision-calibration-bench/src/record.rs`
- `crates/vision-calibration-bench/src/bin/calib_bench.rs`
- `crates/vision-calibration-bench/src/detect.rs`
- `crates/vision-calibration-bench/src/registry.rs`
- `crates/vision-calibration-bench/src/run.rs`
- `crates/vision-calibration-bench/Cargo.toml`
- `Cargo.lock`
- `docs/backlog.md`
- `docs/report/2026-05-31-bench-w2c-compact-puzzle-diagnostics.md`

`crates/vision-calibration-bench/registry/private.json` was updated locally and
remains gitignored.

## Validation Run

- PASS: `cargo fmt -p vision-calibration-bench -- --check`
- PASS: `cargo clippy -p vision-calibration-bench --all-targets --features tier-b --offline -- -D warnings`
- PASS: `cargo test -p vision-calibration-bench --offline`
- PASS: `cargo build -p vision-calibration-bench --features "tier-b laser"`
- PASS: `cargo clippy -p vision-calibration-bench --all-targets --features "tier-b laser" -- -D warnings`

Public dataset validation:

| dataset | status | fit px | headline px | levels |
|---|---|---:|---:|---|
| `stereo_left` | OK | 0.23388 | 0.23388 | intrinsic=0.23388 |
| `stereo_right` | OK | 0.24415 | 0.24415 | intrinsic=0.24415 |
| `stereo_rig` | OK | 0.25031 | 0.25031 | intrinsic=0.23902, rig_extrinsic=0.25031 |
| `kuka_1` | OK | 1.19335 | 1.19335 | intrinsic=0.15722, hand_eye=1.19335 |
| `stereo_charuco` | OK | 0.55087 | 0.55087 | intrinsic=0.40156, rig_extrinsic=0.55087 |
| `ds8` | OK | 4.85526 | 4.85526 | intrinsic=0.18596, hand_eye=4.85526 |

Private dataset validation:

| dataset | status | result |
|---|---|---|
| `charuco_handeye_3536` | OK | fit=10.52753 px, headline=10.52753 px, intrinsic=1.05384 px, hand_eye=10.52753 px |
| `130x130_puzzle` | FAIL/TIME | unseeded run consumed CPU for roughly 12 minutes with no JSON output before being stopped |

CLI sidecar/report smoke test:

- PASS: `calib-bench run --dataset stereo_rig --residuals-out /tmp/calib-bench-validation/stereo_rig.residuals.json`
- PASS: `calib-bench report /tmp/calib-bench-validation/stereo_rig.record.json`
- Confirmed default record schema is v3 and omits full `residuals`; sidecar schema is v3 and carries full residual vectors.

Hand-eye diagnostics:

| dataset | case | intrinsic px | hand-eye px | ratio |
|---|---|---:|---:|---:|
| `kuka_1` | default | 0.157 | 1.193 | 7.590 |
| `kuka_1` | robot_refine_off | 1.734 | 11.313 | 6.524 |
| `kuka_1` | loose_robot_prior | 0.145 | 0.401 | 2.755 |
| `kuka_1` | pose_convention_inverted | 2.637 | 11.203 | 4.249 |
| `ds8` | default | 0.186 | 4.855 | 26.109 |
| `ds8` | robot_refine_off | 0.186 | 14.864 | 79.933 |
| `ds8` | loose_robot_prior | 0.186 | 4.125 | 22.180 |
| `ds8` | pose_convention_inverted | 0.186 | 3.434 | 18.465 |
| `charuco_handeye_3536` | default | 1.054 | 10.519 | 9.981 |
| `charuco_handeye_3536` | robot_refine_off | 1.054 | 61.310 | 58.181 |
| `charuco_handeye_3536` | loose_robot_prior | 1.054 | 11.418 | 10.835 |
| `charuco_handeye_3536` | pose_convention_inverted | 1.054 | 9.812 | 9.311 |

## Follow-Ups / Remaining Risks

- `130x130_puzzle` is not production-ready yet. The first blocker is runtime:
  fixed-board puzzle detection over the tiled 130x130 frames did not reach a
  calibration result in an interactive validation window.
- The public and private hand-eye datasets remain chain-limited. `kuka_1` moves
  materially toward the intrinsic floor with loose robot priors; `ds8` and
  `charuco_handeye_3536` improve under some sweeps but remain far above their
  intrinsic floors.
- Laser extraction currently records extraction counts and timing only. Plane
  residuals remain intentionally unset until a unit-labelled laser-plane fit is
  added.
