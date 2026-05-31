# BENCH-W2D Dashboard And Stage Diagnostics

## Scope

- Added compact benchmark-record mode to `tools/calibration-viewer`, alongside the existing geometry manifest viewer.
- Added robot-pose correction summaries to `BenchRecord` and Markdown reports, using degrees for rotation and millimetres for translation.
- Added `calib-bench diagnose stages` to profile target detection and laser extraction without running calibration solvers.
- Ran focused diagnostics for puzzle runtime, public hand-eye baselines, and the private ChArUco dataset.

## Files Changed

- `crates/vision-calibration-bench/src/record.rs`
- `crates/vision-calibration-bench/src/run.rs`
- `crates/vision-calibration-bench/src/bin/calib_bench.rs`
- `tools/calibration-viewer/src/main.tsx`
- `tools/calibration-viewer/src/schema.ts`
- `tools/calibration-viewer/src/styles.css`
- `tools/calibration-viewer/tests/schema.test.ts`
- `tools/calibration-viewer/tests/viewer.spec.ts`
- `docs/backlog.md`
- `docs/report/2026-05-31-bench-w2d-dashboard-stage-diagnostics.md`

## Validation Run

- PASS: `cargo fmt -p vision-calibration-bench -- --check`
- PASS: `cargo test -p vision-calibration-bench --offline`
- PASS: `cargo clippy -p vision-calibration-bench --all-targets --features tier-b --offline -- -D warnings`
- PASS: `cargo test -p vision-calibration-bench --features tier-b --offline`
- PASS: `cargo build -p vision-calibration-bench --features "tier-b laser"`
- PASS: `npm run test` in `tools/calibration-viewer`
- PASS: `npm run build` in `tools/calibration-viewer`
- PASS: `npm run e2e` in `tools/calibration-viewer`
- PASS: Playwright screenshot smoke check for `?bench=/bench-record.json` dashboard mode.

## Findings

- `130x130_puzzle` target detection works on all six camera tiles, but is the current runtime bottleneck: one image per camera took 37.601 s total, with per-tile target detection from 3.469 s to 8.559 s. Laser extraction works along columns and took 305 ms total for the same one-image-per-camera profile.
- `kuka_1` default hand-eye remains 1.193 px versus 0.157 px intrinsic floor. Robot correction visibility shows 34.893 mm mean and 41.232 mm max translation correction. Loose robot priors reduce hand-eye to 0.401 px, but require 38.298 mm mean and 41.671 mm max translation correction.
- `ds8` default hand-eye remains 4.855 px versus 0.186 px intrinsic floor. Robot correction visibility shows 48.458 mm mean and 354.979 mm max translation correction. Loose robot priors only reduce hand-eye to 4.125 px and increase max translation correction to 462.218 mm.
- `charuco_handeye_3536` current run reports 1.05383 px intrinsic floor and 10.52281 px hand-eye mean. Detection coverage is only 6,876 / 56,144 expected features (12.25%), so the high ChArUco floor is now tied to detection/feature coverage before hand-eye optimization.
- Internal transform units remain metres for existing algorithm contracts. User-facing benchmark correction metrics are reported in millimetres; making millimetres the canonical spatial unit should be a separate API/schema decision.

## Follow-Ups / Remaining Risks

- Add configurable robot-correction acceptance thresholds and fail/warn states in reports.
- Profile and optimize puzzleboard detection before attempting a full unseeded puzzle solve again.
- Investigate ChArUco coverage and corner filtering against the user's <0.4 px reference result.
- Isolate hand-eye objective/initialization with fixed intrinsics and fixed rig extrinsics, because large robot-pose corrections are masking the current chain error rather than proving a good solve.
