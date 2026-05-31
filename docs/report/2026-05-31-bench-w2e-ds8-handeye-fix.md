# BENCH-W2E DS8 Hand-Eye Fix

## Scope

- Verified DS8 board geometry from the registry: `rows=10`, `cols=14`, `cell_size_m=0.052`.
- Added `strict_grid` checkerboard handling so DS8 rejects partial/local-grid detections instead of accepting shifted target frames.
- Switched DS8 from the default `eye_in_hand` assumption to `eye_to_hand`.
- Extended `diagnose handeye` with an `alternate_handeye_mode` case so mode mistakes are visible in the report.

## Files Changed

- `crates/vision-calibration-bench/registry/public.json`
- `crates/vision-calibration-bench/src/bin/calib_bench.rs`
- `crates/vision-calibration-bench/src/detect.rs`
- `crates/vision-calibration-bench/src/fixtures.rs`
- `crates/vision-calibration-bench/src/registry.rs`
- `crates/vision-calibration-bench/src/run.rs`
- `docs/backlog.md`
- `docs/report/2026-05-31-bench-w2e-ds8-handeye-fix.md`

## Validation Run

- PASS: `cargo fmt -p vision-calibration-bench -- --check`
- PASS: `cargo test -p vision-calibration-bench --offline`
- PASS: `cargo clippy -p vision-calibration-bench --all-targets --features tier-b --offline -- -D warnings`
- PASS: `cargo test -p vision-calibration-bench --features tier-b --offline`
- PASS: `calib-bench run --dataset ds8`
- PASS: `calib-bench diagnose handeye --dataset ds8`
- PASS: `calib-bench run --dataset stereo_left` smoke test for non-strict checkerboard datasets.

## Findings

- Original public-registry DS8 run: hand-eye mean `4.855 px`, intrinsic floor `0.186 px`, robot translation correction `48.458 / 354.979 mm` mean/max.
- Strict-grid filtering alone reduced the run to hand-eye mean `1.259 px`, robot translation correction `25.236 / 97.072 mm` mean/max.
- Switching DS8 to `eye_to_hand` with strict-grid detections reduced the run to hand-eye mean `0.26378 px` versus `0.18709 px` intrinsic floor.
- Corrected DS8 robot correction is now `0.341 / 1.267 mm` translation mean/max and `0.123 / 0.260 deg` rotation mean/max.
- The diagnostic sweep proves the previous effective mode is wrong: `alternate_handeye_mode` reports `1.259 px` and `25.236 / 97.072 mm` robot translation correction.

## Follow-Ups / Remaining Risks

- Add report-level warning thresholds for robot pose corrections.
- Consider a first-class known-board checkerboard detector API so strict-grid behavior is not benchmark-only.
- Continue investigating why `robot_refine_off` is still `0.941 px`; the corrected default is usable, but fixed-robot hand-eye remains above the intrinsic floor.
