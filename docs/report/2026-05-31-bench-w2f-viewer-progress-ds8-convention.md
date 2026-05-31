# BENCH-W2F Viewer, Progress, Detector, And DS8 Convention

## Scope

- Added `scripts/bench-viewer.sh <dataset-id>` to run a benchmark record and open `tools/calibration-viewer`.
- Added stderr progress for target detection, laser extraction, and calibration stages while keeping stdout valid JSON.
- Added optional `BenchRecord.artifacts` with camera matrices, Brown-Conrady distortion, Scheimpflug tilt, and named SE(3) transforms.
- Added a clean calibration-artifacts panel to the benchmark dashboard.
- Switched simple chessboard detection to the topological graph builder; ChArUco and puzzleboard dispatch are unchanged.
- Corrected DS8 to the physical EyeInHand setup with `robot_cali.txt` parsed as `gripper_se3_base`.

## Files Changed

- `scripts/bench-viewer.sh`
- `crates/vision-calibration-bench/registry/public.json`
- `crates/vision-calibration-bench/src/bin/calib_bench.rs`
- `crates/vision-calibration-bench/src/detect.rs`
- `crates/vision-calibration-bench/src/record.rs`
- `crates/vision-calibration-bench/src/registry.rs`
- `crates/vision-calibration-bench/src/run.rs`
- `tools/calibration-viewer/src/main.tsx`
- `tools/calibration-viewer/src/schema.ts`
- `tools/calibration-viewer/src/styles.css`
- `tools/calibration-viewer/tests/schema.test.ts`
- `tools/calibration-viewer/tests/viewer.spec.ts`
- `docs/backlog.md`
- `docs/report/2026-05-31-bench-w2f-viewer-progress-ds8-convention.md`

## Validation Run

- PASS: `cargo fmt -p vision-calibration-bench -- --check`
- PASS: `cargo test -p vision-calibration-bench --offline`
- PASS: `cargo test -p vision-calibration-bench --features tier-b --offline`
- PASS: `cargo clippy -p vision-calibration-bench --all-targets --features tier-b --offline -- -D warnings`
- PASS: `cargo build -p vision-calibration-bench --features "tier-b laser"`
- PASS: `npm run test` in `tools/calibration-viewer`
- PASS: `npm run build` in `tools/calibration-viewer`
- PASS: `npm run e2e` in `tools/calibration-viewer`
- PASS: `bash -n scripts/bench-viewer.sh`
- PASS: `calib-bench run --dataset ds8`
- PASS: `calib-bench diagnose handeye --dataset ds8`
- PASS: `calib-bench run --dataset stereo_left`

## Findings

- DS8 is physically EyeInHand. The low-error branch is EyeInHand with the robot file interpreted as `gripper_se3_base`, not EyeToHand.
- Corrected DS8 default: hand-eye `0.26443 px` vs intrinsic floor `0.18709 px`; robot corrections `0.461 / 1.243 mm` translation and `0.103 / 0.214 deg` rotation mean/max.
- Wrong DS8 convention under EyeInHand: hand-eye `1.25888 px`, robot translation correction `25.236 / 97.072 mm`.
- `diagnose handeye` now includes `alternate_mode_inverted_pose`; the compensating EyeToHand case is no longer confused with the physical model.
- Topological chessboard detection keeps `stereo_left` at `0.23388 px` with 100% coverage and reduces DS8 target detection to about `12.5 s`.
- Strict-grid coverage reads 100% for accepted DS8 images because partial/local-grid detections are rejected before the denominator is computed.

## Follow-Ups / Remaining Risks

- Add explicit rejected-image counts to detection reports so strict-grid filtering is obvious in the dashboard.
- Cache detections for diagnostic sweeps; repeated DS8 diagnostics still rerun target detection per case.
- Continue isolating the residual 0.264 px vs 0.187 px DS8 gap with fixed intrinsics/rig and hand-eye-only objective tests.
