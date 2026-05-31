# BENCH-W2H ChArUco Rig-Hand-Eye Correction

## Scope

- Added typed benchmark ChESS detector overrides and wired ChArUco to consume the configured corner detector.
- Added rig-stage reprojection metrics for rig hand-eye reports: Intrinsic, RigExtrinsic, and HandEye.
- Exposed final rig/Scheimpflug hand-eye BA refinement flags for diagnostics while keeping defaults fixed.
- Added robot correction prior flags in compact reports and Markdown output.
- Updated the gitignored private ChArUco registry locally to EyeToHand Scheimpflug staged BA with absolute ChESS threshold 30.

## Files Changed

- `crates/vision-calibration-bench/src/detect.rs`
- `crates/vision-calibration-bench/src/registry.rs`
- `crates/vision-calibration-bench/src/run.rs`
- `crates/vision-calibration-bench/src/record.rs`
- `crates/vision-calibration-bench/src/bin/calib_bench.rs`
- `crates/vision-calibration-pipeline/src/analysis/mod.rs`
- `crates/vision-calibration-pipeline/src/analysis/tests.rs`
- `crates/vision-calibration-pipeline/src/rig_handeye/steps.rs`
- `crates/vision-calibration-pipeline/src/rig_handeye/problem.rs`
- `tools/calibration-viewer/src/main.tsx`
- `tools/calibration-viewer/src/schema.ts`

## Validation Run

- PASS: `cargo fmt -p vision-calibration-bench -- --check`
- PASS: `cargo fmt -p vision-calibration-pipeline -- --check`
- PASS: `cargo test -p vision-calibration-bench --offline`
- PASS: `cargo test -p vision-calibration-pipeline --offline`
- PASS: `cargo clippy -p vision-calibration-bench --all-targets --features tier-b --offline -- -D warnings`
- PASS: `cargo clippy -p vision-calibration-pipeline --all-targets --offline -- -D warnings`
- PASS: `npm run test` in `tools/calibration-viewer`
- PASS: `npm run build` in `tools/calibration-viewer`
- PASS: `calib-bench diagnose stages --dataset charuco_handeye_3536 --registry crates/vision-calibration-bench/registry/private.json`
- PASS: `calib-bench diagnose handeye --dataset charuco_handeye_3536 --registry crates/vision-calibration-bench/registry/private.json`
- PASS: `calib-bench run --dataset charuco_handeye_3536 --registry crates/vision-calibration-bench/registry/private.json --residuals-out /tmp/charuco_handeye_3536-residuals.json`

Private ChArUco result with the new local registry override:

| level | mean px | count |
|---|---:|---:|
| Intrinsic | 0.93638 | 5916 |
| RigExtrinsic | 0.89010 | 5916 |
| HandEye | 0.91058 | 5916 |

Diagnostic sweep highlights:

| case | hand-eye mean px | note |
|---|---:|---|
| default | 0.911 | best fixed-stage default |
| alternate_handeye_mode | 2.043 | worse mode |
| robot_refine_off | 12.507 | robot correction is currently carrying a large error |
| loose_robot_prior | 0.977 | similar fit with larger translation corrections |
| final_rig_tilt_refine | 0.908 | not materially different from default |
| pose_convention_inverted | 2.631 | worse convention |
| alternate_mode_inverted_pose | 0.921 | close to default but with larger corrections |

## Follow-Ups / Remaining Risks

- The ChArUco intrinsic floor is now 0.936 px, still above the external 0.4 px target. Treat this as a detector/model/data finding rather than a hand-eye-only issue.
- Robot-pose corrections exceed the configured priors: max 1.517 deg and 5.316 mm against 0.5 deg / 1 mm.
- Full final BA over intrinsics + rig + Scheimpflug + hand-eye remains out of scope; this task keeps staged BA as the default.
