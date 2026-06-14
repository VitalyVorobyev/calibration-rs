# rtv3d Calibration Quality Update

## Current Result Summary

| path | target reprojection | laser plane residual | note |
|---|---:|---:|---|
| app rig hand-eye warm start | 1.616889 px mean | n/a | centered ROI-local intrinsics; ChESS threshold 30 |
| app frozen upstream diagnostic | 1.616889 px mean | 0.0286-0.0503 mm mean | corrected pose-chain; no longer the old 54 px drift |
| bench V5 joint BA | 1.193 px mean / 1.207 px RMS | 0.018-0.035 mm RMS | benchmark subset/path from V5 report |
| app joint preset | 1.124678 px mean | 0.0410-0.0546 mm mean | all 20 views, `cx/cy` fixed during joint BA |
| app joint unconstrained diagnostic | 0.959179 px mean | 0.0355-0.0478 mm mean | rejected as default: `cy` drifted to 545-625 px |

The 54 px app example was a frozen-path pose-chain bug. The corrected frozen
diagnostic now preserves upstream optimized target poses and applies upstream
robot deltas in the fallback chain. The remaining quality gap is the target
reprojection floor, not the laser plane fit.

## App Joint Intrinsics

ROI-local image center for the rtv3d camera tiles is approximately `(360, 270)`.
The shipped app joint preset fixes `cx/cy` at the hand-eye warm-start values
during joint BA and keeps `p1/p2/k3` fixed at zero.

| cam | fx | fy | cx | cy | p1 | p2 | tau_x | tau_y |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| cam0 | 2116.23 | 2136.59 | 356.52 | 245.87 | 0 | 0 | -0.08522 | 0.00832 |
| cam1 | 2092.17 | 2108.76 | 370.86 | 261.14 | 0 | 0 | -0.10216 | 0.00164 |
| cam2 | 2078.15 | 2099.21 | 372.02 | 275.42 | 0 | 0 | -0.09595 | 0.00370 |
| cam3 | 2068.02 | 2088.87 | 360.05 | 282.19 | 0 | 0 | -0.09109 | 0.00153 |
| cam4 | 2073.64 | 2099.05 | 343.18 | 275.26 | 0 | 0 | -0.06808 | -0.00000 |
| cam5 | 2089.99 | 2111.19 | 345.80 | 258.46 | 0 | 0 | -0.09251 | -0.00178 |

`tau_x` is the dominant Scheimpflug term; `tau_y` remains close to zero.

## Laser Diagnostics

| cam | app joint target mean px | app joint laser mean mm | laser points |
|---|---:|---:|---:|
| cam0 | 1.103 | 0.0410 | 8876 |
| cam1 | 1.063 | 0.0425 | 8694 |
| cam2 | 0.969 | 0.0413 | 8400 |
| cam3 | 0.975 | 0.0462 | 8344 |
| cam4 | 1.158 | 0.0546 | 9169 |
| cam5 | 1.376 | 0.0474 | 9240 |

The new 3D viewer overlay renders six active-pose target-plane intersections
with the solved laser planes. This should make the expected hexagon easier to
read than the translucent full-plane quads.

## Lessons Learned

- The app's 54 px rtv3d laser reprojection was coherent pose drift, not
  detector scatter. Preserving upstream `rig_se3_target` fixes that path.
- The full all-20-pose joint solve exposes a principal-point local-minimum
  valley. Letting `cx/cy` float can reduce reprojection numerically while
  making the solution physically implausible.
- Laser residuals are consistently below the 0.1 mm target in both bench and
  app runs. The unresolved blocker is the target reprojection floor.
- The next likely work is a controlled reprojection diagnostic: detector
  quality, robot-pose priors/deltas, target model consistency, and missing
  camera model terms such as thin-prism.

## Validation Run

- PASS: `cargo fmt --all`
- PASS: `cargo check --workspace --all-features`
- PASS: `cargo test -p vision-calibration-dataset laser -- --nocapture`
- PASS: `cargo test -p vision-calibration-pipeline dataset_runner::laser::tests::rig_laserline_ -- --nocapture`
- PASS: `cargo test -p vision-calibration-pipeline rig_handeye_laserline -- --nocapture`
- PASS: `cargo xtask emit-schemas --check`
- PASS: `npm --prefix app run build`
- PASS: `cargo test --manifest-path app/src-tauri/Cargo.toml rtv3d_laser_end_to_end -- --ignored --nocapture`
- PASS: `cargo clippy --workspace --all-targets --all-features -- -D warnings`
- PASS: `cargo test --workspace --all-features`
- PASS: `cargo run -p vision-calibration-bench --features 'tier-b laser' --bin calib-bench -- run --dataset rtv3d --registry crates/vision-calibration-bench/registry/private.json`
