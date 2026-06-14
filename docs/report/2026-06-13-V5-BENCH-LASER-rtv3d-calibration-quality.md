# V5-BENCH-LASER rtv3d Calibration Quality

## Scope

Implemented the V5 benchmark path for the private rtv3d dataset: target
detection is run once, then `RigHandeye`, `RigLaserlineDevice`, and final
joint hand-eye/laser BA run from the same detected observations. Also added
the app parity hooks needed by the rtv3d presets: detector threshold override,
manual rig-hand-eye intrinsics init, ROI-local image manifests, and a cache-key
marker for the changed detector/ROI contract.

The normal bench run now uses the rtv3d assumptions from the validation notes:
EyeToHand, 5.2 mm ChArUco pitch, ChESS absolute threshold 30.0, Scheimpflug
sensors, `p1/p2 = 0`, free `tau_x/tau_y` in the per-camera stage, and centered
`fx = fy = 2000` seeds.

## Files Changed

- `crates/vision-calibration-bench`: V5 laser/joint runner, typed V5 record
  coverage, and hand-eye diagnostic variants for seed, threshold, tangential,
  Scheimpflug tilt, and robot-pose hypotheses.
- `crates/vision-calibration-dataset`, `crates/vision-calibration-detect`,
  `crates/vision-calibration-pipeline`: detector override schema/application,
  rig-hand-eye manual-init config, ROI-local detection/residual handling.
- `app/src-tauri`, `app/src/workspaces/RunWorkspace`, `app/src/schemas`:
  rtv3d preset parity, ROI-bearing manifests, generated schemas, and the
  ignored local rtv3d test updated to exercise the new seed/threshold path.

## Results

| run | reprojection | laser point-to-plane | note |
|---|---:|---:|---|
| app seeded hand-eye export | 1.617 px mean | n/a | ignored Tauri rtv3d test with threshold/manual-init parity |
| app frozen upstream diagnostic (before pose preservation) | 54.265 px mean | 1.422-1.728 mm mean | old app path recomputed target poses from raw robot poses and ignored upstream optimized per-view poses/deltas |
| bench V5 joint BA | 1.193 px mean / 1.207 px RMS / 8.085 px max | 0.018–0.035 mm RMS | 4 usable laser views, 10,797 laser points |
| best known joint BA floor | 1.16 px mean | 0.017–0.031 mm σ | 2026-06-11 private example run |

The laser criterion passes comfortably: all six planes are below the 0.1 mm
target. The reprojection criterion does not: the current best local floor is
still about 1.16–1.19 px, not the requested <0.4 px.

## Intrinsics

ROI-local image center for the 720x540 tiles is approximately `(360, 270)`.
The V5 solution keeps `cx/cy` near that center and fixes `p1/p2 = 0`.

| cam | fx | fy | cx | cy | p1 | p2 | tau_x | tau_y |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| cam0 | 2036.54 | 2001.09 | 356.19 | 248.19 | 0 | 0 | -0.08332 | 0.00899 |
| cam1 | 2022.94 | 2001.55 | 366.81 | 260.88 | 0 | 0 | -0.07651 | -0.00014 |
| cam2 | 2018.27 | 2004.46 | 367.18 | 273.55 | 0 | 0 | -0.09386 | 0.00609 |
| cam3 | 2047.54 | 2009.43 | 359.42 | 286.74 | 0 | 0 | -0.07030 | 0.00303 |
| cam4 | 2037.78 | 2012.48 | 345.49 | 276.51 | 0 | 0 | -0.03929 | -0.00330 |
| cam5 | 2024.44 | 1997.83 | 348.16 | 262.73 | 0 | 0 | -0.06756 | 0.00188 |

## Rig And Laser Diagnostics

Adjacent camera-center spacing from the V5 `cam*_se3_rig` artifacts is coherent
for the six-device ring: 88.16–89.94 mm, mean 89.27 mm, standard deviation
0.71 mm. This supports the nominal hexagon geometry in the solved rig, but it
does not by itself prove the absolute scale question tracked by V6.

| cam | laser points | plane RMS mm | line RMS px |
|---|---:|---:|---:|
| cam0 | 1832 | 0.0215 | 0.214 |
| cam1 | 1756 | 0.0350 | 0.347 |
| cam2 | 1672 | 0.0195 | 0.194 |
| cam3 | 1801 | 0.0241 | 0.239 |
| cam4 | 1868 | 0.0283 | 0.279 |
| cam5 | 1868 | 0.0175 | 0.173 |

## Lessons Learned

- The low ChESS threshold was a real app hazard; absolute threshold 30.0 avoids
  the false-positive path seen in the rtv3d preset and is now configurable.
- Manual Scheimpflug seeds are necessary for this dataset. The linear estimate
  can land far from the physically plausible `fx/fy` basin; seeded V5 stays in
  the expected 1500–2200 px range with centered principal points.
- ROI coordinates must be solver-local. Mixing full-image coordinates with
  ROI-local calibration data can make good detections look like bad intrinsics.
- The main remaining error is not laser-plane fit. Laser residuals are already
  below target; the unresolved floor is the target reprojection fit.

## Follow-Ups / Remaining Risks

- Run the new `calib-bench diagnose handeye` sweep on rtv3d to isolate seed,
  threshold, `p1/p2`, `tau`, and robot-pose sensitivities.
- Keep the frozen-upstream app path as a diagnostic only; the quality preset
  should use the first-class joint hand-eye laserline app path.
- If detector and robot-pose variants do not move the floor, prioritize M2
  thin-prism support for Scheimpflug optics before adding more local tuning.

## Validation Run

- PASS: `cargo fmt --all`
- PASS: `cargo check -p vision-calibration-dataset -p vision-calibration-detect -p vision-calibration-pipeline --all-features`
- PASS: `cargo check --manifest-path app/src-tauri/Cargo.toml`
- PASS: `npm run build` from `app/`
- PASS: `cargo check -p vision-calibration-bench --features 'tier-b laser'`
- PASS: `cargo check -p vision-calibration-bench`
- PASS: `cargo test -p vision-calibration-dataset -p vision-calibration-detect -p vision-calibration-pipeline --all-features`
- PASS: `cargo test -p vision-calibration-bench --features 'tier-b laser' -- --nocapture`
- PASS: `cargo test --manifest-path app/src-tauri/Cargo.toml`
- PASS: `cargo test --manifest-path app/src-tauri/Cargo.toml rtv3d_laser_end_to_end -- --ignored --nocapture`
- PASS: `cargo run -p vision-calibration-bench --features 'tier-b laser' --bin calib-bench -- run --dataset rtv3d --registry crates/vision-calibration-bench/registry/private.json`
- PASS: `cargo xtask emit-schemas --check`
- PASS: `cargo clippy --workspace --all-targets --all-features -- -D warnings`
- PASS: `cargo test --workspace --all-features`

`npm run build` still emits the existing Vite large-chunk warning.
