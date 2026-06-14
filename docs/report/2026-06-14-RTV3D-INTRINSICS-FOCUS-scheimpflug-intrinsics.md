# RTV3D-INTRINSICS-FOCUS - Scheimpflug Intrinsics

## Scope

- Added a private bench diagnostic command:
  `calib-bench diagnose intrinsics --dataset rtv3d --registry crates/vision-calibration-bench/registry/private.json`.
- Fixed hierarchical intrinsic-floor analysis so rig exports with
  `sensors: Some(_)` project through the Scheimpflug tilted sensor model,
  not the pinhole-only camera.
- Implemented a staged/multistart per-camera Scheimpflug intrinsic solve:
  pose-only fixed-camera polish, robust camera+pose solve, then L2 polish.
  The gate case keeps `cx/cy` fixed at ROI center, fixes `p1/p2/k3=0`,
  frees `fx/fy/k1/k2/tau_x/tau_y`, and recomputes raw residuals from
  `compute_planar_target_residuals`.

## Files Changed

- `crates/vision-calibration-bench/src/run.rs`
- `crates/vision-calibration-bench/src/bin/calib_bench.rs`
- `crates/vision-calibration-pipeline/src/analysis/mod.rs`
- `crates/vision-calibration-pipeline/src/analysis/tests.rs`
- `docs/backlog.md`
- `docs/report/2026-06-14-RTV3D-INTRINSICS-FOCUS-scheimpflug-intrinsics.md`

## RTV3D Result

Final command:

```bash
cargo run -p vision-calibration-bench --features tier-b --no-default-features --bin calib-bench -- \
  diagnose intrinsics \
  --dataset rtv3d \
  --registry crates/vision-calibration-bench/registry/private.json \
  --json-out target/rtv3d-intrinsics-diagnose.json
```

The command applies ChESS absolute threshold `30.0` when the registry does not
specify a detector threshold. All observed/projected coordinates are ROI-local
720x540 coordinates.

| camera | used | raw mean px | rms px | median px | p95 px | max px | <=0.4 | >2 | >5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cam0 | 19/20 | 0.82455 | 0.95959 | 0.82079 | 1.58720 | 4.29748 | 223/997 | 17 | 0 |
| cam1 | 20/20 | 0.82108 | 0.97038 | 0.76233 | 1.65846 | 5.91561 | 205/1000 | 28 | 1 |
| cam2 | 18/20 | 0.74713 | 0.89290 | 0.67917 | 1.69374 | 3.22887 | 224/868 | 22 | 0 |
| cam3 | 17/20 | 0.76791 | 0.88890 | 0.71704 | 1.59401 | 3.08534 | 176/807 | 10 | 0 |
| cam4 | 20/20 | 0.89116 | 1.14607 | 0.76566 | 2.18016 | 6.61701 | 220/1034 | 64 | 4 |
| cam5 | 20/20 | 1.19934 | 1.60950 | 0.90665 | 3.52785 | 7.58522 | 219/1210 | 187 | 13 |

Gate result: **FAIL**. No camera reaches the raw all-corner mean gate
`<0.4 px`.

## Intrinsics

| camera | fx | fy | cx | cy | k1 | k2 | p1 | p2 | k3 | tau_x | tau_y |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cam0 | 1922.20 | 1950.58 | 360.00 | 270.00 | -0.18286 | 1.84751 | 0 | 0 | 0 | -0.21676 | 0.01154 |
| cam1 | 1877.46 | 1894.76 | 360.00 | 270.00 | -0.24341 | 2.64857 | 0 | 0 | 0 | -0.23341 | 0.00665 |
| cam2 | 1882.31 | 1921.98 | 360.00 | 270.00 | -0.14728 | 1.08130 | 0 | 0 | 0 | -0.25126 | 0.00443 |
| cam3 | 1966.35 | 1991.94 | 360.00 | 270.00 | -0.15270 | 0.72143 | 0 | 0 | 0 | -0.19361 | 0.00393 |
| cam4 | 1891.20 | 1921.06 | 360.00 | 270.00 | -0.11948 | 0.05931 | 0 | 0 | 0 | -0.21793 | -0.00516 |
| cam5 | 1725.97 | 1786.82 | 360.00 | 270.00 | 0.10793 | -4.12718 | 0 | 0 | 0 | -0.30279 | -0.01061 |

The expected pattern holds: `tau_x` is material, while `tau_y` remains close to
zero. `cx/cy` remain exactly at the ROI center in the gate case.

## Diagnostics

- ChESS threshold `30.0` improves the floor versus the detector default run
  (previous rough means were 0.914, 1.051, 1.016, 0.967, 1.193, 1.418 px),
  but it does not reach the `<0.4 px` gate.
- 95% trimmed means are still 0.674–1.027 px, so the failure is not just a few
  severe outliers.
- Freeing `cx/cy` changes mean by only about 0.003–0.014 px and usually moves
  the principal point tens to hundreds of pixels from center.
- Freeing `k3` produces very large nonphysical `k3` values with negligible
  improvement.
- Freeing `p1/p2` also gives only hundredths or less of improvement. Tangential
  distortion is not the missing term for this gate.
- Worst-pose means range from 0.90 to 1.50 px, so the residual floor is spread
  across poses rather than a single broken view.

## Validation Run

- PASS: `cargo fmt --all`
- PASS: `cargo test -p vision-calibration-pipeline analysis:: --all-features`
- PASS: `cargo test -p vision-calibration-bench --features tier-b --no-default-features --lib intrinsics -- --nocapture`
- PASS: `cargo test -p vision-calibration-bench --features tier-b --no-default-features --bins`
- PASS: `cargo clippy --workspace --all-targets --all-features -- -D warnings`
- PASS: `cargo test --workspace --all-features`
- PASS: final rtv3d diagnostic command above

## Follow-Ups

- Next likely blocker is target corner localization quality and/or a systematic
  unmodeled image-space residual field, not the hand-eye or rig pose chain.
- Inspect per-corner residual vectors on the threshold-30 detections, especially
  cam5 and cam4 tails, before adding more camera model parameters.
- If residual vector fields remain structured after detector cleanup, test M2
  thin-prism or rational radial terms as diagnostic-only private variants.
