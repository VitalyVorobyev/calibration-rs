# RTV3D-JOINT-LASERLINE-APP Joint App Topology

## Scope

Added a first-class `rig_handeye_laserline` topology for the app and pipeline.
The app path now runs the same stage structure as the V5 benchmark:
`RigHandeye -> RigLaserlineDevice -> optimize_rig_handeye_laserline`, with the
rtv3d preset using EyeToHand, ChESS threshold 30.0, Scheimpflug manual seeds,
fixed `p1/p2/k3`, robot deltas, and a final joint BA.

The rtv3d app preset fixes `cx/cy` during the joint stage. An unconstrained
all-20-pose app run reached 0.959 px, but it moved `cy` to 545-625 px on a
720x540 ROI, so that variant is not the shipped preset.

## Files Changed

- `crates/vision-calibration-dataset`: added `Topology::RigHandeyeLaserline`
  and validation rules.
- `crates/vision-calibration-pipeline`: added the joint pipeline module,
  dataset-runner conversion, export construction, and regression tests.
- `crates/vision-calibration-optim`: made the joint dataset/view serializable
  for pipeline config/export use.
- `crates/vision-calibration`: re-exported the joint topology API.
- `app/src-tauri/src/run.rs`: wired default config, run dispatch, manifest
  export, and the local rtv3d ignored acceptance path.
- `app/src/workspaces/RunWorkspace`, `app/src/store`, `app/src/schemas`,
  `xtask/src/emit_schemas.rs`: added topology registry, preset, labels, and
  schema output.

## Validation Run

- PASS: `cargo fmt --all`
- PASS: `cargo check --workspace --all-features`
- PASS: `cargo test -p vision-calibration-dataset laser -- --nocapture`
- PASS: `cargo test -p vision-calibration-pipeline rig_handeye_laserline -- --nocapture`
- PASS: `cargo xtask emit-schemas --check`
- PASS: `npm --prefix app run build`
- PASS: `cargo test --manifest-path app/src-tauri/Cargo.toml rtv3d_laser_end_to_end -- --ignored --nocapture`

The ignored rtv3d app test now takes about 7 minutes in debug mode because it
runs frozen diagnostic plus full joint BA.

## Results

- Frozen upstream diagnostic after the pose-chain fix: 1.616889 px target
  reprojection, mean laser point-to-plane 0.0286-0.0503 mm.
- Joint app preset with fixed `cx/cy`: 1.124678 px target reprojection over
  20 views, mean laser point-to-plane 0.0410-0.0546 mm.
- Unconstrained joint app diagnostic: 0.959179 px, but nonphysical principal
  point drift, so retained only as evidence of a local-minimum valley.

## Follow-Ups / Remaining Risks

- The app preset still misses the <0.4 px target. The remaining floor appears
  tied to target reprojection/intrinsics/modeling, not laser plane fitting.
- Add a lighter app-side joint smoke fixture or a split ignored test so the
  existing local rtv3d acceptance does not always pay the full joint BA cost.
