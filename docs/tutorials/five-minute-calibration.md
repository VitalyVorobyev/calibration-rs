# Five-minute calibration

> Onboarding tutorial. Runnable companion:
> [`stereo_charuco_session.rs`](../../crates/vision-calibration/examples/stereo_charuco_session.rs).

## Why

The fastest way to develop intuition for `calibration-rs` is to run a real
end-to-end calibration on a public dataset. This tutorial gets you from
"I just cloned the repo" to "I have calibrated intrinsics, distortion, and
multi-camera extrinsics" in five minutes.

## Mental model

A calibration is a **session**: a state container plus a sequence of step
functions you invoke in order.

```
Session::new()
  → set_input(observations)
  → step_intrinsics_init_all     ← Zhang's method per camera
  → step_intrinsics_optimize_all ← non-linear refinement
  → step_rig_init                ← linear cam_se3_rig + rig_se3_target
  → step_rig_optimize            ← joint bundle adjustment
  → export()                     ← stable JSON-serializable result
```

Each step reads the relevant fields from the session state and writes its
own output back; you can inspect the intermediate state between any two
steps. There is also a `run_calibration` convenience wrapper that calls
the canonical step sequence.

## Walkthrough

We'll calibrate a 2-camera ChArUco rig from the dataset bundled in `data/`.

### 1. Run the canned example

```bash
cargo run -p vision-calibration --example stereo_charuco_session --release -- --max-views 8
```

This compiles, detects ChArUco corners in 8 image pairs, and runs the
canonical pipeline. Expected output (excerpt):

```
--- Step 2: Per-Camera Intrinsics Optimization ---
  Camera 0: fx=21896.5, fy=21799.3, cx=918.4, cy=731.5, reproj_err=0.394px
  Camera 1: fx=21652.4, fy=21562.0, cx=1304.9, cy=747.6, reproj_err=0.396px

--- Step 4: Rig Bundle Adjustment ---
  Rig BA mean reprojection error: 0.5435 px
    Camera 0: 0.5969 px
    Camera 1: 0.4869 px
  Rig baseline (after BA): |t(cam1->rig)| = 0.1153 m (115.27 mm)
```

The full source is at
[`crates/vision-calibration/examples/stereo_charuco_session.rs`](../../crates/vision-calibration/examples/stereo_charuco_session.rs).

### 2. Walk the source

The interesting part is just ten lines:

```rust
let mut session = CalibrationSession::<RigExtrinsicsProblem>::new();
session.set_input(input)?;

step_intrinsics_init_all(&mut session, None)?;
step_intrinsics_optimize_all(&mut session, None)?;
step_rig_init(&mut session)?;
step_rig_optimize(&mut session, None)?;

let export = session.export()?;
let mean_reproj = export.mean_reproj_error;
let baseline = (export.cam_se3_rig[1].translation.vector
              - export.cam_se3_rig[0].translation.vector).norm();
```

Notice:

- `RigExtrinsicsProblem` is a phantom marker type. It picks the input,
  state, and export schemas via the `ProblemType` trait — the session
  is fully typed.
- `step_*` functions take an `Option<*Options>` for tweaks
  (`max_iters`, `verbosity`, etc.). Passing `None` uses the config's
  defaults.
- `export()` returns a `RigExtrinsicsExport` — a stable, JSON-serializable
  view of the result. It's the canonical handoff to other tools.

### 3. Choose a different problem type

Different camera setups call for different problem types. The pattern is
the same; only the input shape and the step list change.

| Problem type | Input | Steps |
|---|---|---|
| `PlanarIntrinsicsProblem` | `PlanarDataset` | `step_init` → `step_optimize` |
| `SingleCamHandeyeProblem` | `SingleCamHandeyeInput` | 4 steps (intrinsics ×2, hand-eye ×2) |
| `RigExtrinsicsProblem` | `RigDataset<NoMeta>` | 4 steps (this tutorial). Pinhole or Scheimpflug rig — set `RigExtrinsicsConfig::sensor` to `SensorMode::Pinhole` (default) or `SensorMode::Scheimpflug { … }`. |
| `RigHandeyeProblem` | `RigDataset<RobotPoseMeta>` | 6 steps |
| `LaserlineDeviceProblem` | `LaserlineDataset` | `step_init` → `step_optimize` |
| `ScheimpflugIntrinsicsProblem` | `PlanarDataset` (with tilt) | `step_init` → `step_optimize` |
| `RigScheimpflugHandeyeProblem` | `RigDataset<RobotPoseMeta>` (Scheimpflug) | 6 steps |
| `RigLaserlineDeviceProblem` | `RigLaserlineDeviceInput` | `step_init` → `step_optimize` |

Each problem type's facade module re-exports its types and step functions
(see `vision_calibration::rig_extrinsics`, etc.).

## Common variations

### Pre-detect corners off-line

The `stereo_charuco_session` example calls `calib-targets` to detect
corners as part of loading. For production use you typically pre-detect
once and persist the resulting `RigExtrinsicsInput` (which is pure data:
JSON-serialisable, no images needed). Run calibrations against the cached
detections.

### Use the convenience wrapper

```rust
vision_calibration::rig_extrinsics::run_calibration(&mut session)?;
```

is equivalent to the four step calls above with default options.

### Inspect the session log

Every step appends a `LogEntry` to `session.log`. The log records what
auto-init produced or which fields came from manual seeds:

```rust
for entry in &session.log {
    println!("[{}] {:?} {}",
        entry.operation, entry.success, entry.notes.as_deref().unwrap_or(""));
}
```

### Snapshot and resume

`CalibrationSession` is JSON-serialisable. Persist after `step_rig_init`
and resume later for a fresh `step_rig_optimize` with different solver
options.

## What to read next

- [Manual initialization](./manual-init.md) — seed datasheet values into
  the pipeline.
- [Per-feature residuals](./per-feature-residuals.md) — drill into the
  per-corner errors after a calibration finishes.
- [Puzzle 130×130 walkthrough](./puzzle-130x130-walkthrough.md) — the same
  pattern at industrial scale (Scheimpflug rig, laser plane, robot
  hand-eye, real data).
- [ADR 0007](../adrs/0007-session-framework.md) — why the API is shaped
  around sessions and external step functions instead of a single
  `Calibrator::run()` call.
