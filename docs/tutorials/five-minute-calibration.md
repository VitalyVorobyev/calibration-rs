# Five-minute calibration

> Onboarding tutorial. Runnable companion:
> [`stereo_session.rs`](../../crates/vision-calibration/examples/stereo_session.rs).

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

We'll calibrate a 2-camera rig from the chessboard corners committed in
`regression/observations/stereo_rig.json`: 20 views of a 7×11 board with
30 mm squares, already detected, so no images are needed.

### 1. Run the canned example

```bash
cargo run -p vision-calibration --example stereo_session --release
```

This loads the detected corners and runs the canonical pipeline. Expected
output (excerpt):

```
--- Step 2: Per-Camera Intrinsics Optimization ---
  Camera 0: fx=713.8, fy=724.7, cx=523.4, cy=286.0, reproj_err=0.234px
  Camera 1: fx=724.8, fy=735.6, cx=513.9, cy=293.0, reproj_err=0.244px

--- Step 4: Rig Bundle Adjustment ---
  Rig BA mean reprojection error: 0.2496 px
    Camera 0: 0.2448 px
    Camera 1: 0.2544 px
  Rig baseline (after BA): |c1 - c0| = 0.1895 m (189.45 mm)

--- Uncertainty ---
  Baseline: 189.45 ± 0.112 mm
  Corner noise σ ≈ 0.189 px (6160 residuals)
```

The `±` comes from the export's `uncertainty`: the covariance of the
estimated parameters, here of camera 1's `cam_se3_rig`. It is conditional on
the intrinsics, which the rig solve holds fixed by default. See the book
chapter "Parameter Uncertainty".

The full source is at
[`crates/vision-calibration/examples/stereo_session.rs`](../../crates/vision-calibration/examples/stereo_session.rs).

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
let centre = |i: usize| export.cam_se3_rig[i].inverse().translation.vector;
let baseline = (centre(1) - centre(0)).norm();
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
| `RigHandeyeProblem` | `RigDataset<RobotPoseMeta>` | 6 steps. Pinhole or Scheimpflug rig — set `RigHandeyeConfig::sensor` to `SensorMode::Pinhole` (default) or `SensorMode::Scheimpflug { … }`. |
| `LaserlineDeviceProblem` | `LaserlineDataset` | `step_init` → `step_optimize` |
| `ScheimpflugIntrinsicsProblem` | `PlanarDataset` (with tilt) | `step_init` → `step_optimize` |
| `RigLaserlineDeviceProblem` | `RigLaserlineDeviceInput` | `step_init` → `step_optimize` |

Each problem type's facade module re-exports its types and step functions
(see `vision_calibration::rig_extrinsics`, etc.).

## Common variations

### Detect corners from images

The `planar_real` example detects a chessboard in the `data/kuka_1` images
with `calib-targets` and calibrates the camera, reporting each parameter
with its standard deviation (`fx = 2057.8 ± 0.3 px`):

```bash
cargo run -p vision-calibration --example planar_real --release
```

For production use you typically detect once and persist the resulting
input (`PlanarDataset`, `RigExtrinsicsInput`, …), which is pure data:
JSON-serialisable, no images needed, like the observation file this
tutorial uses. Run calibrations against the cached detections.

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
