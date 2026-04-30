# Puzzle 130×130 walkthrough

> The full-scale "everything turned on" calibration: 6-camera Scheimpflug
> rig + laser plane + robot hand-eye, on real data.
> Runnable companion (private dataset):
> [`puzzle_130x130_rig.rs`](../../crates/vision-calibration-examples-private/examples/puzzle_130x130_rig.rs).

## Why

Most tutorials use one feature at a time. This walkthrough composes them:
how the same session pattern handles a calibration where every advanced
feature in the workspace is in play.

You will not be able to run this on your own machine (the dataset is
internal), but the source is the canonical reference for putting
`calibration-rs` to work end-to-end. Use it as a recipe rather than a demo.

## What's in the dataset

`privatedata/130x130_puzzle/`:

- 20 robot poses (`poses.json`).
- For each pose: a target snapshot + (sometimes) a laser snapshot.
- 6 cameras horizontally concatenated into a single 4320×540 image.
- Puzzleboard target (130×130 cells, 1.014 mm per cell).
- Each camera has a Scheimpflug-tilted sensor.

## Pipeline shape

```
                 ┌─ stage 1 ─────────────────┐
                 │  detect target corners    │
                 │  detect laser pixels      │
                 └─────────────┬─────────────┘
                               ▼
       ┌─ stage 2 ─ RigScheimpflugHandeyeProblem ─────────────┐
       │  step_intrinsics_init_all   (Zhang per camera)       │
       │  step_intrinsics_optimize_all (per-camera BA)        │
       │  step_rig_init               (linear cam_se3_rig)    │
       │  step_rig_optimize           (joint rig BA)          │
       │  step_handeye_init           (Tsai-Lenz)             │
       │  step_handeye_optimize       (final hand-eye BA)     │
       │                                                       │
       │  → RigScheimpflugHandeyeExport                        │
       └─────────────┬─────────────────────────────────────────┘
                     ▼
       ┌─ stage 3 ─ RigLaserlineDeviceProblem ────────────────┐
       │  step_init       (per-cam linear plane fit, frozen)  │
       │  step_optimize   (per-cam laser BA, frozen upstream) │
       │                                                       │
       │  → RigLaserlineDeviceExport (per-cam laser planes)    │
       └─────────────┬─────────────────────────────────────────┘
                     ▼
       ┌─ stage 4 ─ Joint BA (refine everything) ─────────────┐
       │  Reuses the IR built across stage 2 + stage 3 and     │
       │  unfreezes intrinsics, extrinsics, hand-eye, planes,  │
       │  and (optionally) per-view robot pose corrections.    │
       └───────────────────────────────────────────────────────┘
```

Each stage produces a typed `*Export` that the next stage consumes (via
[`RigScheimpflugHandeyeExport::to_upstream_calibration`](../../crates/vision-calibration-pipeline/src/rig_laserline_device/problem.rs)
in particular). The example glues them together.

## Walkthrough

### Stage 1 — detection

Detection is delegated to `calib-targets` (puzzleboard) and
`vision-metrology` (laser line). The output is structured input data for
the calibration sessions:

```rust
let detected = build_datasets(&data_dir, &poses)?;
```

The example prints a per-camera detection diagnostic — how many target
corners and laser pixels each camera produced.

### Stage 2 — Scheimpflug rig + hand-eye

```rust
let mut rig_session =
    CalibrationSession::<RigScheimpflugHandeyeProblem>::with_description("puzzle_130x130_rig");
rig_session.set_input(detected.handeye_views)?;
rig_session.set_config(stage2_cfg)?;

run_calibration(&mut rig_session)?;
let stage2_export = rig_session.export()?;
```

The example uses
[manual init](./manual-init.md) to seed every camera's intrinsics with a
shared datasheet K (homogeneous rig assumption) and recover the
Scheimpflug tilts via the non-linear stages.

Stage 2 produces a mean reprojection error around 0.74 px on this dataset.

### Stage 3 — rig laserline

Wire the upstream Scheimpflug rig + hand-eye into the laserline solver:

```rust
let upstream = stage2_export.to_upstream_calibration(rig_se3_target_per_view);
let laser_input = RigLaserlineDeviceInput {
    dataset: detected.laserline_views,
    upstream,
    initial_planes_cam: None, // default to z=-0.2m per camera
};

let mut laser_session = CalibrationSession::<RigLaserlineDeviceProblem>::new();
laser_session.set_input(laser_input)?;
laser_session.run_calibration()?;
let stage3_export = laser_session.export()?;
```

Stage 3 produces six laser planes (one per camera, both in camera frame
and rig frame).

### Stage 4 — joint BA

The example reaches into `vision-calibration-optim` for `optimize_rig_handeye_laserline`
to refine everything jointly with the upstream stages no longer frozen.
Stage 4 brings the mean reprojection error from 0.74 px down to ~0.45 px.

## Per-feature residuals

After stage 2 every export carries
[`per_feature_residuals`](./per-feature-residuals.md) automatically:

```rust
let pf = &stage2_export.per_feature_residuals;
for (i, h) in pf.target_hist_per_camera.as_ref().unwrap().iter().enumerate() {
    println!("cam {i}: {} corners, mean {:.3}px max {:.3}px",
             h.count, h.mean, h.max);
}
```

`stage3_export.per_feature_residuals` extends this with `laser` records
giving per-pixel point-to-plane and pixel-to-projected-line distances.

The puzzle viewer in
[`puzzle_130x130_rig/viewer.rs`](../../crates/vision-calibration-examples-private/examples/puzzle_130x130_rig/viewer.rs)
predates the `per_feature_residuals` schema — it builds the same shape
inline. A planned follow-up swaps it to consume the export field directly.

## Pixel → gripper mapping

The example demonstrates `pixel_to_gripper_point`, which back-projects a
laser pixel through the calibrated rig and intersects with the laser plane
to recover a 3D point in the robot gripper frame:

```rust
let p = vision_calibration::pixel_to_gripper_point(
    cam_idx,
    &observed_pixel,
    &stage2_export,        // upstream rig + handeye
    &stage3_export.laser_planes_rig,
    &robot_base_se3_gripper, // only required for EyeToHand
)?;
```

This is the load-bearing measurement primitive on top of the calibration
stack.

## What to read next

- [Five-minute calibration](./five-minute-calibration.md) — start small,
  one problem type at a time.
- [Manual initialization](./manual-init.md) — the seeding mechanism stage
  2 uses for intrinsics.
- [Per-feature residuals](./per-feature-residuals.md) — how to drill into
  the per-corner errors emitted at every stage.
- [ADR 0006](../adrs/0006-layered-crate-architecture.md) — why the
  workspace is split into `core` / `linear` / `optim` / `pipeline` /
  facade.
- [Roadmap](../ROADMAP.md) — where the puzzle rig sits in the broader
  4-track plan.
