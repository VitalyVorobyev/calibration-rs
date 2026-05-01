# Manual initialization (warm-start)

> Onboarding tutorial for [ADR 0011](../adrs/0011-manual-initialization-workflow.md).
> Runnable companion: [`manual_init_proof.rs`](../../crates/vision-calibration/examples/manual_init_proof.rs).

## Why

The default pipeline always runs automatic linear initialization (Zhang's
method, Tsai-Lenz DLT). That is the right behaviour when the data is
well-conditioned, but in industrial settings you often have *better*
priors:

- Datasheet focal lengths or principal points.
- Factory-measured Scheimpflug tilts.
- Laser-plane geometry from mechanical drawings.
- A previously calibrated rig you want to nudge with new data.

Manual initialization lets you seed any subset of these into the pipeline
and let the non-linear optimizer do the rest.

## Mental model

For every problem type and every init stage there is:

- A `step_set_*` function that takes a `*ManualInit` struct.
- A typed `*ManualInit` struct whose fields are all `Option<T>`. `None`
  means "auto-initialize this group"; `Some(value)` means "use this value
  verbatim, do not auto-initialize".

```rust
// Automatic path (unchanged):
step_init(&mut session, None)?;
step_optimize(&mut session, None)?;

// Manual path:
step_set_init(&mut session, PlanarManualInit {
    intrinsics: Some(nominal_camera_k),
    distortion: None,  // auto-initialized (or zeros when intrinsics seeded)
    poses: None,       // auto-recovered from homographies using manual intrinsics
}, None)?;
step_optimize(&mut session, None)?;
```

After `step_set_*` returns, the session state is fully initialized — same
postcondition as the auto path. `step_optimize` does not need to know
which fields came from where.

The session's log records the source of every initialization stage, e.g.:

```
[intrinsics_init_all] initialized 2 cameras (manual: per_cam_intrinsics, per_cam_distortion)
[rig_init] ref_cam=0, 8 views (manual: cam_se3_rig, rig_se3_target)
```

## Walkthrough

We'll seed the rig stage of a stereo calibration from a previous run.

### 1. Run the auto pipeline once

```rust
use vision_calibration::prelude::*;
use vision_calibration::rig_extrinsics::{
    RigExtrinsicsProblem, RigIntrinsicsManualInit, RigExtrinsicsManualInit,
    step_set_intrinsics_init_all, step_intrinsics_optimize_all,
    step_set_rig_init, step_rig_optimize,
};

let mut session_a = CalibrationSession::<RigExtrinsicsProblem>::new();
session_a.set_input(load_dataset()?)?;
step_set_intrinsics_init_all(&mut session_a, RigIntrinsicsManualInit::default(), None)?;
step_intrinsics_optimize_all(&mut session_a, None)?;
step_set_rig_init(&mut session_a, RigExtrinsicsManualInit::default())?;
step_rig_optimize(&mut session_a, None)?;
```

`RigIntrinsicsManualInit::default()` and `RigExtrinsicsManualInit::default()`
both have all-`None` fields, so this is exactly the auto path — the same as
`run_calibration`.

### 2. Capture the results

```rust
let cameras_a = session_a.state.per_cam_intrinsics.clone().unwrap();
let cam_se3_rig_a = session_a.state.initial_cam_se3_rig.clone().unwrap();
let rig_se3_target_a = session_a.state.initial_rig_se3_target.clone().unwrap();

let per_cam_k = cameras_a.iter().map(|c| c.k).collect::<Vec<_>>();
let per_cam_dist = cameras_a.iter().map(|c| c.dist).collect::<Vec<_>>();
```

### 3. Replay them on a fresh session

```rust
let mut session_b = CalibrationSession::<RigExtrinsicsProblem>::new();
session_b.set_input(load_dataset()?)?;

step_set_intrinsics_init_all(
    &mut session_b,
    RigIntrinsicsManualInit {
        per_cam_intrinsics: Some(per_cam_k),
        per_cam_distortion: Some(per_cam_dist),
    },
    None,
)?;

// Skip step_intrinsics_optimize_all: the seed is already optimized.

step_set_rig_init(
    &mut session_b,
    RigExtrinsicsManualInit {
        cam_se3_rig: Some(cam_se3_rig_a),
        rig_se3_target: Some(rig_se3_target_a),
    },
)?;
step_rig_optimize(&mut session_b, None)?;
```

The output is bit-exact (`|Δ reproj| < 3e-15` px on the stereo ChArUco
dataset). The example in `manual_init_proof.rs` asserts this.

## Common variations

### Partial seeding

When intrinsics are seeded but distortion is not, distortion defaults to
`BrownConrady5::default()` (zeros) — the auto distortion fit is coupled
with iterative intrinsics estimation and does not run when intrinsics are
seeded.

When intrinsics are seeded but poses are not, poses recover from per-view
homographies using the **manually provided intrinsics** (not auto-estimated
ones). This keeps the geometric chain consistent with the seed.

### Seed from datasheet values

For a homogeneous rig where all cameras share the same optical design:

```rust
use vision_calibration::core::{BrownConrady5, FxFyCxCySkew};

let datasheet_k = FxFyCxCySkew {
    fx: 1800.0, fy: 1800.0,
    cx: 360.0, cy: 270.0,
    skew: 0.0,
};
let nominal_distortion = BrownConrady5 { k1: -0.1, ..Default::default() };

step_set_intrinsics_init_all(
    &mut session,
    RigIntrinsicsManualInit {
        per_cam_intrinsics: Some(vec![datasheet_k; num_cameras]),
        per_cam_distortion: Some(vec![nominal_distortion; num_cameras]),
    },
    None,
)?;
```

This unblocks scenarios where Zhang's method fails on borderline data —
e.g., the puzzle 130x130 rig case that motivated [ADR 0011](../adrs/0011-manual-initialization-workflow.md).

### Coupling rule (rig stage)

`cam_se3_rig` and `rig_se3_target` are geometrically coupled. The rig
stage's `*ManualInit` requires both or neither, enforced at runtime with
a typed `Error::InvalidInput`.

### Hand-eye coupling

`RigHandeye` (pinhole and Scheimpflug variants) has three init stages
(intrinsics, rig, hand-eye), each with its own `*ManualInit`. The hand-eye
stage's `mode_target_pose` is mode-dependent:

- `EyeInHand`: `handeye = T_G_R`, `mode_target_pose = T_B_T`.
- `EyeToHand`: `handeye = T_R_B`, `mode_target_pose = T_G_T`.

If you seed `handeye` but not `mode_target_pose`, the latter is auto-derived
from the chain (see [ADR 0009](../adrs/0009-coordinate-and-pose-conventions.md)
for the convention).

## What to read next

- [ADR 0011](../adrs/0011-manual-initialization-workflow.md) — full design
  rationale, including the partial-seed semantics and the
  `#[non_exhaustive]` rule.
- [Per-feature residuals](./per-feature-residuals.md) — once a calibration
  finishes, drill into per-corner errors.
- [`manual_init_proof.rs`](../../crates/vision-calibration/examples/manual_init_proof.rs)
  — the example this tutorial is built on. Run it with
  `cargo run -p vision-calibration --example manual_init_proof --release -- --max-views=8`.
