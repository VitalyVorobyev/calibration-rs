# Seeding with Prior Knowledge

By default every workflow initializes itself with the closed-form solvers described in [Why Linear Initialization Matters](linear_overview.md): Zhang's method for intrinsics, a DLT for rig extrinsics, Tsai-Lenz for hand-eye. That is the right behaviour when the data is well conditioned. In industrial settings you often have better priors:

- datasheet focal lengths or principal points;
- factory-measured Scheimpflug tilts;
- laser-plane geometry from mechanical drawings;
- a previously calibrated rig that new data should only nudge.

Manual initialization lets you seed any subset of these and leaves the rest to the automatic path and the non-linear optimizer.

## The Seed Types

For every problem type and every initialization stage there are two things:

- a `step_*_with_seed` function that takes a `*ManualInit` struct before its options;
- a typed `*ManualInit` struct whose fields are all `Option<T>`. `None` means "auto-initialize this group"; `Some(value)` means "use this value verbatim, do not auto-initialize".

```rust
// Automatic path
step_init(&mut session, None)?;
step_optimize(&mut session, None)?;

// Seeded path
let mut seed = IntrinsicsManualInit::default();
seed.intrinsics = Some(nominal_camera_k);
// distortion: None  -> auto-initialized (zeros when intrinsics are seeded)
// sensor: None      -> tilt seed, Scheimpflug cameras only (a typed error otherwise)
// poses: None       -> recovered from homographies using the seeded intrinsics
step_init_with_seed(&mut session, seed, None)?;
step_optimize(&mut session, None)?;
```

The seed structs are `#[non_exhaustive]`: build one with `Default::default()` and assign the fields you want, as above. Struct-update syntax (`..Default::default()`) does not compile outside the defining crate.

After `step_*_with_seed` returns, the session state is fully initialized, with the same postcondition as the automatic path; `step_optimize` does not need to know which fields came from where. The session log records the source of every stage, for example:

```text
[intrinsics_init_all] initialized 2 cameras (manual: per_cam_intrinsics, per_cam_distortion)
[rig_init] ref_cam=0, 20 views (manual: cam_se3_rig, rig_se3_target)
```

## Walkthrough: Replaying a Calibration on a Rig

The `manual_init` example seeds the rig stage of the stereo calibration from a previous run. It loads the corners committed in `regression/observations/stereo_rig.json` and runs the pipeline three times:

```bash
cargo run -p vision-calibration --example manual_init --release
```

### 1. Run the automatic pipeline once

```rust
use vision_calibration::prelude::*;
use vision_calibration::rig_extrinsics::{
    IntrinsicsManualInit, RigExtrinsicsProblem, RigManualInit,
    step_intrinsics_init_all_with_seed, step_intrinsics_optimize_all,
    step_rig_init_with_seed, step_rig_optimize,
};

let mut session_a = CalibrationSession::<RigExtrinsicsProblem>::new();
session_a.set_input(load_input()?)?;
step_intrinsics_init_all_with_seed(&mut session_a, IntrinsicsManualInit::default(), None)?;
let intr_opt_a = step_intrinsics_optimize_all(&mut session_a, None)?;
let rig_init_a = step_rig_init_with_seed(&mut session_a, RigManualInit::default(), None)?;
step_rig_optimize(&mut session_a, None)?;
```

`IntrinsicsManualInit::default()` and `RigManualInit::default()` have all fields `None`, so this is exactly the automatic path: the same as `run_calibration`.

### 2. Capture what the stages produced

The values live in the typed results the steps return:

```rust
use vision_calibration::core::{BrownConrady5, DistortionParams};

let cameras_a = intr_opt_a.cameras.clone();                     // Vec<CameraParams>
let cam_se3_rig_a = rig_init_a.initial_cam_se3_rig.clone();     // per-camera T_C_R
let rig_se3_target_a = rig_init_a.initial_rig_se3_target.clone(); // per-view T_R_T

let per_cam_k: Vec<_> = cameras_a.iter().map(|c| c.intrinsics).collect();
let per_cam_dist: Vec<BrownConrady5<f64>> = cameras_a
    .iter()
    .map(|c| match c.distortion {
        DistortionParams::BrownConrady5 { params } => params,
        _ => BrownConrady5::default(),
    })
    .collect();
```

### 3. Replay them on a fresh session

```rust
let mut session_b = CalibrationSession::<RigExtrinsicsProblem>::new();
session_b.set_input(load_input()?)?;

let mut intrinsics_seed = IntrinsicsManualInit::default();
intrinsics_seed.per_cam_intrinsics = Some(per_cam_k);
intrinsics_seed.per_cam_distortion = Some(per_cam_dist);
step_intrinsics_init_all_with_seed(&mut session_b, intrinsics_seed, None)?;

// Skip step_intrinsics_optimize_all: the seed is already optimized.

let mut rig_seed = RigManualInit::default();
rig_seed.cam_se3_rig = Some(cam_se3_rig_a);
rig_seed.rig_se3_target = Some(rig_se3_target_a);
step_rig_init_with_seed(&mut session_b, rig_seed, None)?;
step_rig_optimize(&mut session_b, None)?;
```

On the stereo dataset the replay reproduces the first run exactly (`|Δ reproj| = 0` px); the example asserts this. A third run seeds the intrinsics with a 10 % focal-length error and a 50 px principal-point offset and still converges to the same 0.2496 px solution, which shows that a rough seed is enough for the optimizer.

## Variations

### Partial seeding

When intrinsics are seeded but distortion is not, distortion starts at `BrownConrady5::default()` (zeros): the automatic distortion fit is coupled to the iterative intrinsics estimation and does not run when intrinsics are seeded.

When intrinsics are seeded but poses are not, the poses are recovered from the per-view homographies using the **seeded** intrinsics, not automatically estimated ones. This keeps the geometric chain consistent with the seed.

### Seeding from datasheet values

For a rig whose cameras share one optical design:

```rust
use vision_calibration::core::{BrownConrady5, FxFyCxCy};

let datasheet_k = FxFyCxCy { fx: 1800.0, fy: 1800.0, cx: 360.0, cy: 270.0 };
let nominal_distortion = BrownConrady5 { k1: -0.1, ..Default::default() };

let mut seed = IntrinsicsManualInit::default();
seed.per_cam_intrinsics = Some(vec![datasheet_k; num_cameras]);
seed.per_cam_distortion = Some(vec![nominal_distortion; num_cameras]);
step_intrinsics_init_all_with_seed(&mut session, seed, None)?;
```

This unblocks scenarios where Zhang's method fails on borderline data, for example strongly tilted Scheimpflug rigs. For those, also set `per_cam_sensors` to the nominal mount tilts: a seeded tilt is used verbatim and skips the linear stage's tilt sweep. A non-zero tilt for a camera with an identity sensor is an error; a zero tilt is accepted. `RigHandeyeConfig::manual_init` takes the same `per_cam_sensors`. The single-camera seeds (`planar_intrinsics::IntrinsicsManualInit`, `single_cam_handeye::IntrinsicsManualInit` and `laserline_device::DeviceManualInit`) take one tilt in `sensor`, with the same rule.

The `device_seed` module derives these seeds from a device description (focal length in millimetres, pixel pitch, mount tilt); see [Device Specifications](manifest.md#device-specifications).

### Coupling rule (rig stage)

`cam_se3_rig` and `rig_se3_target` are geometrically coupled. The rig stage's `RigManualInit` requires both or neither, and enforces it at runtime with a typed `Error::InvalidInput`.

### Hand-eye coupling

`RigHandeye` has three initialization stages (intrinsics, rig, hand-eye), each with its own `*ManualInit`. The hand-eye stage's `mode_target_pose` depends on the mounting mode:

- `EyeInHand`: `handeye = T_G_R`, `mode_target_pose = T_B_T`.
- `EyeToHand`: `handeye = T_R_B`, `mode_target_pose = T_G_T`.

If you seed `handeye` but not `mode_target_pose`, the latter is derived from the chain.

## See Also

- [Step Functions vs Pipeline Functions](step_functions.md): the full list of `*_with_seed` variants.
- [Per-Feature Residuals](per_feature_residuals.md): inspect per-corner errors once the calibration finishes.
- The example this chapter follows: `crates/vision-calibration/examples/manual_init.rs`.
