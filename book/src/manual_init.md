# Seeding with Prior Knowledge

By default every workflow initializes itself with the closed-form solvers described in [Why Linear Initialization Matters](linear_overview.md): Zhang's method for intrinsics, a DLT for rig extrinsics, Tsai-Lenz (or another closed-form method) for hand-eye. That is the right behaviour when the data is well conditioned. In industrial settings you often have better priors:

- datasheet focal lengths or principal points;
- factory-measured Scheimpflug tilts;
- laser-plane geometry from mechanical drawings;
- a previously calibrated rig that new data should only nudge.

Manual initialization lets you seed any subset of these and leaves the rest to the automatic path and the non-linear optimizer.

## The Seed Types

For every problem type and every initialization stage there are two things:

- a `step_*_with_seed` function that takes a `*ManualInit` struct before its options;
- a typed `*ManualInit` struct whose fields are all `Option<T>`. `None` means "auto-initialize this group"; `Some(value)` means "use this value verbatim, do not auto-initialize".

The intrinsics stages several problems share take one seed type each, in `common`: `IntrinsicsManualInit` seeds one camera (planar intrinsics, single-camera hand-eye) and `RigIntrinsicsManualInit` each camera of a rig (rig extrinsics, rig hand-eye). The other seeds live in their problem's module.

```rust,no_run
use vision_calibration::core::FxFyCxCy;
use vision_calibration::common::IntrinsicsManualInit;
use vision_calibration::planar_intrinsics::{
    PlanarIntrinsicsProblem, step_init, step_init_with_seed, step_optimize,
};
use vision_calibration::session::CalibrationSession;
# fn main() -> anyhow::Result<()> {
# let mut session = CalibrationSession::<PlanarIntrinsicsProblem>::new();
# let nominal_camera_k: FxFyCxCy<f64> = unimplemented!("from the datasheet");

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
# Ok(())
# }
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

```rust,no_run
use vision_calibration::session::CalibrationSession;
use vision_calibration::common::RigIntrinsicsManualInit;
use vision_calibration::rig_extrinsics::{
    RigExtrinsicsInput, RigExtrinsicsProblem, RigManualInit, RigPoses,
    step_intrinsics_init_all_with_seed, step_intrinsics_optimize_all,
    step_rig_init_with_seed, step_rig_optimize,
};
# fn load_input() -> anyhow::Result<RigExtrinsicsInput> { unimplemented!("the observed rig views") }
# fn main() -> anyhow::Result<()> {

let mut session_a = CalibrationSession::<RigExtrinsicsProblem>::new();
session_a.set_input(load_input()?)?;
step_intrinsics_init_all_with_seed(&mut session_a, RigIntrinsicsManualInit::default(), None)?;
let intr_opt_a = step_intrinsics_optimize_all(&mut session_a, None)?;
let rig_init_a = step_rig_init_with_seed(&mut session_a, RigManualInit::default(), None)?;
step_rig_optimize(&mut session_a, None)?;
# Ok(())
# }
```

`RigIntrinsicsManualInit::default()` and `RigManualInit::default()` have nothing seeded (`None` throughout), so this is exactly the automatic path: the same as `run_calibration`.

### 2. Capture what the stages produced

The values live in the typed results the steps return:

```rust,no_run
use vision_calibration::core::DistortionParams;
# use vision_calibration::rig_extrinsics::{IntrinsicsOptimizeResult, RigInitResult};
# fn main() {
# let intr_opt_a: IntrinsicsOptimizeResult = unimplemented!("from the first run");
# let rig_init_a: RigInitResult = unimplemented!("from the first run");

let cameras_a = intr_opt_a.cameras.clone();                     // Vec<CameraParams>
let cam_se3_rig_a = rig_init_a.initial_cam_se3_rig.clone();     // per-camera T_C_R
let rig_se3_target_a = rig_init_a.initial_rig_se3_target.clone(); // per-view T_R_T

let per_cam_k: Vec<_> = cameras_a.iter().map(|c| c.intrinsics).collect();
let per_cam_dist: Vec<DistortionParams> = cameras_a
    .iter()
    .map(|c| c.distortion.clone())
    .collect();
# }
```

### 3. Replay them on a fresh session

```rust,no_run
use vision_calibration::core::{DistortionParams, FxFyCxCy, Iso3};
use vision_calibration::common::RigIntrinsicsManualInit;
use vision_calibration::rig_extrinsics::{
    RigExtrinsicsInput, RigExtrinsicsProblem, RigManualInit, RigPoses,
    step_intrinsics_init_all_with_seed, step_rig_init_with_seed, step_rig_optimize,
};
use vision_calibration::session::CalibrationSession;
# fn load_input() -> anyhow::Result<RigExtrinsicsInput> { unimplemented!("the observed rig views") }
# fn main() -> anyhow::Result<()> {
# let per_cam_k: Vec<FxFyCxCy<f64>> = unimplemented!("from the first run");
# let per_cam_dist: Vec<DistortionParams> = unimplemented!("from the first run");
# let cam_se3_rig_a: Vec<Iso3> = unimplemented!("from the first run");
# let rig_se3_target_a: Vec<Iso3> = unimplemented!("from the first run");

let mut session_b = CalibrationSession::<RigExtrinsicsProblem>::new();
session_b.set_input(load_input()?)?;

let mut intrinsics_seed = RigIntrinsicsManualInit::default();
intrinsics_seed.per_cam_intrinsics = Some(per_cam_k);
intrinsics_seed.per_cam_distortion = Some(per_cam_dist);
step_intrinsics_init_all_with_seed(&mut session_b, intrinsics_seed, None)?;

// Skip step_intrinsics_optimize_all: the seed is already optimized.

let mut rig_seed = RigManualInit::default();
rig_seed.poses = Some(RigPoses::new(cam_se3_rig_a, rig_se3_target_a));
step_rig_init_with_seed(&mut session_b, rig_seed, None)?;
step_rig_optimize(&mut session_b, None)?;
# Ok(())
# }
```

On the stereo dataset the replay reproduces the first run exactly (`|Δ reproj| = 0` px); the example asserts this. A third run seeds the intrinsics with a 10 % focal-length error and a 50 px principal-point offset and still converges to the same 0.2496 px solution, which shows that a rough seed is enough for the optimizer.

## Variations

### Partial seeding

When intrinsics are seeded but distortion is not, distortion starts at zero in the camera's distortion model: the automatic distortion fit is coupled to the iterative intrinsics estimation and does not run when intrinsics are seeded.

A distortion seed is a `DistortionParams` of the camera's own model: its kind must equal `camera.model.distortion` (a Kannala-Brandt seed for a fisheye camera, a Rational seed for a rational one). It is used as given, never converted; a seed of another kind is a typed error naming the field, raised before any work. In JSON it is the tagged form `{"type": "brown_conrady5", "k1": -0.1, ...}`.

When intrinsics are seeded but poses are not, the poses are recovered from the per-view homographies using the **seeded** intrinsics, not automatically estimated ones. This keeps the geometric chain consistent with the seed.

### Seeding from datasheet values

For a rig whose cameras share one optical design:

```rust,no_run
use vision_calibration::core::{BrownConrady5, DistortionParams, FxFyCxCy};
use vision_calibration::common::RigIntrinsicsManualInit;
use vision_calibration::rig_extrinsics::{
    RigExtrinsicsProblem, step_intrinsics_init_all_with_seed,
};
use vision_calibration::session::CalibrationSession;
# fn main() -> anyhow::Result<()> {
# let mut session = CalibrationSession::<RigExtrinsicsProblem>::new();
# let num_cameras = 2;

let datasheet_k = FxFyCxCy { fx: 1800.0, fy: 1800.0, cx: 360.0, cy: 270.0 };
let nominal_distortion = DistortionParams::BrownConrady5 {
    params: BrownConrady5 { k1: -0.1, ..Default::default() },
};

let mut seed = RigIntrinsicsManualInit::default();
seed.per_cam_intrinsics = Some(vec![datasheet_k; num_cameras]);
seed.per_cam_distortion = Some(vec![nominal_distortion; num_cameras]);
step_intrinsics_init_all_with_seed(&mut session, seed, None)?;
# Ok(())
# }
```

This unblocks scenarios where Zhang's method fails on borderline data, for example strongly tilted Scheimpflug rigs. For those, also set `per_cam_sensors` to the nominal mount tilts: a seeded tilt is used verbatim and skips the linear stage's tilt sweep. A non-zero tilt for a camera with an identity sensor is an error; a zero tilt is accepted. `RigHandeyeConfig::manual_init` takes the same `per_cam_sensors`; it seeds `step_intrinsics_init_all` and `run_calibration` alike, and `step_intrinsics_init_all_with_seed` takes an explicit seed that overrides it. The single-camera seeds (`common::RigIntrinsicsManualInit` and `laserline_device::DeviceManualInit`) take one tilt in `sensor`, with the same rule.

The `device_seed` module derives these seeds from a device description (focal length in millimetres, pixel pitch, mount tilt); see [Device Specifications](manifest.md#device-specifications).

### Rig poses (rig stage)

`cam_se3_rig` and `rig_se3_target` are geometrically coupled, so the rig stage's `RigManualInit` seeds them together: its `poses` is `None` (the linear fit) or a `RigPoses` holding both. A length that does not match the cameras or views is a typed `Error::InvalidInput`.

### Hand-eye seeds

The hand-eye stage's seed has one variant per mounting mode, named and tagged like the export's `handeye` value (`HandeyeSolution` for a rig, `SingleCamHandeyeSolution` for one camera):

```rust,no_run
use vision_calibration::core::Iso3;
use vision_calibration::rig_handeye::{HandeyeManualInit, RigHandeyeProblem, step_handeye_init_with_seed};
use vision_calibration::session::CalibrationSession;
# fn main() -> anyhow::Result<()> {
# let mut session = CalibrationSession::<RigHandeyeProblem>::new();
# let nominal_mount: Iso3 = unimplemented!("the mount from the CAD model");

// EyeInHand: gripper_se3_rig = T_G_R, base_se3_target = T_B_T.
// EyeToHand: rig_se3_base = T_R_B, gripper_se3_target = T_G_T.
let seed = HandeyeManualInit::EyeInHand {
    gripper_se3_rig: Some(nominal_mount),
    base_se3_target: None,
};
step_handeye_init_with_seed(&mut session, seed, None)?;
# Ok(())
# }
```

The variant must be the one of `handeye_init.handeye_mode`; a seed of the other mode is an `Error::InvalidInput`, raised before any work. A seeded pose is used as given, and a pose left `None` is estimated:

- **Both seeded**: the seed is the initialization.
- **One seeded**: the other closes through the first view, so that the mode's chain puts the target where the rig saw it there. The closed-form solve does not run. In eye-in-hand mode the chain is $T_{R,T} = (T_{B,G} \cdot T_{G,R})^{-1} \cdot T_{B,T}$, in eye-to-hand mode $T_{R,T} = T_{R,B} \cdot T_{B,G} \cdot T_{G,T}$.
- **Neither seeded**: the closed-form solver of `handeye_init.method` gives both, as `step_handeye_init` does.

The rules are the same in both modes, for either pose. A run replaces the previous initialization, and a run that fails once it has started (the closed-form solve finding too little rotation, say) leaves none, so `step_handeye_optimize` never starts from an earlier one.

A single camera is a rig of one camera whose frame is the camera's. `single_cam_handeye::HandeyeManualInit` has the same variants and rules, with the camera named where the rig is: `EyeInHand { gripper_se3_cam, base_se3_target }` and `EyeToHand { cam_se3_base, gripper_se3_target }`. The step's result carries the same value: `init.handeye` is the `HandeyeSolution` (or `SingleCamHandeyeSolution`) of the mode, with both poses set. In Python the seed is an `EyeInHandManualInit` or an `EyeToHandManualInit`, from `rig_handeye` or `single_cam_handeye`.

### Staged rig laser calibration

`RigHandeyeLaserline` has no steps: `run_calibration` runs rig hand-eye, fits each camera's laser plane against that frozen geometry, then refines everything jointly. `run_calibration_with_seed` takes a `JointManualInit` that replaces either warm start:

- `rig_handeye`: a `RigHandeyeExport` of the same views, for example from a `RigHandeyeProblem` session you ran (and inspected) first. The rig hand-eye stage is skipped; its cameras, extrinsics, hand-eye transform, target poses and robot-pose corrections start the joint solve.
- `laser_planes_cam`: one laser plane per camera, in camera frame, for example from the mechanical drawing. The plane stage refines them instead of fitting its starting planes linearly.

```rust,no_run
use vision_calibration::core::LaserPlane;
use vision_calibration::rig_handeye::RigHandeyeExport;
use vision_calibration::rig_handeye_laserline::{
    JointManualInit, RigHandeyeLaserlineProblem, run_calibration_with_seed,
};
use vision_calibration::session::CalibrationSession;
# fn main() -> anyhow::Result<()> {
# let mut joint_session = CalibrationSession::<RigHandeyeLaserlineProblem>::new();
# let rig_export: RigHandeyeExport = unimplemented!("a rig hand-eye export");
# let design_planes: Vec<LaserPlane> = unimplemented!("the planes from the design");

// `rig_export` is the export of a rig hand-eye session on the same views,
// with `config.handeye` as its config.
let mut seed = JointManualInit::default();
seed.rig_handeye = Some(rig_export);
seed.laser_planes_cam = Some(design_planes); // Vec<LaserPlane>, one per camera
run_calibration_with_seed(&mut joint_session, seed)?;
# Ok(())
# }
```

With the warm start a rig hand-eye session computes from the joint session's input and `config.handeye`, the result is identical to `run_calibration`. The export must have the input's camera count, each camera in the configured model, one target pose per view, and the configured hand-eye mode with its transforms; the seeded planes must be one per camera. Anything else is an `Error::InvalidInput`, raised before any solve.

## In Python

The seeds are dataclasses with the Rust names and fields; a field left `None` is estimated. The intrinsics seeds `IntrinsicsManualInit` and `RigIntrinsicsManualInit` are importable from the package root, the other seeds from the problem's module. The datasheet seed above:

```python
import vision_calibration as vc

seed = vc.RigIntrinsicsManualInit(
    per_cam_intrinsics=[vc.PinholeIntrinsics(fx=1800.0, fy=1800.0, cx=360.0, cy=270.0)] * 2,
    per_cam_distortion=[vc.BrownConradyDistortion(k1=-0.1, k2=0.0, k3=0.0, p1=0.0, p2=0.0, iters=8)] * 2,
)
init = session.step_intrinsics_init_all_with_seed(seed)
print(session.log[-1].notes)  # the init log names the manual and the estimated parts
```

Distortion seeds are the distortion dataclasses of the camera model (`BrownConradyDistortion`, `KannalaBrandt4Distortion`, …). A seed that does not fit the camera model (a tilt for an untilted camera, a distortion of another kind, one rig pose without the other) raises `ValueError`.

The staged rig laser calibration takes the rig hand-eye session's export as it is:

```python
from vision_calibration import rig_handeye_laserline

rig_export = rig_session.export()  # a RigHandeyeSession on the same views
joint = vc.RigHandeyeLaserlineSession()
joint.set_input(dataset)
joint.run_calibration_with_seed(
    rig_handeye_laserline.JointManualInit(rig_handeye=rig_export, laser_planes_cam=design_planes)
)
```

## See Also

- [Step Functions vs Pipeline Functions](step_functions.md): the full list of `*_with_seed` variants.
- [Per-Feature Residuals](per_feature_residuals.md): inspect per-corner errors once the calibration finishes.
- The example this chapter follows: `crates/vision-calibration/examples/manual_init.rs`.
