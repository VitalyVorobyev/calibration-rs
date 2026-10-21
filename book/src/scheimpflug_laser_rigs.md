# Scheimpflug and Laser Rigs

Four pieces extend the basic workflows to tilted-sensor cameras and laser-triangulation rigs:

| Piece | Solves |
|-------|--------|
| `PlanarIntrinsicsProblem` with a Scheimpflug sensor | Single-camera intrinsics, distortion, and sensor tilt from planar views |
| `SensorMode` | Selects a pinhole or Scheimpflug camera for the rig problems |
| `RigLaserlineDeviceProblem` | One laser plane per camera of an already-calibrated rig |
| `RigHandeyeLaserlineProblem` | Rig, hand-eye, and laser planes jointly |

See [Sensor Models and Scheimpflug Tilt](sensor.md) for the tilt model and [Laserline Device Calibration](laserline.md) for the single-camera laser problem.

## Single Tilted Camera

A single tilted camera is calibrated by [Planar Intrinsics](planar_intrinsics.md)
with `camera.model.sensor = SensorKind::Scheimpflug`: same input
(`PlanarDataset`), two extra parameters, the sensor tilt angles
$(\tau_x, \tau_y)$. Tilt, focal length, and distortion are strongly coupled,
so seed a coarse prior (the nominal focal length and the nominal mount tilt)
instead of relying on the from-scratch initialization:

```rust
use vision_calibration::core::{FxFyCxCy, ScheimpflugParams, SensorKind};
use vision_calibration::planar_intrinsics::{
    PlanarIntrinsicsConfig, PlanarIntrinsicsProblem, PlanarManualInit, step_init_with_seed,
    step_optimize,
};

let mut config = PlanarIntrinsicsConfig::default();
config.camera.model.sensor = SensorKind::Scheimpflug;
let mut session = CalibrationSession::<PlanarIntrinsicsProblem>::new();
session.set_config(config)?;
session.set_input(dataset)?; // PlanarDataset

let mut seed = PlanarManualInit::default();
seed.intrinsics = Some(FxFyCxCy { fx: 1050.0, fy: 1050.0, cx: 360.0, cy: 270.0 });
seed.sensor = Some(ScheimpflugParams { tilt_x: -0.087, tilt_y: 0.0 });

step_init_with_seed(&mut session, seed, None)?;
step_optimize(&mut session, None)?;
let export = session.export()?;
```

`run_calibration(&mut session)` runs the unseeded `step_init` and `step_optimize` in one call. The [Planar Intrinsics](planar_intrinsics.md#tilted-scheimpflug-sensor) chapter describes the seeded and cold-start solves.

## SensorMode for Rigs

`RigExtrinsicsProblem` and `RigHandeyeProblem` cover both camera flavours through their config's `sensor` field:

```rust
use vision_calibration::core::DistortionFixMask;
use vision_calibration::core::DistortionKind;
use vision_calibration::rig_extrinsics::SensorMode;

session.update_config(|c| {
    c.sensor = SensorMode::Scheimpflug {
        init_tilt_x: -0.087,
        init_tilt_y: 0.0,
        fix_scheimpflug: Default::default(),
        distortion_mask_in_percam_ba: DistortionFixMask::radial_only(),
        refine_scheimpflug_in_rig_ba: false,
        distortion_model: DistortionKind::BrownConrady5,
    };
})?;
```

`SensorMode::Pinhole` is the default. The initial tilt is used only when no per-camera sensor seed is supplied through the manual-init struct. Scheimpflug rigs support Brown-Conrady distortion only. With a Scheimpflug mode the exports carry per-camera `sensors`; for pinhole rigs `sensors` is `None`.

## RigLaserlineDevice

Given a frozen rig calibration, `RigLaserlineDeviceProblem` fits one laser plane per camera and reports each plane both in the camera frame and in the rig frame.

```rust
pub struct RigLaserlineDeviceConfig {
    pub solver: SolverConfig,                       // max_iters defaults to 200
    pub laser_residual_type: LaserlineResidualType,
}

pub struct RigLaserlineDeviceInput {
    pub dataset: RigLaserlineDataset,               // target corners + laser pixels per camera/view
    pub upstream: RigUpstreamCalibration,           // intrinsics, distortion, sensors, cam_se3_rig, rig_se3_target
    pub initial_planes_cam: Option<Vec<LaserPlane>>,
}
```

`RigUpstreamCalibration` accepts a pinhole or Scheimpflug rig. The usual source is a rig hand-eye export:

```rust
use vision_calibration::rig_laserline_device::{
    RigLaserlineDeviceInput, RigLaserlineDeviceProblem, run_calibration,
};

let upstream = rig_export.to_upstream_calibration(vec![target_pose; num_views])?;
let mut session = CalibrationSession::<RigLaserlineDeviceProblem>::new();
session.set_input(RigLaserlineDeviceInput { dataset, upstream, initial_planes_cam: None })?;
run_calibration(&mut session)?;
let export = session.export()?; // laser_planes_rig, laser_planes_cam, per_camera_stats, ...
```

The steps `step_init` and `step_optimize` are available for finer control. `dataset` is a `RigLaserlineDataset` (`RigLaserlineDataset::new(views, num_cameras)`, also exported from this module, as is `RigLaserlineView`) whose views hold, per camera, target correspondences and the extracted laser pixels. Laser pixel extraction itself is left to the application.

## RigHandeyeLaserline

`RigHandeyeLaserlineProblem` solves the whole chain jointly: it runs rig hand-eye, initializes the per-camera laser planes from the frozen hand-eye geometry, then refines rig, hand-eye, and laser parameters together.

```rust
pub struct RigHandeyeLaserlineConfig {
    pub handeye: RigHandeyeConfig,                  // warm-start stage (incl. `sensor`)
    pub laserline_init: RigLaserlineDeviceConfig,   // plane-initialization stage
    pub joint_ba: RigHandeyeLaserlineBaConfig,      // solver, calib_loss, laser_loss,
                                                    // calib_weight, laser_weight, fix_handeye, ...
}
```

The input is a `RigHandeyeLaserlineInput { views, num_cameras }`. Each `RigHandeyeLaserlineView` (exported from this module with `RigLaserlineView` and `RobotPoseMeta`) carries the per-camera target observations, the laser pixels, and the robot pose (`base_se3_gripper`). There is a single entry point:

```rust
use vision_calibration::rig_handeye_laserline::{RigHandeyeLaserlineProblem, run_calibration};

let mut session = CalibrationSession::<RigHandeyeLaserlineProblem>::new();
session.set_input(input)?;
run_calibration(&mut session)?;
let export = session.export()?; // cam_se3_rig, gripper_se3_rig, laser_planes_rig, per_camera_stats, ...
```

As in the single-camera laser problem, calibration and laser residuals use separate robust losses (`joint_ba.calib_loss`, `joint_ba.laser_loss`); `joint_ba.solver.robust_loss` is not consulted.

## Examples

- Rust: `cargo run -p vision-calibration --example laserline_device_session` (single-camera laser plane with a Scheimpflug-capable sensor) and `cargo run -p vision-calibration --example rig_handeye_synthetic` (the rig hand-eye stage that feeds the rig laser problems).
- Rust tests showing seeded Scheimpflug calibration: `crates/vision-calibration/tests/planar_intrinsics_scheimpflug.rs`.
- Python: `crates/vision-calibration-py/examples/laserline_device_session.py` and `rig_handeye_synthetic.py`; a tilted single camera runs through `run_planar_intrinsics` with `camera.model.sensor = "scheimpflug"`, and the package also exposes `run_rig_laserline_device` and `run_rig_handeye_laserline`.
