# Scheimpflug and Laser Rigs

Four pieces extend the basic workflows to tilted-sensor cameras and laser-triangulation rigs. A tilted sensor is a value of the camera model, `camera.model.sensor = Scheimpflug`, in every workflow: planar intrinsics, single-camera hand-eye, the laserline device and the rigs.

| Piece | Solves |
|-------|--------|
| `PlanarIntrinsicsProblem` with a Scheimpflug sensor | Single-camera intrinsics, distortion, and sensor tilt from planar views |
| `camera` / `camera_overrides` | Select the lens and sensor of every rig camera, or of single cameras |
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
    PlanarIntrinsicsConfig, PlanarIntrinsicsProblem, IntrinsicsManualInit, step_init_with_seed,
    step_optimize,
};

let mut config = PlanarIntrinsicsConfig::default();
config.camera.model.sensor = SensorKind::Scheimpflug;
let mut session = CalibrationSession::<PlanarIntrinsicsProblem>::new();
session.set_config(config)?;
session.set_input(dataset)?; // PlanarDataset

let mut seed = IntrinsicsManualInit::default();
seed.intrinsics = Some(FxFyCxCy { fx: 1050.0, fy: 1050.0, cx: 360.0, cy: 270.0 });
seed.sensor = Some(ScheimpflugParams { tilt_x: -0.087, tilt_y: 0.0 });

step_init_with_seed(&mut session, seed, None)?;
step_optimize(&mut session, None)?;
let export = session.export()?;
```

`run_calibration(&mut session)` runs the unseeded `step_init` and `step_optimize` in one call. The [Planar Intrinsics](planar_intrinsics.md#tilted-scheimpflug-sensor) chapter describes the seeded and cold-start solves.

## Camera Models for Rigs

`RigExtrinsicsProblem` and `RigHandeyeProblem` take a `camera` (the model every camera uses) and `camera_overrides` (per-camera replacements). Any distortion kind combines with either sensor, per camera:

```rust
use vision_calibration::common::config::{CameraConfig, CameraOverride};
use vision_calibration::core::{CameraModelSpec, DistortionKind, SensorKind};

session.update_config(|c| {
    // Every camera: tilted sensor, Brown-Conrady distortion.
    c.camera = CameraConfig::new(CameraModelSpec::pinhole(
        DistortionKind::BrownConrady5,
        SensorKind::Scheimpflug,
    ));
    // Camera 1 is an untilted camera with a rational lens.
    c.camera_overrides = vec![CameraOverride::new(
        1,
        CameraModelSpec::pinhole(DistortionKind::Rational8, SensorKind::Identity),
    )];
    c.rig.refine_sensor = false; // keep the tilts fixed in the rig bundle adjustment
})?;
```

The default is a Brown-Conrady camera with an identity sensor. `camera.fix` and each override's `fix` name the parameters held fixed, per slot (`intrinsics`, `distortion`, `sensor`); a slot left out keeps the model's default, which for a Scheimpflug camera holds everything but `k1`, `k2`. `rig.refine_intrinsics` keeps these fix lists in the rig bundle adjustment, and `rig.refine_sensor` (hand-eye: `handeye_optimize.refine_sensor`) lets the tilts move there.

Tilt seeds are device data, not config: pass `per_cam_sensors` in the manual-init struct (`RigHandeyeConfig::manual_init`, or `rig_extrinsics::IntrinsicsManualInit` with `step_intrinsics_init_all_with_seed` for rig extrinsics). A seed is used verbatim and skips the linear stage's tilt sweep; without one the linear stage sweeps the tilt grid, which includes zero. A non-zero tilt seed for an untilted camera is an error.

Each tilted camera is first calibrated from its own views, as in single-camera planar intrinsics: the linear stage holds `p1`, `p2` at zero (a tangential term and the tilt trade off, and the default fix list holds them in every later solve), and the camera's own solve frees every target pose. On clean synthetic data this solve reaches the reprojection noise floor, with the principal point and tilt at the truth within their standard deviations.

## RigLaserlineDevice

Given a frozen rig calibration, `RigLaserlineDeviceProblem` fits one laser plane per camera and reports each plane both in the camera frame and in the rig frame.

```rust
pub struct RigLaserlineDeviceConfig {
    pub solver: SolverConfig,                       // max_iters defaults to 200
    pub laser_residual_type: LaserlineResidualType,
}

pub struct RigLaserlineDeviceInput {
    pub dataset: RigLaserlineDataset,               // target corners + laser pixels per camera/view
    pub upstream: RigUpstreamCalibration,           // cameras (CameraParams), cam_se3_rig, rig_se3_target
    pub initial_planes_cam: Option<Vec<LaserPlane>>,
}
```

`RigUpstreamCalibration` holds one `CameraParams` per camera, and each camera's laser residuals are evaluated in its own model, so the rig laser workflows accept every distortion kind and untilted and Scheimpflug cameras alike (a general homography sensor, which no camera model describes, is a typed error). The usual source is a rig hand-eye export:

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
    pub handeye: RigHandeyeConfig,                  // warm-start stage (incl. `camera`)
    pub laserline_init: RigLaserlineDeviceConfig,   // plane-initialization stage
    pub joint_optimize: JointOptimizeConfig,        // solver, calib_loss, laser_loss,
                                                    // calib_weight, laser_weight, fix,
                                                    // fix_handeye, ...
}
```

`joint_optimize.fix` holds one `CameraFix` per camera, named in that camera's model.
Left empty, every camera gets the joint stage's default: intrinsics and the
low-order radial terms (`k1`, `k2`; the division model's `lambda`) refined,
the higher-order distortion terms and the sensor tilt held at their
warm-start values, because they trade off against the extrinsics and the
laser planes. An entry's slot left out keeps the model's default, as in every
`CameraFix`.

The input is a `RigHandeyeLaserlineInput { views, num_cameras }`. Each `RigHandeyeLaserlineView` (exported from this module with `RigLaserlineView` and `RobotPoseMeta`) carries the per-camera target observations, the laser pixels, and the robot pose (`base_se3_gripper`). There is a single entry point:

```rust
use vision_calibration::rig_handeye_laserline::{RigHandeyeLaserlineProblem, run_calibration};

let mut session = CalibrationSession::<RigHandeyeLaserlineProblem>::new();
session.set_input(input)?;
run_calibration(&mut session)?;
let export = session.export()?; // cam_se3_rig, gripper_se3_rig, laser_planes_rig, per_camera_stats, ...
```

As in the single-camera laser problem, calibration and laser residuals use separate robust losses (`joint_optimize.calib_loss`, `joint_optimize.laser_loss`); `joint_optimize.solver.robust_loss` is not consulted.

## Examples

- Rust: `cargo run -p vision-calibration --example laserline_device_session` (single-camera laser plane with a Scheimpflug-capable sensor) and `cargo run -p vision-calibration --example rig_handeye_synthetic` (the rig hand-eye stage that feeds the rig laser problems).
- Rust tests showing seeded Scheimpflug calibration: `crates/vision-calibration/tests/planar_intrinsics_scheimpflug.rs` (one camera) and `crates/vision-calibration/tests/rig_extrinsics_scheimpflug.rs` (a rig).
- Python: `crates/vision-calibration-py/examples/laserline_device_session.py` and `rig_handeye_synthetic.py`; a tilted single camera runs through `run_planar_intrinsics` with `camera.model.sensor = "scheimpflug"`, and the package also exposes `run_rig_laserline_device` and `run_rig_handeye_laserline`.
