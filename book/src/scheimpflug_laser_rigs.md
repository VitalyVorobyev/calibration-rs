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
use vision_calibration::common::IntrinsicsManualInit;
use vision_calibration::core::{FxFyCxCy, ScheimpflugParams, SensorKind};
use vision_calibration::planar_intrinsics::{
    PlanarIntrinsicsConfig, PlanarIntrinsicsProblem, step_init_with_seed, step_optimize,
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
})?;
```

The default is a Brown-Conrady camera with an identity sensor. `camera.fix` and each override's `fix` name the parameters held fixed, per slot (`intrinsics`, `distortion`, `sensor`); a slot left out keeps the model's default, which for a Scheimpflug camera holds everything but `k1`, `k2`. `rig.refine_intrinsics` keeps these fix lists in the rig bundle adjustment, and `rig.refine_sensor` (hand-eye: `handeye_optimize.refine_sensor`) refines the tilts there too. Both default to on: a camera's own views pin its tilt poorly, as it trades with the principal point, while the views of the whole rig pin it several times better. A tilt held at the camera's own estimate leaves its error in the focal length and the principal point, and the covariance, conditional on the held tilt, then states their σ several times too small; setting the flag to `false` holds it there, and the covariance lists it under `fixed`.

Tilt seeds are device data, not config: pass `per_cam_sensors` in the manual-init struct (`RigHandeyeConfig::manual_init`, or `common::RigIntrinsicsManualInit` with `step_intrinsics_init_all_with_seed` for rig extrinsics). A seed is used verbatim and skips the linear stage's tilt sweep; without one the linear stage sweeps the tilt grid, which includes zero. A non-zero tilt seed for an untilted camera is an error.

Each tilted camera is first calibrated from its own views, as in single-camera planar intrinsics: the linear stage holds `p1`, `p2` at zero (a tangential term and the tilt trade off, and the default fix list holds them in every later solve), and the camera's own solve frees every target pose. On clean synthetic data this solve reaches the reprojection noise floor, with the principal point and tilt at the truth within their standard deviations.

From scratch (nothing seeded, for any camera), a tilted camera's own solve can still settle in a wrong tilt / principal-point basin, which the rig bundle adjustment, a local solve, may not leave. Both rig workflows then solve each tilted camera through a wider multi-start over linear tilt seeds, and re-solve a Brown-Conrady tilted camera that still fits worse than 0.5 px from a nominal seed built from the rig's well-fitted cameras. This costs extra solves (several times the per-camera stage on a hard rig), and a seeded rig skips it.

## RigLaserlineDevice

Given a frozen rig calibration, `RigLaserlineDeviceProblem` fits one laser plane per camera and reports each plane both in the camera frame and in the rig frame.

```rust
pub struct RigLaserlineDeviceConfig {
    pub optimize: PlaneOptimizeConfig,              // laser_residual_type, laser_loss
    pub solver: SolverConfig,                       // max_iters defaults to 200
}

pub struct RigLaserlineDeviceInput {
    pub dataset: RigLaserlineDataset,               // target corners + laser pixels per camera/view
    pub upstream: RigUpstreamCalibration,           // cameras (CameraParams), cam_se3_rig,
                                                    // rig_se3_target, uncertainty
}
```

`RigUpstreamCalibration` holds one `CameraParams` per camera, and each camera's laser residuals are evaluated in its own model, so the rig laser workflows accept every distortion kind and untilted and Scheimpflug cameras alike (a general homography sensor, which no camera model describes, is a typed error). The usual source is a rig hand-eye export:

```rust
use vision_calibration::rig_laserline_device::{
    RigLaserlineDeviceInput, RigLaserlineDeviceProblem, run_calibration,
};

// The laser views are the hand-eye run's views: its own board poses.
let upstream = rig_export.to_upstream_calibration(rig_export.rig_se3_target.clone())?;
let mut session = CalibrationSession::<RigLaserlineDeviceProblem>::new();
session.set_input(RigLaserlineDeviceInput { dataset, upstream })?;
run_calibration(&mut session)?;
let export = session.export()?; // laser_planes_rig, laser_planes_cam, per_cam_stats, ...
```

`to_upstream_calibration` takes one board pose per laser view, through the hand-eye chain: the export's own `rig_se3_target` when the laser views are its views, or the chain at the laser views' robot poses (`EyeInHand`: `T_R_T = T_G_R⁻¹ · T_B_G⁻¹ · T_B_T`; `EyeToHand`: `T_R_T = T_R_B · T_B_G · T_G_T`).

**Uncertainty.** The planes are fitted with the upstream calibration held at its values, so a plane moves with the cameras, their extrinsics and the board poses it was fitted against. `RigUpstreamCalibration::uncertainty` is the upstream's covariance over those (`cam1.fx`, `cam1.cam_se3_rig.rx`, `view3.rig_se3_target.rx`, …), and `to_upstream_calibration` fills it from the hand-eye calibration's: the cameras and extrinsics as they are, and the hand-eye and target poses as each view's board pose, correlated across views and with the cameras. The robot poses, with any per-view corrections the hand-eye run estimated, count as exact. The export's covariance then has three parts, all in one matrix:

- each plane's own fit, from its laser pixels;
- the upstream's share, `A Σ_u Aᵀ`, with `A` how the fitted planes move with the upstream parameters: it correlates the planes of different cameras;
- the cameras' parameters and extrinsics with their upstream covariance, and their cross terms `A Σ_u` with the planes.

So a laser point measured with this calibration (`triangulate_laser` on `export.calibrated_rig()`) carries the whole calibration, as with the joint solve below, in the rig frame. The export does not carry the hand-eye: for points in the gripper or base frame with the hand-eye's share, measure with the joint solve's export, whose rig carries the hand-eye correlated with the planes (see [Points in the Robot's Frames](mvg_pose.md#points-in-the-robots-frames)). With the upstream's uncertainty unavailable (an upstream built by hand, the default), the upstream counts as exact: the covariance lists the planes alone, conditional on the upstream, and a laser point's uncertainty is unavailable for want of the cameras'. The rig hand-eye run's covariance holds the extrinsics only when it refines them (`handeye_optimize.refine_cam_se3_rig`); otherwise they count as exact here too.

In Python, `vc.rig_upstream_calibration(export, export.rig_se3_target)` gives the same `RigUpstreamCalibration`, its `uncertainty` the covariance's serde payload.

The steps `step_init` and `step_optimize` are available for finer control. `dataset` is a `RigLaserlineDataset` (`RigLaserlineDataset::new(views, num_cameras)`, also exported from this module; its view type `RigLaserlineView` is in `vision_calibration::common`) whose views hold, per camera, target correspondences and the extracted laser pixels. Laser pixel extraction itself is left to the application.

## RigHandeyeLaserline

`RigHandeyeLaserlineProblem` solves the whole chain jointly: it runs rig hand-eye, initializes the per-camera laser planes from the frozen hand-eye geometry, then refines rig, hand-eye, and laser parameters together.

<a id="one-camera"></a>
**One camera.** A single laser head on a robot is a rig of one camera: `num_cameras: 1`, one target and one laser slot per view. The camera's frame is the rig's, so the hand-eye is the camera's (`gripper_se3_rig` is `gripper_se3_cam`). Its camera, laser plane and hand-eye come out of one solve, in one camera frame.

```rust
pub struct RigHandeyeLaserlineConfig {
    pub handeye: RigHandeyeConfig,                  // warm-start stage (incl. `camera`)
    pub laserline_init: RigLaserlineDeviceConfig,   // plane-initialization stage
    pub joint_optimize: JointOptimizeConfig,        // reprojection_loss, laser_loss,
                                                    // reprojection_weight, laser_weight,
                                                    // laser_residual_type, fix,
                                                    // fix_handeye, ...
    pub solver: SolverConfig,                       // max_iters defaults to 30
}
```

`joint_optimize.fix` holds one `CameraFix` per camera, named in that camera's model.
Left empty, every camera gets the joint stage's default: intrinsics, the
low-order radial terms (`k1`, `k2`; the division model's `lambda`) and the
sensor tilt refined, the higher-order distortion terms held at their
warm-start values, because they trade off against the extrinsics and the
laser planes. An entry's slot left out keeps the model's default, as in every
`CameraFix`.

The input is a `RigHandeyeLaserlineInput { views, num_cameras }`. Each `RigHandeyeLaserlineView` (exported from this module; its parts `RigLaserlineView` and `RobotPoseMeta` are in `vision_calibration::common`) carries the per-camera target observations, the laser pixels, and the robot pose (`base_se3_gripper`). There is a single entry point:

```rust
use vision_calibration::rig_handeye_laserline::{RigHandeyeLaserlineProblem, run_calibration};

let mut session = CalibrationSession::<RigHandeyeLaserlineProblem>::new();
session.set_input(input)?;
run_calibration(&mut session)?;
let export = session.export()?; // cam_se3_rig, handeye (gripper_se3_rig and base_se3_target), laser_planes_rig, per_cam_stats, ...
```

When the rig and its hand-eye are already calibrated (a staged procedure that checks each stage before the next), or the laser planes are known from the design, `run_calibration_with_seed` starts the joint solve from them: its `JointManualInit` takes a `RigHandeyeExport` of the same views (`rig_handeye`) and one plane per camera in camera frame (`laser_planes_cam`). [Seeding with Prior Knowledge](manual_init.md#staged-rig-laser-calibration) shows both.

As in the single-camera laser problem, corner and laser residuals use separate robust losses (`joint_optimize.reprojection_loss`, `joint_optimize.laser_loss`). The embedded `handeye` and `laserline_init` configs keep their own stage losses (`handeye.intrinsics_optimize`, `handeye.rig`, `handeye.handeye_optimize`, `laserline_init.optimize.laser_loss`); the `solver` group holds the engine settings of the joint stage.

## Examples

- Rust: `cargo run -p vision-calibration --example laserline_device_session` (single-camera laser plane with a Scheimpflug-capable sensor) and `cargo run -p vision-calibration --example rig_handeye_synthetic` (the rig hand-eye stage that feeds the rig laser problems).
- Rust tests showing seeded Scheimpflug calibration: `crates/vision-calibration/tests/planar_intrinsics_scheimpflug.rs` (one camera) and `crates/vision-calibration/tests/rig_extrinsics_scheimpflug.rs` (a rig).
- Python: `crates/vision-calibration-py/examples/laserline_device_session.py` and `rig_handeye_synthetic.py`; a tilted single camera runs through `planar_intrinsics.run_calibration` with `camera.model.sensor = "scheimpflug"`, and the package also exposes `rig_laserline_device.run_calibration` and `rig_handeye_laserline.run_calibration`.
