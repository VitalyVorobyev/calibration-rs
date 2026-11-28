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
instead of relying on the from-scratch initialization.

Only the lens distortion determines the tilt. Without distortion a tilted
camera is exactly a pinhole with a shifted principal point whose poses are
rotated, from any views; in a rig or a hand-eye the rotation goes into the
extrinsics or the hand-eye the same way. So a free tilt with no distortion
(`DistortionKind::None`, in `camera` or a rig's `camera_overrides`) is a
configuration error in every workflow, and so is a free tilt with every
distortion term held at zero, refused before the solve. Free a radial term
(`k1`), or hold the tilt (`fix.sensor = ["tilt_x", "tilt_y"]`). Distortion
held at known, non-zero values (a lens calibrated before) does determine
it. Where the distortion is weak the tilt is determined only poorly: the
solve may then need more iterations than the default budget, and the
quality report names the tilt and the principal point as the parameters
the data does not separate.

```rust,no_run
use vision_calibration::common::IntrinsicsManualInit;
use vision_calibration::core::{FxFyCxCy, PlanarDataset, ScheimpflugParams, SensorKind};
use vision_calibration::planar_intrinsics::{
    PlanarIntrinsicsConfig, PlanarIntrinsicsProblem, step_init_with_seed, step_optimize,
};
use vision_calibration::session::CalibrationSession;
# fn main() -> anyhow::Result<()> {
# let dataset: PlanarDataset = unimplemented!("the observed views");

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
# Ok(())
# }
```

`run_calibration(&mut session)` runs the unseeded `step_init` and `step_optimize` in one call. The [Planar Intrinsics](planar_intrinsics.md#tilted-scheimpflug-sensor) chapter describes the seeded and cold-start solves.

## Camera Models for Rigs

`RigExtrinsicsProblem` and `RigHandeyeProblem` take a `camera` (the model every camera uses) and `camera_overrides` (per-camera replacements). Any distortion kind combines with either sensor, per camera:

```rust
use vision_calibration::common::config::{CameraConfig, CameraOverride};
use vision_calibration::core::{CameraModelSpec, DistortionKind, SensorKind};
# use vision_calibration::rig_extrinsics::RigExtrinsicsProblem;
# use vision_calibration::session::CalibrationSession;
# fn main() -> anyhow::Result<()> {
# let mut session = CalibrationSession::<RigExtrinsicsProblem>::new();

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
# Ok(())
# }
```

The default is a Brown-Conrady camera with an identity sensor. `camera.fix` and each override's `fix` name the parameters held fixed, per slot (`intrinsics`, `distortion`, `sensor`); a slot left out keeps the model's default, which for a Scheimpflug camera holds everything but `k1`, `k2`. `rig.refine_intrinsics` keeps these fix lists in the rig bundle adjustment, and `rig.refine_sensor` (hand-eye: `handeye_optimize.refine_sensor`) refines the tilts there too. Both default to on: a camera's own views pin its tilt poorly, as it trades with the principal point, while the views of the whole rig pin it several times better. A tilt held at the camera's own estimate leaves its error in the focal length and the principal point, and the covariance, conditional on the held tilt, then states their σ several times too small; setting the flag to `false` holds it there, and the covariance lists it under `fixed`.

Tilt seeds are device data, not config: pass `per_cam_sensors` in the manual-init struct (`RigHandeyeConfig::manual_init`, or `common::RigIntrinsicsManualInit` with `step_intrinsics_init_all_with_seed` for rig extrinsics). A seed is used verbatim and skips the linear stage's tilt sweep; without one the linear stage sweeps the tilt grid, which includes zero. A non-zero tilt seed for an untilted camera is an error.

Each tilted camera is first calibrated from its own views, as in single-camera planar intrinsics: the linear stage holds `p1`, `p2` at zero (a tangential term and the tilt trade off, and the default fix list holds them in every later solve), and the camera's own solve frees every target pose. On clean synthetic data this solve reaches the reprojection noise floor, with the principal point and tilt at the truth within their standard deviations.

From scratch (nothing seeded, for any camera), a tilted camera's own solve can still settle in a wrong tilt / principal-point basin, which the rig bundle adjustment, a local solve, may not leave. Both rig workflows then solve each tilted camera through a wider multi-start over linear tilt seeds, and re-solve a Brown-Conrady tilted camera that still fits worse than 0.5 px from a nominal seed built from the rig's well-fitted cameras. This costs extra solves (several times the per-camera stage on a hard rig), and a seeded rig skips it.

## RigLaserlineDevice

Given a frozen rig calibration, `RigLaserlineDeviceProblem` fits one laser plane per camera and reports each plane both in the camera frame and in the rig frame.

```rust,ignore
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

```rust,no_run
use vision_calibration::rig_laserline_device::{
    RigLaserlineDataset, RigLaserlineDeviceInput, RigLaserlineDeviceProblem, run_calibration,
};
use vision_calibration::session::CalibrationSession;
# use vision_calibration::rig_handeye::RigHandeyeExport;
# fn main() -> anyhow::Result<()> {
# let rig_export: RigHandeyeExport = unimplemented!("a hand-eye calibration's export");
# let dataset: RigLaserlineDataset = unimplemented!("the laser views");

// The laser views are the hand-eye run's views: its own board poses.
let upstream = rig_export.to_upstream_calibration(rig_export.rig_se3_target.clone())?;
let mut session = CalibrationSession::<RigLaserlineDeviceProblem>::new();
session.set_input(RigLaserlineDeviceInput::new(dataset, upstream))?;
run_calibration(&mut session)?;
let export = session.export()?; // laser_planes_rig, laser_planes_cam, per_cam_stats, ...
# Ok(())
# }
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

```rust,ignore
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

The input is a `RigHandeyeLaserlineInput::new(views, num_cameras)`. Each `RigHandeyeLaserlineView` (exported from this module; its parts `RigLaserlineView` and `RobotPoseMeta` are in `vision_calibration::common`) carries the per-camera target observations, the laser pixels, and the robot pose (`base_se3_gripper`). There is a single entry point:

```rust,no_run
use vision_calibration::rig_handeye_laserline::{
    RigHandeyeLaserlineInput, RigHandeyeLaserlineProblem, run_calibration,
};
use vision_calibration::session::CalibrationSession;
# fn main() -> anyhow::Result<()> {
# let input: RigHandeyeLaserlineInput = unimplemented!("the observed views, laser pixels and robot poses");

let mut session = CalibrationSession::<RigHandeyeLaserlineProblem>::new();
session.set_input(input)?;
run_calibration(&mut session)?;
let export = session.export()?; // cam_se3_rig, handeye (gripper_se3_rig and base_se3_target), laser_planes_rig, per_cam_stats, ...
# Ok(())
# }
```

When the rig and its hand-eye are already calibrated (a staged procedure that checks each stage before the next), or the laser planes are known from the design, `run_calibration_with_seed` starts the joint solve from them: its `JointManualInit` takes a `RigHandeyeExport` of the same views (`rig_handeye`) and one plane per camera in camera frame (`laser_planes_cam`). [Seeding with Prior Knowledge](manual_init.md#staged-rig-laser-calibration) shows both.

As in the single-camera laser problem, corner and laser residuals use separate robust losses (`joint_optimize.reprojection_loss`, `joint_optimize.laser_loss`). The embedded `handeye` and `laserline_init` configs keep their own stage losses (`handeye.intrinsics_optimize`, `handeye.rig`, `handeye.handeye_optimize`, `laserline_init.optimize.laser_loss`); the `solver` group holds the engine settings of the joint stage.

## A Rig Without a Robot

A rig that holds still while the board moves (or moves with no robot pose to read) has no hand-eye chain to close, and `RigLaserlineDevice` holds the board poses fixed. The joint solve is then a problem of your own ([Your Own Residual](custom_residual.md)) of the library's residuals alone: one board pose per capture shared by every camera that saw it, each camera's laser plane in its own frame, the cameras and their extrinsics free or held as you choose. Its residuals are the corner reprojections and the laser residual, `add_laser`, through the same chains; the board is the plane `z = 0` of its frame.

- **The gauge.** A rig frame is a convention: make it one camera's frame. That reference camera reaches a board through the board pose alone (`PoseChain::Pose`); every other camera through its `cam_se3_rig` (`PoseChain::Rig`). Nothing else fixes the frame, so give no camera both.
- **What is held.** `CameraFix` names each camera's held parameters (`tilt_y`, `k3`, …), `fix` holds a known extrinsic or board pose, `fix_plane` a known plane. A held block keeps its value exactly and leaves the covariance.
- **Coverage.** A capture needs corners from at least one camera; a camera's laser line on a capture whose corners another camera saw still constrains its plane through the shared board pose. Each camera must share captures with the reference camera (directly or through others), and each plane needs laser lines on boards in at least two orientations. A camera that shares no capture leaves its extrinsic and its boards' poses free to trade off: the solve still runs, and its covariance is unavailable as `RankDeficient`.
- **The residual.** `LaserlineResidualType::PointToPlane` is metric: the pixel's ray meets the board and the residual is that point's distance from the plane, in metres. `LineDistNormalized` is the pixel's distance from the imaged laser line, in pixels; a fisheye camera takes point-to-plane. Each kind's noise level is estimated from its residuals (`ResidualKind::LaserPlane`, metres; `LaserLine`, pixels). Since every residual's noise is estimated, the problem cannot also take a calibration added with `Use::Prior`.

```rust,no_run
use vision_calibration::core::{CalibratedRig, CameraFix, CameraParams, Iso3, LaserPlane, Pt2, Pt3};
use vision_calibration::optim::problem::{
    LaserNoise, PixelNoise, PoseChain, Problem, SolveOptions, TargetPoint, solve,
};
use vision_calibration::optim::{LaserlineResidualType, RobustLoss};
# fn main() -> anyhow::Result<()> {
# let cameras: Vec<CameraParams> = unimplemented!("the seeded cameras");
# let cam_se3_rig: Vec<Iso3> = unimplemented!("each camera's seeded extrinsic; camera 0 is the rig");
# let planes: Vec<LaserPlane> = unimplemented!("each camera's seeded laser plane, in its frame");
# let rig_se3_target: Vec<Iso3> = unimplemented!("each capture's seeded board pose");
# let corners: Vec<Vec<Option<Vec<(Pt3, Pt2)>>>> = unimplemented!("[capture][camera] corners");
# let laser: Vec<Vec<Option<Vec<Pt2>>>> = unimplemented!("[capture][camera] laser pixels");

let mut problem = Problem::new();
// Camera 1 is tilted: hold its tilt_y, refine the rest.
let mut tilted = CameraFix::default();
tilted.sensor = Some(vec!["tilt_y".into()]);
let fixes = [CameraFix::default(), tilted];
let mut cams = Vec::new();
let mut extrinsics = Vec::new();
let mut lasers = Vec::new();
for (i, camera) in cameras.iter().enumerate() {
    cams.push(problem.add_camera(camera, &fixes[i])?);
    // Camera 0 is the rig frame: no extrinsic block (the gauge).
    extrinsics.push(if i == 0 {
        None
    } else {
        Some(problem.add_pose(format!("cam{i}.cam_se3_rig"), &cam_se3_rig[i])?)
    });
    lasers.push(problem.add_plane(format!("cam{i}.plane"), &planes[i])?);
}
// A known extrinsic is held: problem.fix(extrinsics[1].unwrap())?;
let mut boards = Vec::new();
for (k, pose) in rig_se3_target.iter().enumerate() {
    let board = problem.add_pose(format!("capture{k}.rig_se3_target"), pose)?;
    for (i, &cam) in cams.iter().enumerate() {
        let chain = match extrinsics[i] {
            None => PoseChain::Pose(board),
            Some(cam_se3_rig) => PoseChain::Rig { cam_se3_rig, rig_se3_target: board },
        };
        for (point, pixel) in corners[k][i].iter().flatten() {
            problem.add_reprojection(
                cam,
                chain,
                TargetPoint::Known(*point),
                *pixel,
                PixelNoise::Estimated,
                RobustLoss::None,
            )?;
        }
        for pixel in laser[k][i].iter().flatten() {
            problem.add_laser(
                cam,
                chain,
                lasers[i],
                *pixel,
                LaserlineResidualType::PointToPlane,
                LaserNoise::Estimated,
                RobustLoss::None,
            )?;
        }
    }
    boards.push(board);
}

let solution = solve(&problem, &SolveOptions::default())?;
let cameras: Vec<CameraParams> = cams.iter().map(|&c| solution.camera(c).unwrap()).collect();
let cam_se3_rig: Vec<Iso3> = extrinsics
    .iter()
    .map(|e| e.map_or(Iso3::identity(), |b| solution.pose(b).unwrap()))
    .collect();
let planes: Vec<Option<LaserPlane>> = lasers.iter().map(|&p| solution.plane(p)).collect();
let boards: Vec<Iso3> = boards.iter().map(|&b| solution.pose(b).unwrap()).collect();
// Labelled as the workflows' results, the covariance carries over to a
// rig for laser measurements.
let rig = CalibratedRig::new(cameras, cam_se3_rig, solution.uncertainty.clone())?
    .with_laser_planes(planes)?;
# Ok(())
# }
```

`solution.noise` holds the reprojection noise in pixels and the laser noise in its unit, estimated from the residuals at the solution; recomputing them from the solved cameras, poses and planes gives the same values. The names `cam<i>.cam_se3_rig` and `cam<i>.plane` give the covariance the labels of the workflows' results (`cam1.cam_se3_rig.rx`, `cam1.plane.nx`, `cam1.fx`), which `CalibratedRig` reads; the reference camera has no extrinsic, which counts as exact.

The solve is Rust-only, as the builder is; in Python, the rig laser workflows above cover a rig whose board poses come from a robot or from an earlier calibration.

## Examples

- Rust: `cargo run -p vision-calibration --example laserline_device_session` (single-camera laser plane with a Scheimpflug-capable sensor) and `cargo run -p vision-calibration --example rig_handeye_synthetic` (the rig hand-eye stage that feeds the rig laser problems).
- Rust tests showing seeded Scheimpflug calibration: `crates/vision-calibration/tests/planar_intrinsics_scheimpflug.rs` (one camera) and `crates/vision-calibration/tests/rig_extrinsics_scheimpflug.rs` (a rig).
- Python: `crates/vision-calibration-py/examples/laserline_device_session.py` and `rig_handeye_synthetic.py`; a tilted single camera runs through `planar_intrinsics.run_calibration` with `camera.model.sensor = "scheimpflug"`, and the package also exposes `rig_laserline_device.run_calibration` and `rig_handeye_laserline.run_calibration`.
