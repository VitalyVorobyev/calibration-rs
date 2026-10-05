//! Per-problem drivers: run a scene through the public step functions and
//! time the init and optimize phases separately.

use std::time::{Duration, Instant};

use anyhow::{Result, bail};
use vision_calibration::core::{
    CameraParams, DistortionParams, IntrinsicsParams, PinholeCamera, ScheimpflugParams,
    SensorParams,
};
use vision_calibration::laserline_device::{
    LaserlineDeviceProblem, step_init as laser_init, step_optimize as laser_optimize,
};
use vision_calibration::planar_intrinsics::{
    PlanarIntrinsicsProblem, step_init as planar_init, step_optimize as planar_optimize,
};
use vision_calibration::rig_extrinsics::{
    RigExtrinsicsOutput, RigExtrinsicsProblem, step_intrinsics_init_all_with_seed,
    step_intrinsics_optimize_all, step_rig_init, step_rig_optimize,
};
use vision_calibration::rig_handeye::{
    RigHandeyeOutput, RigHandeyeProblem, step_handeye_init as rh_handeye_init,
    step_handeye_optimize as rh_handeye_optimize,
    step_intrinsics_init_all_with_seed as rh_intrinsics_init,
    step_intrinsics_optimize_all as rh_intrinsics_optimize, step_rig_init as rh_rig_init,
    step_rig_optimize as rh_rig_optimize,
};
use vision_calibration::rig_handeye_laserline::{
    RigHandeyeLaserlineProblem, run_calibration as rhl_run,
};
use vision_calibration::rig_laserline_device::{
    RigLaserlineDeviceProblem, step_init as rl_init, step_optimize as rl_optimize,
};
use vision_calibration::scheimpflug_intrinsics::{
    ScheimpflugIntrinsicsProblem, step_init_with_seed as sch_init, step_optimize as sch_optimize,
};
use vision_calibration::session::CalibrationSession;
use vision_calibration::single_cam_handeye::{
    SingleCamHandeyeProblem, step_handeye_init, step_handeye_optimize, step_intrinsics_init,
    step_intrinsics_optimize,
};
use vision_calibration_core::PerFeatureResiduals;
use vision_calibration_optim::SolveReport;

use super::metrics::{CameraParamsSet, ModelParams};
use super::record::{SolverTiming, TimingSample};
use super::scenes::{Scene, SceneData};

/// The result of timing a scene.
#[derive(Debug, Clone)]
pub struct Measured {
    /// Median timing and raw samples.
    pub timing: SolverTiming,
    /// Estimated model from the last timed run.
    pub params: ModelParams,
    /// Per-feature residuals from the last timed run.
    pub residuals: PerFeatureResiduals,
    /// Backend report from the last timed run, when exposed.
    pub report: Option<SolveReport>,
}

/// One solve: timing plus the extracted model.
struct Solved {
    sample: TimingSample,
    params: ModelParams,
    residuals: PerFeatureResiduals,
    report: Option<SolveReport>,
}

/// Accumulates the time spent in init and optimize steps.
#[derive(Default)]
struct Phases {
    init: Option<Duration>,
    optimize: Duration,
}

impl Phases {
    fn init<T>(&mut self, f: impl FnOnce() -> Result<T, vision_calibration::Error>) -> Result<T> {
        let t = Instant::now();
        let out = f();
        *self.init.get_or_insert_default() += t.elapsed();
        Ok(out?)
    }

    fn optimize<T>(
        &mut self,
        f: impl FnOnce() -> Result<T, vision_calibration::Error>,
    ) -> Result<T> {
        let t = Instant::now();
        let out = f();
        self.optimize += t.elapsed();
        Ok(out?)
    }

    fn sample(&self) -> TimingSample {
        TimingSample {
            init_ms: self.init.map(|d| d.as_secs_f64() * 1e3),
            optimize_ms: self.optimize.as_secs_f64() * 1e3,
        }
    }
}

/// Run `scene` once untimed (warm-up), then `repeats` timed times, each from a
/// fresh session.
///
/// # Errors
///
/// Returns the first pipeline error.
pub fn measure(scene: &Scene, repeats: usize) -> Result<Measured> {
    if repeats == 0 {
        bail!("repeats must be at least 1");
    }
    solve_once(scene)?;
    let mut samples = Vec::with_capacity(repeats);
    let mut last = None;
    for _ in 0..repeats {
        let solved = solve_once(scene)?;
        samples.push(solved.sample.clone());
        last = Some(solved);
    }
    let last = last.expect("repeats >= 1");
    Ok(Measured {
        timing: SolverTiming::from_samples(samples).expect("repeats >= 1"),
        params: last.params,
        residuals: last.residuals,
        report: last.report,
    })
}

fn camera_set(camera: &CameraParams) -> Result<CameraParamsSet> {
    let IntrinsicsParams::FxFyCxCySkew { params: k } = camera.intrinsics;
    let DistortionParams::BrownConrady5 { params: dist } = camera.distortion else {
        bail!(
            "expected Brown-Conrady distortion, got {:?}",
            camera.distortion
        );
    };
    let sensor = match camera.sensor {
        SensorParams::Identity => None,
        SensorParams::Scheimpflug { params } => Some(params),
        ref other => bail!("unexpected sensor params {other:?}"),
    };
    Ok(CameraParamsSet { k, dist, sensor })
}

fn rig_cameras(
    cameras: &[PinholeCamera],
    sensors: Option<&[ScheimpflugParams]>,
) -> Vec<CameraParamsSet> {
    cameras
        .iter()
        .enumerate()
        .map(|(i, c)| CameraParamsSet {
            k: c.k,
            dist: c.dist,
            sensor: sensors.map(|s| s[i]),
        })
        .collect()
}

fn solve_once(scene: &Scene) -> Result<Solved> {
    let mut ph = Phases::default();
    let (params, residuals, report) = match &scene.data {
        SceneData::Planar { input, config } => {
            let mut s = CalibrationSession::<PlanarIntrinsicsProblem>::new();
            s.set_config(config.clone())?;
            s.set_input(input.clone())?;
            ph.init(|| planar_init(&mut s, None))?;
            ph.optimize(|| planar_optimize(&mut s, None))?;
            let e = s.export()?;
            (
                ModelParams {
                    cameras: vec![camera_set(&e.params.camera)?],
                    ..ModelParams::default()
                },
                e.per_feature_residuals,
                Some(e.report),
            )
        }
        SceneData::Scheimpflug {
            input,
            config,
            seed,
        } => {
            let mut s = CalibrationSession::<ScheimpflugIntrinsicsProblem>::new();
            s.set_config(config.clone())?;
            s.set_input(input.clone())?;
            ph.init(|| sch_init(&mut s, seed.clone(), None))?;
            ph.optimize(|| sch_optimize(&mut s, None))?;
            let e = s.export()?;
            (
                ModelParams {
                    cameras: vec![camera_set(&e.params.camera)?],
                    ..ModelParams::default()
                },
                e.per_feature_residuals,
                Some(e.report),
            )
        }
        SceneData::SingleCamHandeye { input, config } => {
            let mut s = CalibrationSession::<SingleCamHandeyeProblem>::new();
            s.set_config(config.clone())?;
            s.set_input(input.clone())?;
            ph.init(|| step_intrinsics_init(&mut s, None))?;
            ph.optimize(|| step_intrinsics_optimize(&mut s, None))?;
            ph.init(|| step_handeye_init(&mut s, None))?;
            ph.optimize(|| step_handeye_optimize(&mut s, None))?;
            let report = s.output().map(|o| o.report.clone());
            let e = s.export()?;
            (
                ModelParams {
                    cameras: rig_cameras(std::slice::from_ref(&e.camera), None),
                    handeye: e.gripper_se3_camera,
                    ..ModelParams::default()
                },
                e.per_feature_residuals,
                report,
            )
        }
        SceneData::Laserline { input, config } => {
            let mut s = CalibrationSession::<LaserlineDeviceProblem>::new();
            s.set_config(config.clone())?;
            s.set_input(input.clone())?;
            ph.init(|| laser_init(&mut s, None))?;
            ph.optimize(|| laser_optimize(&mut s, None))?;
            let e = s.export()?;
            let p = &e.estimate.params;
            (
                ModelParams {
                    cameras: vec![CameraParamsSet {
                        k: p.intrinsics,
                        dist: p.distortion,
                        sensor: Some(p.sensor),
                    }],
                    planes_cam: vec![p.plane.clone()],
                    ..ModelParams::default()
                },
                e.per_feature_residuals,
                Some(e.estimate.report.clone()),
            )
        }
        SceneData::Rig {
            input,
            config,
            seed,
        } => {
            let mut s = CalibrationSession::<RigExtrinsicsProblem>::new();
            s.set_config(config.clone())?;
            s.set_input(input.clone())?;
            ph.init(|| step_intrinsics_init_all_with_seed(&mut s, seed.clone(), None))?;
            ph.optimize(|| step_intrinsics_optimize_all(&mut s, None))?;
            ph.init(|| step_rig_init(&mut s))?;
            ph.optimize(|| step_rig_optimize(&mut s, None))?;
            let report = match s.output() {
                Some(RigExtrinsicsOutput::Pinhole(e)) => Some(e.report.clone()),
                Some(RigExtrinsicsOutput::Scheimpflug(e)) => Some(e.report.clone()),
                _ => None,
            };
            let e = s.export()?;
            (
                ModelParams {
                    cameras: rig_cameras(&e.cameras, e.sensors.as_deref()),
                    cam_se3_rig: e.cam_se3_rig,
                    ..ModelParams::default()
                },
                e.per_feature_residuals,
                report,
            )
        }
        SceneData::RigHandeye { input, config } => {
            let mut s = CalibrationSession::<RigHandeyeProblem>::new();
            let manual = config.manual_init.clone().unwrap_or_default();
            s.set_config(config.clone())?;
            s.set_input(input.clone())?;
            ph.init(|| rh_intrinsics_init(&mut s, manual, None))?;
            ph.optimize(|| rh_intrinsics_optimize(&mut s, None))?;
            ph.init(|| rh_rig_init(&mut s))?;
            ph.optimize(|| rh_rig_optimize(&mut s, None))?;
            ph.init(|| rh_handeye_init(&mut s, None))?;
            ph.optimize(|| rh_handeye_optimize(&mut s, None))?;
            let report = match s.output() {
                Some(RigHandeyeOutput::Pinhole(e)) => Some(e.report.clone()),
                Some(RigHandeyeOutput::Scheimpflug(e)) => Some(e.report.clone()),
                _ => None,
            };
            let e = s.export()?;
            (
                ModelParams {
                    cameras: rig_cameras(&e.cameras, e.sensors.as_deref()),
                    cam_se3_rig: e.cam_se3_rig,
                    handeye: e.gripper_se3_rig,
                    ..ModelParams::default()
                },
                e.per_feature_residuals,
                report,
            )
        }
        SceneData::RigLaserline { input, config } => {
            let mut s = CalibrationSession::<RigLaserlineDeviceProblem>::new();
            s.set_config(config.clone())?;
            s.set_input(input.clone())?;
            ph.init(|| rl_init(&mut s))?;
            ph.optimize(|| rl_optimize(&mut s, None))?;
            let e = s.export()?;
            (
                ModelParams {
                    planes_cam: e.laser_planes_cam,
                    ..ModelParams::default()
                },
                e.per_feature_residuals,
                None,
            )
        }
        SceneData::RigHandeyeLaserline { input, config } => {
            let mut s = CalibrationSession::<RigHandeyeLaserlineProblem>::new();
            s.set_config(config.clone())?;
            s.set_input(input.clone())?;
            // Only a whole-pipeline entry point exists: time it as one phase.
            ph.optimize(|| rhl_run(&mut s))?;
            let report = s.output().map(|o| o.estimate.report.clone());
            let e = s.export()?;
            (
                ModelParams {
                    cameras: rig_cameras(&e.cameras, Some(&e.sensors)),
                    cam_se3_rig: e.cam_se3_rig,
                    handeye: e.gripper_se3_rig,
                    planes_cam: e.laser_planes_cam,
                },
                e.per_feature_residuals,
                report,
            )
        }
    };
    Ok(Solved {
        sample: ph.sample(),
        params,
        residuals,
        report,
    })
}
