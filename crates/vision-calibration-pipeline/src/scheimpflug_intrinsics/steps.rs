//! Step functions for Scheimpflug intrinsics calibration.

use anyhow::{Context, Result, anyhow, ensure};
use vision_calibration_core::{
    BrownConrady5, CameraParams, DistortionParams, FxFyCxCySkew, IntrinsicsParams,
    ProjectionParams, ScheimpflugParams, SensorParams,
};
use vision_calibration_linear::{DistortionFitOptions, IterativeIntrinsicsOptions};
use vision_calibration_optim::{
    BackendSolveOptions, ScheimpflugFixMask as OptimScheimpflugFixMask,
    ScheimpflugIntrinsicsParams as OptimScheimpflugIntrinsicsParams,
    ScheimpflugIntrinsicsSolveOptions as OptimScheimpflugIntrinsicsSolveOptions,
    optimize_scheimpflug_intrinsics,
};

use crate::planar_family::bootstrap_planar_intrinsics;
use crate::session::CalibrationSession;

use super::problem::{
    ScheimpflugIntrinsicsConfig, ScheimpflugIntrinsicsParams, ScheimpflugIntrinsicsProblem,
    ScheimpflugIntrinsicsResult,
};

/// Options for the initialization step.
#[derive(Debug, Clone, Default)]
pub struct IntrinsicsInitOptions {
    /// Override the number of iterative initialization rounds.
    pub iterations: Option<usize>,
}

/// Options for the optimization step.
#[derive(Debug, Clone, Default)]
pub struct IntrinsicsOptimizeOptions {
    /// Override the maximum number of optimization iterations.
    pub max_iters: Option<usize>,
    /// Override solver verbosity.
    pub verbosity: Option<usize>,
}

/// Initialize intrinsics and poses for Scheimpflug refinement.
pub fn step_init(
    session: &mut CalibrationSession<ScheimpflugIntrinsicsProblem>,
    opts: Option<IntrinsicsInitOptions>,
) -> Result<()> {
    session.validate()?;
    let dataset = session.require_input()?.clone();

    let opts = opts.unwrap_or_default();
    let mut init_iterations = session.config.init_iterations;
    if let Some(iterations) = opts.iterations {
        init_iterations = iterations;
    }
    ensure!(init_iterations > 0, "init_iterations must be positive");

    // Shared planar-family initialization helper (homographies + intrinsics + poses).
    let bootstrap = bootstrap_planar_intrinsics(
        &dataset,
        IterativeIntrinsicsOptions {
            iterations: init_iterations,
            distortion_opts: DistortionFitOptions {
                fix_k3: session.config.fix_k3_in_init,
                fix_tangential: true,
                iters: 8,
            },
            zero_skew: session.config.zero_skew,
        },
    )
    .context("scheimpflug intrinsics initialization failed")?;
    let mut initial_camera = bootstrap.camera;

    // Tangential terms are intentionally fixed for this workflow.
    initial_camera.dist.p1 = 0.0;
    initial_camera.dist.p2 = 0.0;

    let poses = bootstrap.poses;

    session.state.initial_intrinsics = Some(initial_camera.k);
    session.state.initial_distortion = Some(initial_camera.dist);
    session.state.initial_sensor = Some(ScheimpflugParams::default());
    session.state.initial_poses = Some(poses);
    session.state.clear_optimization();

    session.log_success_with_notes(
        "init",
        format!(
            "fx={:.1}, fy={:.1}, views={}",
            initial_camera.k.fx,
            initial_camera.k.fy,
            dataset.num_views()
        ),
    );

    Ok(())
}

/// Optimize Scheimpflug intrinsics, distortion, sensor tilt, and target poses.
pub fn step_optimize(
    session: &mut CalibrationSession<ScheimpflugIntrinsicsProblem>,
    opts: Option<IntrinsicsOptimizeOptions>,
) -> Result<()> {
    session.validate()?;
    let dataset = session.require_input()?.clone();

    let (initial_intrinsics, initial_distortion, initial_sensor, initial_poses) = session
        .state
        .initial_values()
        .ok_or_else(|| anyhow!("initialization not run - call step_init first"))?;

    let opts = opts.unwrap_or_default();
    let mut max_iters = session.config.max_iters;
    let mut verbosity = session.config.verbosity;
    if let Some(v) = opts.max_iters {
        max_iters = v;
    }
    if let Some(v) = opts.verbosity {
        verbosity = v;
    }
    ensure!(max_iters > 0, "max_iters must be positive");

    let initial = OptimScheimpflugIntrinsicsParams::new(
        initial_intrinsics,
        initial_distortion,
        initial_sensor,
        initial_poses,
    )?;
    let solve_opts = OptimScheimpflugIntrinsicsSolveOptions {
        robust_loss: session.config.robust_loss,
        fix_intrinsics: session.config.fix_intrinsics,
        fix_distortion: session.config.fix_distortion,
        fix_scheimpflug: to_optim_scheimpflug_fix_mask(session.config.fix_scheimpflug),
        fix_poses: if session.config.fix_first_pose {
            vec![0]
        } else {
            Vec::new()
        },
    };

    // Shared planar-family optimization path implemented in `vision-calibration-optim`.
    let estimate = optimize_scheimpflug_intrinsics(
        &dataset,
        &initial,
        solve_opts,
        BackendSolveOptions {
            max_iters,
            verbosity,
            ..Default::default()
        },
    )
    .context("scheimpflug optimization failed")?;

    let result = ScheimpflugIntrinsicsResult {
        params: ScheimpflugIntrinsicsParams {
            camera: scheimpflug_camera_params(
                estimate.params.intrinsics,
                estimate.params.distortion,
                estimate.params.sensor,
            ),
            camera_se3_target: estimate.params.camera_se3_target,
        },
        report: estimate.report,
        mean_reproj_error: estimate.mean_reproj_error,
    };

    session.state.final_cost = Some(result.report.final_cost);
    session.state.mean_reproj_error = Some(result.mean_reproj_error);
    session.set_output(result.clone());

    session.log_success_with_notes(
        "optimize",
        format!(
            "cost={:.2e}, reproj_err={:.3}px",
            result.report.final_cost, result.mean_reproj_error
        ),
    );

    Ok(())
}

/// Run full Scheimpflug calibration pipeline on a session: init -> optimize.
pub fn run_calibration(
    session: &mut CalibrationSession<ScheimpflugIntrinsicsProblem>,
    config: Option<ScheimpflugIntrinsicsConfig>,
) -> Result<()> {
    if let Some(cfg) = config {
        session.set_config(cfg)?;
    }
    step_init(session, None)?;
    step_optimize(session, None)?;
    Ok(())
}

fn to_optim_scheimpflug_fix_mask(
    mask: super::problem::ScheimpflugFixMask,
) -> OptimScheimpflugFixMask {
    OptimScheimpflugFixMask {
        tilt_x: mask.tilt_x,
        tilt_y: mask.tilt_y,
    }
}

fn scheimpflug_camera_params(
    intrinsics: FxFyCxCySkew<f64>,
    distortion: BrownConrady5<f64>,
    sensor: ScheimpflugParams,
) -> CameraParams {
    CameraParams {
        projection: ProjectionParams::Pinhole,
        distortion: DistortionParams::BrownConrady5 { params: distortion },
        sensor: SensorParams::Scheimpflug { params: sensor },
        intrinsics: IntrinsicsParams::FxFyCxCySkew { params: intrinsics },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vision_calibration_core::{
        Camera, FxFyCxCySkew, Pinhole, PlanarDataset, View, make_pinhole_camera, synthetic::planar,
    };

    fn make_dataset(sensor: ScheimpflugParams) -> PlanarDataset {
        let base = make_pinhole_camera(
            FxFyCxCySkew {
                fx: 800.0,
                fy: 780.0,
                cx: 640.0,
                cy: 360.0,
                skew: 0.0,
            },
            BrownConrady5::default(),
        );
        let camera = Camera::new(Pinhole, base.dist, sensor.compile(), base.k);

        let board_points = planar::grid_points(6, 5, 0.03);
        let poses = planar::poses_yaw_y_z(5, 0.0, 0.08, 0.55, 0.03);
        let views = planar::project_views_all(&camera, &board_points, &poses).expect("views");
        PlanarDataset::new(views.into_iter().map(View::without_meta).collect()).expect("dataset")
    }

    #[test]
    fn step_optimize_requires_initialization() {
        let mut session = CalibrationSession::<ScheimpflugIntrinsicsProblem>::new();
        session
            .set_input(make_dataset(ScheimpflugParams::default()))
            .expect("input");

        let err = step_optimize(&mut session, None).expect_err("init should be required");
        assert!(err.to_string().contains("step_init"));
    }

    #[test]
    fn run_calibration_sets_output_and_state() {
        let sensor_gt = ScheimpflugParams {
            tilt_x: 0.01,
            tilt_y: -0.008,
        };
        let mut session = CalibrationSession::<ScheimpflugIntrinsicsProblem>::new();
        session.set_input(make_dataset(sensor_gt)).expect("input");

        run_calibration(&mut session, None).expect("run_calibration");

        assert!(session.has_output());
        assert!(session.state.is_initialized());
        assert!(session.state.is_optimized());

        let output = session.output().expect("output");
        assert!(output.mean_reproj_error.is_finite());
        assert!(output.mean_reproj_error < 1.0);
        assert_eq!(output.params.camera_se3_target.len(), 5);
    }
}
