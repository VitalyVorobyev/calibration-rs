//! Step functions for Scheimpflug intrinsics calibration.

use crate::Error;
use serde::{Deserialize, Serialize};
use vision_calibration_core::{
    BrownConrady5, CameraParams, DistortionParams, FxFyCxCySkew, IntrinsicsParams, Iso3,
    ProjectionParams, Real, ScheimpflugParams, SensorParams,
};
use vision_calibration_linear::{DistortionFitOptions, IterativeIntrinsicsOptions};
use vision_calibration_optim::{
    BackendSolveOptions, ScheimpflugFixMask as OptimScheimpflugFixMask,
    ScheimpflugIntrinsicsParams as OptimScheimpflugIntrinsicsParams,
    ScheimpflugIntrinsicsSolveOptions as OptimScheimpflugIntrinsicsSolveOptions,
    optimize_scheimpflug_intrinsics,
};

use crate::planar_family::{
    bootstrap_planar_intrinsics, estimate_view_homographies,
    recover_planar_poses_from_homographies,
};
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

/// Manual initialization seeds for Scheimpflug intrinsics calibration.
///
/// All fields are `Option<T>`:
/// - `None` means *auto-initialize this group* (same path as plain `step_init`).
/// - `Some(value)` means *use this value*; do not auto-initialize.
///
/// Partial-seed semantics mirror `PlanarManualInit`:
/// - `intrinsics: Some` skips the planar bootstrap; distortion defaults to zeros and
///   sensor to identity tilt unless also seeded; poses recover from homographies using
///   the manual intrinsics.
/// - `intrinsics: None` runs the bootstrap; any seeded field overrides the
///   corresponding bootstrap output. The auto path's tangential-distortion zeroing
///   (workflow invariant — Scheimpflug pipelines fix tangential) is **not** applied
///   when the user supplies a manual `distortion` — they get exactly what they pass.
///
/// See ADR 0011 for the design rationale.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ScheimpflugManualInit {
    /// Manual intrinsics seed. `None` means auto-init via Zhang's method.
    pub intrinsics: Option<FxFyCxCySkew<Real>>,
    /// Manual distortion seed. `None` means auto-init (or zeros when intrinsics are
    /// seeded).
    pub distortion: Option<BrownConrady5<Real>>,
    /// Manual Scheimpflug sensor tilts. `None` means start at identity tilt
    /// (`ScheimpflugParams::default()`); the optimizer refines from there.
    pub sensor: Option<ScheimpflugParams>,
    /// Manual per-view poses (`camera_se3_target`). `None` means recover from
    /// homographies using whichever intrinsics are in effect.
    pub poses: Option<Vec<Iso3>>,
}

/// Options for the optimization step.
#[derive(Debug, Clone, Default)]
pub struct IntrinsicsOptimizeOptions {
    /// Override the maximum number of optimization iterations.
    pub max_iters: Option<usize>,
    /// Override solver verbosity.
    pub verbosity: Option<usize>,
}

/// Initialize Scheimpflug intrinsics, distortion, sensor tilt, and per-view poses
/// from any combination of manual seeds and auto-estimation.
///
/// This is the load-bearing init function. [`step_init`] is a thin delegate that
/// passes `ScheimpflugManualInit::default()` (all-`None`, full auto path).
///
/// See [`ScheimpflugManualInit`] for partial-seed semantics.
///
/// # Errors
///
/// - Input not set, or fewer than 3 views.
/// - `init_iterations == 0` when running the bootstrap auto-fit.
/// - Homography or auto-init computation fails.
/// - `manual.poses` is `Some` but its length does not match the view count.
pub fn step_set_init(
    session: &mut CalibrationSession<ScheimpflugIntrinsicsProblem>,
    manual: ScheimpflugManualInit,
    opts: Option<IntrinsicsInitOptions>,
) -> Result<(), Error> {
    session.validate()?;
    let dataset = session.require_input()?.clone();

    let opts = opts.unwrap_or_default();
    let mut init_iterations = session.config.init_iterations;
    if let Some(iterations) = opts.iterations {
        init_iterations = iterations;
    }

    let mut manual_fields: Vec<&'static str> = Vec::new();
    let mut auto_fields: Vec<&'static str> = Vec::new();

    let (intrinsics, distortion, sensor, poses) = if let Some(k) = manual.intrinsics {
        manual_fields.push("intrinsics");

        let dist = match manual.distortion {
            Some(d) => {
                manual_fields.push("distortion");
                d
            }
            None => {
                auto_fields.push("distortion");
                BrownConrady5::default()
            }
        };

        let sensor = match manual.sensor {
            Some(s) => {
                manual_fields.push("sensor");
                s
            }
            None => {
                auto_fields.push("sensor");
                ScheimpflugParams::default()
            }
        };

        let poses = match manual.poses {
            Some(p) => {
                manual_fields.push("poses");
                if p.len() != dataset.num_views() {
                    let msg = format!(
                        "manual poses count ({}) does not match view count ({})",
                        p.len(),
                        dataset.num_views()
                    );
                    session.log_failure("init", msg.clone());
                    return Err(Error::invalid_input(msg));
                }
                p
            }
            None => {
                auto_fields.push("poses");
                let homographies = estimate_view_homographies(&dataset).map_err(|e| {
                    Error::numerical(format!(
                        "scheimpflug intrinsics homography estimation failed: {e}"
                    ))
                })?;
                recover_planar_poses_from_homographies(&homographies, &k).map_err(|e| {
                    Error::numerical(format!(
                        "scheimpflug intrinsics pose recovery failed: {e}"
                    ))
                })?
            }
        };

        (k, dist, sensor, poses)
    } else {
        if init_iterations == 0 {
            return Err(Error::invalid_input("init_iterations must be positive"));
        }
        auto_fields.push("intrinsics");

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
        .map_err(|e| {
            Error::numerical(format!("scheimpflug intrinsics initialization failed: {e}"))
        })?;

        let dist = match manual.distortion {
            Some(d) => {
                manual_fields.push("distortion");
                d
            }
            None => {
                auto_fields.push("distortion");
                // Workflow invariant: Scheimpflug pipelines fix tangential distortion.
                let mut d = bootstrap.camera.dist;
                d.p1 = 0.0;
                d.p2 = 0.0;
                d
            }
        };

        let sensor = match manual.sensor {
            Some(s) => {
                manual_fields.push("sensor");
                s
            }
            None => {
                auto_fields.push("sensor");
                ScheimpflugParams::default()
            }
        };

        let poses = match manual.poses {
            Some(p) => {
                manual_fields.push("poses");
                if p.len() != dataset.num_views() {
                    let msg = format!(
                        "manual poses count ({}) does not match view count ({})",
                        p.len(),
                        dataset.num_views()
                    );
                    session.log_failure("init", msg.clone());
                    return Err(Error::invalid_input(msg));
                }
                p
            }
            None => {
                auto_fields.push("poses");
                bootstrap.poses
            }
        };

        (bootstrap.camera.k, dist, sensor, poses)
    };

    session.state.initial_intrinsics = Some(intrinsics);
    session.state.initial_distortion = Some(distortion);
    session.state.initial_sensor = Some(sensor);
    session.state.initial_poses = Some(poses);
    session.state.clear_optimization();

    let source = format_init_source(&manual_fields, &auto_fields);
    session.log_success_with_notes(
        "init",
        format!(
            "fx={:.1}, fy={:.1}, views={} {}",
            intrinsics.fx,
            intrinsics.fy,
            dataset.num_views(),
            source
        ),
    );

    Ok(())
}

fn format_init_source(manual: &[&str], auto: &[&str]) -> String {
    match (manual.is_empty(), auto.is_empty()) {
        (false, false) => format!(
            "(manual: {}; auto: {})",
            manual.join(", "),
            auto.join(", ")
        ),
        (false, true) => format!("(manual: {})", manual.join(", ")),
        (true, false) => format!("(auto: {})", auto.join(", ")),
        (true, true) => "(empty)".to_string(),
    }
}

/// Initialize intrinsics, distortion, sensor tilt, and poses from observations
/// using full auto-init.
///
/// Convenience wrapper around [`step_set_init`] with `ScheimpflugManualInit::default()`.
pub fn step_init(
    session: &mut CalibrationSession<ScheimpflugIntrinsicsProblem>,
    opts: Option<IntrinsicsInitOptions>,
) -> Result<(), Error> {
    step_set_init(session, ScheimpflugManualInit::default(), opts)
}

/// Optimize Scheimpflug intrinsics, distortion, sensor tilt, and target poses.
pub fn step_optimize(
    session: &mut CalibrationSession<ScheimpflugIntrinsicsProblem>,
    opts: Option<IntrinsicsOptimizeOptions>,
) -> Result<(), Error> {
    session.validate()?;
    let dataset = session.require_input()?.clone();

    let (initial_intrinsics, initial_distortion, initial_sensor, initial_poses) = session
        .state
        .initial_values()
        .ok_or_else(|| Error::not_available("initial params (call step_init first)"))?;

    let opts = opts.unwrap_or_default();
    let mut max_iters = session.config.max_iters;
    let mut verbosity = session.config.verbosity;
    if let Some(v) = opts.max_iters {
        max_iters = v;
    }
    if let Some(v) = opts.verbosity {
        verbosity = v;
    }
    if max_iters == 0 {
        return Err(Error::invalid_input("max_iters must be positive"));
    }

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
    )?;

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
) -> Result<(), Error> {
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

    // ─────────────────────────────────────────────────────────────────────────
    // Manual init (ADR 0011) tests
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn step_set_init_default_matches_step_init() {
        let sensor_gt = ScheimpflugParams::default();
        let mut session_a = CalibrationSession::<ScheimpflugIntrinsicsProblem>::new();
        session_a.set_input(make_dataset(sensor_gt)).expect("input");
        step_init(&mut session_a, None).expect("step_init");

        let mut session_b = CalibrationSession::<ScheimpflugIntrinsicsProblem>::new();
        session_b.set_input(make_dataset(sensor_gt)).expect("input");
        step_set_init(&mut session_b, ScheimpflugManualInit::default(), None)
            .expect("step_set_init");

        let k_a = session_a.state.initial_intrinsics.unwrap();
        let k_b = session_b.state.initial_intrinsics.unwrap();
        assert!((k_a.fx - k_b.fx).abs() < 1e-9);
        assert!((k_a.fy - k_b.fy).abs() < 1e-9);
    }

    #[test]
    fn step_set_init_with_intrinsics_and_sensor_seed_converges() {
        let sensor_gt = ScheimpflugParams {
            tilt_x: 0.01,
            tilt_y: -0.008,
        };
        let mut session = CalibrationSession::<ScheimpflugIntrinsicsProblem>::new();
        session.set_input(make_dataset(sensor_gt)).expect("input");

        let manual = ScheimpflugManualInit {
            intrinsics: Some(FxFyCxCySkew {
                fx: 800.0,
                fy: 780.0,
                cx: 640.0,
                cy: 360.0,
                skew: 0.0,
            }),
            distortion: Some(BrownConrady5::default()),
            sensor: Some(sensor_gt),
            poses: None,
        };
        step_set_init(&mut session, manual, None).expect("step_set_init");
        step_optimize(&mut session, None).expect("step_optimize");

        let output = session.output().expect("output");
        assert!(
            output.mean_reproj_error < 1.0,
            "got {:.4}",
            output.mean_reproj_error
        );
    }

    #[test]
    fn step_set_init_rejects_wrong_pose_count() {
        use vision_calibration_core::Iso3;

        let mut session = CalibrationSession::<ScheimpflugIntrinsicsProblem>::new();
        session
            .set_input(make_dataset(ScheimpflugParams::default()))
            .expect("input");

        // Test dataset has 5 views; supply only 1 pose.
        let manual = ScheimpflugManualInit {
            poses: Some(vec![Iso3::identity()]),
            ..Default::default()
        };
        let err = step_set_init(&mut session, manual, None).unwrap_err();
        assert!(
            err.to_string().contains("manual poses count"),
            "unexpected error: {}",
            err
        );
    }
}
