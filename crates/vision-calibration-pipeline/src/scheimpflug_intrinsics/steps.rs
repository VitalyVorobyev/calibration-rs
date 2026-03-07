//! Step functions for Scheimpflug intrinsics calibration.

use std::collections::HashMap;

use anyhow::{Context, Result, anyhow, ensure};
use nalgebra::DVector;
use vision_calibration_core::{
    BrownConrady5, Camera, CameraParams, DistortionParams, FxFyCxCySkew, IntrinsicsParams, Iso3,
    Mat3, Pinhole, PlanarDataset, ProjectionParams, Pt2, ScheimpflugParams, SensorParams,
};
use vision_calibration_linear::{
    DistortionFitOptions, IterativeIntrinsicsOptions, dlt_homography,
    estimate_intrinsics_iterative, estimate_planar_pose_from_h,
};
use vision_calibration_optim::{
    BackendKind, BackendSolveOptions, FactorKind, FixedMask, ManifoldKind, ProblemIR,
    ResidualBlock, iso3_to_se3_dvec, se3_dvec_to_iso3, solve_with_backend,
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

    let mut initial_camera = estimate_intrinsics_iterative(
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

    // Tangential terms are intentionally fixed for this workflow.
    initial_camera.dist.p1 = 0.0;
    initial_camera.dist.p2 = 0.0;

    let poses = estimate_initial_poses(&dataset, &initial_camera.k)
        .context("scheimpflug pose initialization failed")?;

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

    let (ir, initial_map) = build_problem_ir(
        &dataset,
        &initial_intrinsics,
        &initial_distortion,
        initial_sensor,
        &initial_poses,
        &session.config,
    )?;

    let solution = solve_with_backend(
        BackendKind::TinySolver,
        &ir,
        &initial_map,
        &BackendSolveOptions {
            max_iters,
            verbosity,
            ..Default::default()
        },
    )
    .context("scheimpflug optimization failed")?;

    let intrinsics = unpack_intrinsics(
        solution
            .params
            .get("cam")
            .ok_or_else(|| anyhow!("missing intrinsics solution block"))?,
    )?;
    let distortion = unpack_distortion(
        solution
            .params
            .get("dist")
            .ok_or_else(|| anyhow!("missing distortion solution block"))?,
    )?;
    let sensor = unpack_scheimpflug(
        solution
            .params
            .get("sensor")
            .ok_or_else(|| anyhow!("missing sensor solution block"))?,
    )?;

    let mut optimized_poses = Vec::with_capacity(dataset.num_views());
    for view_idx in 0..dataset.num_views() {
        let key = format!("pose/{view_idx}");
        let pose = solution
            .params
            .get(&key)
            .ok_or_else(|| anyhow!("missing {key} solution block"))?;
        optimized_poses.push(se3_dvec_to_iso3(pose.as_view())?);
    }

    let result = ScheimpflugIntrinsicsResult {
        params: ScheimpflugIntrinsicsParams {
            camera: scheimpflug_camera_params(intrinsics, distortion, sensor),
            camera_se3_target: optimized_poses.clone(),
        },
        report: solution.solve_report,
        mean_reproj_error: compute_mean_reproj_error(
            &dataset,
            intrinsics,
            distortion,
            sensor,
            &optimized_poses,
        ),
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

fn estimate_initial_poses(
    dataset: &PlanarDataset,
    intrinsics: &FxFyCxCySkew<f64>,
) -> Result<Vec<Iso3>> {
    let k = intrinsics_k_matrix(intrinsics);
    let mut poses = Vec::with_capacity(dataset.num_views());

    for (view_idx, view) in dataset.views.iter().enumerate() {
        let board_xy: Vec<Pt2> = view
            .obs
            .points_3d
            .iter()
            .map(|p| Pt2::new(p.x, p.y))
            .collect();
        let homography = dlt_homography(&board_xy, &view.obs.points_2d)
            .with_context(|| format!("failed to estimate homography for view {}", view_idx))?;
        let pose = estimate_planar_pose_from_h(&k, &homography)
            .with_context(|| format!("failed to estimate pose for view {}", view_idx))?;
        poses.push(pose);
    }

    Ok(poses)
}

fn build_problem_ir(
    dataset: &PlanarDataset,
    intrinsics: &FxFyCxCySkew<f64>,
    distortion: &BrownConrady5<f64>,
    sensor: ScheimpflugParams,
    poses: &[Iso3],
    config: &ScheimpflugIntrinsicsConfig,
) -> Result<(ProblemIR, HashMap<String, DVector<f64>>)> {
    ensure!(
        dataset.num_views() == poses.len(),
        "pose count ({}) must match dataset views ({})",
        poses.len(),
        dataset.num_views()
    );

    let mut ir = ProblemIR::new();
    let mut initial_map = HashMap::new();

    let cam_id = ir.add_param_block(
        "cam",
        4,
        ManifoldKind::Euclidean,
        fixed_mask_from_indices(4, &config.fix_intrinsics.to_indices()),
        None,
    );
    initial_map.insert(
        "cam".to_string(),
        DVector::from_vec(vec![
            intrinsics.fx,
            intrinsics.fy,
            intrinsics.cx,
            intrinsics.cy,
        ]),
    );

    let dist_id = ir.add_param_block(
        "dist",
        5,
        ManifoldKind::Euclidean,
        fixed_mask_from_indices(5, &config.fix_distortion.to_indices()),
        None,
    );
    initial_map.insert(
        "dist".to_string(),
        DVector::from_vec(vec![
            distortion.k1,
            distortion.k2,
            distortion.k3,
            distortion.p1,
            distortion.p2,
        ]),
    );

    let sensor_id = ir.add_param_block(
        "sensor",
        2,
        ManifoldKind::Euclidean,
        fixed_mask_from_indices(2, &config.fix_scheimpflug.to_indices()),
        None,
    );
    initial_map.insert(
        "sensor".to_string(),
        DVector::from_vec(vec![sensor.tilt_x, sensor.tilt_y]),
    );

    let mut pose_ids = Vec::with_capacity(poses.len());
    for (view_idx, pose) in poses.iter().enumerate() {
        let fixed = if config.fix_first_pose && view_idx == 0 {
            FixedMask::all_fixed(7)
        } else {
            FixedMask::all_free()
        };
        let key = format!("pose/{view_idx}");
        let pose_id = ir.add_param_block(&key, 7, ManifoldKind::SE3, fixed, None);
        pose_ids.push(pose_id);
        initial_map.insert(key, iso3_to_se3_dvec(pose));
    }

    for (view_idx, view) in dataset.views.iter().enumerate() {
        let pose_id = pose_ids[view_idx];
        for (point_idx, (pw, uv)) in view
            .obs
            .points_3d
            .iter()
            .zip(view.obs.points_2d.iter())
            .enumerate()
        {
            ir.add_residual_block(ResidualBlock {
                params: vec![cam_id, dist_id, sensor_id, pose_id],
                loss: config.robust_loss,
                factor: FactorKind::ReprojPointPinhole4Dist5Scheimpflug2 {
                    pw: [pw.x, pw.y, pw.z],
                    uv: [uv.x, uv.y],
                    w: view.obs.weight(point_idx),
                },
                residual_dim: 2,
            });
        }
    }

    ir.validate()
        .context("scheimpflug problem IR validation failed")?;
    Ok((ir, initial_map))
}

fn unpack_intrinsics(values: &DVector<f64>) -> Result<FxFyCxCySkew<f64>> {
    ensure!(values.len() == 4, "intrinsics block must have 4 entries");
    Ok(FxFyCxCySkew {
        fx: values[0],
        fy: values[1],
        cx: values[2],
        cy: values[3],
        skew: 0.0,
    })
}

fn unpack_distortion(values: &DVector<f64>) -> Result<BrownConrady5<f64>> {
    ensure!(values.len() == 5, "distortion block must have 5 entries");
    Ok(BrownConrady5 {
        k1: values[0],
        k2: values[1],
        k3: values[2],
        p1: values[3],
        p2: values[4],
        iters: 8,
    })
}

fn unpack_scheimpflug(values: &DVector<f64>) -> Result<ScheimpflugParams> {
    ensure!(values.len() == 2, "scheimpflug block must have 2 entries");
    Ok(ScheimpflugParams {
        tilt_x: values[0],
        tilt_y: values[1],
    })
}

fn compute_mean_reproj_error(
    dataset: &PlanarDataset,
    intrinsics: FxFyCxCySkew<f64>,
    distortion: BrownConrady5<f64>,
    sensor: ScheimpflugParams,
    poses: &[Iso3],
) -> f64 {
    let camera = Camera::new(Pinhole, distortion, sensor.compile(), intrinsics);
    let mut sum = 0.0;
    let mut count = 0usize;

    for (view, pose) in dataset.views.iter().zip(poses.iter()) {
        for (p3d, p2d) in view.obs.points_3d.iter().zip(view.obs.points_2d.iter()) {
            let p_cam = pose.transform_point(p3d);
            let Some(projected) = camera.project_point(&p_cam) else {
                continue;
            };
            let err = (projected - *p2d).norm();
            if err.is_finite() {
                sum += err;
                count += 1;
            }
        }
    }

    if count == 0 { 0.0 } else { sum / count as f64 }
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

fn fixed_mask_from_indices(dim: usize, indices: &[usize]) -> FixedMask {
    if indices.is_empty() {
        FixedMask::all_free()
    } else if indices.len() == dim {
        FixedMask::all_fixed(dim)
    } else {
        FixedMask::fix_indices(indices)
    }
}

fn intrinsics_k_matrix(k: &FxFyCxCySkew<f64>) -> Mat3 {
    Mat3::new(k.fx, k.skew, k.cx, 0.0, k.fy, k.cy, 0.0, 0.0, 1.0)
}
