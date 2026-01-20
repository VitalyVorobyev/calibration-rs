//! Step functions for planar intrinsics calibration.
//!
//! This module provides step functions that operate on
//! `CalibrationSession<PlanarIntrinsicsProblemV2>` to perform calibration.
//!
//! # Example
//!
//! ```ignore
//! use calib_pipeline::session::v2::CalibrationSession;
//! use calib_pipeline::planar_intrinsics::{
//!     PlanarIntrinsicsProblemV2, step_init, step_optimize, step_filter,
//!     FilterOptions,
//! };
//!
//! let mut session = CalibrationSession::<PlanarIntrinsicsProblemV2>::new();
//! session.set_input(dataset)?;
//!
//! // Run initialization
//! step_init(&mut session, None)?;
//!
//! // Run optimization
//! step_optimize(&mut session, None)?;
//!
//! // Optionally filter outliers and re-run
//! step_filter(&mut session, FilterOptions::default())?;
//! step_init(&mut session, None)?;
//! step_optimize(&mut session, None)?;
//!
//! // Export result
//! let export = session.export()?;
//! ```

use anyhow::{ensure, Context, Result};
use calib_core::{CorrespondenceView, FxFyCxCySkew, Iso3, Mat3, Pt2, Real, View};
use calib_linear::prelude::*;
use calib_optim::optimize_planar_intrinsics;
use serde::{Deserialize, Serialize};

use crate::session::v2::CalibrationSession;

use super::problem_v2::PlanarIntrinsicsProblemV2;

// ─────────────────────────────────────────────────────────────────────────────
// Step Options
// ─────────────────────────────────────────────────────────────────────────────

/// Options for the initialization step.
///
/// These options override session config for a single step invocation.
#[derive(Debug, Clone, Default)]
pub struct InitOptions {
    /// Override the number of iterations for iterative estimation.
    pub iterations: Option<usize>,
}

/// Options for the optimization step.
///
/// These options override session config for a single step invocation.
#[derive(Debug, Clone, Default)]
pub struct OptimizeOptions {
    /// Override the maximum number of iterations.
    pub max_iters: Option<usize>,
    /// Override verbosity level.
    pub verbosity: Option<usize>,
}

/// Options for filtering observations based on reprojection error.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FilterOptions {
    /// Maximum reprojection error threshold (pixels).
    /// Points with error above this are removed.
    pub max_reproj_error: f64,

    /// Minimum number of points per view after filtering.
    pub min_points_per_view: usize,

    /// If true, remove entire views that have fewer than `min_points_per_view`.
    pub remove_sparse_views: bool,
}

impl Default for FilterOptions {
    fn default() -> Self {
        Self {
            max_reproj_error: 2.0,
            min_points_per_view: 4,
            remove_sparse_views: true,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Helper Functions
// ─────────────────────────────────────────────────────────────────────────────

fn board_and_pixel_points(view: &CorrespondenceView) -> (Vec<Pt2>, Vec<Pt2>) {
    let board_2d: Vec<Pt2> = view.points_3d.iter().map(|p| Pt2::new(p.x, p.y)).collect();
    let pixel_2d: Vec<Pt2> = view.points_2d.iter().map(|v| Pt2::new(v.x, v.y)).collect();
    (board_2d, pixel_2d)
}

fn k_matrix_from_intrinsics(k: &FxFyCxCySkew<Real>) -> Mat3 {
    Mat3::new(k.fx, k.skew, k.cx, 0.0, k.fy, k.cy, 0.0, 0.0, 1.0)
}

// ─────────────────────────────────────────────────────────────────────────────
// Step Functions
// ─────────────────────────────────────────────────────────────────────────────

/// Initialize intrinsics and poses from observations.
///
/// This step computes:
/// 1. Homographies from each view's 2D-3D correspondences
/// 2. Initial intrinsics using Zhang's method with iterative distortion estimation
/// 3. Initial poses from homographies and estimated intrinsics
///
/// Updates `session.state` with intermediate results (homographies, initial estimates).
///
/// # Arguments
///
/// * `session` - The calibration session
/// * `opts` - Optional overrides for initialization parameters
///
/// # Errors
///
/// - Input not set
/// - Fewer than 3 views
/// - Homography computation fails
/// - Intrinsics estimation fails
pub fn step_init(
    session: &mut CalibrationSession<PlanarIntrinsicsProblemV2>,
    opts: Option<InitOptions>,
) -> Result<()> {
    // Validate preconditions
    session.validate()?;
    let input = session.require_input()?;

    // Get effective options
    let opts = opts.unwrap_or_default();
    let mut init_opts = session.config.init_opts();
    if let Some(iters) = opts.iterations {
        init_opts.iterations = iters;
    }

    // Step 1: Compute homographies
    let mut homographies = Vec::with_capacity(input.views.len());
    for (idx, view) in input.views.iter().enumerate() {
        let (board_2d, pixel_2d) = board_and_pixel_points(&view.obs);
        let h = dlt_homography(&board_2d, &pixel_2d).with_context(|| {
            format!(
                "failed to compute homography for view {} (need >=4 well-conditioned points)",
                idx
            )
        })?;
        homographies.push(h);
    }

    // Step 2: Estimate intrinsics with iterative distortion
    let camera = match estimate_intrinsics_iterative(input, init_opts) {
        Ok(c) => c,
        Err(e) => {
            session.log_failure("init", e.to_string());
            return Err(e);
        }
    };

    // Step 3: Recover poses from homographies
    let kmtx = k_matrix_from_intrinsics(&camera.k);
    let poses: Vec<Iso3> = homographies
        .iter()
        .enumerate()
        .map(|(idx, h)| {
            estimate_planar_pose_from_h(&kmtx, h)
                .with_context(|| format!("failed to recover pose for view {}", idx))
        })
        .collect::<Result<Vec<_>>>()?;

    // Update state
    session.state.homographies = Some(homographies);
    session.state.initial_intrinsics = Some(camera.k);
    session.state.initial_distortion = Some(camera.dist);
    session.state.initial_poses = Some(poses);

    // Clear any previous optimization results (since we have new init)
    session.state.clear_optimization();

    // Log success
    session.log_success_with_notes(
        "init",
        format!(
            "fx={:.1}, fy={:.1}, cx={:.1}, cy={:.1}",
            camera.k.fx, camera.k.fy, camera.k.cx, camera.k.cy
        ),
    );

    Ok(())
}

/// Optimize camera parameters using non-linear least squares.
///
/// This step refines the initial estimates by minimizing reprojection error
/// using bundle adjustment.
///
/// Requires that [`step_init`] has been run first.
///
/// Updates `session.state` with optimization metrics and sets `session.output`.
///
/// # Arguments
///
/// * `session` - The calibration session
/// * `opts` - Optional overrides for optimization parameters
///
/// # Errors
///
/// - Input not set
/// - Initialization not run (state.initial_intrinsics is None)
/// - Optimization fails
pub fn step_optimize(
    session: &mut CalibrationSession<PlanarIntrinsicsProblemV2>,
    opts: Option<OptimizeOptions>,
) -> Result<()> {
    // Validate preconditions
    session.validate()?;
    let input = session.require_input()?;

    let initial = session
        .state
        .initial_params()
        .ok_or_else(|| anyhow::anyhow!("initialization not run - call step_init first"))?;

    // Get effective options
    let opts = opts.unwrap_or_default();
    let solve_opts = session.config.solve_opts();
    let mut backend_opts = session.config.backend_opts();
    if let Some(max_iters) = opts.max_iters {
        backend_opts.max_iters = max_iters;
    }
    if let Some(verbosity) = opts.verbosity {
        backend_opts.verbosity = verbosity;
    }

    // Run optimization
    let result = match optimize_planar_intrinsics(input, &initial, solve_opts, backend_opts) {
        Ok(r) => r,
        Err(e) => {
            session.log_failure("optimize", e.to_string());
            return Err(e);
        }
    };

    // Update state
    session.state.final_cost = Some(result.report.final_cost);
    session.state.mean_reproj_error = Some(result.mean_reproj_error);

    // Set output
    session.set_output(result.clone());

    // Log success
    session.log_success_with_notes(
        "optimize",
        format!(
            "cost={:.2e}, reproj_err={:.3}px",
            result.report.final_cost, result.mean_reproj_error
        ),
    );

    Ok(())
}

/// Filter observations based on reprojection error.
///
/// This step removes outlier points (and optionally entire views) based on
/// reprojection error computed from the current output.
///
/// Requires that [`step_optimize`] has been run first.
///
/// **Important**: This modifies `session.input` in-place, which triggers the
/// invalidation policy and clears state and output. You should run [`step_init`]
/// and [`step_optimize`] again after filtering.
///
/// # Arguments
///
/// * `session` - The calibration session
/// * `opts` - Filtering parameters
///
/// # Errors
///
/// - Output not computed (need reprojection errors)
/// - Filtering would remove all observations
/// - Not enough points remaining per view
pub fn step_filter(
    session: &mut CalibrationSession<PlanarIntrinsicsProblemV2>,
    opts: FilterOptions,
) -> Result<()> {
    ensure!(
        opts.min_points_per_view >= 4,
        "min_points_per_view must be >= 4 for homography"
    );

    let output = session.require_output()?.clone();
    let input = session.require_input()?.clone();

    let poses = output.params.poses();
    ensure!(
        input.views.len() == poses.len(),
        "pose count ({}) must match view count ({})",
        poses.len(),
        input.views.len()
    );

    let camera = &output.params.camera;

    let mut filtered_views = Vec::new();
    let mut total_removed = 0usize;

    for (view, pose) in input.views.iter().zip(poses) {
        let mut points_3d = Vec::new();
        let mut points_2d = Vec::new();
        let mut weights = view.obs.weights.as_ref().map(|_| Vec::<f64>::new());

        for (i, (p3d, p2d)) in view
            .obs
            .points_3d
            .iter()
            .zip(view.obs.points_2d.iter())
            .enumerate()
        {
            let p_cam = pose.transform_point(p3d);
            let Some(projected) = camera.project_point_c(&p_cam.coords) else {
                total_removed += 1;
                continue;
            };

            let error = (projected - *p2d).norm();
            if error <= opts.max_reproj_error {
                points_3d.push(*p3d);
                points_2d.push(*p2d);
                if let Some(ref mut w) = weights {
                    w.push(view.obs.weight(i));
                }
            } else {
                total_removed += 1;
            }
        }

        if points_3d.len() >= opts.min_points_per_view {
            let obs = if let Some(w) = weights {
                CorrespondenceView::new_with_weights(points_3d, points_2d, w)?
            } else {
                CorrespondenceView::new(points_3d, points_2d)?
            };
            filtered_views.push(View::without_meta(obs));
        } else if !opts.remove_sparse_views {
            // Keep sparse views if requested
            filtered_views.push(view.clone());
        }
        // else: view is removed entirely
    }

    let views_removed = input.views.len() - filtered_views.len();
    ensure!(
        !filtered_views.is_empty(),
        "filtering would remove all views"
    );

    let filtered_dataset =
        calib_core::PlanarDataset::new(filtered_views).context("failed to create filtered dataset")?;

    // This will clear state and output per invalidation policy
    session.set_input(filtered_dataset)?;

    session.log_success_with_notes(
        "filter",
        format!(
            "removed {} points, {} views",
            total_removed, views_removed
        ),
    );

    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Pipeline Functions
// ─────────────────────────────────────────────────────────────────────────────

/// Run the full calibration pipeline: init -> optimize.
///
/// Convenience function that runs initialization and optimization in sequence.
///
/// # Errors
///
/// Any error from [`step_init`] or [`step_optimize`].
pub fn run_calibration(
    session: &mut CalibrationSession<PlanarIntrinsicsProblemV2>,
) -> Result<()> {
    step_init(session, None)?;
    step_optimize(session, None)?;
    Ok(())
}

/// Run calibration with outlier filtering: init -> optimize -> filter -> init -> optimize.
///
/// Performs an initial calibration pass, filters outliers based on reprojection error,
/// then re-runs calibration on the cleaned data.
///
/// # Arguments
///
/// * `session` - The calibration session
/// * `filter_opts` - Filtering parameters
///
/// # Errors
///
/// Any error from the constituent steps.
pub fn run_calibration_with_filtering(
    session: &mut CalibrationSession<PlanarIntrinsicsProblemV2>,
    filter_opts: FilterOptions,
) -> Result<()> {
    // First pass
    step_init(session, None)?;
    step_optimize(session, None)?;

    // Filter outliers
    step_filter(session, filter_opts)?;

    // Second pass on cleaned data
    step_init(session, None)?;
    step_optimize(session, None)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use calib_core::{make_pinhole_camera, synthetic::planar, BrownConrady5, FxFyCxCySkew};

    fn make_test_camera() -> calib_core::PinholeCamera {
        make_pinhole_camera(
            FxFyCxCySkew {
                fx: 800.0,
                fy: 780.0,
                cx: 640.0,
                cy: 360.0,
                skew: 0.0,
            },
            BrownConrady5 {
                k1: 0.0,
                k2: 0.0,
                k3: 0.0,
                p1: 0.0,
                p2: 0.0,
                iters: 8,
            },
        )
    }

    fn make_test_dataset() -> calib_core::PlanarDataset {
        let cam_gt = make_test_camera();
        let board_points = planar::grid_points(6, 5, 0.05);
        let poses = planar::poses_yaw_y_z(4, 0.0, 0.1, 0.6, 0.1);
        let views = planar::project_views_all(&cam_gt, &board_points, &poses).unwrap();
        calib_core::PlanarDataset::new(views.into_iter().map(View::without_meta).collect()).unwrap()
    }

    #[test]
    fn step_init_computes_initial_estimate() {
        let mut session = CalibrationSession::<PlanarIntrinsicsProblemV2>::new();
        session.set_input(make_test_dataset()).unwrap();

        step_init(&mut session, None).unwrap();

        assert!(session.state.is_initialized());
        assert!(session.state.homographies.is_some());
        assert_eq!(session.state.initial_poses.as_ref().unwrap().len(), 4);

        // Check intrinsics are reasonable
        let k = session.state.initial_intrinsics.unwrap();
        assert!((k.fx - 800.0).abs() < 100.0); // Within 12.5%
        assert!((k.fy - 780.0).abs() < 100.0);
    }

    #[test]
    fn step_optimize_requires_init() {
        let mut session = CalibrationSession::<PlanarIntrinsicsProblemV2>::new();
        session.set_input(make_test_dataset()).unwrap();

        let result = step_optimize(&mut session, None);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("init"));
    }

    #[test]
    fn step_optimize_sets_output() {
        let mut session = CalibrationSession::<PlanarIntrinsicsProblemV2>::new();
        session.set_input(make_test_dataset()).unwrap();

        step_init(&mut session, None).unwrap();
        step_optimize(&mut session, None).unwrap();

        assert!(session.has_output());
        assert!(session.state.is_optimized());

        let output = session.output().unwrap();
        // Synthetic data should have very low reprojection error
        assert!(output.mean_reproj_error < 1.0);
    }

    #[test]
    fn step_filter_modifies_input() {
        let cam_gt = make_test_camera();
        let board_points = planar::grid_points(6, 5, 0.05);
        let poses = planar::poses_yaw_y_z(4, 0.0, 0.1, 0.6, 0.1);
        let mut views = planar::project_views_all(&cam_gt, &board_points, &poses).unwrap();

        // Add an outlier
        if let Some(first_view) = views.first_mut() {
            if let Some(p) = first_view.points_2d.first_mut() {
                p.x += 50.0;
            }
        }

        let dataset =
            calib_core::PlanarDataset::new(views.into_iter().map(View::without_meta).collect())
                .unwrap();
        let original_points: usize = dataset.views.iter().map(|v| v.obs.len()).sum();

        let mut session = CalibrationSession::<PlanarIntrinsicsProblemV2>::new();
        session.set_input(dataset).unwrap();

        step_init(&mut session, None).unwrap();
        step_optimize(&mut session, None).unwrap();

        step_filter(&mut session, FilterOptions::default()).unwrap();

        // Should have removed at least the outlier
        let filtered_points: usize = session
            .require_input()
            .unwrap()
            .views
            .iter()
            .map(|v| v.obs.len())
            .sum();
        assert!(filtered_points < original_points);

        // State and output should be cleared
        assert!(!session.state.is_initialized());
        assert!(!session.has_output());
    }

    #[test]
    fn run_calibration_full_pipeline() {
        let mut session = CalibrationSession::<PlanarIntrinsicsProblemV2>::new();
        session.set_input(make_test_dataset()).unwrap();

        run_calibration(&mut session).unwrap();

        assert!(session.has_output());
        let output = session.output().unwrap();

        // Check accuracy on synthetic data (no noise)
        let k = output.params.intrinsics();
        assert!((k.fx - 800.0).abs() < 10.0);
        assert!((k.fy - 780.0).abs() < 10.0);
        assert!((k.cx - 640.0).abs() < 10.0);
        assert!((k.cy - 360.0).abs() < 10.0);
    }

    #[test]
    fn session_json_checkpoint() {
        let mut session = CalibrationSession::<PlanarIntrinsicsProblemV2>::with_description(
            "Test calibration",
        );
        session.set_input(make_test_dataset()).unwrap();
        run_calibration(&mut session).unwrap();
        session.export().unwrap();

        // Checkpoint
        let json = session.to_json().unwrap();

        // Restore
        let restored =
            CalibrationSession::<PlanarIntrinsicsProblemV2>::from_json(&json).unwrap();

        assert_eq!(
            restored.metadata.description,
            Some("Test calibration".to_string())
        );
        assert!(restored.has_input());
        assert!(restored.has_output());
        assert_eq!(restored.exports.len(), 1);
        assert!(restored.state.is_initialized());
        assert!(restored.state.is_optimized());
    }

    #[test]
    fn log_entries_recorded_through_pipeline() {
        let mut session = CalibrationSession::<PlanarIntrinsicsProblemV2>::new();
        session.set_input(make_test_dataset()).unwrap();

        run_calibration(&mut session).unwrap();

        // Should have log entries for init and optimize
        assert!(session.log.len() >= 2);
        assert!(session.log.iter().any(|e| e.operation == "init"));
        assert!(session.log.iter().any(|e| e.operation == "optimize"));
        assert!(session.log.iter().all(|e| e.success));
    }
}
