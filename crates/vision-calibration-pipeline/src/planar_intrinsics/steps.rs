//! Step functions for planar intrinsics calibration.
//!
//! This module provides step functions that operate on
//! `CalibrationSession<PlanarIntrinsicsProblem>` to perform calibration.
//!
//! # Example
//!
//! ```no_run
//! use vision_calibration_pipeline::session::CalibrationSession;
//! use vision_calibration_pipeline::planar_intrinsics::{
//!     PlanarIntrinsicsProblem, step_init, step_optimize, step_filter,
//!     FilterOptions,
//! };
//! # fn main() -> anyhow::Result<()> {
//! # let dataset = unimplemented!();
//!
//! let mut session = CalibrationSession::<PlanarIntrinsicsProblem>::new();
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
//! # Ok(())
//! # }
//! ```

use crate::Error;
use serde::{Deserialize, Serialize};
use vision_calibration_core::{
    BrownConrady5, CorrespondenceView, FxFyCxCySkew, Iso3, Real, View,
};
use vision_calibration_optim::optimize_planar_intrinsics;

use crate::planar_family::{
    bootstrap_planar_intrinsics, estimate_view_homographies,
    recover_planar_poses_from_homographies,
};
use crate::session::CalibrationSession;

use super::problem::PlanarIntrinsicsProblem;

// ─────────────────────────────────────────────────────────────────────────────
// Step Options
// ─────────────────────────────────────────────────────────────────────────────

/// Options for the initialization step.
///
/// These options override session config for a single step invocation.
#[derive(Debug, Clone, Default)]
pub struct IntrinsicsInitOptions {
    /// Override the number of iterations for iterative estimation.
    pub iterations: Option<usize>,
}

/// Options for the optimization step.
///
/// These options override session config for a single step invocation.
#[derive(Debug, Clone, Default)]
pub struct IntrinsicsOptimizeOptions {
    /// Override the maximum number of iterations.
    pub max_iters: Option<usize>,
    /// Override verbosity level.
    pub verbosity: Option<usize>,
}

/// Manual initialization seeds for planar intrinsics calibration.
///
/// All fields are `Option<T>`:
/// - `None` means *auto-initialize this group* (same path as plain `step_init`).
/// - `Some(value)` means *use this value*; do not auto-initialize.
///
/// Partial-seed semantics:
/// - When `intrinsics` is `Some` but `poses` is `None`, poses are recovered from
///   homographies using the **manual** intrinsics (not auto-estimated ones). This keeps
///   the geometric chain consistent with the seed.
/// - When `intrinsics` is `Some` and `distortion` is `None`, distortion defaults to
///   `BrownConrady5::default()` (zeros) — the auto distortion fit is coupled with the
///   iterative intrinsics estimator and does not run when intrinsics are seeded.
/// - When `intrinsics` is `None`, the bootstrap auto-fit runs and any `Some` field
///   overrides the corresponding bootstrap output.
///
/// See ADR 0011 for the design rationale.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PlanarManualInit {
    /// Manual intrinsics seed. `None` means auto-init via Zhang's method.
    pub intrinsics: Option<FxFyCxCySkew<Real>>,
    /// Manual distortion seed. `None` means auto-init (or zeros when intrinsics are
    /// also seeded — see struct docs).
    pub distortion: Option<BrownConrady5<Real>>,
    /// Manual per-view poses (camera-from-target). `None` means recover from
    /// homographies using whichever intrinsics are in effect.
    pub poses: Option<Vec<Iso3>>,
}

/// Options for filtering observations based on reprojection error.
#[derive(Debug, Clone)]
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
// Step Functions
// ─────────────────────────────────────────────────────────────────────────────

/// Initialize intrinsics, distortion, and per-view poses from any combination of
/// manual seeds and auto-estimation.
///
/// This is the load-bearing init function. [`step_init`] is a thin delegate that
/// passes `PlanarManualInit::default()` (all-`None`, full auto path).
///
/// # Field-by-field behavior
///
/// - `intrinsics`: `Some` skips Zhang's auto-fit and uses the seed; `None` runs the
///   iterative auto-fit (`bootstrap_planar_intrinsics`).
/// - `distortion`: `Some` overrides the bootstrap's fitted distortion; `None` keeps
///   the bootstrap value (or `BrownConrady5::default()` when intrinsics are seeded
///   and the bootstrap doesn't run).
/// - `poses`: `Some` uses the seed verbatim (count must match view count); `None`
///   recovers poses from homographies using the in-effect intrinsics — manual when
///   seeded, auto otherwise.
///
/// Updates `session.state.{homographies, initial_intrinsics, initial_distortion,
/// initial_poses}` and clears any previous optimization results. Same postcondition
/// as `step_init`.
///
/// # Errors
///
/// - Input not set, or fewer than 3 views.
/// - Homography or auto-init computation fails.
/// - `manual.poses` is `Some` but its length does not match the view count.
pub fn step_set_init(
    session: &mut CalibrationSession<PlanarIntrinsicsProblem>,
    manual: PlanarManualInit,
    opts: Option<IntrinsicsInitOptions>,
) -> Result<(), Error> {
    session.validate()?;
    let input = session.require_input()?;

    let opts = opts.unwrap_or_default();
    let mut init_opts = session.config.init_opts();
    if let Some(iters) = opts.iterations {
        init_opts.iterations = iters;
    }

    let mut manual_fields: Vec<&'static str> = Vec::new();
    let mut auto_fields: Vec<&'static str> = Vec::new();

    let (intrinsics, distortion, homographies, poses) = if let Some(k) = manual.intrinsics {
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

        let homographies = match estimate_view_homographies(input) {
            Ok(hs) => hs,
            Err(e) => {
                session.log_failure("init", e.to_string());
                return Err(Error::from(e));
            }
        };

        let poses = match manual.poses {
            Some(p) => {
                manual_fields.push("poses");
                if p.len() != input.num_views() {
                    let msg = format!(
                        "manual poses count ({}) does not match view count ({})",
                        p.len(),
                        input.num_views()
                    );
                    session.log_failure("init", msg.clone());
                    return Err(Error::invalid_input(msg));
                }
                p
            }
            None => {
                auto_fields.push("poses");
                match recover_planar_poses_from_homographies(&homographies, &k) {
                    Ok(ps) => ps,
                    Err(e) => {
                        session.log_failure("init", e.to_string());
                        return Err(Error::from(e));
                    }
                }
            }
        };

        (k, dist, homographies, poses)
    } else {
        auto_fields.push("intrinsics");

        let bootstrap = match bootstrap_planar_intrinsics(input, init_opts) {
            Ok(b) => b,
            Err(e) => {
                session.log_failure("init", e.to_string());
                return Err(Error::from(e));
            }
        };

        let dist = match manual.distortion {
            Some(d) => {
                manual_fields.push("distortion");
                d
            }
            None => {
                auto_fields.push("distortion");
                bootstrap.camera.dist
            }
        };

        let poses = match manual.poses {
            Some(p) => {
                manual_fields.push("poses");
                if p.len() != input.num_views() {
                    let msg = format!(
                        "manual poses count ({}) does not match view count ({})",
                        p.len(),
                        input.num_views()
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

        (bootstrap.camera.k, dist, bootstrap.homographies, poses)
    };

    session.state.homographies = Some(homographies);
    session.state.initial_intrinsics = Some(intrinsics);
    session.state.initial_distortion = Some(distortion);
    session.state.initial_poses = Some(poses);
    session.state.clear_optimization();

    let source = format_init_source(&manual_fields, &auto_fields);
    session.log_success_with_notes(
        "init",
        format!(
            "fx={:.1}, fy={:.1}, cx={:.1}, cy={:.1} {}",
            intrinsics.fx, intrinsics.fy, intrinsics.cx, intrinsics.cy, source
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

/// Initialize intrinsics and poses from observations using full auto-init.
///
/// Convenience wrapper around [`step_set_init`] with `PlanarManualInit::default()`
/// (all-`None`). Auto-fits intrinsics + distortion via Zhang's method with iterative
/// distortion, then recovers poses from homographies.
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
/// See [`step_set_init`].
pub fn step_init(
    session: &mut CalibrationSession<PlanarIntrinsicsProblem>,
    opts: Option<IntrinsicsInitOptions>,
) -> Result<(), Error> {
    step_set_init(session, PlanarManualInit::default(), opts)
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
    session: &mut CalibrationSession<PlanarIntrinsicsProblem>,
    opts: Option<IntrinsicsOptimizeOptions>,
) -> Result<(), Error> {
    // Validate preconditions
    session.validate()?;
    let input = session.require_input()?;

    let initial = session
        .state
        .initial_params()
        .ok_or_else(|| Error::not_available("initial params (call step_init first)"))?;

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
            return Err(Error::from(e));
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
    session: &mut CalibrationSession<PlanarIntrinsicsProblem>,
    opts: FilterOptions,
) -> Result<(), Error> {
    if opts.min_points_per_view < 4 {
        return Err(Error::invalid_input(
            "min_points_per_view must be >= 4 for homography",
        ));
    }

    let output = session.require_output()?.clone();
    let input = session.require_input()?.clone();

    let poses = output.params.poses();
    if input.views.len() != poses.len() {
        return Err(Error::invalid_input(format!(
            "pose count ({}) must match view count ({})",
            poses.len(),
            input.views.len()
        )));
    }

    let camera = &output.params.camera;

    let mut filtered_views = Vec::new();
    let mut total_removed = 0usize;

    for (view, pose) in input.views.iter().zip(poses) {
        let mut points_3d = Vec::new();
        let mut points_2d = Vec::new();
        let mut weights: Vec<f64> = Vec::new();
        let has_weights = !view.obs.weights.is_empty();

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
                if has_weights {
                    weights.push(view.obs.weight(i));
                }
            } else {
                total_removed += 1;
            }
        }

        if points_3d.len() >= opts.min_points_per_view {
            let obs = if has_weights {
                CorrespondenceView::new_with_weights(points_3d, points_2d, weights)?
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
    if filtered_views.is_empty() {
        return Err(Error::invalid_input("filtering would remove all views"));
    }

    let filtered_dataset = vision_calibration_core::PlanarDataset::new(filtered_views)
        .map_err(|e| Error::invalid_input(format!("failed to create filtered dataset: {e}")))?;

    // This will clear state and output per invalidation policy
    session.set_input(filtered_dataset)?;

    session.log_success_with_notes(
        "filter",
        format!("removed {} points, {} views", total_removed, views_removed),
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
    session: &mut CalibrationSession<PlanarIntrinsicsProblem>,
) -> Result<(), Error> {
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
    session: &mut CalibrationSession<PlanarIntrinsicsProblem>,
    filter_opts: FilterOptions,
) -> Result<(), Error> {
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
    use vision_calibration_core::{
        BrownConrady5, FxFyCxCySkew, make_pinhole_camera, synthetic::planar,
    };

    fn make_test_camera() -> vision_calibration_core::PinholeCamera {
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

    fn make_test_dataset() -> vision_calibration_core::PlanarDataset {
        let cam_gt = make_test_camera();
        let board_points = planar::grid_points(6, 5, 0.05);
        let poses = planar::poses_yaw_y_z(4, 0.0, 0.1, 0.6, 0.1);
        let views = planar::project_views_all(&cam_gt, &board_points, &poses).unwrap();
        vision_calibration_core::PlanarDataset::new(
            views.into_iter().map(View::without_meta).collect(),
        )
        .unwrap()
    }

    #[test]
    fn step_init_computes_initial_estimate() {
        let mut session = CalibrationSession::<PlanarIntrinsicsProblem>::new();
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
        let mut session = CalibrationSession::<PlanarIntrinsicsProblem>::new();
        session.set_input(make_test_dataset()).unwrap();

        let result = step_optimize(&mut session, None);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("init"));
    }

    #[test]
    fn step_optimize_sets_output() {
        let mut session = CalibrationSession::<PlanarIntrinsicsProblem>::new();
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
        if let Some(first_view) = views.first_mut()
            && let Some(p) = first_view.points_2d.first_mut()
        {
            p.x += 50.0;
        }

        let dataset = vision_calibration_core::PlanarDataset::new(
            views.into_iter().map(View::without_meta).collect(),
        )
        .unwrap();
        let original_points: usize = dataset.views.iter().map(|v| v.obs.len()).sum();

        let mut session = CalibrationSession::<PlanarIntrinsicsProblem>::new();
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
        let mut session = CalibrationSession::<PlanarIntrinsicsProblem>::new();
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
        let mut session =
            CalibrationSession::<PlanarIntrinsicsProblem>::with_description("Test calibration");
        session.set_input(make_test_dataset()).unwrap();
        run_calibration(&mut session).unwrap();
        session.export().unwrap();

        // Checkpoint
        let json = session.to_json().unwrap();

        // Restore
        let restored = CalibrationSession::<PlanarIntrinsicsProblem>::from_json(&json).unwrap();

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
        let mut session = CalibrationSession::<PlanarIntrinsicsProblem>::new();
        session.set_input(make_test_dataset()).unwrap();

        run_calibration(&mut session).unwrap();

        // Should have log entries for init and optimize
        assert!(session.log.len() >= 2);
        assert!(session.log.iter().any(|e| e.operation == "init"));
        assert!(session.log.iter().any(|e| e.operation == "optimize"));
        assert!(session.log.iter().all(|e| e.success));
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Manual init (ADR 0011) tests
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn step_set_init_default_matches_step_init() {
        let mut session_a = CalibrationSession::<PlanarIntrinsicsProblem>::new();
        session_a.set_input(make_test_dataset()).unwrap();
        step_init(&mut session_a, None).unwrap();

        let mut session_b = CalibrationSession::<PlanarIntrinsicsProblem>::new();
        session_b.set_input(make_test_dataset()).unwrap();
        step_set_init(&mut session_b, PlanarManualInit::default(), None).unwrap();

        // Both paths should produce identical state — step_init delegates to
        // step_set_init with default fields.
        let k_a = session_a.state.initial_intrinsics.unwrap();
        let k_b = session_b.state.initial_intrinsics.unwrap();
        assert!((k_a.fx - k_b.fx).abs() < 1e-9);
        assert!((k_a.fy - k_b.fy).abs() < 1e-9);
        assert!((k_a.cx - k_b.cx).abs() < 1e-9);
        assert!((k_a.cy - k_b.cy).abs() < 1e-9);
    }

    #[test]
    fn step_set_init_with_intrinsics_seed_converges() {
        let mut session = CalibrationSession::<PlanarIntrinsicsProblem>::new();
        session.set_input(make_test_dataset()).unwrap();

        let manual = PlanarManualInit {
            intrinsics: Some(FxFyCxCySkew {
                fx: 800.0,
                fy: 780.0,
                cx: 640.0,
                cy: 360.0,
                skew: 0.0,
            }),
            ..Default::default()
        };
        step_set_init(&mut session, manual, None).unwrap();
        step_optimize(&mut session, None).unwrap();

        let output = session.output().unwrap();
        // Synthetic data with GT-matching intrinsics seed should converge tightly.
        assert!(
            output.mean_reproj_error < 1.0,
            "got {:.4}",
            output.mean_reproj_error
        );
    }

    #[test]
    fn step_set_init_with_full_seeds_converges_tightly() {
        // Run auto-init first to harvest a plausible pose set.
        let mut session_auto = CalibrationSession::<PlanarIntrinsicsProblem>::new();
        session_auto.set_input(make_test_dataset()).unwrap();
        step_init(&mut session_auto, None).unwrap();
        let auto_poses = session_auto.state.initial_poses.clone().unwrap();

        let mut session = CalibrationSession::<PlanarIntrinsicsProblem>::new();
        session.set_input(make_test_dataset()).unwrap();

        let manual = PlanarManualInit {
            intrinsics: Some(FxFyCxCySkew {
                fx: 800.0,
                fy: 780.0,
                cx: 640.0,
                cy: 360.0,
                skew: 0.0,
            }),
            distortion: Some(BrownConrady5::default()),
            poses: Some(auto_poses),
        };
        step_set_init(&mut session, manual, None).unwrap();
        step_optimize(&mut session, None).unwrap();

        let output = session.output().unwrap();
        assert!(
            output.mean_reproj_error < 1.0,
            "got {:.4}",
            output.mean_reproj_error
        );
    }

    #[test]
    fn step_set_init_rejects_wrong_pose_count() {
        use vision_calibration_core::Iso3;

        let mut session = CalibrationSession::<PlanarIntrinsicsProblem>::new();
        session.set_input(make_test_dataset()).unwrap();

        // Test dataset has 4 views; supply only 1 pose.
        let manual = PlanarManualInit {
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

    #[test]
    fn step_set_init_logs_manual_and_auto_sources() {
        let mut session = CalibrationSession::<PlanarIntrinsicsProblem>::new();
        session.set_input(make_test_dataset()).unwrap();

        let manual = PlanarManualInit {
            intrinsics: Some(FxFyCxCySkew {
                fx: 800.0,
                fy: 780.0,
                cx: 640.0,
                cy: 360.0,
                skew: 0.0,
            }),
            ..Default::default()
        };
        step_set_init(&mut session, manual, None).unwrap();

        let init_entry = session
            .log
            .iter()
            .find(|e| e.operation == "init")
            .expect("init log entry");
        let notes = init_entry.notes.as_deref().unwrap_or("");
        assert!(notes.contains("manual: intrinsics"), "notes: {}", notes);
        assert!(notes.contains("auto: distortion"), "notes: {}", notes);
        assert!(notes.contains("poses"), "notes: {}", notes);
    }
}
