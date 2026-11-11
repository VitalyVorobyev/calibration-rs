//! Step functions for multi-camera rig extrinsics calibration.
//!
//! This module provides step functions that operate on
//! `CalibrationSession<RigExtrinsicsProblem>` to perform calibration. The
//! steps are the rig stages every rig problem type shares.

use crate::Error;
use serde::{Deserialize, Serialize};
use vision_calibration_core::CameraParams;

use crate::common::{
    IntrinsicsInitOptions, IntrinsicsOptimizeOptions, RigInitOptions, RigOptimizeOptions,
};
use crate::rig_family::{self, rig_ba_notes};
pub use crate::rig_family::{
    IntrinsicsInitResult, IntrinsicsManualInit, RigInitResult, RigManualInit, RigPoses,
};
use crate::session::CalibrationSession;

use super::problem::RigExtrinsicsProblem;

// ─────────────────────────────────────────────────────────────────────────────
// Step Results
// ─────────────────────────────────────────────────────────────────────────────

/// Typed return value of [`step_intrinsics_optimize_all`].
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct IntrinsicsOptimizeResult {
    /// Per-camera refined parameters, each in its camera's model.
    pub cameras: Vec<CameraParams>,
    /// Per-camera mean reprojection error in pixels.
    pub per_cam_reproj_errors: Vec<f64>,
}

/// Typed return value of [`step_rig_optimize`].
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct RigOptimizeResult {
    /// Mean reprojection error in pixels across the whole rig.
    pub mean_reproj_error: f64,
    /// Per-camera mean reprojection error in pixels.
    pub per_cam_reproj_errors: Vec<f64>,
    /// Per-camera parameters after rig BA.
    pub cameras: Vec<CameraParams>,
}

// ─────────────────────────────────────────────────────────────────────────────
// Step Functions
// ─────────────────────────────────────────────────────────────────────────────

/// Initialize intrinsics for all cameras from any combination of manual seeds
/// and auto-estimation (Zhang's method per camera).
///
/// This is the load-bearing intrinsics-stage init. [`step_intrinsics_init_all`]
/// is a thin delegate with `IntrinsicsManualInit::default()`.
///
/// See [`IntrinsicsManualInit`] for partial-seed semantics. Per-camera target
/// poses are *always* recovered from per-camera homographies — using the seeded
/// intrinsics if provided, otherwise the auto-fit intrinsics. Each camera is
/// initialized in its own model (`config.camera_for(i)`), and the linear
/// stage fits only the distortion terms that camera's solves estimate.
///
/// # Errors
///
/// - Input not set, or any camera has fewer than 3 views with observations.
/// - `manual.per_cam_intrinsics` (or `per_cam_distortion`, `per_cam_sensors`)
///   length mismatches `input.num_cameras`, or a non-zero tilt seed is given
///   for an untilted camera.
pub fn step_intrinsics_init_all_with_seed(
    session: &mut CalibrationSession<RigExtrinsicsProblem>,
    manual: IntrinsicsManualInit,
    opts: Option<IntrinsicsInitOptions>,
) -> Result<IntrinsicsInitResult, Error> {
    rig_family::init_intrinsics(session, manual, opts)
}

/// Initialize intrinsics for all cameras using full auto-init (Zhang's per camera).
///
/// Convenience wrapper around [`step_intrinsics_init_all_with_seed`] with default seeds.
pub fn step_intrinsics_init_all(
    session: &mut CalibrationSession<RigExtrinsicsProblem>,
    opts: Option<IntrinsicsInitOptions>,
) -> Result<IntrinsicsInitResult, Error> {
    step_intrinsics_init_all_with_seed(session, IntrinsicsManualInit::default(), opts)
}

/// Optimize intrinsics for all cameras.
///
/// Refines each camera independently in its own model, holding its `fix`
/// lists, with every pose free. An untilted camera runs a plain planar
/// solve. A Scheimpflug camera runs the staged multi-start tilt solve;
/// after a full auto-init it runs a wider multi-start over
/// linear tilt seeds instead, and a Brown-Conrady tilted camera that still
/// fits badly is re-solved from a nominal seed built from the well-fitted
/// ones.
///
/// Requires [`step_intrinsics_init_all`] to be run first.
///
/// # Errors
///
/// - Input not set
/// - Initialization not run, or the camera models changed since
/// - Optimization fails for any camera
pub fn step_intrinsics_optimize_all(
    session: &mut CalibrationSession<RigExtrinsicsProblem>,
    opts: Option<IntrinsicsOptimizeOptions>,
) -> Result<IntrinsicsOptimizeResult, Error> {
    let estimate = rig_family::optimize_intrinsics(session, opts)?;
    Ok(IntrinsicsOptimizeResult {
        cameras: estimate.cameras,
        per_cam_reproj_errors: estimate.per_cam_reproj_errors,
    })
}

/// Initialize rig extrinsics from any combination of manual seeds and auto-
/// estimation (linear extrinsics fit).
///
/// This is the load-bearing rig-stage init. [`step_rig_init`] is a thin delegate
/// with `RigManualInit::default()`.
///
/// **Coupling rule**: `cam_se3_rig` and `rig_se3_target` are
/// geometrically coupled. Both must be `Some` or both `None`. Mismatched
/// configurations return `Error::InvalidInput`.
///
/// # Errors
///
/// - Input not set, or per-camera intrinsics not computed.
/// - Coupling rule violated (one of `cam_se3_rig` / `rig_se3_target` seeded
///   without the other).
/// - Vector length mismatches: `cam_se3_rig.len() != input.num_cameras`, or
///   `rig_se3_target.len() != input.num_views()`.
/// - Auto-init linear estimation fails.
pub fn step_rig_init_with_seed(
    session: &mut CalibrationSession<RigExtrinsicsProblem>,
    manual: RigManualInit,
    opts: Option<RigInitOptions>,
) -> Result<RigInitResult, Error> {
    rig_family::init_rig(session, manual, opts)
}

/// Initialize rig extrinsics using full auto-init (linear extrinsics fit).
///
/// Convenience wrapper around [`step_rig_init_with_seed`] with default seeds.
pub fn step_rig_init(
    session: &mut CalibrationSession<RigExtrinsicsProblem>,
    opts: Option<RigInitOptions>,
) -> Result<RigInitResult, Error> {
    step_rig_init_with_seed(session, RigManualInit::default(), opts)
}

/// Optimize rig extrinsics using bundle adjustment.
///
/// Jointly optimizes:
/// - Per-camera extrinsics (camera-to-rig)
/// - Per-view rig poses (rig-to-target)
/// - Per-camera intrinsics and distortion (unless `rig.refine_intrinsics`
///   is false), holding each camera's `fix` lists
/// - The tilt of each Scheimpflug camera (unless `rig.refine_sensor` is
///   false), holding its `fix.sensor` list
///
/// The reference camera (`rig.reference_camera_idx`) alone removes the rig's
/// full 6-DOF gauge freedom; no rig-from-target pose is fixed; an extra
/// per-view pose constraint would be redundant and mildly pessimizing.
///
/// Requires [`step_rig_init`] to be run first.
///
/// # Errors
///
/// - Input not set
/// - Rig initialization not run
/// - Optimization fails
pub fn step_rig_optimize(
    session: &mut CalibrationSession<RigExtrinsicsProblem>,
    opts: Option<RigOptimizeOptions>,
) -> Result<RigOptimizeResult, Error> {
    let output = rig_family::optimize_rig(session, opts)?;
    let notes = rig_ba_notes(&output);
    let result = RigOptimizeResult {
        mean_reproj_error: output.mean_reproj_error,
        per_cam_reproj_errors: output.per_cam_reproj_errors.clone(),
        cameras: output.params.cameras.clone(),
    };
    session.state.rig_ba_final_cost = Some(output.report.final_cost);
    session.set_output(output);
    session.log_success_with_notes("rig_optimize", notes);
    Ok(result)
}

// ─────────────────────────────────────────────────────────────────────────────
// Pipeline Function
// ─────────────────────────────────────────────────────────────────────────────

/// Run the full rig extrinsics calibration pipeline.
///
/// Runs: intrinsics_init_all → intrinsics_optimize_all → rig_init → rig_optimize.
///
/// # Errors
///
/// Any error from the constituent steps.
pub fn run_calibration(
    session: &mut CalibrationSession<RigExtrinsicsProblem>,
) -> Result<(), Error> {
    session.begin_step("intrinsics_init_all")?;
    let _ = step_intrinsics_init_all(session, None)?;
    session.begin_step("intrinsics_optimize_all")?;
    let _ = step_intrinsics_optimize_all(session, None)?;
    session.begin_step("rig_init")?;
    let _ = step_rig_init(session, None)?;
    session.begin_step("rig_optimize")?;
    let _ = step_rig_optimize(session, None)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::problem::RigExtrinsicsInput;
    use super::*;
    use nalgebra::{Rotation3, Translation3};
    use vision_calibration_core::{
        BrownConrady5, CorrespondenceView, FxFyCxCy, PinholeCamera, Pt2, Pt3, RigDataset, RigView,
        RigViewObs, make_pinhole_camera,
    };
    use vision_calibration_core::{Iso3, NoMeta};

    fn make_test_camera(offset: f64) -> PinholeCamera {
        make_pinhole_camera(
            FxFyCxCy {
                fx: 800.0 + offset,
                fy: 780.0 + offset,
                cx: 640.0,
                cy: 360.0,
            },
            BrownConrady5::default(),
        )
    }

    fn make_iso(angles: (f64, f64, f64), t: (f64, f64, f64)) -> Iso3 {
        let rot = Rotation3::from_euler_angles(angles.0, angles.1, angles.2);
        let tr = Translation3::new(t.0, t.1, t.2);
        Iso3::from_parts(tr, rot.into())
    }

    fn make_test_input() -> RigExtrinsicsInput {
        // Ground truth cameras and rig
        let cam0 = make_test_camera(0.0);
        let cam1 = make_test_camera(10.0);
        let cam0_se3_rig = Iso3::identity();
        let cam1_se3_rig = make_iso((0.0, 0.0, 0.1), (0.2, 0.0, 0.0));

        // Board points
        let board_pts: Vec<Pt3> = (0..6)
            .flat_map(|i| (0..5).map(move |j| Pt3::new(i as f64 * 0.05, j as f64 * 0.05, 0.0)))
            .collect();

        // Rig poses
        let rig_poses = [
            make_iso((0.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
            make_iso((0.1, 0.0, 0.0), (0.1, 0.0, 1.0)),
            make_iso((0.0, 0.1, 0.0), (0.0, 0.1, 1.0)),
            make_iso((0.05, 0.05, 0.0), (-0.1, 0.0, 1.0)),
        ];

        let views: Vec<RigView<NoMeta>> = rig_poses
            .iter()
            .map(|rig_se3_target| {
                // cam_se3_target = cam_se3_rig * rig_se3_target
                let cam0_se3_target = cam0_se3_rig * rig_se3_target;
                let cam1_se3_target = cam1_se3_rig * rig_se3_target;

                let project_points = |cam: &PinholeCamera, pose: &Iso3| -> Vec<Pt2> {
                    board_pts
                        .iter()
                        .map(|p| {
                            let p_cam = pose.transform_point(p);
                            cam.project_point_c(&p_cam.coords).unwrap()
                        })
                        .collect()
                };

                let pts0 = project_points(&cam0, &cam0_se3_target);
                let pts1 = project_points(&cam1, &cam1_se3_target);

                RigView {
                    meta: NoMeta,
                    obs: RigViewObs {
                        cameras: vec![
                            Some(CorrespondenceView::new(board_pts.clone(), pts0).unwrap()),
                            Some(CorrespondenceView::new(board_pts.clone(), pts1).unwrap()),
                        ],
                    },
                }
            })
            .collect();

        RigDataset::new(views, 2).unwrap()
    }

    #[test]
    fn step_intrinsics_init_all_computes_estimate() {
        let mut session = CalibrationSession::<RigExtrinsicsProblem>::new();
        session.set_input(make_test_input()).unwrap();

        step_intrinsics_init_all(&mut session, None).unwrap();

        assert!(session.state.rig.has_per_cam_intrinsics());
        let cameras = session.state.rig.cameras.as_ref().unwrap();
        assert_eq!(cameras.len(), 2);
    }

    #[test]
    fn step_rig_init_requires_intrinsics() {
        let mut session = CalibrationSession::<RigExtrinsicsProblem>::new();
        session.set_input(make_test_input()).unwrap();

        let result = step_rig_init(&mut session, None);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("intrinsics"));
    }

    #[test]
    fn step_rig_optimize_requires_init() {
        let mut session = CalibrationSession::<RigExtrinsicsProblem>::new();
        session.set_input(make_test_input()).unwrap();
        step_intrinsics_init_all(&mut session, None).unwrap();
        step_intrinsics_optimize_all(&mut session, None).unwrap();

        let result = step_rig_optimize(&mut session, None);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("rig"));
    }

    #[test]
    fn set_input_clears_state() {
        let mut session = CalibrationSession::<RigExtrinsicsProblem>::new();
        session.set_input(make_test_input()).unwrap();
        step_intrinsics_init_all(&mut session, None).unwrap();

        assert!(session.state.rig.has_per_cam_intrinsics());

        session.set_input(make_test_input()).unwrap();
        assert!(!session.state.rig.has_per_cam_intrinsics());
    }

    #[test]
    fn set_config_keeps_output() {
        let mut session = CalibrationSession::<RigExtrinsicsProblem>::new();
        session.set_input(make_test_input()).unwrap();
        run_calibration(&mut session).unwrap();

        assert!(session.has_output());

        session
            .set_config(super::super::problem::RigExtrinsicsConfig {
                solver: crate::common::config::SolverConfig {
                    max_iters: 100,
                    ..Default::default()
                },
                ..Default::default()
            })
            .unwrap();

        assert!(session.has_output());
    }

    /// `run_calibration` reports its steps in order, an observer leaves the
    /// export unchanged, and a token cancelled at a step boundary stops the
    /// run there with no output.
    #[test]
    fn run_calibration_reports_steps_and_stops_between_them() {
        use std::sync::{Arc, Mutex};
        use vision_calibration_core::{CancelToken, Progress, RunControl};

        let mut plain = CalibrationSession::<RigExtrinsicsProblem>::new();
        plain.set_input(make_test_input()).unwrap();
        run_calibration(&mut plain).unwrap();

        let run = |stop_at: Option<&'static str>| {
            let seen = Arc::new(Mutex::new(Vec::new()));
            let log = Arc::clone(&seen);
            let token = CancelToken::new();
            let trigger = token.clone();
            let control = RunControl::new()
                .with_cancel(token)
                .with_observer(move |progress| {
                    if let Progress::Step { name, .. } = progress {
                        log.lock().unwrap().push(name);
                        if stop_at == Some(name) {
                            trigger.cancel();
                        }
                    }
                });
            let mut session = CalibrationSession::<RigExtrinsicsProblem>::new();
            session.set_input(make_test_input()).unwrap();
            session.set_control(control);
            let result = run_calibration(&mut session);
            let steps = seen.lock().unwrap().clone();
            (session, result, steps)
        };

        let (observed, result, steps) = run(None);
        result.unwrap();
        assert_eq!(
            steps,
            [
                "intrinsics_init_all",
                "intrinsics_optimize_all",
                "rig_init",
                "rig_optimize"
            ]
        );
        assert_eq!(
            serde_json::to_value(plain.export().unwrap()).unwrap(),
            serde_json::to_value(observed.export_peek().unwrap()).unwrap()
        );

        // Cancelled as `rig_optimize` is announced: the step never starts.
        let (session, result, steps) = run(Some("rig_optimize"));
        assert!(matches!(result, Err(Error::Cancelled)), "{result:?}");
        assert_eq!(steps.len(), 4);
        assert!(!session.has_output());
    }

    #[test]
    fn json_roundtrip() {
        let mut session =
            CalibrationSession::<RigExtrinsicsProblem>::with_description("Test rig extrinsics");
        session.set_input(make_test_input()).unwrap();
        run_calibration(&mut session).unwrap();
        session.export().unwrap();

        let json = session.to_json().unwrap();
        let restored = CalibrationSession::<RigExtrinsicsProblem>::from_json(&json).unwrap();

        assert_eq!(
            restored.metadata.description,
            Some("Test rig extrinsics".to_string())
        );
        assert!(restored.has_input());
        assert!(restored.has_output());
        assert_eq!(restored.exports.len(), 1);
    }

    #[test]
    fn rig_optimize_keeps_reprojection_error_reasonable() {
        let input = make_test_input();
        let mut session = CalibrationSession::<RigExtrinsicsProblem>::new();
        session.set_input(input.clone()).unwrap();

        run_calibration(&mut session).unwrap();
        let output = session.require_output().unwrap().clone();

        let cam_se3_rig: Vec<Iso3> = output
            .params
            .rig_se3_cam
            .iter()
            .map(|t| t.inverse())
            .collect();
        let models = crate::rig_family::rig_camera_models(&output.params.cameras).unwrap();
        let reproj_stats = vision_calibration_core::compute_rig_reprojection_stats(
            &models,
            &input,
            &cam_se3_rig,
            &output.params.rig_se3_target,
        )
        .unwrap();
        let per_cam_stats = vision_calibration_core::compute_rig_reprojection_stats_per_camera(
            &models,
            &input,
            &cam_se3_rig,
            &output.params.rig_se3_target,
        )
        .unwrap();
        let mean_err = reproj_stats.mean;
        assert!(
            mean_err.is_finite() && mean_err < 1.0e-3,
            "mean reprojection error too large: {} px",
            mean_err
        );

        let state_mean = session.state.rig.rig_ba_reproj_error.unwrap();
        assert!(
            (state_mean - mean_err).abs() < 1e-12,
            "state reprojection error mismatch: state={}, computed={}",
            state_mean,
            mean_err
        );

        let state_per_cam = session
            .state
            .rig
            .rig_ba_per_cam_reproj_errors
            .as_ref()
            .unwrap();
        assert_eq!(state_per_cam.len(), per_cam_stats.len());
        for (i, (state_mean, computed)) in state_per_cam
            .iter()
            .zip(per_cam_stats.iter().map(|s| s.mean))
            .enumerate()
        {
            assert!(
                (state_mean - computed).abs() < 1e-12,
                "camera {} error mismatch: state={}, computed={}",
                i,
                state_mean,
                computed
            );
        }

        // Export should use `cam_se3_rig` convention.
        let export = session.export().unwrap();
        assert_eq!(export.cam_se3_rig.len(), cam_se3_rig.len());
        for (a, b) in export.cam_se3_rig.iter().zip(cam_se3_rig.iter()) {
            let dt = (a.translation.vector - b.translation.vector).norm();
            assert!(dt < 1e-10, "export translation error {}", dt);
        }
    }
}
