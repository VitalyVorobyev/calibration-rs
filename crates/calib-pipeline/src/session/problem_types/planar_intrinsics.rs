//! Planar intrinsics session problem.

use crate::{
    optimize_planar_intrinsics_with_init, pinhole_camera_params, planar_init_seed_from_views,
    PlanarIntrinsicsConfig, PlanarIntrinsicsInput, PlanarIntrinsicsReport, PlanarViewData,
};
use anyhow::{ensure, Result};
use calib_core::{BrownConrady5, Camera, FxFyCxCySkew, IdentitySensor, Iso3, Pinhole, Real};
use serde::{Deserialize, Serialize};

use crate::session::ProblemType;

/// Planar intrinsics calibration problem (Zhang's method with distortion).
///
/// Estimates camera intrinsics (fx, fy, cx, cy, skew) and Brown-Conrady distortion
/// (k1, k2, k3, p1, p2) from observations of a planar calibration pattern.
pub struct PlanarIntrinsicsProblem;

/// Observations for planar intrinsics calibration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanarIntrinsicsObservations {
    pub views: Vec<PlanarViewData>,
}

/// Initial values from linear initialization (iterative Zhang's method).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanarIntrinsicsInitial {
    /// Initial intrinsics and distortion estimates (for compatibility).
    pub report: PlanarIntrinsicsReport,
    /// Serialized seed for nonlinear refinement.
    #[serde(default)]
    pub init: Option<PlanarIntrinsicsInitState>,
}

/// Optimized results from non-linear refinement.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanarIntrinsicsOptimized {
    /// Final calibration report with refined parameters.
    pub report: PlanarIntrinsicsReport,
    /// Refined board-to-camera poses.
    pub poses: Vec<Iso3>,
    /// Mean reprojection error after optimization (pixels).
    pub mean_reproj_error: Real,
}

/// Serializable seed for planar intrinsics optimization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanarIntrinsicsInitState {
    pub intrinsics: FxFyCxCySkew<Real>,
    pub distortion: BrownConrady5<Real>,
    pub poses: Vec<Iso3>,
}

impl PlanarIntrinsicsInitState {
    pub fn from_seed(seed: &calib_optim::planar_intrinsics::PlanarIntrinsicsInit) -> Self {
        Self {
            intrinsics: FxFyCxCySkew {
                fx: seed.intrinsics.fx,
                fy: seed.intrinsics.fy,
                cx: seed.intrinsics.cx,
                cy: seed.intrinsics.cy,
                skew: 0.0,
            },
            distortion: BrownConrady5 {
                k1: seed.distortion.k1,
                k2: seed.distortion.k2,
                k3: seed.distortion.k3,
                p1: seed.distortion.p1,
                p2: seed.distortion.p2,
                iters: 8,
            },
            poses: seed.poses.clone(),
        }
    }

    pub fn to_seed(&self) -> Result<calib_optim::planar_intrinsics::PlanarIntrinsicsInit> {
        calib_optim::planar_intrinsics::PlanarIntrinsicsInit::new(
            self.intrinsics,
            self.distortion,
            self.poses.clone(),
        )
    }
}

/// Options for linear initialization.
///
/// Currently uses default options from the existing pipeline.
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct PlanarIntrinsicsInitOptions {
    /// Placeholder for future init-specific options.
    #[serde(skip)]
    _private: (),
}

/// Options for non-linear optimization.
///
/// Wraps the existing PlanarIntrinsicsConfig.
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct PlanarIntrinsicsOptimOptions {
    #[serde(flatten)]
    pub config: PlanarIntrinsicsConfig,
}

impl ProblemType for PlanarIntrinsicsProblem {
    type Observations = PlanarIntrinsicsObservations;
    type InitialValues = PlanarIntrinsicsInitial;
    type OptimizedResults = PlanarIntrinsicsOptimized;
    type InitOptions = PlanarIntrinsicsInitOptions;
    type OptimOptions = PlanarIntrinsicsOptimOptions;

    fn problem_name() -> &'static str {
        "planar_intrinsics"
    }

    fn initialize(
        obs: &Self::Observations,
        _opts: &Self::InitOptions,
    ) -> Result<Self::InitialValues> {
        let (seed, cam) = planar_init_seed_from_views(&obs.views)?;
        let init_state = PlanarIntrinsicsInitState::from_seed(&seed);

        let report = PlanarIntrinsicsReport {
            camera: pinhole_camera_params(&cam),
            final_cost: 0.0,
        };

        Ok(PlanarIntrinsicsInitial {
            report,
            init: Some(init_state),
        })
    }

    fn optimize(
        obs: &Self::Observations,
        init: &Self::InitialValues,
        opts: &Self::OptimOptions,
    ) -> Result<Self::OptimizedResults> {
        let input = PlanarIntrinsicsInput {
            views: obs.views.clone(),
        };

        let dataset = crate::build_planar_dataset(&input)?;
        let seed = if let Some(state) = init.init.as_ref() {
            state.to_seed()?
        } else {
            planar_init_seed_from_views(&input.views)?.0
        };

        let result = optimize_planar_intrinsics_with_init(dataset, seed, &opts.config)?;

        let mean_reproj_error = mean_reproj_error_planar(
            &obs.views,
            &result.camera.k,
            &result.camera.dist,
            &result.poses,
        )?;

        let report = PlanarIntrinsicsReport {
            camera: pinhole_camera_params(&result.camera),
            final_cost: result.final_cost,
        };

        Ok(PlanarIntrinsicsOptimized {
            report,
            poses: result.poses,
            mean_reproj_error,
        })
    }
}

fn mean_reproj_error_planar(
    views: &[PlanarViewData],
    intrinsics: &FxFyCxCySkew<Real>,
    distortion: &BrownConrady5<Real>,
    poses: &[Iso3],
) -> Result<Real> {
    let camera = Camera::new(Pinhole, *distortion, IdentitySensor, *intrinsics);

    let mut total_error = 0.0;
    let mut total_points = 0usize;

    for (view, pose) in views.iter().zip(poses.iter()) {
        for (p3d, p2d) in view.points_3d.iter().zip(view.points_2d.iter()) {
            let p_cam = pose.transform_point(p3d);
            if let Some(projected) = camera.project_point_c(&p_cam.coords) {
                let error = (projected - *p2d).norm();
                total_error += error;
                total_points += 1;
            }
        }
    }

    ensure!(
        total_points > 0,
        "no valid projections for error computation"
    );
    Ok(total_error / total_points as Real)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::session::CalibrationSession;
    use calib_core::{BrownConrady5, FxFyCxCySkew, Iso3, Pt3, Vec2};
    use nalgebra::{UnitQuaternion, Vector3};

    #[test]
    fn planar_intrinsics_problem_full_pipeline() {
        // Generate synthetic calibration data
        let k_gt = FxFyCxCySkew {
            fx: 800.0,
            fy: 780.0,
            cx: 640.0,
            cy: 360.0,
            skew: 0.0,
        };
        let dist_gt = BrownConrady5 {
            k1: 0.0,
            k2: 0.0,
            k3: 0.0,
            p1: 0.0,
            p2: 0.0,
            iters: 8,
        };
        let cam_gt = crate::make_pinhole_camera(k_gt, dist_gt);

        // Generate checkerboard points
        let nx = 5;
        let ny = 4;
        let spacing = 0.05_f64;
        let mut board_points = Vec::new();
        for j in 0..ny {
            for i in 0..nx {
                board_points.push(Pt3::new(i as f64 * spacing, j as f64 * spacing, 0.0));
            }
        }

        // Generate views
        let mut views = Vec::new();
        for view_idx in 0..3 {
            let angle = 0.1 * (view_idx as f64);
            let axis = Vector3::new(0.0, 1.0, 0.0);
            let rotation = UnitQuaternion::from_scaled_axis(axis * angle);
            let translation = Vector3::new(0.0, 0.0, 0.6 + 0.1 * view_idx as f64);
            let pose = Iso3::from_parts(translation.into(), rotation);

            let mut points_2d = Vec::new();
            for pw in &board_points {
                let pc = pose.transform_point(pw);
                let proj = cam_gt.project_point(&pc).unwrap();
                points_2d.push(Vec2::new(proj.x, proj.y));
            }

            views.push(PlanarViewData {
                points_3d: board_points.clone(),
                points_2d,
                weights: None,
            });
        }

        // Create session and run pipeline
        let mut session = CalibrationSession::<PlanarIntrinsicsProblem>::new_with_description(
            "Synthetic planar intrinsics test".to_string(),
        );

        let obs = PlanarIntrinsicsObservations { views };
        session.set_observations(obs);

        // Initialize
        let init_result = session.initialize(PlanarIntrinsicsInitOptions::default());
        assert!(init_result.is_ok(), "Initialization failed");
        assert_eq!(
            session.stage(),
            crate::session::SessionStage::Initialized,
            "Should be in Initialized stage"
        );

        // Optimize
        let optim_result = session.optimize(PlanarIntrinsicsOptimOptions::default());
        assert!(optim_result.is_ok(), "Optimization failed");
        assert_eq!(
            session.stage(),
            crate::session::SessionStage::Optimized,
            "Should be in Optimized stage"
        );

        // Export
        let export_result = session.export();
        assert!(export_result.is_ok(), "Export failed");
        assert_eq!(
            session.stage(),
            crate::session::SessionStage::Exported,
            "Should be in Exported stage"
        );

        let final_report = export_result.unwrap().report;
        assert!(
            final_report.final_cost < 1e-6,
            "Final cost too high: {}",
            final_report.final_cost
        );

        // Verify intrinsics are close to ground truth
        let cam_cfg = &final_report.camera;
        let k_est = match &cam_cfg.intrinsics {
            calib_core::IntrinsicsParams::FxFyCxCySkew { params } => FxFyCxCySkew {
                fx: params.fx,
                fy: params.fy,
                cx: params.cx,
                cy: params.cy,
                skew: 0.0,
            },
        };

        assert!((k_est.fx - k_gt.fx).abs() < 20.0, "fx estimate off");
        assert!((k_est.fy - k_gt.fy).abs() < 20.0, "fy estimate off");
        assert!((k_est.cx - k_gt.cx).abs() < 20.0, "cx estimate off");
        assert!((k_est.cy - k_gt.cy).abs() < 20.0, "cy estimate off");
    }

    #[test]
    fn planar_intrinsics_session_json_checkpoint() {
        // Create a simple session with observations
        let views = vec![PlanarViewData {
            points_3d: vec![
                Pt3::new(0.0, 0.0, 0.0),
                Pt3::new(0.05, 0.0, 0.0),
                Pt3::new(0.05, 0.05, 0.0),
                Pt3::new(0.0, 0.05, 0.0),
            ],
            points_2d: vec![
                Vec2::new(100.0, 100.0),
                Vec2::new(200.0, 100.0),
                Vec2::new(200.0, 200.0),
                Vec2::new(100.0, 200.0),
            ],
            weights: None,
        }];

        let mut session = CalibrationSession::<PlanarIntrinsicsProblem>::new();
        session.set_observations(PlanarIntrinsicsObservations { views });

        // Serialize
        let json = session.to_json().unwrap();
        assert!(json.contains("planar_intrinsics"));
        assert!(json.contains("Uninitialized"));

        // Deserialize
        let restored: CalibrationSession<PlanarIntrinsicsProblem> =
            CalibrationSession::from_json(&json).unwrap();
        assert_eq!(
            restored.stage(),
            crate::session::SessionStage::Uninitialized
        );
        assert_eq!(restored.observations().unwrap().views.len(), 1);
    }
}
