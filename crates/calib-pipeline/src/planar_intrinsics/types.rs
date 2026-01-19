//! Types for planar intrinsics calibration.
//!
//! Defines observation, initial value, optimized result, and report types
//! for planar intrinsics calibration (Zhang's method with distortion).

use calib_core::{BrownConrady5, CameraParams, CorrespondenceView, FxFyCxCySkew, Iso3, Real};
use serde::{Deserialize, Serialize};

/// Observations for planar intrinsics calibration.
///
/// Contains multiple views of a planar calibration pattern with
/// 3D-2D point correspondences.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanarIntrinsicsObservations {
    /// Views of the calibration pattern.
    pub views: Vec<CorrespondenceView>,
}

impl PlanarIntrinsicsObservations {
    /// Create new observations from views.
    pub fn new(views: Vec<CorrespondenceView>) -> Self {
        Self { views }
    }

    /// Number of views.
    pub fn num_views(&self) -> usize {
        self.views.len()
    }

    /// Total number of points across all views.
    pub fn num_points(&self) -> usize {
        self.views.iter().map(|v| v.points_3d.len()).sum()
    }
}

/// Initial values from linear initialization.
///
/// Contains camera intrinsics, distortion parameters, and per-view poses
/// estimated using closed-form methods (Zhang's algorithm + iterative refinement).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanarIntrinsicsInitial {
    /// Initial camera intrinsics (fx, fy, cx, cy, skew).
    pub intrinsics: FxFyCxCySkew<Real>,
    /// Initial distortion parameters (k1, k2, k3, p1, p2).
    pub distortion: BrownConrady5<Real>,
    /// Initial board-to-camera poses (one per view).
    pub poses: Vec<Iso3>,
}

/// Optimized results from non-linear refinement.
///
/// Contains refined camera parameters, poses, and per-view error statistics
/// for filtering support.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanarIntrinsicsOptimized {
    /// Refined camera intrinsics.
    pub intrinsics: FxFyCxCySkew<Real>,
    /// Refined distortion parameters.
    pub distortion: BrownConrady5<Real>,
    /// Refined board-to-camera poses.
    pub poses: Vec<Iso3>,
    /// Final optimization cost.
    pub final_cost: Real,
    /// Per-view mean reprojection errors (for filtering).
    pub per_view_errors: Vec<Real>,
}

/// Export report for planar intrinsics calibration.
///
/// User-facing report containing calibrated camera parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanarIntrinsicsReport {
    /// Calibrated camera configuration.
    pub camera: CameraParams,
    /// Final optimization cost.
    pub final_cost: Real,
    /// Mean reprojection error across all points (pixels).
    pub mean_reproj_error: Real,
    /// Refined poses (optional, based on export options).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub poses: Option<Vec<Iso3>>,
}

/// Options for linear initialization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanarIntrinsicsInitOptions {
    /// Number of iterations for iterative intrinsics refinement.
    /// Default is 2.
    #[serde(default = "default_init_iterations")]
    pub iterations: usize,
    /// Whether to fix k3 during distortion estimation.
    #[serde(default = "default_true")]
    pub fix_k3: bool,
    /// Whether to fix tangential distortion (p1, p2).
    #[serde(default)]
    pub fix_tangential: bool,
}

impl Default for PlanarIntrinsicsInitOptions {
    fn default() -> Self {
        Self {
            iterations: 2,
            fix_k3: true,
            fix_tangential: false,
        }
    }
}

fn default_init_iterations() -> usize {
    2
}

fn default_true() -> bool {
    true
}

/// Options for non-linear optimization.
///
/// Wraps the solve options from calib-optim.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PlanarIntrinsicsOptimOptions {
    /// Solve options for the optimizer.
    #[serde(flatten)]
    pub solve_opts: calib_optim::PlanarIntrinsicsSolveOptions,
    /// Backend options for the solver.
    #[serde(default)]
    pub backend_opts: calib_optim::BackendSolveOptions,
}

#[cfg(test)]
mod tests {
    use super::*;
    use calib_core::{Pt3, Vec2};

    #[test]
    fn observations_counts() {
        let view = CorrespondenceView {
            points_3d: vec![Pt3::new(0.0, 0.0, 0.0), Pt3::new(1.0, 0.0, 0.0)],
            points_2d: vec![Vec2::new(100.0, 100.0), Vec2::new(200.0, 100.0)],
            weights: None,
        };
        let obs = PlanarIntrinsicsObservations::new(vec![view.clone(), view]);

        assert_eq!(obs.num_views(), 2);
        assert_eq!(obs.num_points(), 4);
    }

    #[test]
    fn init_options_defaults() {
        let opts = PlanarIntrinsicsInitOptions::default();
        assert_eq!(opts.iterations, 2);
        assert!(opts.fix_k3);
        assert!(!opts.fix_tangential);
    }

    #[test]
    fn report_serialization() {
        let report = PlanarIntrinsicsReport {
            camera: CameraParams {
                projection: calib_core::ProjectionParams::Pinhole,
                distortion: calib_core::DistortionParams::BrownConrady5 {
                    params: BrownConrady5::default(),
                },
                sensor: calib_core::SensorParams::Identity,
                intrinsics: calib_core::IntrinsicsParams::FxFyCxCySkew {
                    params: FxFyCxCySkew {
                        fx: 800.0,
                        fy: 800.0,
                        cx: 640.0,
                        cy: 360.0,
                        skew: 0.0,
                    },
                },
            },
            final_cost: 1e-6,
            mean_reproj_error: 0.5,
            poses: None,
        };

        let json = serde_json::to_string(&report).unwrap();
        let restored: PlanarIntrinsicsReport = serde_json::from_str(&json).unwrap();
        assert!((restored.final_cost - 1e-6).abs() < 1e-12);
    }
}
