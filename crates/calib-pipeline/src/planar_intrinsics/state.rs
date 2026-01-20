//! Intermediate state for planar intrinsics calibration.
//!
//! This module defines `PlanarState`, which holds intermediate results
//! computed during the calibration pipeline (homographies, initial estimates, etc.).

use calib_core::{make_pinhole_camera, BrownConrady5, FxFyCxCySkew, Iso3, Mat3, Real};
use calib_optim::PlanarIntrinsicsParams;
use serde::{Deserialize, Serialize};

/// Intermediate state for planar intrinsics calibration.
///
/// Stores by-products of the calibration pipeline including:
/// - Homographies from pattern detection
/// - Initial estimates from linear initialization
/// - Metrics from optimization
///
/// This struct is updated by step functions as the calibration progresses.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PlanarState {
    // ─────────────────────────────────────────────────────────────────────────
    // From homography computation
    // ─────────────────────────────────────────────────────────────────────────
    /// Homographies for each view (board_2d -> pixel_2d).
    pub homographies: Option<Vec<Mat3>>,

    // ─────────────────────────────────────────────────────────────────────────
    // From initialization
    // ─────────────────────────────────────────────────────────────────────────
    /// Initial intrinsics estimate (fx, fy, cx, cy, skew).
    pub initial_intrinsics: Option<FxFyCxCySkew<Real>>,

    /// Initial distortion estimate.
    pub initial_distortion: Option<BrownConrady5<Real>>,

    /// Initial pose estimates for each view (camera_T_target).
    pub initial_poses: Option<Vec<Iso3>>,

    // ─────────────────────────────────────────────────────────────────────────
    // From optimization
    // ─────────────────────────────────────────────────────────────────────────
    /// Final cost from optimizer.
    pub final_cost: Option<f64>,

    /// Mean reprojection error (pixels).
    pub mean_reproj_error: Option<f64>,

    /// Number of optimization iterations.
    pub iterations: Option<usize>,
}

impl PlanarState {
    /// Check if initialization has been run.
    ///
    /// Initialization is considered complete when both initial intrinsics
    /// and initial poses are available.
    pub fn is_initialized(&self) -> bool {
        self.initial_intrinsics.is_some() && self.initial_poses.is_some()
    }

    /// Check if optimization has been run.
    pub fn is_optimized(&self) -> bool {
        self.final_cost.is_some()
    }

    /// Get initial parameters for optimization.
    ///
    /// Constructs `PlanarIntrinsicsParams` from the initial estimates.
    /// Returns `None` if initialization hasn't been run.
    pub fn initial_params(&self) -> Option<PlanarIntrinsicsParams> {
        let intrinsics = self.initial_intrinsics?;
        let distortion = self.initial_distortion.unwrap_or_default();
        let poses = self.initial_poses.clone()?;

        // Construct camera from intrinsics and distortion
        let camera = make_pinhole_camera(intrinsics, distortion);
        PlanarIntrinsicsParams::new(camera, poses).ok()
    }

    /// Clear optimization results, keeping initialization.
    pub fn clear_optimization(&mut self) {
        self.final_cost = None;
        self.mean_reproj_error = None;
        self.iterations = None;
    }

    /// Clear everything including initialization.
    pub fn clear(&mut self) {
        *self = Self::default();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use calib_core::Iso3;

    #[test]
    fn default_state_not_initialized() {
        let state = PlanarState::default();
        assert!(!state.is_initialized());
        assert!(!state.is_optimized());
    }

    #[test]
    fn is_initialized_requires_both() {
        let mut state = PlanarState::default();

        // Only intrinsics - not initialized
        state.initial_intrinsics = Some(FxFyCxCySkew {
            fx: 800.0,
            fy: 800.0,
            cx: 320.0,
            cy: 240.0,
            skew: 0.0,
        });
        assert!(!state.is_initialized());

        // Add poses - now initialized
        state.initial_poses = Some(vec![Iso3::identity()]);
        assert!(state.is_initialized());
    }

    #[test]
    fn is_optimized() {
        let mut state = PlanarState::default();
        assert!(!state.is_optimized());

        state.final_cost = Some(0.001);
        assert!(state.is_optimized());
    }

    #[test]
    fn initial_params_returns_none_when_not_initialized() {
        let state = PlanarState::default();
        assert!(state.initial_params().is_none());
    }

    #[test]
    fn initial_params_with_default_distortion() {
        let mut state = PlanarState::default();
        state.initial_intrinsics = Some(FxFyCxCySkew {
            fx: 800.0,
            fy: 800.0,
            cx: 320.0,
            cy: 240.0,
            skew: 0.0,
        });
        state.initial_poses = Some(vec![Iso3::identity()]);
        // No distortion set - should use default

        let params = state.initial_params();
        assert!(params.is_some());
        let params = params.unwrap();
        assert_eq!(params.poses().len(), 1);
    }

    #[test]
    fn clear_optimization_keeps_init() {
        let mut state = PlanarState::default();
        state.initial_intrinsics = Some(FxFyCxCySkew {
            fx: 800.0,
            fy: 800.0,
            cx: 320.0,
            cy: 240.0,
            skew: 0.0,
        });
        state.initial_poses = Some(vec![Iso3::identity()]);
        state.final_cost = Some(0.001);
        state.mean_reproj_error = Some(0.5);
        state.iterations = Some(10);

        state.clear_optimization();

        assert!(state.is_initialized()); // Kept
        assert!(!state.is_optimized()); // Cleared
        assert!(state.final_cost.is_none());
        assert!(state.mean_reproj_error.is_none());
        assert!(state.iterations.is_none());
    }

    #[test]
    fn clear_removes_everything() {
        let mut state = PlanarState::default();
        state.initial_intrinsics = Some(FxFyCxCySkew {
            fx: 800.0,
            fy: 800.0,
            cx: 320.0,
            cy: 240.0,
            skew: 0.0,
        });
        state.initial_poses = Some(vec![Iso3::identity()]);
        state.final_cost = Some(0.001);

        state.clear();

        assert!(!state.is_initialized());
        assert!(!state.is_optimized());
    }

    #[test]
    fn json_roundtrip() {
        let mut state = PlanarState::default();
        state.initial_intrinsics = Some(FxFyCxCySkew {
            fx: 800.0,
            fy: 780.0,
            cx: 320.0,
            cy: 240.0,
            skew: 0.1,
        });
        state.initial_distortion = Some(BrownConrady5 {
            k1: -0.1,
            k2: 0.05,
            k3: 0.0,
            p1: 0.001,
            p2: -0.001,
            iters: 8,
        });
        state.initial_poses = Some(vec![Iso3::identity(), Iso3::identity()]);
        state.final_cost = Some(0.001);
        state.mean_reproj_error = Some(0.5);
        state.iterations = Some(25);

        let json = serde_json::to_string_pretty(&state).unwrap();
        let restored: PlanarState = serde_json::from_str(&json).unwrap();

        assert!(restored.is_initialized());
        assert!(restored.is_optimized());
        assert_eq!(restored.initial_intrinsics.unwrap().fx, 800.0);
        assert_eq!(restored.initial_distortion.unwrap().k1, -0.1);
        assert_eq!(restored.initial_poses.unwrap().len(), 2);
        assert_eq!(restored.final_cost.unwrap(), 0.001);
        assert_eq!(restored.mean_reproj_error.unwrap(), 0.5);
        assert_eq!(restored.iterations.unwrap(), 25);
    }
}
