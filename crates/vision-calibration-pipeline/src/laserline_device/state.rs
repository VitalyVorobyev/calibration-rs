//! Intermediate state for laserline device calibration.

use serde::{Deserialize, Serialize};
use vision_calibration_optim::LaserlineParams;

/// Intermediate state for laserline device calibration.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LaserlineDeviceState {
    /// Initial parameters estimated from linear steps.
    pub initial_params: Option<LaserlineParams>,

    /// RMSE of the initial laser plane fit (meters).
    pub initial_plane_rmse: Option<f64>,

    // Optimization metrics
    /// Final objective value after non-linear optimization.
    pub final_cost: Option<f64>,
    /// Mean reprojection error in pixels.
    pub mean_reproj_error: Option<f64>,
    /// Mean laser residual (units depend on selected residual type).
    pub mean_laser_error: Option<f64>,
    /// Per-view mean reprojection errors in pixels.
    pub per_view_reproj_errors: Option<Vec<f64>>,
    /// Per-view mean laser residuals.
    pub per_view_laser_errors: Option<Vec<f64>>,
}

impl LaserlineDeviceState {
    /// Check if initialization has been run.
    pub fn is_initialized(&self) -> bool {
        self.initial_params.is_some()
    }

    /// Check if optimization has been run.
    pub fn is_optimized(&self) -> bool {
        self.final_cost.is_some()
    }

    /// Clear optimization results, keeping initialization.
    pub fn clear_optimization(&mut self) {
        self.final_cost = None;
        self.mean_reproj_error = None;
        self.mean_laser_error = None;
        self.per_view_reproj_errors = None;
        self.per_view_laser_errors = None;
    }

    /// Clear everything including initialization.
    pub fn clear(&mut self) {
        *self = Self::default();
    }
}
