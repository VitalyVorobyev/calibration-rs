//! Intermediate state for Scheimpflug intrinsics calibration.

use serde::{Deserialize, Serialize};
use vision_calibration_core::{BrownConrady5, FxFyCxCySkew, Iso3, Real, ScheimpflugParams};

/// Initial parameter bundle consumed by the optimization step.
pub type ScheimpflugInitialValues = (
    FxFyCxCySkew<Real>,
    BrownConrady5<Real>,
    ScheimpflugParams,
    Vec<Iso3>,
);

/// Intermediate state for the Scheimpflug intrinsics pipeline.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ScheimpflugIntrinsicsState {
    /// Initial intrinsics estimated from iterative linear initialization.
    pub initial_intrinsics: Option<FxFyCxCySkew<Real>>,

    /// Initial distortion estimated from iterative linear initialization.
    pub initial_distortion: Option<BrownConrady5<Real>>,

    /// Initial Scheimpflug sensor parameters.
    pub initial_sensor: Option<ScheimpflugParams>,

    /// Initial pose estimates for each view (`camera_se3_target`).
    pub initial_poses: Option<Vec<Iso3>>,

    /// Final solver cost after non-linear optimization.
    pub final_cost: Option<f64>,

    /// Mean reprojection error in pixels.
    pub mean_reproj_error: Option<f64>,
}

impl ScheimpflugIntrinsicsState {
    /// Check if initialization has been run.
    pub fn is_initialized(&self) -> bool {
        self.initial_intrinsics.is_some()
            && self.initial_distortion.is_some()
            && self.initial_sensor.is_some()
            && self.initial_poses.is_some()
    }

    /// Check if optimization has been run.
    pub fn is_optimized(&self) -> bool {
        self.final_cost.is_some()
    }

    /// Return initial values required for optimization.
    pub fn initial_values(&self) -> Option<ScheimpflugInitialValues> {
        Some((
            self.initial_intrinsics?,
            self.initial_distortion?,
            self.initial_sensor?,
            self.initial_poses.clone()?,
        ))
    }

    /// Clear optimization results while keeping initialization.
    pub fn clear_optimization(&mut self) {
        self.final_cost = None;
        self.mean_reproj_error = None;
    }

    /// Clear all intermediate state.
    pub fn clear(&mut self) {
        *self = Self::default();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_state_is_empty() {
        let state = ScheimpflugIntrinsicsState::default();
        assert!(!state.is_initialized());
        assert!(!state.is_optimized());
    }

    #[test]
    fn initialization_and_clear_optimization() {
        let mut state = ScheimpflugIntrinsicsState {
            initial_intrinsics: Some(FxFyCxCySkew {
                fx: 800.0,
                fy: 780.0,
                cx: 640.0,
                cy: 360.0,
                skew: 0.0,
            }),
            initial_distortion: Some(BrownConrady5::default()),
            initial_sensor: Some(ScheimpflugParams::default()),
            initial_poses: Some(vec![Iso3::identity()]),
            final_cost: Some(1.0),
            mean_reproj_error: Some(0.5),
        };

        assert!(state.is_initialized());
        assert!(state.is_optimized());

        state.clear_optimization();

        assert!(state.is_initialized());
        assert!(!state.is_optimized());
        assert!(state.mean_reproj_error.is_none());
    }

    #[test]
    fn json_roundtrip() {
        let state = ScheimpflugIntrinsicsState {
            initial_intrinsics: Some(FxFyCxCySkew {
                fx: 800.0,
                fy: 780.0,
                cx: 640.0,
                cy: 360.0,
                skew: 0.0,
            }),
            initial_distortion: Some(BrownConrady5 {
                k1: -0.05,
                k2: 0.01,
                k3: 0.0,
                p1: 0.0,
                p2: 0.0,
                iters: 8,
            }),
            initial_sensor: Some(ScheimpflugParams {
                tilt_x: 0.01,
                tilt_y: -0.008,
            }),
            initial_poses: Some(vec![Iso3::identity()]),
            final_cost: Some(0.002),
            mean_reproj_error: Some(0.3),
        };

        let json = serde_json::to_string(&state).expect("serialize state");
        let restored: ScheimpflugIntrinsicsState =
            serde_json::from_str(&json).expect("deserialize state");

        assert!(restored.is_initialized());
        assert!(restored.is_optimized());
        assert_eq!(restored.initial_poses.expect("poses").len(), 1);
        assert_eq!(restored.final_cost.expect("final cost"), 0.002);
    }
}
