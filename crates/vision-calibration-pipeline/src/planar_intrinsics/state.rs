//! Intermediate state for planar intrinsics calibration.
//!
//! This module defines `PlanarState`, which holds intermediate results
//! computed during the calibration pipeline (homographies, initial estimates, etc.).

use serde::{Deserialize, Serialize};
use vision_calibration_core::{
    BrownConrady5, CameraModelSpec, CameraParams, FxFyCxCy, Iso3, Mat3, ProjectionParams, Real,
    ScheimpflugParams, SensorKind, SensorParams,
};
use vision_calibration_optim::PlanarIntrinsicsParams;

use crate::planar_family::embed_distortion;

/// Intermediate state for planar intrinsics calibration.
///
/// Stores by-products of the calibration pipeline including:
/// - Homographies from pattern detection
/// - Initial estimates from linear initialization
/// - Metrics from optimization
///
/// This struct is updated by step functions as the calibration progresses.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub(crate) struct PlanarState {
    // ─────────────────────────────────────────────────────────────────────────
    // From homography computation
    // ─────────────────────────────────────────────────────────────────────────
    /// Homographies for each view (board_2d -> pixel_2d). Set by the
    /// untilted initialization; the tilted one estimates its own.
    pub homographies: Option<Vec<Mat3>>,

    // ─────────────────────────────────────────────────────────────────────────
    // From initialization
    // ─────────────────────────────────────────────────────────────────────────
    /// Initial intrinsics estimate (fx, fy, cx, cy).
    pub initial_intrinsics: Option<FxFyCxCy<Real>>,

    /// Initial distortion estimate.
    ///
    /// Always Brown-Conrady coefficients: this is the linear-init seed. The
    /// distortion **model** lives in the config, not in state, so a
    /// `set_config(model)` between init and optimize is honoured;
    /// `step_optimize` embeds these coefficients into the configured model.
    pub initial_distortion: Option<BrownConrady5<Real>>,

    /// Initial sensor tilt; set exactly when the camera has a Scheimpflug
    /// sensor.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub initial_sensor: Option<ScheimpflugParams>,

    /// Whether `initial_sensor` was a user-provided tilt seed (vs
    /// auto-estimated). When `true`, the optimization trusts that tilt basin
    /// directly instead of running the cold-start multi-start tilt sweep.
    #[serde(default)]
    pub initial_sensor_manual: bool,

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
    #[cfg(test)]
    pub fn is_initialized(&self) -> bool {
        self.initial_intrinsics.is_some() && self.initial_poses.is_some()
    }

    /// Check if optimization has been run.
    #[cfg(test)]
    pub fn is_optimized(&self) -> bool {
        self.final_cost.is_some()
    }

    /// Initial parameters for optimizing a camera of `model`.
    ///
    /// The Brown-Conrady linear-init coefficients are embedded into
    /// `model.distortion` (extra coefficients start at zero); the sensor is
    /// the initial tilt for a Scheimpflug model.
    ///
    /// Returns `None` if initialization hasn't been run.
    pub fn initial_params(&self, model: CameraModelSpec) -> Option<PlanarIntrinsicsParams> {
        let poses = self.initial_poses.clone()?;
        let intrinsics = self.initial_intrinsics?;
        let distortion = self.initial_distortion.unwrap_or_default();
        let sensor = match model.sensor {
            SensorKind::Identity => SensorParams::Identity,
            SensorKind::Scheimpflug => SensorParams::Scheimpflug {
                params: self.initial_sensor.unwrap_or_default(),
            },
        };
        let camera = CameraParams {
            projection: ProjectionParams::Pinhole,
            distortion: embed_distortion(&distortion, model.distortion),
            sensor,
            intrinsics,
        };
        PlanarIntrinsicsParams::new(camera, poses).ok()
    }

    /// Clear optimization results, keeping initialization.
    pub fn clear_optimization(&mut self) {
        self.final_cost = None;
        self.mean_reproj_error = None;
        self.iterations = None;
    }

    /// Clear everything including initialization.
    #[cfg(test)]
    pub fn clear(&mut self) {
        *self = Self::default();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vision_calibration_core::Iso3;
    use vision_calibration_core::{DistortionKind, DistortionParams};

    fn pinhole_bc5() -> CameraModelSpec {
        CameraModelSpec::pinhole(DistortionKind::BrownConrady5, SensorKind::Identity)
    }

    #[test]
    fn default_state_not_initialized() {
        let state = PlanarState::default();
        assert!(!state.is_initialized());
        assert!(!state.is_optimized());
    }

    #[test]
    fn is_initialized_requires_both() {
        let intrinsics = FxFyCxCy {
            fx: 800.0,
            fy: 800.0,
            cx: 320.0,
            cy: 240.0,
        };

        let state = PlanarState {
            initial_intrinsics: Some(intrinsics),
            ..Default::default()
        };
        assert!(!state.is_initialized()); // missing poses

        let state = PlanarState {
            initial_intrinsics: Some(intrinsics),
            initial_poses: Some(vec![Iso3::identity()]),
            ..Default::default()
        };
        assert!(state.is_initialized());
    }

    #[test]
    fn is_optimized() {
        let state = PlanarState::default();
        assert!(!state.is_optimized());

        let state = PlanarState {
            final_cost: Some(0.001),
            ..Default::default()
        };
        assert!(state.is_optimized());
    }

    #[test]
    fn initial_params_returns_none_when_not_initialized() {
        let state = PlanarState::default();
        assert!(state.initial_params(pinhole_bc5()).is_none());
    }

    #[test]
    fn initial_params_with_default_distortion() {
        let state = PlanarState {
            initial_intrinsics: Some(FxFyCxCy {
                fx: 800.0,
                fy: 800.0,
                cx: 320.0,
                cy: 240.0,
            }),
            initial_poses: Some(vec![Iso3::identity()]),
            ..Default::default()
        };

        let params = state.initial_params(pinhole_bc5());
        assert!(params.is_some());
        let params = params.unwrap();
        assert_eq!(params.poses().len(), 1);
        assert!(matches!(params.camera.sensor, SensorParams::Identity));

        // A Scheimpflug model carries the initial tilt; an extended
        // distortion embeds the Brown-Conrady seed.
        let tilted = PlanarState {
            initial_sensor: Some(ScheimpflugParams {
                tilt_x: 0.1,
                tilt_y: -0.02,
            }),
            ..state
        };
        let params = tilted
            .initial_params(CameraModelSpec::pinhole(
                DistortionKind::Rational8,
                SensorKind::Scheimpflug,
            ))
            .unwrap();
        assert!(matches!(
            params.camera.sensor,
            SensorParams::Scheimpflug { params } if params.tilt_x == 0.1
        ));
        assert!(matches!(
            params.camera.distortion,
            DistortionParams::Rational { .. }
        ));
    }

    #[test]
    fn clear_optimization_keeps_init() {
        let mut state = PlanarState {
            initial_intrinsics: Some(FxFyCxCy {
                fx: 800.0,
                fy: 800.0,
                cx: 320.0,
                cy: 240.0,
            }),
            initial_poses: Some(vec![Iso3::identity()]),
            final_cost: Some(0.001),
            mean_reproj_error: Some(0.5),
            iterations: Some(10),
            ..Default::default()
        };

        state.clear_optimization();

        assert!(state.is_initialized()); // Kept
        assert!(!state.is_optimized()); // Cleared
        assert!(state.final_cost.is_none());
        assert!(state.mean_reproj_error.is_none());
        assert!(state.iterations.is_none());
    }

    #[test]
    fn clear_removes_everything() {
        let mut state = PlanarState {
            initial_intrinsics: Some(FxFyCxCy {
                fx: 800.0,
                fy: 800.0,
                cx: 320.0,
                cy: 240.0,
            }),
            initial_poses: Some(vec![Iso3::identity()]),
            final_cost: Some(0.001),
            ..Default::default()
        };

        state.clear();

        assert!(!state.is_initialized());
        assert!(!state.is_optimized());
    }

    #[test]
    fn json_roundtrip() {
        let state = PlanarState {
            initial_intrinsics: Some(FxFyCxCy {
                fx: 800.0,
                fy: 780.0,
                cx: 320.0,
                cy: 240.0,
            }),
            initial_distortion: Some(BrownConrady5 {
                k1: -0.1,
                k2: 0.05,
                k3: 0.0,
                p1: 0.001,
                p2: -0.001,
                iters: 8,
            }),
            initial_poses: Some(vec![Iso3::identity(), Iso3::identity()]),
            final_cost: Some(0.001),
            mean_reproj_error: Some(0.5),
            iterations: Some(25),
            ..Default::default()
        };

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
