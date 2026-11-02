//! Intermediate state for planar intrinsics calibration.
//!
//! This module defines `PlanarState`, which holds intermediate results
//! computed during the calibration pipeline (homographies, initial estimates, etc.).

use serde::{Deserialize, Serialize};
use vision_calibration_core::{CameraModelSpec, CameraParams, Iso3, Mat3};
use vision_calibration_optim::PlanarIntrinsicsParams;

use crate::planar_family::convert_distortion;

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
    /// untilted pinhole initialization; the others estimate none.
    pub homographies: Option<Vec<Mat3>>,

    // ─────────────────────────────────────────────────────────────────────────
    // From initialization
    // ─────────────────────────────────────────────────────────────────────────
    /// Initial camera.
    ///
    /// Its distortion holds the linear stage's coefficients (Brown-Conrady
    /// for a pinhole camera), not necessarily in the configured model: the
    /// distortion **model** lives in the config, so a `set_config(model)`
    /// between init and optimize is honoured — [`Self::initial_params`]
    /// converts the coefficients by name. A change of projection or sensor
    /// needs a new init.
    pub initial_camera: Option<CameraParams>,

    /// Whether the initial sensor tilt was a user-provided seed (vs
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
    /// Initialization is considered complete when both the initial camera
    /// and initial poses are available.
    #[cfg(test)]
    pub fn is_initialized(&self) -> bool {
        self.initial_camera.is_some() && self.initial_poses.is_some()
    }

    /// Check if optimization has been run.
    #[cfg(test)]
    pub fn is_optimized(&self) -> bool {
        self.final_cost.is_some()
    }

    /// Initial parameters for optimizing a camera of `model`: the initial
    /// camera with its distortion converted into `model.distortion` (extra
    /// coefficients start at zero).
    ///
    /// Returns `None` if initialization hasn't been run.
    pub fn initial_params(&self, model: CameraModelSpec) -> Option<PlanarIntrinsicsParams> {
        let poses = self.initial_poses.clone()?;
        let mut camera = self.initial_camera.clone()?;
        camera.distortion = convert_distortion(&camera.distortion, model.distortion);
        PlanarIntrinsicsParams::new(camera, poses).ok()
    }

    /// Whether the initial camera has the projection and sensor of `model`,
    /// so [`Self::initial_params`] applies to it.
    pub fn initialized_for(&self, model: CameraModelSpec) -> bool {
        self.initial_camera.as_ref().is_some_and(|c| {
            c.projection.kind() == model.projection && c.sensor.kind() == Some(model.sensor)
        })
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
    use vision_calibration_core::{
        BrownConrady5, DistortionKind, DistortionParams, FxFyCxCy, ProjectionParams,
        ScheimpflugParams, SensorKind, SensorParams,
    };

    fn pinhole_bc5() -> CameraModelSpec {
        CameraModelSpec::pinhole(DistortionKind::BrownConrady5, SensorKind::Identity)
    }

    fn camera(sensor: SensorParams) -> CameraParams {
        CameraParams {
            projection: ProjectionParams::Pinhole,
            distortion: DistortionParams::BrownConrady5 {
                params: BrownConrady5 {
                    k1: -0.1,
                    k2: 0.05,
                    k3: 0.0,
                    p1: 0.001,
                    p2: -0.001,
                    iters: 8,
                },
            },
            sensor,
            intrinsics: FxFyCxCy {
                fx: 800.0,
                fy: 780.0,
                cx: 320.0,
                cy: 240.0,
            },
        }
    }

    #[test]
    fn default_state_not_initialized() {
        let state = PlanarState::default();
        assert!(!state.is_initialized());
        assert!(!state.is_optimized());
    }

    #[test]
    fn is_initialized_requires_both() {
        let state = PlanarState {
            initial_camera: Some(camera(SensorParams::Identity)),
            ..Default::default()
        };
        assert!(!state.is_initialized()); // missing poses

        let state = PlanarState {
            initial_poses: Some(vec![Iso3::identity()]),
            ..state
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
    fn initial_params_convert_the_distortion_into_the_model() {
        let tilt = ScheimpflugParams {
            tilt_x: 0.1,
            tilt_y: -0.02,
        };
        let state = PlanarState {
            initial_camera: Some(camera(SensorParams::Scheimpflug { params: tilt })),
            initial_poses: Some(vec![Iso3::identity()]),
            ..Default::default()
        };
        let tilted = CameraModelSpec::pinhole(DistortionKind::Rational8, SensorKind::Scheimpflug);
        assert!(state.initialized_for(tilted));
        assert!(!state.initialized_for(pinhole_bc5()));

        let params = state.initial_params(tilted).unwrap();
        assert_eq!(params.camera_se3_target.len(), 1);
        assert!(matches!(
            params.camera.sensor,
            SensorParams::Scheimpflug { params } if params.tilt_x == 0.1
        ));
        let DistortionParams::Rational { params } = params.camera.distortion else {
            panic!("a rational distortion");
        };
        assert_eq!((params.k1, params.p2, params.k4), (-0.1, -0.001, 0.0));
    }

    #[test]
    fn clear_optimization_keeps_init() {
        let mut state = PlanarState {
            initial_camera: Some(camera(SensorParams::Identity)),
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
            initial_camera: Some(camera(SensorParams::Identity)),
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
            initial_camera: Some(camera(SensorParams::Identity)),
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
        let camera = restored.initial_camera.unwrap();
        assert_eq!(camera.intrinsics.fx, 800.0);
        assert_eq!(camera.distortion.values()[0], -0.1);
        assert_eq!(restored.initial_poses.unwrap().len(), 2);
        assert_eq!(restored.final_cost.unwrap(), 0.001);
        assert_eq!(restored.mean_reproj_error.unwrap(), 0.5);
        assert_eq!(restored.iterations.unwrap(), 25);
    }
}
