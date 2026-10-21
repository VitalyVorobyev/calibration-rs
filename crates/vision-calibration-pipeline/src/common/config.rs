//! Shared config sub-structs embedded by grouped top-level `*Config` types.
//!
//! The same handful of concepts — per-camera linear init, non-linear solve
//! settings, robot-pose refinement, hand-eye linear init — recur across the
//! eight problem configs. This module holds the single canonical
//! definition of each; top-level `*Config` types embed them as named groups
//! (`init: IntrinsicsInitConfig`, `solver: SolverConfig`, ...) instead of
//! re-declaring the same fields flat.
//!
//! Per-problem defaults still differ (e.g. `SolverConfig::max_iters` is 50
//! for most solves but 120 for planar intrinsics, sized for a tilted
//! sensor's tilt valley) — that variation lives in each problem's own
//! `Default` impl, which overrides the field after `..Default::default()`.
//! The struct's own [`Default`] documents the plain value used when a
//! problem has no reason to deviate.

use serde::{Deserialize, Serialize};
use vision_calibration_core::{CameraFix, CameraModelSpec, SensorKind};
use vision_calibration_linear::prelude::{DistortionFitOptions, IterativeIntrinsicsOptions};
use vision_calibration_optim::{
    BackendSolveOptions, DistortionKind, HandEyeMode, RobustLoss, SolverBackend,
};

/// Fixed iteration count for the inner Brown-Conrady distortion fit loop.
///
/// Shared by every [`IntrinsicsInitConfig::iterative_opts`] /
/// [`IntrinsicsInitConfig::distortion_fit_opts`] call site — never
/// user-configurable, so it lives as a single named constant rather than a
/// copy-pasted literal `8`.
const DISTORTION_FIT_ITERS: u32 = 8;

/// The camera model a workflow calibrates, and which of its parameters stay
/// fixed.
///
/// The model names one kind per slot — projection, distortion and sensor —
/// so a tilted (Scheimpflug) sensor is a value of `model.sensor`, not a
/// separate workflow. Behaviour that depends on the sensor (default fix
/// lists, the linear stage's tangential terms, the tilt multi-start) follows
/// from the model.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[non_exhaustive]
#[serde(deny_unknown_fields)]
pub struct CameraConfig {
    /// The camera model, one kind per slot. Default: pinhole projection,
    /// Brown-Conrady (`k1, k2, k3, p1, p2`) distortion and an identity
    /// (untilted) sensor — the model every workflow accepts.
    pub model: CameraModelSpec,
    /// Parameters held fixed during the non-linear solve, named per slot
    /// (`{"intrinsics": ["cx", "cy"], "distortion": ["k3"], "sensor":
    /// ["tilt_y"]}`). A slot left out keeps the model's default: `k3` for
    /// Brown-Conrady, and everything but the low-order radial terms (`k1`,
    /// `k2`; the division model's `lambda`) under a Scheimpflug sensor,
    /// because tangential and higher-order terms trade off against the tilt.
    /// An unknown name is a configuration error.
    #[serde(default)]
    pub fix: CameraFix,
}

impl Default for CameraConfig {
    fn default() -> Self {
        Self {
            model: CameraModelSpec::pinhole(DistortionKind::BrownConrady5, SensorKind::Identity),
            fix: CameraFix::default(),
        }
    }
}

impl CameraConfig {
    /// A camera of `model` with the model's default fix lists.
    pub fn new(model: CameraModelSpec) -> Self {
        Self {
            model,
            fix: CameraFix::default(),
        }
    }

    /// Check the fix lists name parameters of the model.
    ///
    /// # Errors
    ///
    /// [`crate::Error::InvalidInput`] naming the unknown parameter.
    pub fn validate(&self) -> Result<(), crate::Error> {
        self.fix
            .resolve(self.model)
            .map(|_| ())
            .map_err(|e| crate::Error::invalid_input(format!("camera.fix: {e}")))
    }
}

/// One camera of a rig whose model differs from the rig's `camera`, so a
/// rig can mix lenses (a tilted laser camera beside a frontal one, a fisheye
/// beside a narrow lens).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[non_exhaustive]
#[serde(deny_unknown_fields)]
pub struct CameraOverride {
    /// Index of the camera in the rig (`0..num_cameras`).
    pub camera: usize,
    /// The camera's model, one kind per slot.
    pub model: CameraModelSpec,
    /// Parameters held fixed during the non-linear solves, named per slot;
    /// a slot left out keeps the model's default (see [`CameraConfig::fix`]).
    #[serde(default)]
    pub fix: CameraFix,
}

impl CameraOverride {
    /// Camera `camera` with `model` and the model's default fix lists.
    pub fn new(camera: usize, model: CameraModelSpec) -> Self {
        Self {
            camera,
            model,
            fix: CameraFix::default(),
        }
    }
}

/// The configuration of rig camera `index`: its override if `overrides` has
/// one, otherwise the rig's `camera`.
pub(crate) fn rig_camera_for(
    camera: &CameraConfig,
    overrides: &[CameraOverride],
    index: usize,
) -> CameraConfig {
    overrides.iter().find(|o| o.camera == index).map_or_else(
        || camera.clone(),
        |o| CameraConfig {
            model: o.model,
            fix: o.fix.clone(),
        },
    )
}

/// Check a rig's camera configuration: every fix list names parameters of
/// its model, and no camera is overridden twice.
///
/// # Errors
///
/// [`crate::Error::InvalidInput`] naming the field.
pub(crate) fn validate_rig_cameras(
    camera: &CameraConfig,
    overrides: &[CameraOverride],
) -> Result<(), crate::Error> {
    camera.validate()?;
    for (i, o) in overrides.iter().enumerate() {
        if overrides[..i].iter().any(|prev| prev.camera == o.camera) {
            return Err(crate::Error::invalid_input(format!(
                "camera_overrides: camera {} is listed twice",
                o.camera
            )));
        }
        o.fix
            .resolve(o.model)
            .map_err(|e| crate::Error::invalid_input(format!("camera_overrides[{i}].fix: {e}")))?;
    }
    Ok(())
}

/// Check every override names a camera of a rig of `num_cameras`.
///
/// # Errors
///
/// [`crate::Error::InvalidInput`] naming the out-of-range camera.
pub(crate) fn validate_rig_camera_indices(
    overrides: &[CameraOverride],
    num_cameras: usize,
) -> Result<(), crate::Error> {
    match overrides.iter().find(|o| o.camera >= num_cameras) {
        Some(o) => Err(crate::Error::invalid_input(format!(
            "camera_overrides: camera {} is out of range (num_cameras = {num_cameras})",
            o.camera
        ))),
        None => Ok(()),
    }
}

/// Per-camera linear-initialization stage (Zhang's method + iterative
/// Brown-Conrady distortion fit).
///
/// Shared by every intrinsics-bearing problem's `init` config group.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[non_exhaustive]
#[serde(deny_unknown_fields)]
pub struct IntrinsicsInitConfig {
    /// Number of iterations for iterative intrinsics estimation.
    pub init_iterations: usize,
    /// Fix k3 during initialization (recommended for typical lenses).
    pub fix_k3: bool,
    /// Fix tangential distortion (p1, p2) during initialization.
    ///
    /// `None` (the default) takes the camera's default: `true` for a
    /// Scheimpflug sensor in planar intrinsics — the tilt/tangential
    /// coupling makes a free tangential term ill-posed during the linear
    /// stage — and `false` otherwise. The multi-camera workflows resolve
    /// `None` to `false` for every camera, whatever its sensor: their
    /// per-camera linear stage has always kept the tangential terms free.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fix_tangential: Option<bool>,
}

impl Default for IntrinsicsInitConfig {
    fn default() -> Self {
        Self {
            init_iterations: 2,
            fix_k3: true,
            fix_tangential: None,
        }
    }
}

impl IntrinsicsInitConfig {
    /// Convert to `vision-calibration-linear`'s
    /// [`IterativeIntrinsicsOptions`], the shape every per-camera Zhang's
    /// method bootstrap consumes.
    ///
    /// `iterations_override` — when `Some`, overrides `self.init_iterations`
    /// (used by step functions that accept a per-call `IntrinsicsInitOptions`
    /// override, e.g. `RigExtrinsicsProblem::step_intrinsics_init_all`).
    /// `None` uses the config's own `init_iterations` verbatim (the shape
    /// problems without a step-level override need).
    ///
    /// `default_fix_tangential` is the camera's default for an unset
    /// [`Self::fix_tangential`].
    pub fn iterative_opts(
        &self,
        iterations_override: Option<usize>,
        default_fix_tangential: bool,
    ) -> IterativeIntrinsicsOptions {
        IterativeIntrinsicsOptions {
            iterations: iterations_override.unwrap_or(self.init_iterations),
            distortion_opts: self.distortion_fit_opts(default_fix_tangential),
        }
    }

    /// The `{fix_k3, fix_tangential, iters}` sub-literal shared by every
    /// intrinsics-bootstrap options struct, including the Scheimpflug
    /// linear-init path (which builds a different top-level options struct
    /// and so cannot use [`Self::iterative_opts`] directly).
    ///
    /// `default_fix_tangential` is the camera's default for an unset
    /// [`Self::fix_tangential`].
    pub fn distortion_fit_opts(&self, default_fix_tangential: bool) -> DistortionFitOptions {
        DistortionFitOptions {
            fix_k3: self.fix_k3,
            fix_tangential: self.fix_tangential.unwrap_or(default_fix_tangential),
            iters: DISTORTION_FIT_ITERS,
        }
    }
}

/// Non-linear solve stage options shared across problem types.
///
/// `max_iters` has no single workspace-wide default — problems override it
/// per solve-stage cost. Documented per-problem defaults: 50 standard; 120
/// for planar intrinsics (sized for a tilted sensor's tilt valley; the
/// solver stops at convergence, so the cap only bounds runs that do not
/// converge); 30 for a warm-started joint
/// rig+hand-eye+laser polish; 200 for the rig-laserline frozen-geometry
/// stage (cheap 1-DOF-per-view problem). This struct's own [`Default`] is
/// 50 — the plain value used by problems with no reason to deviate.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[non_exhaustive]
#[serde(deny_unknown_fields)]
pub struct SolverConfig {
    /// Maximum iterations for the non-linear optimizer.
    pub max_iters: usize,
    /// Verbosity level (0 = silent, 1 = summary, 2+ = detailed).
    pub verbosity: usize,
    /// Robust loss function for outlier handling.
    pub robust_loss: RobustLoss,
    /// Engine that linearizes the problem for the Levenberg–Marquardt loop.
    /// Both reach the same minimizer; see [`SolverBackend`].
    #[serde(default)]
    pub backend: SolverBackend,
}

impl Default for SolverConfig {
    fn default() -> Self {
        Self {
            max_iters: 50,
            verbosity: 0,
            robust_loss: RobustLoss::None,
            backend: SolverBackend::default(),
        }
    }
}

impl SolverConfig {
    /// The solver options this stage runs with. A step that adjusts them (an
    /// iteration budget, say) overrides fields of this value, so the backend
    /// choice always comes from the config.
    pub fn backend_options(&self) -> BackendSolveOptions {
        BackendSolveOptions {
            backend: self.backend,
            max_iters: self.max_iters,
            verbosity: self.verbosity,
            ..BackendSolveOptions::default()
        }
    }
}

/// Robot-pose refinement options for hand-eye bundle adjustment.
///
/// Shared by `SingleCamHandeyeConfig`, `RigHandeyeBaConfig`, and the joint
/// rig+hand-eye+laser BA config.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[non_exhaustive]
#[serde(deny_unknown_fields)]
pub struct RobotPoseConfig {
    /// Refine robot poses with per-view se(3) corrections.
    pub refine: bool,
    /// Robot rotation prior sigma (radians).
    pub rot_sigma: f64,
    /// Robot translation prior sigma (meters).
    pub trans_sigma: f64,
}

impl Default for RobotPoseConfig {
    fn default() -> Self {
        Self {
            refine: true,
            rot_sigma: 0.5_f64.to_radians(),
            trans_sigma: 1.0e-3,
        }
    }
}

impl RobotPoseConfig {
    /// Validate that `rot_sigma`/`trans_sigma` are strictly positive.
    ///
    /// Called from `validate_config` of `SingleCamHandeyeConfig`,
    /// `RigHandeyeConfig`, and `RigHandeyeLaserlineConfig`. Whether the check is gated on
    /// `self.refine` is each caller's own choice — the problem types
    /// disagree on this (`SingleCamHandeyeProblem` only requires positive
    /// sigmas when `refine` is enabled; the rig problem types always
    /// validate) and this helper preserves each site's own
    /// behavior rather than picking one.
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::InvalidInput`] naming the non-positive field.
    pub fn validate(&self) -> Result<(), crate::Error> {
        if self.rot_sigma <= 0.0 {
            return Err(crate::Error::invalid_input(
                "robot_rot_sigma must be positive",
            ));
        }
        if self.trans_sigma <= 0.0 {
            return Err(crate::Error::invalid_input(
                "robot_trans_sigma must be positive",
            ));
        }
        Ok(())
    }
}

/// Hand-eye linear-initialization stage options (Tsai-Lenz DLT).
///
/// Shared by every hand-eye problem's `handeye_init` config group.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[non_exhaustive]
#[serde(deny_unknown_fields)]
pub struct HandeyeInitConfig {
    /// Hand-eye mode: `EyeInHand` or `EyeToHand`.
    pub handeye_mode: HandEyeMode,
    /// Minimum motion angle (degrees) for linear hand-eye initialization.
    pub min_motion_angle_deg: f64,
}

impl Default for HandeyeInitConfig {
    fn default() -> Self {
        Self {
            handeye_mode: HandEyeMode::EyeInHand,
            min_motion_angle_deg: 5.0,
        }
    }
}

/// Multi-camera rig frame options: reference-camera gauge and rig-BA scope.
///
/// The `rig` group of `RigExtrinsicsConfig` and `RigHandeyeConfig`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[non_exhaustive]
#[serde(deny_unknown_fields)]
pub struct RigConfig {
    /// Reference camera index for the rig frame (identity extrinsics).
    pub reference_camera_idx: usize,
    /// Re-refine per-camera intrinsics and distortion jointly during rig
    /// bundle adjustment, holding the camera's `fix` lists. Default:
    /// `false` — the per-camera solves have already fitted them.
    pub refine_intrinsics_in_rig_ba: bool,
    /// Re-refine the sensor tilt of each Scheimpflug camera during rig
    /// bundle adjustment, holding its `fix.sensor` list. Default: `false` —
    /// the tilt is fitted per camera and trades off against the extrinsics
    /// in the joint solve. Cameras with an untilted sensor have no tilt.
    #[serde(default)]
    pub refine_sensor: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn intrinsics_init_config_default() {
        let cfg = IntrinsicsInitConfig::default();
        assert_eq!(cfg.init_iterations, 2);
        assert!(cfg.fix_k3);
        assert_eq!(cfg.fix_tangential, None);
        assert!(!cfg.distortion_fit_opts(false).fix_tangential);
        assert!(cfg.distortion_fit_opts(true).fix_tangential);
    }

    #[test]
    fn solver_config_default() {
        let cfg = SolverConfig::default();
        assert_eq!(cfg.max_iters, 50);
        assert_eq!(cfg.verbosity, 0);
        assert_eq!(cfg.robust_loss, RobustLoss::None);
    }

    #[test]
    fn robot_pose_config_default() {
        let cfg = RobotPoseConfig::default();
        assert!(cfg.refine);
        assert!((cfg.rot_sigma - 0.5_f64.to_radians()).abs() < 1e-15);
        assert!((cfg.trans_sigma - 1.0e-3).abs() < 1e-15);
    }

    #[test]
    fn handeye_init_config_default() {
        let cfg = HandeyeInitConfig::default();
        assert_eq!(cfg.handeye_mode, HandEyeMode::EyeInHand);
        assert!((cfg.min_motion_angle_deg - 5.0).abs() < 1e-15);
    }

    #[test]
    fn rig_config_default() {
        let cfg = RigConfig::default();
        assert_eq!(cfg.reference_camera_idx, 0);
        assert!(!cfg.refine_intrinsics_in_rig_ba);
        assert!(!cfg.refine_sensor);
    }

    #[test]
    fn rig_camera_overrides_select_and_validate() {
        let camera = CameraConfig::default();
        let tilted =
            CameraModelSpec::pinhole(DistortionKind::BrownConrady5, SensorKind::Scheimpflug);
        let overrides = vec![CameraOverride::new(1, tilted)];
        assert_eq!(rig_camera_for(&camera, &overrides, 0), camera);
        assert_eq!(rig_camera_for(&camera, &overrides, 1).model, tilted);
        assert!(validate_rig_cameras(&camera, &overrides).is_ok());
        assert!(validate_rig_camera_indices(&overrides, 2).is_ok());

        let err = validate_rig_camera_indices(&overrides, 1)
            .unwrap_err()
            .to_string();
        assert!(err.contains("out of range"), "{err}");
        let twice = vec![overrides[0].clone(), overrides[0].clone()];
        let err = validate_rig_cameras(&camera, &twice)
            .unwrap_err()
            .to_string();
        assert!(err.contains("listed twice"), "{err}");
        let mut bad = overrides[0].clone();
        bad.model = CameraModelSpec::pinhole(DistortionKind::BrownConrady5, SensorKind::Identity);
        bad.fix.sensor = Some(vec!["tilt_x".into()]);
        let err = validate_rig_cameras(&camera, &[bad])
            .unwrap_err()
            .to_string();
        assert!(err.contains("tilt_x"), "{err}");

        let json = r#"{"camera": 1, "model": {"projection": "pinhole",
            "distortion": "division1", "sensor": "identity"}}"#;
        let parsed: CameraOverride = serde_json::from_str(json).unwrap();
        assert_eq!(parsed.model.distortion, DistortionKind::Division1);
        assert!(serde_json::from_str::<CameraOverride>(r#"{"camera": 0, "lens": 1}"#).is_err());
    }

    #[test]
    fn iterative_opts_uses_config_iterations_when_no_override() {
        let cfg = IntrinsicsInitConfig {
            init_iterations: 3,
            fix_k3: false,
            fix_tangential: Some(true),
        };
        // An explicit value wins over the camera's default.
        let opts = cfg.iterative_opts(None, false);
        assert_eq!(opts.iterations, 3);
        assert!(!opts.distortion_opts.fix_k3);
        assert!(opts.distortion_opts.fix_tangential);
        assert_eq!(opts.distortion_opts.iters, 8);
    }

    #[test]
    fn iterative_opts_override_wins_over_config_iterations() {
        let cfg = IntrinsicsInitConfig::default();
        let opts = cfg.iterative_opts(Some(9), false);
        assert_eq!(opts.iterations, 9);
    }

    #[test]
    fn distortion_fit_opts_mirrors_init_fields() {
        let cfg = IntrinsicsInitConfig {
            init_iterations: 2,
            fix_k3: true,
            fix_tangential: Some(false),
        };
        let dist_opts = cfg.distortion_fit_opts(true);
        assert!(dist_opts.fix_k3);
        assert!(!dist_opts.fix_tangential);
        assert_eq!(dist_opts.iters, 8);
    }

    #[test]
    fn robot_pose_config_validate_accepts_default() {
        assert!(RobotPoseConfig::default().validate().is_ok());
    }

    #[test]
    fn robot_pose_config_validate_rejects_non_positive_rot_sigma() {
        let cfg = RobotPoseConfig {
            rot_sigma: 0.0,
            ..Default::default()
        };
        let err = cfg.validate().expect_err("zero rot_sigma must be rejected");
        assert!(err.to_string().contains("robot_rot_sigma"));
    }

    #[test]
    fn robot_pose_config_validate_rejects_non_positive_trans_sigma() {
        let cfg = RobotPoseConfig {
            trans_sigma: -1.0,
            ..Default::default()
        };
        let err = cfg
            .validate()
            .expect_err("negative trans_sigma must be rejected");
        assert!(err.to_string().contains("robot_trans_sigma"));
    }

    #[test]
    fn json_roundtrip() {
        let cfg = IntrinsicsInitConfig {
            init_iterations: 3,
            fix_k3: false,
            fix_tangential: Some(true),
        };
        let json = serde_json::to_string(&cfg).unwrap();
        let restored: IntrinsicsInitConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.init_iterations, 3);
        assert!(!restored.fix_k3);
        assert_eq!(restored.fix_tangential, Some(true));
        // An unset value is omitted and reads back as `None`.
        let json = serde_json::to_string(&IntrinsicsInitConfig::default()).unwrap();
        assert!(!json.contains("fix_tangential"), "{json}");
    }

    #[test]
    fn camera_config_defaults_to_pinhole_brown_conrady() {
        let cfg = CameraConfig::default();
        assert_eq!(cfg.model.distortion, DistortionKind::BrownConrady5);
        assert_eq!(cfg.model.sensor, SensorKind::Identity);
        assert_eq!(cfg.fix, CameraFix::default());
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn camera_config_json_roundtrip_and_validation() {
        let json = r#"{
            "model": {"projection": "pinhole", "distortion": "brown_conrady5", "sensor": "scheimpflug"},
            "fix": {"intrinsics": ["cx", "cy"], "sensor": ["tilt_y"]}
        }"#;
        let cfg: CameraConfig = serde_json::from_str(json).unwrap();
        assert_eq!(cfg.model.sensor, SensorKind::Scheimpflug);
        assert!(cfg.validate().is_ok());
        let back: CameraConfig =
            serde_json::from_str(&serde_json::to_string(&cfg).unwrap()).unwrap();
        assert_eq!(back, cfg);

        let bad: CameraConfig = serde_json::from_str(
            r#"{"model": {"projection": "pinhole", "distortion": "brown_conrady5", "sensor": "identity"},
                "fix": {"sensor": ["tilt_x"]}}"#,
        )
        .unwrap();
        let err = bad.validate().unwrap_err().to_string();
        assert!(err.contains("tilt_x"), "{err}");
        assert!(serde_json::from_str::<CameraConfig>(r#"{"model": {}, "extra": 1}"#).is_err());
    }
}
