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
    BackendSolveOptions, DistortionKind, HandeyeMode, RobustLoss, SolverBackend,
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

    /// Check a single camera calibrated from its own planar views can be:
    /// its fix lists name parameters of the model, and a Scheimpflug sensor
    /// comes with a distortion model — the tilt is estimated jointly with the
    /// radial terms, which tell it apart from a principal-point shift.
    ///
    /// # Errors
    ///
    /// [`crate::Error::InvalidInput`] naming the problem.
    pub(crate) fn validate_single_camera(&self) -> Result<(), crate::Error> {
        self.validate()?;
        if self.model.sensor == SensorKind::Scheimpflug
            && self.model.distortion == DistortionKind::None
        {
            return Err(crate::Error::invalid_input(
                "a Scheimpflug sensor needs a distortion model: the tilt is estimated \
                 jointly with the radial terms",
            ));
        }
        Ok(())
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
    pub iterations: usize,
    /// Hold `k3` at zero in the linear stage even where the camera's
    /// non-linear solves estimate it (default `true`: a sixth-order term
    /// fitted linearly overfits typical data, and the solve that frees it
    /// starts it from zero).
    ///
    /// Which terms the linear stage fits otherwise follows the camera: a
    /// term the camera holds fixed, or its
    /// model lacks, stays at zero, and so do the tangential terms of a
    /// camera whose tilt is estimated.
    pub fix_k3: bool,
}

impl Default for IntrinsicsInitConfig {
    fn default() -> Self {
        Self {
            iterations: 2,
            fix_k3: true,
        }
    }
}

impl IntrinsicsInitConfig {
    /// Convert to `vision-calibration-linear`'s
    /// [`IterativeIntrinsicsOptions`], the shape every per-camera Zhang's
    /// method bootstrap consumes.
    ///
    /// `iterations_override` — when `Some`, overrides `self.iterations`
    /// (used by step functions that accept a per-call `IntrinsicsInitOptions`
    /// override, e.g. `RigExtrinsicsProblem::step_intrinsics_init_all`).
    /// `None` uses the config's own `iterations` verbatim (the shape
    /// problems without a step-level override need).
    ///
    /// `camera` is the camera being initialized, with the fix lists its
    /// non-linear solves apply; it sets which distortion terms the linear
    /// stage fits ([`Self::distortion_fit_opts`]).
    pub(crate) fn iterative_opts(
        &self,
        iterations_override: Option<usize>,
        camera: &CameraConfig,
    ) -> IterativeIntrinsicsOptions {
        IterativeIntrinsicsOptions {
            iterations: iterations_override.unwrap_or(self.iterations),
            distortion_opts: self.distortion_fit_opts(camera),
        }
    }

    /// The linear stage's distortion-fit options for `camera`, shared by
    /// every intrinsics bootstrap, including the Scheimpflug linear-init
    /// path (which builds a different top-level options struct and so
    /// cannot use [`Self::iterative_opts`] directly).
    ///
    /// The linear stage fits a Brown-Conrady term only if the camera's
    /// model has it and its distortion fix list (the model's default when
    /// unset) does not hold it. A term the non-linear solves hold keeps its
    /// initial value, so a linearly fitted one would stay at that fit
    /// rather than at zero. On top of that:
    /// - `k3` is held under [`Self::fix_k3`];
    /// - the tangential pair is held while the solves estimate a sensor
    ///   tilt: the two trade off in the linear fit, so the pair starts at
    ///   zero and the non-linear solve, which separates them, moves it when
    ///   the camera frees it;
    /// - the pair is fitted only when both terms are free, since the linear
    ///   fit has no single-term option.
    pub(crate) fn distortion_fit_opts(&self, camera: &CameraConfig) -> DistortionFitOptions {
        let default = camera.model.default_fix();
        let held = |list: &Option<Vec<String>>, default: Option<Vec<String>>, name: &str| {
            list.clone()
                .or(default)
                .is_some_and(|l| l.iter().any(|h| h == name))
        };
        let distortion = camera.model.distortion.param_names();
        let free = |n: &str| {
            distortion.contains(&n) && !held(&camera.fix.distortion, default.distortion.clone(), n)
        };
        let estimates_tilt = camera
            .model
            .sensor
            .param_names()
            .iter()
            .any(|n| !held(&camera.fix.sensor, default.sensor.clone(), n));
        DistortionFitOptions {
            fix_k3: self.fix_k3 || !free("k3"),
            fix_tangential: estimates_tilt || !(free("p1") && free("p2")),
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
    pub(crate) fn backend_options(&self) -> BackendSolveOptions {
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
/// Shared by `SingleCamHandeyeConfig`, `HandeyeOptimizeConfig`, and the joint
/// rig+hand-eye+laser BA config (`JointOptimizeConfig`).
///
/// With `refine`, every view's measured robot pose gets an se(3) correction
/// in the gripper frame, `base_se3_gripper · exp(δ)`, under a zero-mean
/// prior with the standard deviations below: the tool's orientation error
/// about each gripper axis and its position error along each, the figures
/// a robot's accuracy is stated in. The prior is weighed
/// against the corner residuals at their noise level, which the solve
/// estimates (one solve at 1 px, then one from that solution at the noise
/// its residuals show), so the result and its covariance are the
/// maximum-likelihood ones for a robot whose error has these standard
/// deviations. The covariance takes them as the robot's true accuracy:
/// state them, rather than leave a loose default, to get honest hand-eye
/// and target standard deviations. Without `refine` the robot poses are
/// taken as exact.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[non_exhaustive]
#[serde(deny_unknown_fields)]
pub struct RobotPoseConfig {
    /// Refine the robot poses with per-view se(3) corrections. On by
    /// default: robot pose errors are rarely below the corner noise, and
    /// taking them as exact pushes them into the camera parameters.
    pub refine: bool,
    /// Standard deviation of the tool's orientation error about each gripper
    /// axis (radians, > 0). Default 0.5°, a loose bound for an industrial
    /// arm.
    pub rot_sigma: f64,
    /// Standard deviation of the tool's position error along each gripper
    /// axis (metres, > 0). Default 1 mm, a loose bound for an industrial
    /// arm.
    pub trans_sigma: f64,
    /// Estimate the robot's error from the corrections, starting from
    /// `rot_sigma` and `trans_sigma`: each prior's variance factor is
    /// estimated from the corrections and their redundancy, and the solve
    /// repeats until it settles. The export reports the estimate
    /// (`robot_pose_sigma`). On by default: the stated accuracy of an arm is
    /// rarely its error in one calibration, and a prior that is too loose
    /// or too tight misstates the hand-eye and target standard deviations.
    /// Off, `rot_sigma` and `trans_sigma` are taken as the robot's true
    /// accuracy.
    pub estimate_sigma: bool,
}

impl Default for RobotPoseConfig {
    fn default() -> Self {
        Self {
            refine: true,
            rot_sigma: 0.5_f64.to_radians(),
            trans_sigma: 1.0e-3,
            estimate_sigma: true,
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
    pub handeye_mode: HandeyeMode,
    /// Minimum motion angle (degrees) for linear hand-eye initialization.
    pub min_motion_angle_deg: f64,
}

impl Default for HandeyeInitConfig {
    fn default() -> Self {
        Self {
            handeye_mode: HandeyeMode::EyeInHand,
            min_motion_angle_deg: 5.0,
        }
    }
}

/// Multi-camera rig frame options: reference-camera gauge and rig-BA scope.
///
/// The `rig` group of `RigExtrinsicsConfig` and `RigHandeyeConfig`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[non_exhaustive]
#[serde(deny_unknown_fields)]
pub struct RigConfig {
    /// Reference camera index for the rig frame (identity extrinsics).
    pub reference_camera_idx: usize,
    /// Refine per-camera intrinsics and distortion jointly with the rig in
    /// its bundle adjustment, holding the camera's `fix` lists. Default:
    /// `true`. Each camera's solve fits them from its own views only; the
    /// joint solve is the estimate that uses every view, and its covariance
    /// carries the intrinsics and their correlation with the extrinsics.
    /// Held fixed, the extrinsics' standard deviations are conditional on
    /// the intrinsics and understate the uncertainty (several-fold on real
    /// stereo rigs).
    pub refine_intrinsics: bool,
    /// Re-refine the sensor tilt of each Scheimpflug camera during rig
    /// bundle adjustment, holding its `fix.sensor` list. Default: `false` —
    /// the tilt is fitted per camera and trades off against the extrinsics
    /// in the joint solve. Cameras with an untilted sensor have no tilt.
    #[serde(default)]
    pub refine_sensor: bool,
}

impl Default for RigConfig {
    fn default() -> Self {
        Self {
            reference_camera_idx: 0,
            refine_intrinsics: true,
            refine_sensor: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn intrinsics_init_config_default() {
        let cfg = IntrinsicsInitConfig::default();
        assert_eq!(cfg.iterations, 2);
        assert!(cfg.fix_k3);
    }

    fn camera(distortion: DistortionKind, sensor: SensorKind) -> CameraConfig {
        CameraConfig::new(CameraModelSpec::pinhole(distortion, sensor))
    }

    #[test]
    fn the_linear_stage_fits_only_what_the_camera_estimates() {
        let cfg = IntrinsicsInitConfig::default();
        let opts = |c: &CameraConfig| {
            let o = cfg.distortion_fit_opts(c);
            (o.fix_k3, o.fix_tangential)
        };
        // Default fix lists: an untilted camera fits p1, p2; a tilted one,
        // whose default list holds them, does not.
        let frontal = camera(DistortionKind::BrownConrady5, SensorKind::Identity);
        assert_eq!(opts(&frontal), (true, false));
        let tilted = camera(DistortionKind::BrownConrady5, SensorKind::Scheimpflug);
        assert_eq!(opts(&tilted), (true, true));
        // A camera holding p1, p2 keeps them at zero in the linear stage.
        let mut held = frontal.clone();
        held.fix.distortion = Some(vec!["k3".into(), "p1".into(), "p2".into()]);
        assert_eq!(opts(&held), (true, true));
        // One held term is enough: the linear fit has no single-term option.
        held.fix.distortion = Some(vec!["p2".into()]);
        assert_eq!(opts(&held), (true, true));
        // A tilted camera that frees them still starts them at zero while
        // its tilt is estimated, and fits them once the tilt is held.
        let mut freed = tilted.clone();
        freed.fix.distortion = Some(Vec::new());
        assert_eq!(opts(&freed), (true, true));
        freed.fix.sensor = Some(vec!["tilt_x".into(), "tilt_y".into()]);
        assert_eq!(opts(&freed), (true, false));
        // A model without the terms never fits them.
        let division = camera(DistortionKind::Division1, SensorKind::Identity);
        assert_eq!(opts(&division), (true, true));
        // `fix_k3: false` fits k3 only where the camera estimates it.
        let free_k3 = IntrinsicsInitConfig {
            fix_k3: false,
            ..IntrinsicsInitConfig::default()
        };
        assert!(free_k3.distortion_fit_opts(&frontal).fix_k3);
        assert!(!free_k3.distortion_fit_opts(&freed).fix_k3);
        let rational = camera(DistortionKind::Rational8, SensorKind::Identity);
        assert!(!free_k3.distortion_fit_opts(&rational).fix_k3);
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
        assert_eq!(cfg.handeye_mode, HandeyeMode::EyeInHand);
        assert!((cfg.min_motion_angle_deg - 5.0).abs() < 1e-15);
    }

    #[test]
    fn rig_config_default() {
        let cfg = RigConfig::default();
        assert_eq!(cfg.reference_camera_idx, 0);
        assert!(cfg.refine_intrinsics);
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
            iterations: 3,
            fix_k3: false,
        };
        let opts = cfg.iterative_opts(None, &CameraConfig::default());
        assert_eq!(opts.iterations, 3);
        assert_eq!(opts.distortion_opts.iters, 8);
    }

    #[test]
    fn iterative_opts_override_wins_over_config_iterations() {
        let cfg = IntrinsicsInitConfig::default();
        let opts = cfg.iterative_opts(Some(9), &CameraConfig::default());
        assert_eq!(opts.iterations, 9);
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
            iterations: 3,
            fix_k3: false,
        };
        let json = serde_json::to_string(&cfg).unwrap();
        let restored: IntrinsicsInitConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.iterations, 3);
        assert!(!restored.fix_k3);
        // The removed field is an unknown field, not silently ignored.
        let old = r#"{"iterations": 2, "fix_k3": true, "fix_tangential": true}"#;
        assert!(serde_json::from_str::<IntrinsicsInitConfig>(old).is_err());
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
