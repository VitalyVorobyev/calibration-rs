//! Shared config sub-structs embedded by grouped top-level `*Config` types.
//!
//! ADR 0024 (field-inventory survey, 2026-07-08) found the same handful of
//! concepts — per-camera linear init, non-linear solve settings, robot-pose
//! refinement, hand-eye linear init — spelled up to four different ways
//! across the eight problem configs. This module holds the single canonical
//! definition of each; top-level `*Config` types embed them as named groups
//! (`init: IntrinsicsInitConfig`, `solver: SolverConfig`, ...) instead of
//! re-declaring the same fields flat.
//!
//! Per-problem defaults still differ (e.g. `SolverConfig::max_iters` is 50
//! for a plain intrinsics solve but 120 for the Scheimpflug tilt valley) —
//! that variation lives in each problem's own `Default` impl, which
//! overrides the field after `..Default::default()`, or via a
//! `field(default_factory = ...)`-style helper. The struct's own [`Default`]
//! documents the plain value used when a problem has no reason to deviate.

use serde::{Deserialize, Serialize};
use vision_calibration_optim::{HandEyeMode, RobustLoss};

/// Per-camera linear-initialization stage (Zhang's method + iterative
/// Brown-Conrady distortion fit).
///
/// Shared by every intrinsics-bearing problem's `init` config group.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[non_exhaustive]
pub struct IntrinsicsInitConfig {
    /// Number of iterations for iterative intrinsics estimation.
    pub init_iterations: usize,
    /// Fix k3 during initialization (recommended for typical lenses).
    pub fix_k3: bool,
    /// Fix tangential distortion (p1, p2) during initialization.
    ///
    /// Defaults to `false` here; the Scheimpflug intrinsics problem
    /// overrides its own `init` default to `true` (was hard-coded before
    /// ADR 0024 — the tilt/tangential-distortion coupling makes a free
    /// tangential term ill-posed during the linear stage).
    pub fix_tangential: bool,
    /// Enforce zero skew during initialization.
    pub zero_skew: bool,
}

impl Default for IntrinsicsInitConfig {
    fn default() -> Self {
        Self {
            init_iterations: 2,
            fix_k3: true,
            fix_tangential: false,
            zero_skew: true,
        }
    }
}

/// Non-linear solve stage options shared across problem types.
///
/// `max_iters` has no single workspace-wide default — problems override it
/// per solve-stage cost. Documented per-problem defaults: 50 standard; 120
/// for the Scheimpflug single-cam tilt valley; 30 for a warm-started joint
/// rig+hand-eye+laser polish; 200 for the rig-laserline frozen-geometry
/// stage (cheap 1-DOF-per-view problem). This struct's own [`Default`] is
/// 50 — the plain value used by problems with no reason to deviate.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[non_exhaustive]
pub struct SolverConfig {
    /// Maximum iterations for the non-linear optimizer.
    pub max_iters: usize,
    /// Verbosity level (0 = silent, 1 = summary, 2+ = detailed).
    pub verbosity: usize,
    /// Robust loss function for outlier handling.
    pub robust_loss: RobustLoss,
}

impl Default for SolverConfig {
    fn default() -> Self {
        Self {
            max_iters: 50,
            verbosity: 0,
            robust_loss: RobustLoss::None,
        }
    }
}

/// Robot-pose refinement options for hand-eye bundle adjustment.
///
/// Replaces the `robot_rot_sigma` / `robot_trans_sigma` /
/// `refine_robot_poses` trio that used to be duplicated verbatim across
/// `SingleCamHandeyeConfig`, `RigHandeyeBaConfig`, and the joint
/// rig+hand-eye+laser BA config.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[non_exhaustive]
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

/// Hand-eye linear-initialization stage options (Tsai-Lenz DLT).
///
/// Shared by every hand-eye problem's `handeye_init` config group.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[non_exhaustive]
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn intrinsics_init_config_default() {
        let cfg = IntrinsicsInitConfig::default();
        assert_eq!(cfg.init_iterations, 2);
        assert!(cfg.fix_k3);
        assert!(!cfg.fix_tangential);
        assert!(cfg.zero_skew);
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
    fn json_roundtrip() {
        let cfg = IntrinsicsInitConfig {
            init_iterations: 3,
            fix_k3: false,
            fix_tangential: true,
            zero_skew: false,
        };
        let json = serde_json::to_string(&cfg).unwrap();
        let restored: IntrinsicsInitConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.init_iterations, 3);
        assert!(!restored.fix_k3);
        assert!(restored.fix_tangential);
        assert!(!restored.zero_skew);
    }
}
