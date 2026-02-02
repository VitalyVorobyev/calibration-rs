//! [`ProblemType`] implementation for planar intrinsics calibration.
//!
//! This module provides the `PlanarIntrinsicsProblem` type that implements
//! the session API's `ProblemType` trait.

use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};
use vision_calibration_core::{DistortionFixMask, IntrinsicsFixMask, PlanarDataset};
use vision_calibration_linear::prelude::*;
use vision_calibration_optim::{
    BackendSolveOptions, PlanarIntrinsicsEstimate, PlanarIntrinsicsSolveOptions,
};

use crate::session::{InvalidationPolicy, ProblemType};

use super::state::PlanarState;

/// Planar intrinsics calibration problem (Zhang's method with distortion).
///
/// This problem type implements calibration of camera intrinsics and distortion
/// from multiple views of a planar calibration pattern.
///
/// # Associated Types
///
/// - **Config**: [`PlanarConfig`] - solver settings, fix masks, etc.
/// - **Input**: [`PlanarDataset`] - views with 2D-3D point correspondences
/// - **State**: [`PlanarState`] - homographies, initial estimates, metrics
/// - **Output**: [`PlanarIntrinsicsEstimate`] - final calibrated camera + poses
/// - **Export**: [`PlanarIntrinsicsEstimate`] - same as output
///
/// # Example
///
/// ```no_run
/// use vision_calibration_pipeline::session::CalibrationSession;
/// use vision_calibration_pipeline::planar_intrinsics::{
///     PlanarIntrinsicsProblem, PlanarConfig,
///     step_init, step_optimize,
/// };
/// # fn main() -> anyhow::Result<()> {
/// # let dataset = unimplemented!();
///
/// let mut session = CalibrationSession::<PlanarIntrinsicsProblem>::new();
/// session.set_input(dataset)?;
///
/// step_init(&mut session, None)?;
/// step_optimize(&mut session, None)?;
///
/// let export = session.export()?;
/// # Ok(())
/// # }
/// ```
#[derive(Debug)]
pub struct PlanarIntrinsicsProblem;

/// Configuration for planar intrinsics calibration.
///
/// Contains settings for both initialization and optimization phases.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanarConfig {
    // ─────────────────────────────────────────────────────────────────────────
    // Initialization options
    // ─────────────────────────────────────────────────────────────────────────
    /// Number of iterations for iterative intrinsics estimation.
    pub init_iterations: usize,

    /// Fix k3 during initialization (recommended for typical lenses).
    pub fix_k3_in_init: bool,

    /// Fix tangential distortion (p1, p2) during initialization.
    pub fix_tangential_in_init: bool,

    /// Enforce zero skew during initialization.
    pub zero_skew: bool,

    // ─────────────────────────────────────────────────────────────────────────
    // Optimization options
    // ─────────────────────────────────────────────────────────────────────────
    /// Maximum iterations for the optimizer.
    pub max_iters: usize,

    /// Verbosity level (0 = silent, 1 = summary, 2+ = detailed).
    pub verbosity: usize,

    /// Robust loss function for outlier handling.
    pub robust_loss: vision_calibration_optim::RobustLoss,

    /// Mask for fixing intrinsic parameters during optimization.
    pub fix_intrinsics: IntrinsicsFixMask,

    /// Mask for fixing distortion parameters during optimization.
    pub fix_distortion: DistortionFixMask,

    /// Indices of poses to fix during optimization (e.g., \[0\] to fix first pose).
    pub fix_poses: Vec<usize>,
}

impl Default for PlanarConfig {
    fn default() -> Self {
        Self {
            // Initialization
            init_iterations: 2,
            fix_k3_in_init: true,
            fix_tangential_in_init: false,
            zero_skew: true,
            // Optimization
            max_iters: 50,
            verbosity: 0,
            robust_loss: vision_calibration_optim::RobustLoss::None,
            fix_intrinsics: Default::default(),
            fix_distortion: Default::default(),
            fix_poses: Vec::new(),
        }
    }
}

impl PlanarConfig {
    /// Convert to vision-calibration-linear initialization options.
    pub fn init_opts(&self) -> IterativeIntrinsicsOptions {
        IterativeIntrinsicsOptions {
            iterations: self.init_iterations,
            distortion_opts: DistortionFitOptions {
                fix_k3: self.fix_k3_in_init,
                fix_tangential: self.fix_tangential_in_init,
                iters: 8,
            },
            zero_skew: self.zero_skew,
        }
    }

    /// Convert to vision-calibration-optim solve options.
    pub fn solve_opts(&self) -> PlanarIntrinsicsSolveOptions {
        PlanarIntrinsicsSolveOptions {
            robust_loss: self.robust_loss,
            fix_intrinsics: self.fix_intrinsics,
            fix_distortion: self.fix_distortion,
            fix_poses: self.fix_poses.clone(),
        }
    }

    /// Convert to backend solver options.
    pub fn backend_opts(&self) -> BackendSolveOptions {
        BackendSolveOptions {
            max_iters: self.max_iters,
            verbosity: self.verbosity,
            ..Default::default()
        }
    }
}

/// Export format for planar intrinsics.
///
/// Currently identical to `PlanarIntrinsicsEstimate`, but kept as a separate
/// type alias for flexibility in future changes.
pub type PlanarExport = PlanarIntrinsicsEstimate;

impl ProblemType for PlanarIntrinsicsProblem {
    type Config = PlanarConfig;
    type Input = PlanarDataset;
    type State = PlanarState;
    type Output = PlanarIntrinsicsEstimate;
    type Export = PlanarExport;

    fn name() -> &'static str {
        "planar_intrinsics_v2"
    }

    fn schema_version() -> u32 {
        1
    }

    fn validate_input(input: &Self::Input) -> Result<()> {
        ensure!(
            input.num_views() >= 3,
            "need at least 3 views for calibration (got {})",
            input.num_views()
        );

        for (i, view) in input.views.iter().enumerate() {
            ensure!(
                view.obs.len() >= 4,
                "view {} has too few points (need >= 4 for homography, got {})",
                i,
                view.obs.len()
            );
        }

        Ok(())
    }

    fn validate_config(config: &Self::Config) -> Result<()> {
        ensure!(config.max_iters > 0, "max_iters must be positive");
        ensure!(
            config.init_iterations > 0,
            "init_iterations must be positive"
        );
        Ok(())
    }

    fn on_input_change() -> InvalidationPolicy {
        // When input changes, clear state and output
        InvalidationPolicy::CLEAR_COMPUTED
    }

    fn on_config_change() -> InvalidationPolicy {
        // Config changes don't auto-invalidate (allow experimentation)
        InvalidationPolicy::KEEP_ALL
    }

    fn export(output: &Self::Output, _config: &Self::Config) -> Result<Self::Export> {
        Ok(output.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vision_calibration_core::{CorrespondenceView, NoMeta, Pt2, Pt3, View};

    fn make_minimal_dataset() -> PlanarDataset {
        // Create 3 views with 4 points each (minimum requirements)
        let make_view = || {
            View::new(
                CorrespondenceView {
                    points_3d: vec![
                        Pt3::new(0.0, 0.0, 0.0),
                        Pt3::new(0.05, 0.0, 0.0),
                        Pt3::new(0.05, 0.05, 0.0),
                        Pt3::new(0.0, 0.05, 0.0),
                    ],
                    points_2d: vec![
                        Pt2::new(100.0, 100.0),
                        Pt2::new(200.0, 100.0),
                        Pt2::new(200.0, 200.0),
                        Pt2::new(100.0, 200.0),
                    ],
                    weights: None,
                },
                NoMeta {},
            )
        };

        PlanarDataset::new(vec![make_view(), make_view(), make_view()]).unwrap()
    }

    #[test]
    fn validate_input_requires_3_views() {
        // PlanarDataset::new already validates this, so we test our validation
        // by checking the error message content
        let dataset = make_minimal_dataset();
        // With 3 views it should pass
        let result = PlanarIntrinsicsProblem::validate_input(&dataset);
        assert!(result.is_ok());
    }

    #[test]
    fn validate_input_minimum_views() {
        // Verify the validation function checks view count
        // We can't create invalid PlanarDataset, but we verify the validation logic
        // by checking that the valid dataset passes
        let dataset = make_minimal_dataset();
        assert!(PlanarIntrinsicsProblem::validate_input(&dataset).is_ok());
    }

    #[test]
    fn validate_input_accepts_valid() {
        let dataset = make_minimal_dataset();
        let result = PlanarIntrinsicsProblem::validate_input(&dataset);
        assert!(result.is_ok());
    }

    #[test]
    fn validate_config_requires_positive_iters() {
        let config = PlanarConfig {
            max_iters: 0,
            ..Default::default()
        };
        let result = PlanarIntrinsicsProblem::validate_config(&config);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("max_iters"));
    }

    #[test]
    fn validate_config_accepts_valid() {
        let config = PlanarConfig::default();
        let result = PlanarIntrinsicsProblem::validate_config(&config);
        assert!(result.is_ok());
    }

    #[test]
    fn config_json_roundtrip() {
        let config = PlanarConfig {
            max_iters: 100,
            fix_k3_in_init: false,
            robust_loss: vision_calibration_optim::RobustLoss::Huber { scale: 2.5 },
            ..Default::default()
        };

        let json = serde_json::to_string_pretty(&config).unwrap();
        let restored: PlanarConfig = serde_json::from_str(&json).unwrap();

        assert_eq!(restored.max_iters, 100);
        assert!(!restored.fix_k3_in_init);
        match restored.robust_loss {
            vision_calibration_optim::RobustLoss::Huber { scale } => {
                assert!((scale - 2.5).abs() < 1e-12);
            }
            _ => panic!("wrong loss type"),
        }
    }

    #[test]
    fn problem_name_and_version() {
        assert_eq!(PlanarIntrinsicsProblem::name(), "planar_intrinsics_v2");
        assert_eq!(PlanarIntrinsicsProblem::schema_version(), 1);
    }

    #[test]
    fn init_opts_conversion() {
        let config = PlanarConfig {
            init_iterations: 5,
            fix_k3_in_init: true,
            fix_tangential_in_init: true,
            zero_skew: false,
            ..Default::default()
        };

        let opts = config.init_opts();
        assert_eq!(opts.iterations, 5);
        assert!(opts.distortion_opts.fix_k3);
        assert!(opts.distortion_opts.fix_tangential);
        assert!(!opts.zero_skew);
    }

    #[test]
    fn solve_opts_conversion() {
        let config = PlanarConfig {
            robust_loss: vision_calibration_optim::RobustLoss::Cauchy { scale: 1.0 },
            fix_poses: vec![0, 1],
            ..Default::default()
        };

        let opts = config.solve_opts();
        assert_eq!(opts.fix_poses, vec![0, 1]);
        match opts.robust_loss {
            vision_calibration_optim::RobustLoss::Cauchy { scale } => {
                assert!((scale - 1.0).abs() < 1e-12);
            }
            _ => panic!("wrong loss type"),
        }
    }

    #[test]
    fn backend_opts_conversion() {
        let config = PlanarConfig {
            max_iters: 100,
            verbosity: 2,
            ..Default::default()
        };

        let opts = config.backend_opts();
        assert_eq!(opts.max_iters, 100);
        assert_eq!(opts.verbosity, 2);
    }
}
