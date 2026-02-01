//! [`ProblemType`] implementation for single laserline device calibration.

use anyhow::{Result, anyhow, ensure};
use serde::{Deserialize, Serialize};
use vision_calibration_core::ScheimpflugParams;
use vision_calibration_linear::prelude::*;
use vision_calibration_optim::{
    BackendSolveOptions, LaserlineDataset, LaserlineEstimate, LaserlineResidualType,
    LaserlineSolveOptions, LaserlineStats,
};

use crate::session::{InvalidationPolicy, ProblemType};

use super::state::LaserlineDeviceState;

/// Laserline device calibration problem (single camera + laser plane).
#[derive(Debug)]
pub struct LaserlineDeviceProblem;

/// Input type for laserline device calibration.
pub type LaserlineDeviceInput = LaserlineDataset;

/// Configuration for laserline device calibration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LaserlineDeviceConfig {
    // ─────────────────────────────────────────────────────────────────────────
    // Initialization options
    // ─────────────────────────────────────────────────────────────────────────
    /// Number of iterations for iterative intrinsics estimation.
    pub init_iterations: usize,
    /// Fix k3 during initialization (recommended for typical lenses).
    pub fix_k3_in_init: bool,
    /// Fix tangential distortion during initialization.
    pub fix_tangential_in_init: bool,
    /// Enforce zero skew during initialization.
    pub zero_skew: bool,
    /// Initial Scheimpflug sensor parameters (use zeros for pinhole/identity).
    pub sensor_init: ScheimpflugParams,

    // ─────────────────────────────────────────────────────────────────────────
    // Optimization options
    // ─────────────────────────────────────────────────────────────────────────
    /// Maximum iterations for the optimizer.
    pub max_iters: usize,
    /// Verbosity level (0 = silent, 1 = summary, 2+ = detailed).
    pub verbosity: usize,
    /// Robust loss for calibration reprojection residuals.
    pub calib_loss: vision_calibration_optim::RobustLoss,
    /// Robust loss for laser residuals.
    pub laser_loss: vision_calibration_optim::RobustLoss,
    /// Global weight for calibration residuals.
    pub calib_weight: f64,
    /// Global weight for laser residuals.
    pub laser_weight: f64,
    /// Fix camera intrinsics during optimization.
    pub fix_intrinsics: bool,
    /// Fix distortion parameters during optimization.
    pub fix_distortion: bool,
    /// Fix k3 distortion parameter during optimization.
    pub fix_k3: bool,
    /// Fix Scheimpflug sensor parameters during optimization.
    pub fix_sensor: bool,
    /// Indices of poses to fix (e.g., \[0\] to fix first pose).
    pub fix_poses: Vec<usize>,
    /// Fix laser plane parameters during optimization.
    pub fix_plane: bool,
    /// Laser residual type.
    pub laser_residual_type: LaserlineResidualType,
}

impl Default for LaserlineDeviceConfig {
    fn default() -> Self {
        Self {
            // Init
            init_iterations: 2,
            fix_k3_in_init: true,
            fix_tangential_in_init: false,
            zero_skew: true,
            sensor_init: ScheimpflugParams::default(),
            // Optimize
            max_iters: 50,
            verbosity: 0,
            calib_loss: vision_calibration_optim::RobustLoss::Huber { scale: 1.0 },
            laser_loss: vision_calibration_optim::RobustLoss::Huber { scale: 0.01 },
            calib_weight: 1.0,
            laser_weight: 1.0,
            fix_intrinsics: false,
            fix_distortion: false,
            fix_k3: true,
            fix_sensor: true,
            fix_poses: vec![0],
            fix_plane: false,
            laser_residual_type: LaserlineResidualType::LineDistNormalized,
        }
    }
}

impl LaserlineDeviceConfig {
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
    pub fn solve_opts(&self) -> LaserlineSolveOptions {
        LaserlineSolveOptions {
            calib_loss: self.calib_loss,
            calib_weight: self.calib_weight,
            laser_loss: self.laser_loss,
            laser_weight: self.laser_weight,
            fix_intrinsics: self.fix_intrinsics,
            fix_distortion: self.fix_distortion,
            fix_k3: self.fix_k3,
            fix_sensor: self.fix_sensor,
            fix_poses: self.fix_poses.clone(),
            fix_plane: self.fix_plane,
            laser_residual_type: self.laser_residual_type,
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

/// Pipeline output including optimized parameters and summary statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LaserlineDeviceOutput {
    pub estimate: LaserlineEstimate,
    pub stats: LaserlineStats,
}

/// Export type for laserline device calibration.
pub type LaserlineDeviceExport = LaserlineDeviceOutput;

impl ProblemType for LaserlineDeviceProblem {
    type Config = LaserlineDeviceConfig;
    type Input = LaserlineDeviceInput;
    type State = LaserlineDeviceState;
    type Output = LaserlineDeviceOutput;
    type Export = LaserlineDeviceExport;

    fn name() -> &'static str {
        "laserline_device_v1"
    }

    fn validate_input(input: &Self::Input) -> Result<()> {
        ensure!(
            input.len() >= 3,
            "need at least 3 views for intrinsics initialization (got {})",
            input.len()
        );

        for (i, view) in input.iter().enumerate() {
            ensure!(
                view.obs.len() >= 4,
                "view {} has too few points (need >= 4 for homography, got {})",
                i,
                view.obs.len()
            );
            view.meta
                .validate()
                .map_err(|e| anyhow!("view {}: {}", i, e))?;
        }

        Ok(())
    }

    fn validate_config(config: &Self::Config) -> Result<()> {
        ensure!(config.max_iters > 0, "max_iters must be positive");
        ensure!(
            config.init_iterations > 0,
            "init_iterations must be positive"
        );
        ensure!(config.calib_weight > 0.0, "calib_weight must be positive");
        ensure!(config.laser_weight > 0.0, "laser_weight must be positive");
        Ok(())
    }

    fn on_input_change() -> InvalidationPolicy {
        InvalidationPolicy::CLEAR_COMPUTED
    }

    fn export(output: &Self::Output, _config: &Self::Config) -> Result<Self::Export> {
        Ok(output.clone())
    }
}
