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
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LaserlineDeviceConfig {
    /// Initialization options.
    pub init: LaserlineDeviceInitConfig,
    /// Shared solver options.
    pub solver: LaserlineDeviceSolverConfig,
    /// Bundle-adjustment options.
    pub optimize: LaserlineDeviceOptimizeConfig,
}

/// Initialization options for laserline device calibration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LaserlineDeviceInitConfig {
    /// Number of iterations for iterative intrinsics estimation.
    pub iterations: usize,
    /// Fix k3 during initialization (recommended for typical lenses).
    pub fix_k3: bool,
    /// Fix tangential distortion during initialization.
    pub fix_tangential: bool,
    /// Enforce zero skew during initialization.
    pub zero_skew: bool,
    /// Initial Scheimpflug sensor parameters (use zeros for pinhole/identity).
    pub sensor_init: ScheimpflugParams,
}

impl Default for LaserlineDeviceInitConfig {
    fn default() -> Self {
        Self {
            iterations: 2,
            fix_k3: true,
            fix_tangential: false,
            zero_skew: true,
            sensor_init: ScheimpflugParams::default(),
        }
    }
}

/// Shared solver options for laserline device calibration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LaserlineDeviceSolverConfig {
    /// Maximum iterations for the optimizer.
    pub max_iters: usize,
    /// Verbosity level (0 = silent, 1 = summary, 2+ = detailed).
    pub verbosity: usize,
}

impl Default for LaserlineDeviceSolverConfig {
    fn default() -> Self {
        Self {
            max_iters: 50,
            verbosity: 0,
        }
    }
}

/// Bundle-adjustment options for laserline device calibration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LaserlineDeviceOptimizeConfig {
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

impl Default for LaserlineDeviceOptimizeConfig {
    fn default() -> Self {
        Self {
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
            iterations: self.init.iterations,
            distortion_opts: DistortionFitOptions {
                fix_k3: self.init.fix_k3,
                fix_tangential: self.init.fix_tangential,
                iters: 8,
            },
            zero_skew: self.init.zero_skew,
        }
    }

    /// Convert to vision-calibration-optim solve options.
    pub fn solve_opts(&self) -> LaserlineSolveOptions {
        LaserlineSolveOptions {
            calib_loss: self.optimize.calib_loss,
            calib_weight: self.optimize.calib_weight,
            laser_loss: self.optimize.laser_loss,
            laser_weight: self.optimize.laser_weight,
            fix_intrinsics: self.optimize.fix_intrinsics,
            fix_distortion: self.optimize.fix_distortion,
            fix_k3: self.optimize.fix_k3,
            fix_sensor: self.optimize.fix_sensor,
            fix_poses: self.optimize.fix_poses.clone(),
            fix_plane: self.optimize.fix_plane,
            laser_residual_type: self.optimize.laser_residual_type,
        }
    }

    /// Convert to backend solver options.
    pub fn backend_opts(&self) -> BackendSolveOptions {
        BackendSolveOptions {
            max_iters: self.solver.max_iters,
            verbosity: self.solver.verbosity,
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
        ensure!(config.solver.max_iters > 0, "max_iters must be positive");
        ensure!(
            config.init.iterations > 0,
            "init_iterations must be positive"
        );
        ensure!(
            config.optimize.calib_weight > 0.0,
            "calib_weight must be positive"
        );
        ensure!(
            config.optimize.laser_weight > 0.0,
            "laser_weight must be positive"
        );
        Ok(())
    }

    fn on_input_change() -> InvalidationPolicy {
        InvalidationPolicy::CLEAR_COMPUTED
    }

    fn export(output: &Self::Output, _config: &Self::Config) -> Result<Self::Export> {
        Ok(output.clone())
    }
}
