//! Backend adapters that compile the IR into solver-specific problems.
//!
//! Backends are responsible for translating the IR into solver-native graphs,
//! applying manifolds and constraints, and returning a solved parameter map.

/// Camera-model dispatch table: maps a [`CameraModelDesc`](crate::ir::CameraModelDesc)
/// to concrete kernel types and expands `$mk!(P, D, S)` for the matched row.
///
/// Adding a camera model = one descriptor enum variant + one kernel type +
/// one row here. Chains are factor data and do not multiply rows. Every
/// backend compiles its factors through this one table.
macro_rules! dispatch_camera_model {
    ($model:expr, $mk:ident) => {{
        use $crate::factors::camera_kernels as kernels;
        use $crate::ir::{DistortionKind as Dist, ProjectionKind as Proj, SensorKind as Sensor};
        match ($model.projection, $model.distortion, $model.sensor) {
            (Proj::Pinhole, Dist::None, Sensor::None) => {
                $mk!(
                    kernels::PinholeKernel,
                    kernels::NoDistortionKernel,
                    kernels::IdentitySensorKernel
                )
            }
            (Proj::Pinhole, Dist::BrownConrady5, Sensor::None) => {
                $mk!(
                    kernels::PinholeKernel,
                    kernels::BrownConrady5Kernel,
                    kernels::IdentitySensorKernel
                )
            }
            (Proj::Pinhole, Dist::None, Sensor::Scheimpflug2) => {
                $mk!(
                    kernels::PinholeKernel,
                    kernels::NoDistortionKernel,
                    kernels::Scheimpflug2Kernel
                )
            }
            (Proj::Pinhole, Dist::BrownConrady5, Sensor::Scheimpflug2) => {
                $mk!(
                    kernels::PinholeKernel,
                    kernels::BrownConrady5Kernel,
                    kernels::Scheimpflug2Kernel
                )
            }
            (Proj::Pinhole, Dist::Rational8, Sensor::None) => {
                $mk!(
                    kernels::PinholeKernel,
                    kernels::RationalKernel,
                    kernels::IdentitySensorKernel
                )
            }
            (Proj::Pinhole, Dist::Rational8, Sensor::Scheimpflug2) => {
                $mk!(
                    kernels::PinholeKernel,
                    kernels::RationalKernel,
                    kernels::Scheimpflug2Kernel
                )
            }
            (Proj::Pinhole, Dist::ThinPrism9, Sensor::None) => {
                $mk!(
                    kernels::PinholeKernel,
                    kernels::ThinPrismKernel,
                    kernels::IdentitySensorKernel
                )
            }
            (Proj::Pinhole, Dist::ThinPrism9, Sensor::Scheimpflug2) => {
                $mk!(
                    kernels::PinholeKernel,
                    kernels::ThinPrismKernel,
                    kernels::Scheimpflug2Kernel
                )
            }
            (Proj::Pinhole, Dist::Division1, Sensor::None) => {
                $mk!(
                    kernels::PinholeKernel,
                    kernels::DivisionKernel,
                    kernels::IdentitySensorKernel
                )
            }
            (Proj::Pinhole, Dist::Division1, Sensor::Scheimpflug2) => {
                $mk!(
                    kernels::PinholeKernel,
                    kernels::DivisionKernel,
                    kernels::Scheimpflug2Kernel
                )
            }
        }
    }};
}
pub(crate) use dispatch_camera_model;

mod factrs_backend;
mod lm;
#[cfg(test)]
mod parity_tests;
mod s2;
mod tiny_solver_backend;
mod tiny_solver_manifolds;

use crate::Error;
use nalgebra::DVector;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::ir::ProblemIR;

use factrs_backend::FactrsBackend;
use tiny_solver_backend::TinySolverBackend;

/// Which engine linearizes the problem.
///
/// Both run the same Levenberg–Marquardt loop over the same residual
/// kernels and reach the same minimizer; they differ in how they build the
/// Jacobian and how they fold in a robust loss.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case")]
pub enum SolverBackend {
    /// `tiny-solver`: dynamic-size forward-mode dual numbers; a robust loss
    /// enters through the Triggs correction (second-order exact).
    #[default]
    TinySolver,
    /// `factrs`: static-size forward-mode dual numbers; a robust loss enters
    /// through iterative reweighting (each block scaled by `√ρ′`).
    Factrs,
}

/// Backend-agnostic solver options.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackendSolveOptions {
    /// Linearization engine.
    #[serde(default)]
    pub backend: SolverBackend,
    /// Maximum number of iterations for the optimizer.
    pub max_iters: usize,
    /// Verbosity level (backend-specific).
    pub verbosity: usize,
    /// Optional linear solver selection.
    pub linear_solver: Option<LinearSolverKind>,
    /// Absolute error decrease threshold for early termination.
    pub min_abs_decrease: Option<f64>,
    /// Relative error decrease threshold for early termination.
    pub min_rel_decrease: Option<f64>,
    /// Error threshold for early termination.
    pub min_error: Option<f64>,
}

impl Default for BackendSolveOptions {
    fn default() -> Self {
        Self {
            backend: SolverBackend::default(),
            max_iters: 100,
            verbosity: 0,
            linear_solver: Some(LinearSolverKind::SparseCholesky),
            min_abs_decrease: Some(1e-5),
            min_rel_decrease: Some(1e-5),
            min_error: Some(1e-10),
        }
    }
}

/// Linear solver selection (backend-agnostic).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LinearSolverKind {
    /// Sparse Cholesky decomposition.
    SparseCholesky,
    /// Sparse QR decomposition.
    SparseQR,
}

/// Summary of backend solve outcome.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
pub struct SolveReport {
    /// The objective at the solution: `½ Σᵢ ρᵢ(‖rᵢ‖²)` over the residual
    /// blocks, where `ρᵢ` is the block's robust loss (`ρ(s) = s` without
    /// one) — so `½‖r‖²` for a plain least-squares problem.
    pub final_cost: f64,
    /// Number of outer solver iterations executed.
    #[serde(default)]
    pub num_iters: usize,
}

/// Solver output from a backend.
///
/// The `params` map uses the IR parameter block names.
#[derive(Debug, Clone)]
pub struct BackendSolution {
    /// Optimized parameter vectors keyed by block name.
    pub params: HashMap<String, DVector<f64>>,
    /// Final robustified cost if supported by the backend.
    pub solve_report: SolveReport,
}

/// Backend interface implemented by solver adapters.
pub trait OptimBackend {
    /// Solve a compiled IR with the provided initial parameters.
    fn solve(
        &self,
        ir: &ProblemIR,
        initial: &HashMap<String, DVector<f64>>,
        opts: &BackendSolveOptions,
    ) -> Result<BackendSolution, Error>;
}

/// Solve `ir` from `initial` with the backend `opts.backend` selects.
///
/// This is the backend-agnostic entry point every problem uses.
///
/// # Errors
///
/// Returns [`Error::InvalidInput`] for an IR the backend cannot compile and
/// [`Error::Numerical`] if the solve fails.
pub(crate) fn solve(
    ir: &ProblemIR,
    initial: &HashMap<String, DVector<f64>>,
    opts: &BackendSolveOptions,
) -> Result<BackendSolution, Error> {
    match opts.backend {
        SolverBackend::TinySolver => TinySolverBackend.solve(ir, initial, opts),
        SolverBackend::Factrs => FactrsBackend.solve(ir, initial, opts),
    }
}
