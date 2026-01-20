//! Planar intrinsics calibration (Zhang's method with distortion).
//!
//! This module provides multiple APIs for planar intrinsics calibration:
//!
//! ## New Session API (v2) - Recommended
//!
//! The new session API uses a mutable state container with step functions.
//!
//! ```ignore
//! use calib_pipeline::session::v2::CalibrationSession;
//! use calib_pipeline::planar_intrinsics::{
//!     PlanarIntrinsicsProblemV2, PlanarConfig,
//!     step_init, step_optimize, run_calibration,
//! };
//!
//! let mut session = CalibrationSession::<PlanarIntrinsicsProblemV2>::new();
//! session.set_input(dataset)?;
//!
//! // Option 1: Step-by-step
//! step_init(&mut session, None)?;
//! step_optimize(&mut session, None)?;
//!
//! // Option 2: Pipeline function
//! // run_calibration(&mut session)?;
//!
//! let export = session.export()?;
//! ```
//!
//! ## Legacy Session API (v1) - Deprecated
//!
//! The original artifact-based session API is deprecated but still available.
//!
//! ```ignore
//! use calib_pipeline::session::CalibrationSession;
//! use calib_pipeline::planar_intrinsics::{PlanarIntrinsicsConfig, PlanarIntrinsicsProblem};
//!
//! let mut session = CalibrationSession::<PlanarIntrinsicsProblem>::new();
//! let obs_id = session.add_observations(dataset);
//! let config = PlanarIntrinsicsConfig::default();
//! let init_id = session.run_init(obs_id, config.clone())?;
//! let result_id = session.run_optimize(obs_id, init_id, config)?;
//! ```
//!
//! ## Imperative API
//!
//! Direct function calls without session management.
//!
//! ```ignore
//! use calib_pipeline::planar_intrinsics::{
//!     planar_init_seed_from_views, run_planar_intrinsics, PlanarIntrinsicsConfig,
//! };
//!
//! let seed = planar_init_seed_from_views(&dataset, config.init_opts.clone())?;
//! let report = run_planar_intrinsics(&dataset, &config)?;
//! ```

// Legacy session API (v1)
mod functions;
mod session;

// New session API (v2)
mod problem_v2;
mod state;
mod steps;

// ─────────────────────────────────────────────────────────────────────────────
// New API (v2) - Recommended
// ─────────────────────────────────────────────────────────────────────────────

/// New session API types and step functions.
pub mod v2 {
    pub use super::problem_v2::{PlanarConfig, PlanarExport, PlanarIntrinsicsProblemV2};
    pub use super::state::PlanarState;
    pub use super::steps::{
        run_calibration, run_calibration_with_filtering, step_filter, step_init, step_optimize,
        FilterOptions, InitOptions, OptimizeOptions,
    };
}

// Re-export v2 types at module level for convenience
pub use problem_v2::{PlanarConfig, PlanarExport, PlanarIntrinsicsProblemV2};
pub use state::PlanarState;
pub use steps::{
    run_calibration, run_calibration_with_filtering, step_filter, step_init, step_optimize,
    FilterOptions, InitOptions, OptimizeOptions,
};

// ─────────────────────────────────────────────────────────────────────────────
// Legacy API (v1) - Deprecated
// ─────────────────────────────────────────────────────────────────────────────

// Session problem type (legacy)
pub use session::PlanarIntrinsicsProblem;

// Imperative functions (still useful)
pub use functions::{planar_init_seed_from_views, run_planar_intrinsics, PlanarIntrinsicsConfig};

// ─────────────────────────────────────────────────────────────────────────────
// Re-exports from calib-optim
// ─────────────────────────────────────────────────────────────────────────────

pub use calib_optim::{
    PlanarIntrinsicsEstimate, PlanarIntrinsicsParams, PlanarIntrinsicsSolveOptions,
};
