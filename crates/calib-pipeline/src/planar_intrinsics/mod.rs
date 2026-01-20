//! Planar intrinsics calibration (Zhang's method with distortion).
//!
//! This module provides both a session-based API and imperative functions
//! for planar intrinsics calibration.
//!
//! # Session API
//!
//! ```ignore
//! use calib_pipeline::session::CalibrationSession;
//! use calib_pipeline::planar_intrinsics::{PlanarIntrinsicsConfig, PlanarIntrinsicsProblem};
//! use calib_pipeline::PlanarDataset;
//!
//! let mut session = CalibrationSession::<PlanarIntrinsicsProblem>::new();
//! let obs_id = session.add_observations(dataset);
//! let config = PlanarIntrinsicsConfig::default();
//! let init_id = session.run_init(obs_id, config.clone())?;
//! let result_id = session.run_optimize(obs_id, init_id, config)?;
//! let report = session.run_export(result_id, Default::default())?;
//! ```
//!
//! # Imperative API
//!
//! ```ignore
//! use calib_pipeline::planar_intrinsics::{
//!     planar_init_seed_from_views, run_planar_intrinsics, PlanarIntrinsicsConfig,
//! };
//!
//! let seed = planar_init_seed_from_views(&dataset, config.init_opts.clone())?;
//! let report = run_planar_intrinsics(&dataset, &config)?;
//! ```

mod functions;
mod session;

// Session problem type
pub use session::PlanarIntrinsicsProblem;

// Imperative functions
pub use functions::{planar_init_seed_from_views, run_planar_intrinsics, PlanarIntrinsicsConfig};

// Re-export useful types from calib-optim
pub use calib_optim::{
    PlanarIntrinsicsEstimate, PlanarIntrinsicsParams, PlanarIntrinsicsSolveOptions,
};
