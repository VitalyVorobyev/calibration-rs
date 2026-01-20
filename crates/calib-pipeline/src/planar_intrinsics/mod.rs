//! Planar intrinsics calibration (Zhang's method with distortion).
//!
//! This module provides both a session-based API and imperative functions
//! for planar intrinsics calibration.
//!
//! # Session API
//!
//! ```ignore
//! use calib_pipeline::session::CalibrationSession;
//! use calib_pipeline::planar_intrinsics::{
//!     PlanarIntrinsicsProblem, PlanarIntrinsicsObservations,
//! };
//!
//! let mut session = CalibrationSession::<PlanarIntrinsicsProblem>::new();
//! let obs_id = session.add_observations(PlanarIntrinsicsObservations { views });
//! let init_id = session.run_init(obs_id, Default::default())?;
//! let result_id = session.run_optimize(obs_id, init_id, Default::default())?;
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
//! let seed = planar_init_seed_from_views(&views)?;
//! let report = run_planar_intrinsics(&dataset, &config)?;
//! ```

mod functions;
mod session;

// Session problem type
pub use session::PlanarIntrinsicsProblem;

// Imperative functions
pub use functions::{planar_init_seed_from_views, run_planar_intrinsics, PlanarIntrinsicsConfig};

// Re-export useful types from calib-optim
pub use calib_optim::{PlanarIntrinsicsParams, PlanarIntrinsicsSolveOptions};
