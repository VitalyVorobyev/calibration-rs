//! Calibration session framework.
//!
//! This module provides session API:
//!
//! ## API: Mutable State Container
//!
//! The session API uses a mutable state container with step functions.
//! Sessions store configuration, input data, intermediate state, and a single
//! final output. Step functions mutate the session in-place.
//!
//! ```ignore
//! use calib_pipeline::session::{CalibrationSession, ProblemType};
//! use calib_pipeline::planar_intrinsics::{PlanarIntrinsicsProblem, step_init, step_optimize};
//!
//! let mut session = CalibrationSession::<PlanarIntrinsicsProblem>::new();
//! session.set_input(dataset);
//!
//! step_init(&mut session, None);
//! step_optimize(&mut session, None);
//!
//! let export = session.export();
//! ```

// Session API
pub mod problem_type;
pub mod session;
pub mod types;

/// Session API with mutable state container.
pub use problem_type::{InvalidationPolicy, ProblemType};
pub use session::CalibrationSession;
pub use types::{current_timestamp, ExportRecord, LogEntry, SessionMetadata};
