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
//! [`CalibrationSession`] has a runnable example.

// Session API
mod calibsession;
mod problem_type;
mod types;

pub use calibsession::CalibrationSession;
pub(crate) use problem_type::ProblemState;
/// Session API with mutable state container.
pub use problem_type::{InvalidationPolicy, ProblemType};
pub use types::{ExportRecord, LogEntry, SessionMetadata};
