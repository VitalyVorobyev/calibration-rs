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
//! ```no_run
//! use vision_calibration_pipeline::session::{CalibrationSession, ProblemType};
//! use vision_calibration_pipeline::planar_intrinsics::{PlanarIntrinsicsProblem, step_init, step_optimize};
//! # fn main() -> anyhow::Result<()> {
//! # let dataset = unimplemented!();
//!
//! let mut session = CalibrationSession::<PlanarIntrinsicsProblem>::new();
//! session.set_input(dataset)?;
//!
//! step_init(&mut session, None)?;
//! step_optimize(&mut session, None)?;
//!
//! let export = session.export()?;
//! # Ok(())
//! # }
//! ```

// Session API
pub mod calibsession;
pub mod problem_type;
pub mod types;

pub use calibsession::CalibrationSession;
/// Session API with mutable state container.
pub use problem_type::{InvalidationPolicy, ProblemType};
pub use types::{ExportRecord, LogEntry, SessionMetadata, current_timestamp};
