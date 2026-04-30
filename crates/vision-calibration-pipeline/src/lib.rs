//! High-level camera calibration pipelines.
//!
//! This crate provides ready-to-use calibration workflows with multiple APIs:
//!
//! ## Session API
//!
//! The session API uses a mutable state container with step functions.
//!
//! ```no_run
//! use vision_calibration_pipeline::session::CalibrationSession;
//! use vision_calibration_pipeline::planar_intrinsics::{
//!     PlanarIntrinsicsProblem, step_init, step_optimize, run_calibration,
//! };
//! # fn main() -> anyhow::Result<()> {
//! # let dataset = unimplemented!();
//!
//! let mut session = CalibrationSession::<PlanarIntrinsicsProblem>::new();
//! session.set_input(dataset)?;
//!
//! // Option 1: Step-by-step control
//! step_init(&mut session, None)?;
//! step_optimize(&mut session, None)?;
//!
//! // Option 2: Pipeline function
//! // run_calibration(&mut session)?;
//!
//! let export = session.export()?;
//! # Ok(())
//! # }
//! ```

mod error;
pub use error::Error;

// Core session framework
mod planar_family;
mod rig_family;
pub mod session;

// Problem-specific modules
pub mod laserline_device;
pub mod planar_intrinsics;
pub mod rig_extrinsics;
pub mod rig_handeye;
pub mod rig_laserline_device;
pub mod rig_scheimpflug_extrinsics;
pub mod rig_scheimpflug_handeye;
pub mod scheimpflug_intrinsics;
pub mod single_cam_handeye;
