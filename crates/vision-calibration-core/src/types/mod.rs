//! Common types shared across the calibration workspace.
//!
//! This module provides canonical data structures for observations, results,
//! and configuration options used throughout the calibration pipeline.

mod observation;
mod options;
mod residual;

pub use observation::*;
pub use options::*;
pub use residual::*;
