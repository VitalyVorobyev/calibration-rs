//! Common types shared across the calibration workspace.
//!
//! This module provides canonical data structures for observations and
//! results used throughout the calibration pipeline.

mod calibrated_rig;
mod image_manifest;
mod observation;
mod residual;
mod uncertainty;

pub use calibrated_rig::*;
pub use image_manifest::*;
pub use observation::*;
pub use residual::*;
pub use uncertainty::*;
