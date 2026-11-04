//! Common types shared across the calibration workspace.
//!
//! This module provides canonical data structures for observations and
//! results used throughout the calibration pipeline.

mod calibrated_rig;
mod handeye;
mod image_manifest;
mod laser_plane;
mod observation;
mod residual;
mod uncertainty;

pub use calibrated_rig::*;
pub use handeye::*;
pub use image_manifest::*;
pub use laser_plane::*;
pub use observation::*;
pub use residual::*;
pub use uncertainty::*;
