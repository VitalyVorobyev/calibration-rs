//! Detector output: 2D image points paired with their 3D world points
//! in the target frame.

use serde::{Deserialize, Serialize};

#[cfg(feature = "schemars")]
use schemars::JsonSchema;

/// Single 2D-3D feature correspondence in canonical units.
///
/// `image_xy` is in source-image pixels (pre-ROI-crop if a ROI is
/// declared in the manifest, the converter adjusts as needed).
/// `world_xyz` is in metres, expressed in the calibration target's
/// own frame (target origin at `(0, 0, 0)`, target plane at `z = 0`).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(JsonSchema))]
pub struct Feature {
    /// Pixel coordinates `[x, y]` (column, row).
    pub image_xy: [f64; 2],
    /// World coordinates `[x, y, z]` in metres in the target frame.
    pub world_xyz: [f64; 3],
}
