//! Dense diagnostic maps over an image.
//!
//! A map is a set of planes over an [`ImageGrid`]: row-major `Vec<f64>`
//! values, one per node, with `NaN` where the map has no value. Maps are
//! derived on demand from a calibration result; they are not part of any
//! export.
//!
//! Every displacement is `actual − reference`, in pixels: a
//! [`distortion_map`] layer is where the camera puts a ray relative to where
//! the simpler model without that stage puts it, and a reprojection residual
//! is `observed − projected`
//! ([`TargetFeatureResidual::residual_px`](crate::TargetFeatureResidual::residual_px)).
//! A distortion the model lacks then shows in the residuals with the sign
//! and units it has in a distortion map.

mod distortion_map;
mod grid;

pub use distortion_map::{DisplacementPlane, DistortionMap, distortion_map};
pub use grid::ImageGrid;

#[cfg(test)]
mod tests;
