//! Bearings: the unit rays a rig camera's own unprojection gives for pixels.
//!
//! Closed-form multiple-view steps work on bearings, so they serve every camera
//! model, including a fisheye that sees beyond 90° from its axis, where
//! normalized coordinates do not exist. Every projection model is central (the
//! ray starts at the camera centre), so the ray's origin is ignored.

use crate::{MvgError, Result};
use vision_calibration_core::{CalibratedRig, CameraModel, Pt2, Vec3};

/// Camera `camera` of the rig, or `InvalidInput` when out of range.
pub(crate) fn camera_model(rig: &CalibratedRig, camera: usize) -> Result<&CameraModel> {
    rig.model(camera).ok_or_else(|| {
        MvgError::invalid_input(format!(
            "camera {camera} out of range: the rig has {} cameras",
            rig.num_cameras()
        ))
    })
}

/// The unit bearing of `pixel` through `model`. `index` names the pixel in
/// the error when its ray is zero or not finite.
pub(crate) fn bearing(model: &CameraModel, pixel: &Pt2, index: usize) -> Result<Vec3> {
    let d = model.backproject_pixel(pixel).direction;
    let n = d.norm();
    if n.is_finite() && n > 0.0 {
        Ok(d / n)
    } else {
        Err(MvgError::invalid_input(format!(
            "pixel {index} ({}, {}) has no finite viewing ray",
            pixel.x, pixel.y
        )))
    }
}

/// The unit bearings of `pixels` through rig camera `camera`.
pub(crate) fn bearings(
    rig: &CalibratedRig,
    camera: usize,
    pixels: impl IntoIterator<Item = Pt2>,
) -> Result<Vec<Vec3>> {
    let model = camera_model(rig, camera)?;
    pixels
        .into_iter()
        .enumerate()
        .map(|(i, px)| bearing(model, &px, i))
        .collect()
}
