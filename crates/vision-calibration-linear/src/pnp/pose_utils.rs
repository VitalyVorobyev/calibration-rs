//! Pose recovery utilities for PnP solvers.
//!
//! Provides helper functions for computing camera poses from world-camera
//! point correspondences, used by minimal solvers like P3P and EPnP.

use crate::Error;
use nalgebra::{Isometry3, Rotation3, Translation3, UnitQuaternion};
use vision_calibration_core::{Iso3, Mat3, Pt3, Real, Vec3};

/// Recover camera pose from corresponding world and camera-frame points.
///
/// Uses the Kabsch algorithm (SVD-based rotation alignment followed by
/// translation computation). Returns `T_C_W`: the transform from world
/// coordinates to camera coordinates.
pub(super) fn pose_from_points(world: &[Pt3], camera: &[Vec3]) -> Result<Iso3, Error> {
    if world.len() != camera.len() || world.len() < 3 {
        return Err(Error::invalid_input(
            "degenerate 3d point configuration for normalization",
        ));
    }

    let n = world.len() as Real;
    let mut c_w = Vec3::zeros();
    let mut c_c = Vec3::zeros();
    for (pw, pc) in world.iter().zip(camera.iter()) {
        c_w += pw.coords;
        c_c += pc;
    }
    c_w /= n;
    c_c /= n;

    let mut h = Mat3::zeros();
    for (pw, pc) in world.iter().zip(camera.iter()) {
        let dw = pw.coords - c_w;
        let dc = pc - c_c;
        h += dc * dw.transpose();
    }

    // Polar decomposition to the nearest rotation (see `math::project_to_so3`).
    let r = crate::math::project_to_so3(&h)?;

    let t = c_c - r * c_w;
    let rot = UnitQuaternion::from_rotation_matrix(&Rotation3::from_matrix_unchecked(r));
    let trans = Translation3::from(t);
    Ok(Isometry3::from_parts(trans, rot))
}
