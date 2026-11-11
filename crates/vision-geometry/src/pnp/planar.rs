//! Pose of a planar point set from its plane homography.
//!
//! Used by EPnP for points that lie on a plane, where its control-point
//! formulation degenerates.

use crate::GeometryError;
use crate::homography::dlt_homography;
use nalgebra::{Rotation3, UnitQuaternion};
use vision_calibration_core::linalg::project_to_so3;
use vision_calibration_core::{Iso3, Mat3, Pt2, Pt3, Real, Vec3};

/// `T_C_W` for the points `world`, which lie on the plane through `centre`
/// spanned by the first two columns of `axes` (the third is the normal; the
/// columns are a right-handed orthonormal basis), seen at the normalized
/// coordinates `image`.
///
/// With in-plane coordinates `(a, b)` the homography is `H ∝ [r₁ r₂ t]`; the
/// rotation columns are scaled to unit length, completed by their cross
/// product and projected onto SO(3), and the sign chosen so the plane lies
/// ahead of the camera.
pub(super) fn planar_pose(
    world: &[Pt3],
    image: &[Pt2],
    centre: &Vec3,
    axes: &Mat3,
) -> Result<Iso3, GeometryError> {
    let plane: Vec<Pt2> = world
        .iter()
        .map(|p| {
            let d = axes.transpose() * (p.coords - centre);
            Pt2::new(d.x, d.y)
        })
        .collect();
    let h = dlt_homography(&plane, image)?;
    let (h1, h2, h3) = (
        h.column(0).into_owned(),
        h.column(1).into_owned(),
        h.column(2).into_owned(),
    );
    let scale: Real = 2.0 / (h1.norm() + h2.norm());
    let scale = if h3.z < 0.0 { -scale } else { scale };
    let (r1, r2, t) = (h1 * scale, h2 * scale, h3 * scale);
    let r = project_to_so3(&Mat3::from_columns(&[r1, r2, r1.cross(&r2)]))?;
    let cam_se3_plane = Iso3::from_parts(
        t.into(),
        UnitQuaternion::from_rotation_matrix(&Rotation3::from_matrix_unchecked(r)),
    );
    // World to plane coordinates: Eᵀ (p − centre).
    let plane_se3_world = Iso3::from_parts(
        (-(axes.transpose() * centre)).into(),
        UnitQuaternion::from_rotation_matrix(&Rotation3::from_matrix_unchecked(axes.transpose())),
    );
    Ok(cam_se3_plane * plane_se3_world)
}
