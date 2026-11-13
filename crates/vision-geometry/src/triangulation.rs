//! Triangulation of a 3D point from bearings in several views.
//!
//! [`triangulate_point_linear`] solves the homogeneous DLT system on unit
//! bearings (rays from each camera's own unprojection), so it works for every
//! camera model, including a fisheye that sees beyond 90° from its axis. The
//! estimate is algebraic; refining it against pixels belongs to a calibrated
//! measurement, where the camera model is known.

use crate::math::{dlt_rank_ok, ensure_finite};
use crate::{GeometryError, Result};
use nalgebra::{DMatrix, Matrix3x4};
use vision_calibration_core::linalg::null_space;
use vision_calibration_core::{Iso3, Pt3, Real, Vec3};

/// Linear (DLT) triangulation from N ≥ 2 views, on bearings.
///
/// `cam_se3_world[i]` is the pose of view `i` (world → camera `i`), and
/// `bearings[i]` the ray along which the point is seen in camera `i`'s frame
/// (any nonzero length; it is normalized). A pinhole's `(x, y, 1)` of
/// normalized coordinates is one. The returned point is in the world frame.
///
/// With `P = [R | t]` and unit bearing `b`, the point `X` satisfies
/// `b × (P X) = 0`; the three rows of `[b]× P` (rank 2) of every view stack into
/// a `3N×4` system. It is solved via [`null_space`] (the smallest-singular
/// direction, from the `AᵀA` symmetric eigendecomposition rather than a dense
/// SVD — see [`null_space`] for why the SVD path is deliberately avoided).
///
/// # Errors
///
/// Returns an error if fewer than 2 views are given, the pose/bearing counts
/// disagree, a bearing is zero or not finite, the system is rank-deficient
/// (identical cameras or coincident rays / zero parallax), or the recovered
/// point is at infinity.
pub fn triangulate_point_linear(cam_se3_world: &[Iso3], bearings: &[Vec3]) -> Result<Pt3> {
    ensure_finite(
        "camera pose",
        cam_se3_world
            .iter()
            .flat_map(|p| p.translation.vector.iter().chain(p.rotation.coords.iter())),
    )?;
    ensure_finite("bearing", bearings.iter().flat_map(|v| v.iter()))?;
    if cam_se3_world.len() < 2 {
        return Err(GeometryError::InsufficientData {
            need: 2,
            got: cam_se3_world.len(),
        });
    }
    if cam_se3_world.len() != bearings.len() {
        return Err(GeometryError::CountMismatch {
            expected: cam_se3_world.len(),
            got: bearings.len(),
        });
    }

    let mut a = DMatrix::<Real>::zeros(3 * cam_se3_world.len(), 4);
    for (i, (pose, b)) in cam_se3_world.iter().zip(bearings.iter()).enumerate() {
        let norm = b.norm();
        if !(norm.is_finite() && norm > 0.0) {
            return Err(GeometryError::degenerate(
                "triangulation bearing is zero or not finite",
            ));
        }
        let b = b / norm;
        let skew = b.cross_matrix();
        let r = pose.rotation.to_rotation_matrix();
        let mut p = Matrix3x4::<Real>::zeros();
        p.fixed_view_mut::<3, 3>(0, 0).copy_from(r.matrix());
        p.set_column(3, &pose.translation.vector);
        let rows = skew * p;
        a.view_mut((3 * i, 0), (3, 4)).copy_from(&rows);
    }

    let ns = null_space(&a)?;

    // The 4-column DLT triangulation matrix is well-posed at rank 3 (1-D null
    // space). Identical cameras or coincident rays collapse the boundary
    // singular value toward zero.
    if !dlt_rank_ok(&ns.singular_values, 4, 1, 1e-7) {
        return Err(GeometryError::degenerate(
            "zero-parallax / rank-deficient triangulation system \
             (identical cameras or coincident rays)",
        ));
    }

    let x_h = &ns.vector;
    let w = x_h[3];
    if w.abs() <= Real::EPSILON {
        return Err(GeometryError::degenerate(
            "triangulation produced an invalid point",
        ));
    }

    Ok(Pt3::new(x_h[0] / w, x_h[1] / w, x_h[2] / w))
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{Rotation3, Translation3, UnitQuaternion};

    fn pose(rx: Real, ry: Real, rz: Real, t: Vec3) -> Iso3 {
        Iso3::from_parts(
            Translation3::from(t),
            UnitQuaternion::from_rotation_matrix(&Rotation3::from_euler_angles(rx, ry, rz)),
        )
    }

    fn bearing(cam: &Iso3, p: &Pt3) -> Vec3 {
        (cam * p).coords
    }

    #[test]
    fn two_views_recover_point() {
        let cam1 = Iso3::identity();
        let cam2 = pose(0.0, 0.0, 0.0, Vec3::new(-0.2, 0.0, 0.0));
        let pw = Pt3::new(0.1, -0.05, 2.0);
        let est =
            triangulate_point_linear(&[cam1, cam2], &[bearing(&cam1, &pw), bearing(&cam2, &pw)])
                .unwrap();
        assert!((est - pw).norm() < 1e-9, "error {}", (est - pw).norm());
    }

    /// The bearing's length does not matter.
    #[test]
    fn bearing_scale_is_irrelevant() {
        let cam1 = Iso3::identity();
        let cam2 = pose(0.0, 0.1, 0.0, Vec3::new(-0.2, 0.0, 0.0));
        let pw = Pt3::new(0.1, -0.05, 2.0);
        let est = triangulate_point_linear(
            &[cam1, cam2],
            &[bearing(&cam1, &pw) * 7.0, bearing(&cam2, &pw) * 0.01],
        )
        .unwrap();
        assert!((est - pw).norm() < 1e-9);
    }

    /// Regression: two identical cameras (zero baseline) must return Err.
    #[test]
    fn rejects_identical_cameras() {
        let cam = Iso3::identity();
        let pw = Pt3::new(0.05, -0.02, 1.5);
        let b = bearing(&cam, &pw);
        assert!(
            triangulate_point_linear(&[cam, cam], &[b, b]).is_err(),
            "identical cameras (zero baseline) must return Err"
        );
    }

    #[test]
    fn rejects_too_few_views_count_mismatch_and_zero_bearing() {
        let cam = Iso3::identity();
        let b = Vec3::new(0.0, 0.0, 1.0);
        assert!(matches!(
            triangulate_point_linear(&[cam], &[b]),
            Err(GeometryError::InsufficientData { .. })
        ));
        assert!(matches!(
            triangulate_point_linear(&[cam, cam], &[b]),
            Err(GeometryError::CountMismatch { .. })
        ));
        assert!(triangulate_point_linear(&[cam, cam], &[b, Vec3::zeros()]).is_err());
    }

    /// Four spread-out views recover a noiseless point essentially exactly.
    #[test]
    fn nview_recovers_point() {
        let cams = [
            pose(0.0, 0.0, 0.0, Vec3::new(0.0, 0.0, 0.0)),
            pose(0.02, -0.03, 0.01, Vec3::new(-0.4, 0.05, 0.02)),
            pose(-0.05, 0.04, 0.0, Vec3::new(0.3, -0.2, 0.1)),
            pose(0.01, 0.06, -0.02, Vec3::new(0.1, 0.35, -0.05)),
        ];
        let pw = Pt3::new(0.15, -0.1, 3.0);
        let bs: Vec<Vec3> = cams.iter().map(|c| bearing(c, &pw)).collect();
        let est = triangulate_point_linear(&cams, &bs).unwrap();
        assert!((est - pw).norm() < 1e-9, "error {}", (est - pw).norm());
    }

    /// A point behind one camera and beyond 90° from its axis still
    /// triangulates: the bearing has `z < 0`.
    #[test]
    fn point_behind_a_camera_triangulates() {
        let cams = [
            Iso3::identity(),
            pose(0.0, 0.3, 0.0, Vec3::new(-1.0, 0.1, 0.2)),
        ];
        let pw = Pt3::new(1.5, 0.2, -0.5);
        let bs: Vec<Vec3> = cams.iter().map(|c| bearing(c, &pw)).collect();
        assert!(bs[0].z < 0.0);
        let est = triangulate_point_linear(&cams, &bs).unwrap();
        assert!((est - pw).norm() < 1e-9, "error {}", (est - pw).norm());
    }
}
