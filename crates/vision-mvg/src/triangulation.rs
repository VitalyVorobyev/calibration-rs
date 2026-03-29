//! Enhanced triangulation with quality diagnostics.
//!
//! Wraps the low-level DLT triangulation from `vision-geometry` and adds
//! reprojection error computation, parallax angle measurement, and cheirality
//! validation.

use crate::types::TriangulatedPoint;
use anyhow::Result;
use vision_calibration_core::{Mat3, Pt2, Vec3};
use vision_geometry::camera_matrix::Mat34;

/// Triangulate a single point from two views with diagnostics.
///
/// `p1` and `p2` are 3×4 projection matrices. Returns a [`TriangulatedPoint`]
/// with reprojection error, parallax angle, and cheirality flag.
pub fn triangulate_point_two_view(
    p1: &Mat34,
    p2: &Mat34,
    pt1: &Pt2,
    pt2: &Pt2,
) -> Result<TriangulatedPoint> {
    let pt3 = vision_geometry::triangulation::triangulate_point_linear(&[*p1, *p2], &[*pt1, *pt2])?;

    // Reprojection error.
    let x1 = p1 * nalgebra::Vector4::new(pt3.x, pt3.y, pt3.z, 1.0);
    let x2 = p2 * nalgebra::Vector4::new(pt3.x, pt3.y, pt3.z, 1.0);

    let proj1 = Pt2::new(x1.x / x1.z, x1.y / x1.z);
    let proj2 = Pt2::new(x2.x / x2.z, x2.y / x2.z);

    let e1 = ((proj1.x - pt1.x).powi(2) + (proj1.y - pt1.y).powi(2)).sqrt();
    let e2 = ((proj2.x - pt2.x).powi(2) + (proj2.y - pt2.y).powi(2)).sqrt();
    let reprojection_error = ((e1 * e1 + e2 * e2) / 2.0).sqrt();

    // Cheirality: positive depth in both cameras.
    let z1 = x1.z;
    let z2 = x2.z;
    let in_front = z1 > 0.0 && z2 > 0.0;

    // Parallax angle: angle between rays from each camera center to the point.
    let c1 = camera_center(p1);
    let c2 = camera_center(p2);
    let ray1 = (pt3 - c1).normalize();
    let ray2 = (pt3 - c2).normalize();
    let cos_angle = ray1.dot(&ray2).clamp(-1.0, 1.0);
    let parallax_deg = cos_angle.acos().to_degrees();

    Ok(TriangulatedPoint {
        point: pt3,
        reprojection_error,
        parallax_deg,
        in_front,
    })
}

/// Triangulate multiple correspondences from two calibrated views.
///
/// Camera 1 is at the origin: `P₁ = [I | 0]`. Camera 2 has relative pose
/// `(R, t)`: `P₂ = [R | t]`. Points must be in normalized camera coordinates.
pub fn triangulate_two_view(
    r: &Mat3,
    t: &Vec3,
    pts1: &[Pt2],
    pts2: &[Pt2],
) -> Result<Vec<TriangulatedPoint>> {
    if pts1.len() != pts2.len() {
        anyhow::bail!("point count mismatch: {} vs {}", pts1.len(), pts2.len());
    }

    let p1 = Mat34::new(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0);
    let mut p2 = Mat34::zeros();
    p2.fixed_view_mut::<3, 3>(0, 0).copy_from(r);
    p2.set_column(3, t);

    let mut results = Vec::with_capacity(pts1.len());
    for (q1, q2) in pts1.iter().zip(pts2.iter()) {
        results.push(triangulate_point_two_view(&p1, &p2, q1, q2)?);
    }

    Ok(results)
}

/// Extract camera center from a projection matrix.
///
/// For `P = [M | p₄]`, the camera center is `C = -M⁻¹ p₄`.
fn camera_center(p: &Mat34) -> vision_calibration_core::Pt3 {
    let m = p.fixed_view::<3, 3>(0, 0).into_owned();
    let p4 = p.column(3).into_owned();
    match m.try_inverse() {
        Some(m_inv) => {
            let c = -m_inv * p4;
            vision_calibration_core::Pt3::new(c.x, c.y, c.z)
        }
        None => vision_calibration_core::Pt3::origin(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Rotation3;
    use vision_calibration_core::Pt3;

    #[test]
    fn triangulate_two_view_noiseless() {
        let rot = Rotation3::from_euler_angles(0.05, -0.03, 0.02);
        let r = *rot.matrix();
        let t = Vec3::new(0.3, 0.01, 0.005);

        let world = [
            Pt3::new(0.1, 0.2, 3.0),
            Pt3::new(-0.2, 0.1, 2.5),
            Pt3::new(0.3, -0.1, 4.0),
        ];

        let mut pts1 = Vec::new();
        let mut pts2 = Vec::new();
        for pw in &world {
            let pc1 = pw.coords;
            let pc2 = r * pw.coords + t;
            pts1.push(Pt2::new(pc1.x / pc1.z, pc1.y / pc1.z));
            pts2.push(Pt2::new(pc2.x / pc2.z, pc2.y / pc2.z));
        }

        let results = triangulate_two_view(&r, &t, &pts1, &pts2).unwrap();

        assert_eq!(results.len(), world.len());
        for (tp, pw) in results.iter().zip(world.iter()) {
            let err = (tp.point - pw).norm();
            assert!(err < 1e-6, "triangulation error too large: {}", err);
            assert!(tp.reprojection_error < 1e-8);
            assert!(tp.in_front);
            assert!(tp.parallax_deg > 0.0);
        }
    }

    #[test]
    fn triangulate_small_baseline_has_small_parallax() {
        let r = Mat3::identity();
        let t = Vec3::new(0.001, 0.0, 0.0); // very small baseline

        let pw = Pt3::new(0.0, 0.0, 10.0); // far away point
        let pc2 = r * pw.coords + t;
        let pt1 = Pt2::new(pw.x / pw.z, pw.y / pw.z);
        let pt2 = Pt2::new(pc2.x / pc2.z, pc2.y / pc2.z);

        let results = triangulate_two_view(&r, &t, &[pt1], &[pt2]).unwrap();

        assert!(
            results[0].parallax_deg < 0.01,
            "expected small parallax for narrow baseline, got {}",
            results[0].parallax_deg
        );
    }
}
