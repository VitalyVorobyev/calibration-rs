//! Enhanced triangulation with quality diagnostics.
//!
//! Wraps the low-level DLT triangulation from `vision-geometry` and adds
//! reprojection error computation, parallax angle measurement, and cheirality
//! validation.

use crate::types::TriangulatedPoint;
use crate::{MvgError, Result};
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
        return Err(MvgError::CountMismatch {
            expected: pts1.len(),
            got: pts2.len(),
        });
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

/// Like [`triangulate_two_view`] but tolerant of per-point degeneracies.
///
/// Correspondences that fail to triangulate (e.g. a single on-baseline /
/// zero-parallax outlier) are skipped instead of discarding the whole set.
/// Returns the successfully triangulated points, which may be empty if none
/// triangulate. The two slices must have equal length (callers pass
/// [`Correspondence2D::split`](crate::types::Correspondence2D::split) output).
pub fn triangulate_two_view_partial(
    r: &Mat3,
    t: &Vec3,
    pts1: &[Pt2],
    pts2: &[Pt2],
) -> Vec<TriangulatedPoint> {
    debug_assert_eq!(pts1.len(), pts2.len(), "point count mismatch");

    let p1 = Mat34::new(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0);
    let mut p2 = Mat34::zeros();
    p2.fixed_view_mut::<3, 3>(0, 0).copy_from(r);
    p2.set_column(3, t);

    pts1.iter()
        .zip(pts2.iter())
        .filter_map(|(q1, q2)| triangulate_point_two_view(&p1, &p2, q1, q2).ok())
        .collect()
}

/// Triangulate a single point from N ≥ 2 views with refinement and diagnostics.
///
/// `cameras` are 3×4 projection matrices and `points` their corresponding image
/// coordinates. Uses [`vision_geometry::triangulation::triangulate_point`]
/// (linear DLT + Gauss-Newton reprojection refinement), then reports:
///
/// - `reprojection_error`: RMS reprojection error over all views,
/// - `parallax_deg`: the **widest** parallax angle across all camera-center
///   pairs (the best-conditioned baseline; larger ⇒ better depth constraint),
/// - `in_front`: whether the point has positive depth in **every** view.
///
/// # Errors
///
/// Returns an error if the camera/point counts disagree, fewer than 2 views are
/// given, or the system is degenerate (see the linear solver).
pub fn triangulate_nview(cameras: &[Mat34], points: &[Pt2]) -> Result<TriangulatedPoint> {
    if cameras.len() != points.len() {
        return Err(MvgError::CountMismatch {
            expected: cameras.len(),
            got: points.len(),
        });
    }

    let pt3 = vision_geometry::triangulation::triangulate_point(cameras, points)?;
    let xh = nalgebra::Vector4::new(pt3.x, pt3.y, pt3.z, 1.0);

    // RMS reprojection error + all-views cheirality.
    let mut sq_sum = 0.0;
    let mut in_front = true;
    for (cam, obs) in cameras.iter().zip(points.iter()) {
        let x = cam * xh;
        in_front &= x.z > 0.0;
        let proj = Pt2::new(x.x / x.z, x.y / x.z);
        sq_sum += (proj.x - obs.x).powi(2) + (proj.y - obs.y).powi(2);
    }
    let reprojection_error = (sq_sum / cameras.len() as f64).sqrt();

    // Widest pairwise parallax: the best baseline angle subtended at the point.
    let centers: Vec<vision_calibration_core::Pt3> = cameras.iter().map(camera_center).collect();
    let mut parallax_deg = 0.0_f64;
    for i in 0..centers.len() {
        for j in (i + 1)..centers.len() {
            let ray_i = (pt3 - centers[i]).normalize();
            let ray_j = (pt3 - centers[j]).normalize();
            let cos_angle = ray_i.dot(&ray_j).clamp(-1.0, 1.0);
            let angle = cos_angle.acos().to_degrees();
            if angle > parallax_deg {
                parallax_deg = angle;
            }
        }
    }

    Ok(TriangulatedPoint {
        point: pt3,
        reprojection_error,
        parallax_deg,
        in_front,
    })
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

    #[test]
    fn triangulate_two_view_partial_skips_degenerate_points() {
        // r = identity, t = 0 → p1 == p2 → every point's DLT is rank-deficient.
        // The partial variant must return an empty vec WITHOUT erroring
        // (contrast: triangulate_two_view is all-or-nothing and would fail).
        let r = Mat3::identity();
        let t = Vec3::zeros();
        let pts1 = vec![Pt2::new(0.1, 0.2), Pt2::new(-0.3, 0.4)];
        let pts2 = pts1.clone();
        let got = triangulate_two_view_partial(&r, &t, &pts1, &pts2);
        assert!(
            got.is_empty(),
            "degenerate (zero-baseline) points must be skipped, got {}",
            got.len()
        );
    }

    fn camera_rt(rx: f64, ry: f64, rz: f64, t: Vec3) -> Mat34 {
        let r = *Rotation3::from_euler_angles(rx, ry, rz).matrix();
        let mut p = Mat34::zeros();
        p.fixed_view_mut::<3, 3>(0, 0).copy_from(&r);
        p.set_column(3, &t);
        p
    }

    #[test]
    fn triangulate_nview_noiseless_diagnostics() {
        let cams = [
            camera_rt(0.0, 0.0, 0.0, Vec3::new(0.0, 0.0, 0.0)),
            camera_rt(0.02, -0.03, 0.01, Vec3::new(-0.4, 0.05, 0.02)),
            camera_rt(-0.05, 0.04, 0.0, Vec3::new(0.3, -0.2, 0.1)),
            camera_rt(0.01, 0.06, -0.02, Vec3::new(0.1, 0.35, -0.05)),
        ];
        let pw = Pt3::new(0.15, -0.1, 3.0);
        let pts: Vec<Pt2> = cams
            .iter()
            .map(|c| {
                let x = c * nalgebra::Vector4::new(pw.x, pw.y, pw.z, 1.0);
                Pt2::new(x.x / x.z, x.y / x.z)
            })
            .collect();

        let tp = triangulate_nview(&cams, &pts).unwrap();
        assert!((tp.point - pw).norm() < 1e-9, "n-view 3D error too large");
        assert!(tp.reprojection_error < 1e-8);
        assert!(tp.in_front);
        assert!(tp.parallax_deg > 0.0);
    }

    #[test]
    fn triangulate_nview_rejects_count_mismatch() {
        let cam = Mat34::new(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0);
        assert!(triangulate_nview(&[cam, cam], &[Pt2::new(0.0, 0.0)]).is_err());
    }

    #[test]
    fn triangulate_two_view_partial_keeps_good_points() {
        let rot = Rotation3::from_euler_angles(0.05, -0.03, 0.02);
        let r = *rot.matrix();
        let t = Vec3::new(0.3, 0.01, 0.005);
        let world = [
            Pt3::new(0.5, 0.3, 3.0),
            Pt3::new(-0.4, 0.2, 4.0),
            Pt3::new(0.6, -0.3, 5.0),
        ];
        let pts1: Vec<Pt2> = world
            .iter()
            .map(|pw| Pt2::new(pw.x / pw.z, pw.y / pw.z))
            .collect();
        let pts2: Vec<Pt2> = world
            .iter()
            .map(|pw| {
                let pc = r * pw.coords + t;
                Pt2::new(pc.x / pc.z, pc.y / pc.z)
            })
            .collect();
        let got = triangulate_two_view_partial(&r, &t, &pts1, &pts2);
        assert_eq!(got.len(), 3, "all valid points must be retained");
    }
}
