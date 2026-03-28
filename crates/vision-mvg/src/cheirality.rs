//! Cheirality (positive depth) tests for pose disambiguation.
//!
//! When decomposing an essential matrix, four candidate (R, t) pairs are
//! returned. Only one places the triangulated points in front of both
//! cameras. These functions count and select the correct pose.

use anyhow::Result;
use vision_calibration_core::{Mat3, Pt2, Vec3};
use vision_geometry::camera_matrix::Mat34;
use vision_geometry::triangulation::triangulate_point_linear;

/// Count the number of points that are in front of both cameras.
///
/// Camera 1 is at the origin with `P₁ = [I | 0]`. Camera 2 has rotation
/// `r` and translation `t`, giving `P₂ = [R | t]`. Points are in
/// **normalized camera coordinates**.
pub fn cheirality_count(r: &Mat3, t: &Vec3, pts1: &[Pt2], pts2: &[Pt2]) -> usize {
    let p1 = Mat34::new(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0);
    let mut p2 = Mat34::zeros();
    p2.fixed_view_mut::<3, 3>(0, 0).copy_from(r);
    p2.set_column(3, t);

    let mut count = 0;
    for (q1, q2) in pts1.iter().zip(pts2.iter()) {
        let Ok(pt3) = triangulate_point_linear(&[p1, p2], &[*q1, *q2]) else {
            continue;
        };

        // Depth in camera 1: z coordinate directly (camera is at origin).
        let z1 = pt3.z;

        // Depth in camera 2: z component of R * X + t.
        let z2 = (r.row(2) * pt3.coords)[0] + t.z;

        if z1 > 0.0 && z2 > 0.0 {
            count += 1;
        }
    }

    count
}

/// Select the correct (R, t) from candidate poses by cheirality.
///
/// Returns the pose that places the most points in front of both cameras,
/// along with the number of points passing the cheirality check. Returns
/// `None` if no candidate has any valid points.
pub fn select_pose(
    candidates: &[(Mat3, Vec3)],
    pts1: &[Pt2],
    pts2: &[Pt2],
) -> Option<(Mat3, Vec3, usize)> {
    let mut best: Option<(Mat3, Vec3, usize)> = None;

    for (r, t) in candidates {
        let count = cheirality_count(r, t, pts1, pts2);
        if count > best.as_ref().map_or(0, |b| b.2) {
            best = Some((*r, *t, count));
        }
    }

    best.filter(|b| b.2 > 0)
}

/// Recover the unique relative pose from an essential matrix and correspondences.
///
/// Decomposes `E` into 4 candidate (R, t) pairs, then selects the one with
/// the most points in front of both cameras.
pub fn recover_pose_from_essential(e: &Mat3, pts1: &[Pt2], pts2: &[Pt2]) -> Result<(Mat3, Vec3)> {
    let candidates = vision_geometry::epipolar::decompose_essential(e)?;
    select_pose(&candidates, pts1, pts2)
        .map(|(r, t, _)| (r, t))
        .ok_or_else(|| anyhow::anyhow!("no valid pose found (all candidates fail cheirality)"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Rotation3;
    use vision_calibration_core::Pt3;

    fn skew(v: &Vec3) -> Mat3 {
        Mat3::new(0.0, -v.z, v.y, v.z, 0.0, -v.x, -v.y, v.x, 0.0)
    }

    fn make_stereo_data() -> (Mat3, Vec3, Vec<Pt2>, Vec<Pt2>) {
        let rot = Rotation3::from_euler_angles(0.05, -0.03, 0.02);
        let r = *rot.matrix();
        let t = Vec3::new(0.3, 0.01, 0.005);

        let world = [
            Pt3::new(0.1, 0.2, 3.0),
            Pt3::new(-0.2, 0.1, 2.5),
            Pt3::new(0.3, -0.1, 4.0),
            Pt3::new(-0.15, -0.2, 3.2),
            Pt3::new(0.05, 0.3, 2.8),
            Pt3::new(0.2, -0.3, 3.5),
            Pt3::new(-0.1, 0.15, 2.0),
            Pt3::new(0.15, 0.25, 4.5),
        ];

        let mut pts1 = Vec::new();
        let mut pts2 = Vec::new();
        for pw in &world {
            let pc1 = pw.coords;
            let pc2 = r * pw.coords + t;
            pts1.push(Pt2::new(pc1.x / pc1.z, pc1.y / pc1.z));
            pts2.push(Pt2::new(pc2.x / pc2.z, pc2.y / pc2.z));
        }

        (r, t, pts1, pts2)
    }

    #[test]
    fn cheirality_count_correct_for_true_pose() {
        let (r, t, pts1, pts2) = make_stereo_data();
        let count = cheirality_count(&r, &t, &pts1, &pts2);
        assert_eq!(count, pts1.len(), "all points should be in front");
    }

    #[test]
    fn cheirality_count_zero_for_wrong_pose() {
        let (r, t, pts1, pts2) = make_stereo_data();
        // Negate translation — should put points behind camera 2.
        let count = cheirality_count(&r, &(-t), &pts1, &pts2);
        assert_eq!(count, 0, "negated t should fail cheirality");
    }

    #[test]
    fn select_pose_picks_correct_candidate() {
        let (r, t, pts1, pts2) = make_stereo_data();
        let e = skew(&t) * r;

        let candidates = vision_geometry::epipolar::decompose_essential(&e).unwrap();
        let (r_est, t_est, count) = select_pose(&candidates, &pts1, &pts2).unwrap();

        assert_eq!(count, pts1.len());

        // Check rotation is close.
        let r_diff = r_est.transpose() * r;
        let cos_theta = ((r_diff.trace() - 1.0) * 0.5).clamp(-1.0, 1.0);
        let ang = cos_theta.acos();
        assert!(ang < 1e-6, "rotation error: {} rad", ang);

        // Check translation direction.
        let cos_t = t_est.normalize().dot(&t.normalize());
        assert!(
            (cos_t.abs() - 1.0).abs() < 1e-6,
            "translation direction error"
        );
    }

    #[test]
    fn recover_pose_from_essential_works() {
        let (r, t, pts1, pts2) = make_stereo_data();
        let e = skew(&t) * r;

        let (r_est, t_est) = recover_pose_from_essential(&e, &pts1, &pts2).unwrap();

        let r_diff = r_est.transpose() * r;
        let cos_theta = ((r_diff.trace() - 1.0) * 0.5).clamp(-1.0, 1.0);
        let ang = cos_theta.acos();
        assert!(ang < 1e-6, "rotation error: {} rad", ang);

        let cos_t = t_est.normalize().dot(&t.normalize());
        assert!((cos_t.abs() - 1.0).abs() < 1e-6);
    }
}
