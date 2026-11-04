//! Cheirality (positive depth) tests for pose disambiguation.
//!
//! When decomposing an essential matrix, four candidate (R, t) pairs are
//! returned. Only one places the triangulated points in front of both
//! cameras. These functions count and select the correct pose.

#[cfg(test)]
use crate::{MvgError, Result};
use vision_calibration_core::{Mat3, Vec3};

/// Count the matches whose point lies ahead along both bearings.
///
/// A candidate `(r, t)` maps the first camera's frame into the second's:
/// `p2 = r p1 + t`. A point seen along the unit bearings `b1` and `b2` is
/// `p1 = λ1 b1`, `p2 = λ2 b2`, so `λ1 r b1 − λ2 b2 = −t`; the least-squares
/// `λ` of this 3×2 system decide, and the point is in front iff both are
/// positive. This holds beyond 90° from the optical axis, where depth along `z`
/// means nothing. A pair of parallel rays (no solution) does not count.
pub fn cheirality_count(r: &Mat3, t: &Vec3, b1: &[Vec3], b2: &[Vec3]) -> usize {
    b1.iter()
        .zip(b2.iter())
        .filter(|(q1, q2)| {
            let rq1 = r * **q1;
            // Normal equations of [r q1, -q2] [λ1, λ2]ᵀ = -t.
            let a11 = rq1.dot(&rq1);
            let a12 = -rq1.dot(q2);
            let a22 = q2.dot(q2);
            let det = a11 * a22 - a12 * a12;
            if det <= 1e-12 * a11 * a22 {
                return false;
            }
            let c1 = -rq1.dot(t);
            let c2 = q2.dot(t);
            let lambda1 = (c1 * a22 - a12 * c2) / det;
            let lambda2 = (a11 * c2 - a12 * c1) / det;
            lambda1 > 0.0 && lambda2 > 0.0
        })
        .count()
}

/// Select the correct (R, t) from candidate poses by cheirality on bearings.
///
/// Returns the pose that places the most points in front of both cameras,
/// along with the number of points passing the cheirality check. Returns
/// `None` if no candidate has any valid points.
pub fn select_pose(
    candidates: &[(Mat3, Vec3)],
    pts1: &[Vec3],
    pts2: &[Vec3],
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
///
/// Returns `Err` if the point slices have different lengths, if there are no
/// correspondences, or if every candidate pose fails the cheirality check.
///
/// The tests' reference: the robust pose recovery votes with
/// [`select_pose`] on the decompositions of its refined `E` itself.
#[cfg(test)]
pub fn recover_pose_from_essential(e: &Mat3, pts1: &[Vec3], pts2: &[Vec3]) -> Result<(Mat3, Vec3)> {
    if pts1.len() != pts2.len() {
        return Err(MvgError::CountMismatch {
            expected: pts1.len(),
            got: pts2.len(),
        });
    }
    if pts1.is_empty() {
        return Err(MvgError::InsufficientData { need: 1, got: 0 });
    }
    let candidates = vision_geometry::epipolar::decompose_essential(e)?;
    select_pose(&candidates, pts1, pts2)
        .map(|(r, t, _)| (r, t))
        .ok_or(MvgError::NoValidPose)
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Rotation3;
    use vision_calibration_core::Pt3;

    fn skew(v: &Vec3) -> Mat3 {
        Mat3::new(0.0, -v.z, v.y, v.z, 0.0, -v.x, -v.y, v.x, 0.0)
    }

    fn make_stereo_data() -> (Mat3, Vec3, Vec<Vec3>, Vec<Vec3>) {
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
            pts1.push(pc1.normalize());
            pts2.push(pc2.normalize());
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

    /// Regression: mismatched slice lengths must return Err, not silently truncate.
    #[test]
    fn recover_pose_from_essential_rejects_length_mismatch() {
        let (r, t, pts1, pts2) = make_stereo_data();
        let e = skew(&t) * r;
        // Give pts2 one extra point — lengths differ by 1.
        let mut pts2_long = pts2.clone();
        pts2_long.push(Vec3::new(0.0, 0.0, 1.0));
        assert!(
            recover_pose_from_essential(&e, &pts1, &pts2_long).is_err(),
            "mismatched lengths must return Err"
        );
        // Symmetric: pts1 longer.
        let mut pts1_long = pts1.clone();
        pts1_long.push(Vec3::new(0.0, 0.0, 1.0));
        assert!(
            recover_pose_from_essential(&e, &pts1_long, &pts2).is_err(),
            "mismatched lengths must return Err"
        );
    }

    /// Regression: empty slices must return Err.
    #[test]
    fn recover_pose_from_essential_rejects_empty_input() {
        let (r, t, _pts1, _pts2) = make_stereo_data();
        let e = skew(&t) * r;
        assert!(
            recover_pose_from_essential(&e, &[], &[]).is_err(),
            "empty input must return Err"
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

    /// A point behind the first camera (`z < 0`) is still "in front" along its
    /// bearing: cheirality is decided by the rays, not by the sign of `z`.
    #[test]
    fn cheirality_counts_points_beyond_90_degrees() {
        let rot = Rotation3::from_euler_angles(0.0, 0.4, 0.0);
        let r = *rot.matrix();
        let t = Vec3::new(0.5, 0.0, 0.0);
        let world = [
            Pt3::new(-2.0, 0.3, -1.0),
            Pt3::new(1.5, -0.2, -0.5),
            Pt3::new(0.5, 0.5, 2.0),
        ];
        let b1: Vec<Vec3> = world.iter().map(|p| p.coords.normalize()).collect();
        let b2: Vec<Vec3> = world
            .iter()
            .map(|p| (r * p.coords + t).normalize())
            .collect();
        assert!(b1.iter().filter(|b| b.z < 0.0).count() >= 2);
        assert_eq!(cheirality_count(&r, &t, &b1, &b2), 3);
        assert_eq!(cheirality_count(&r, &(-t), &b1, &b2), 0);
    }
}
