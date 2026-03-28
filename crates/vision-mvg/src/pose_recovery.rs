//! Relative pose recovery from calibrated correspondences.
//!
//! Orchestrates the 5-point algorithm, essential matrix decomposition, and
//! cheirality testing to recover the unique relative pose between two views.

use crate::cheirality;
use crate::residuals;
use crate::triangulation;
use crate::types::{Correspondence2D, EssentialMatrix, TriangulatedPoint};
use anyhow::Result;
use vision_calibration_core::{Mat3, Real, Vec3};

/// Result of relative pose recovery.
#[derive(Debug, Clone)]
pub struct RelativePose {
    /// Rotation from camera 1 to camera 2.
    pub r: Mat3,
    /// Unit translation direction from camera 1 to camera 2.
    pub t: Vec3,
    /// The essential matrix used.
    pub essential: EssentialMatrix,
    /// Triangulated points from the inlier correspondences.
    pub points: Vec<TriangulatedPoint>,
}

/// Recover relative pose from calibrated correspondences using the 5-point algorithm.
///
/// This is a convenience function that:
/// 1. Runs the 5-point essential matrix solver on all correspondences
/// 2. For each candidate E, decomposes into (R, t) pairs
/// 3. Selects the pose with the most points passing cheirality
/// 4. Triangulates all correspondences with the selected pose
///
/// Input points must be in **normalized camera coordinates**.
/// Requires at least 5 correspondences.
pub fn recover_relative_pose(corrs: &[Correspondence2D]) -> Result<RelativePose> {
    if corrs.len() < 5 {
        anyhow::bail!("need at least 5 correspondences, got {}", corrs.len());
    }

    let (pts1, pts2) = Correspondence2D::split(corrs);

    // Use first 5 points for the 5-point algorithm.
    let candidates_e = vision_geometry::epipolar::essential_5point(&pts1[..5], &pts2[..5])?;

    let mut best_r = None;
    let mut best_t = None;
    let mut best_e = None;
    let mut best_count = 0;
    let mut best_residual = Real::INFINITY;

    for e in &candidates_e {
        // Score this E by mean Sampson distance over all correspondences.
        let mean_sampson: Real = corrs
            .iter()
            .map(|c| residuals::sampson_distance(e, c))
            .sum::<Real>()
            / corrs.len() as Real;

        let decompositions = vision_geometry::epipolar::decompose_essential(e)?;
        if let Some((r, t, count)) = cheirality::select_pose(&decompositions, &pts1, &pts2) {
            // Prefer more cheirality inliers; break ties by residual.
            let is_better =
                count > best_count || (count == best_count && mean_sampson < best_residual);
            if is_better {
                best_count = count;
                best_residual = mean_sampson;
                best_r = Some(r);
                best_t = Some(t);
                best_e = Some(*e);
            }
        }
    }

    let r = best_r.ok_or_else(|| anyhow::anyhow!("no valid pose found"))?;
    let t = best_t.unwrap();
    let e = best_e.unwrap();

    let points = triangulation::triangulate_two_view(&r, &t, &pts1, &pts2)?;

    Ok(RelativePose {
        r,
        t,
        essential: e,
        points,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Rotation3;
    use vision_calibration_core::{Pt2, Pt3};

    #[test]
    fn recover_relative_pose_noiseless() {
        let rot = Rotation3::from_euler_angles(0.1, -0.05, 0.2);
        let r_gt = *rot.matrix();
        let t_gt = Vec3::new(0.5, 0.1, 0.05);

        // Points well-spread in depth and position for robust estimation.
        let world = [
            Pt3::new(0.5, 0.3, 3.0),
            Pt3::new(-0.4, 0.2, 4.0),
            Pt3::new(0.6, -0.3, 5.0),
            Pt3::new(-0.3, -0.4, 3.5),
            Pt3::new(0.1, 0.6, 4.5),
            Pt3::new(0.4, -0.5, 6.0),
            Pt3::new(-0.2, 0.3, 3.0),
            Pt3::new(0.3, 0.5, 5.5),
        ];

        let corrs: Vec<_> = world
            .iter()
            .map(|pw| {
                let pc1 = pw.coords;
                let pc2 = r_gt * pw.coords + t_gt;
                Correspondence2D::new(
                    Pt2::new(pc1.x / pc1.z, pc1.y / pc1.z),
                    Pt2::new(pc2.x / pc2.z, pc2.y / pc2.z),
                )
            })
            .collect();

        // First verify that the 5-point solver + decomposition works on its own.
        let (p1, p2) = Correspondence2D::split(&corrs);
        let es = vision_geometry::epipolar::essential_5point(&p1[..5], &p2[..5]).unwrap();

        let mut found_good = false;
        for e in &es {
            let decomps = vision_geometry::epipolar::decompose_essential(e).unwrap();
            for (r, t) in &decomps {
                let rd = r.transpose() * r_gt;
                let ca = ((rd.trace() - 1.0) * 0.5).clamp(-1.0, 1.0);
                let ad = ca.acos().to_degrees();
                let ct = t.normalize().dot(&t_gt.normalize()).abs();
                if ad < 1.0 && ct > 0.99 {
                    found_good = true;
                }
            }
        }
        assert!(
            found_good,
            "5-point solver did not produce a valid E candidate"
        );

        let result = recover_relative_pose(&corrs).unwrap();

        // Check rotation.
        let r_diff = result.r.transpose() * r_gt;
        let cos_theta = ((r_diff.trace() - 1.0) * 0.5).clamp(-1.0, 1.0);
        let ang_deg = cos_theta.acos().to_degrees();
        assert!(ang_deg < 1.0, "rotation error too large: {} deg", ang_deg);

        // Check translation direction.
        let cos_t = result.t.normalize().dot(&t_gt.normalize());
        assert!(
            (cos_t.abs() - 1.0).abs() < 1e-3,
            "translation direction error: cos = {}",
            cos_t
        );

        // Check triangulated points.
        assert_eq!(result.points.len(), world.len());
        for tp in &result.points {
            assert!(tp.in_front, "point should be in front of cameras");
            assert!(tp.reprojection_error < 1e-6);
        }
    }
}
