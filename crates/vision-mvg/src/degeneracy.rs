//! Degeneracy detection for two-view geometry.
//!
//! Identifies degenerate or near-degenerate configurations that can cause
//! estimation failures:
//!
//! - **Pure rotation**: no translation baseline, triangulation is impossible
//! - **Planar scene**: homography model is more appropriate than essential
//! - **Poor baseline**: near-parallel rays produce unreliable depth

use crate::types::Correspondence2D;
use vision_calibration_core::{Mat3, Real, Vec3};

/// Diagnostics about a two-view scene configuration.
#[derive(Debug, Clone)]
pub struct SceneDiagnostics {
    /// Median parallax angle in degrees across triangulated points.
    pub median_parallax_deg: Real,
    /// Whether the motion is (approximately) a pure rotation.
    pub is_pure_rotation: bool,
    /// Whether the scene appears planar.
    pub is_planar: bool,
    /// Ratio of translation magnitude to scene depth (baseline ratio).
    pub baseline_ratio: Real,
}

/// Detect a pure rotation from an essential matrix.
///
/// A pure rotation has `t ≈ 0`, so the essential matrix `E = [t]× R ≈ 0`.
/// Returns `true` if the Frobenius norm of E (after normalizing the largest
/// singular value to 1) is below a threshold.
pub fn detect_pure_rotation(e: &Mat3) -> bool {
    let svd = e.svd(false, false);
    let s1 = svd.singular_values[0];
    if s1 < 1e-12 {
        return true; // zero matrix ⇒ pure rotation
    }
    // For a proper essential matrix, sv = (σ, σ, 0).
    // For pure rotation, all singular values ≈ 0.
    // Check ratio of second singular value to first.
    let s2 = svd.singular_values[1];
    s2 / s1 < 0.01
}

/// Detect a planar scene by comparing homography and essential inlier counts.
///
/// If the homography model explains almost as many inliers as the essential
/// matrix, the scene is likely planar. The `ratio_threshold` controls
/// sensitivity (typical value: 0.45 — if H inlier ratio exceeds this
/// fraction of E inlier ratio, the scene is planar).
pub fn detect_planar_scene(
    h_inliers: usize,
    e_inliers: usize,
    total: usize,
    ratio_threshold: Real,
) -> bool {
    if total == 0 || e_inliers == 0 {
        return false;
    }
    let h_ratio = h_inliers as Real / total as Real;
    let e_ratio = e_inliers as Real / total as Real;
    h_ratio / e_ratio > ratio_threshold
}

/// Detect insufficient baseline from triangulated parallax angles.
///
/// Returns `true` if fewer than half the angles exceed `threshold_deg`.
pub fn detect_poor_baseline(parallax_angles: &[Real], threshold_deg: Real) -> bool {
    if parallax_angles.is_empty() {
        return true;
    }
    let good = parallax_angles
        .iter()
        .filter(|&&a| a > threshold_deg)
        .count();
    good < parallax_angles.len() / 2
}

/// Analyze a two-view scene for degeneracies.
///
/// Given correspondences, the estimated essential matrix, and the recovered
/// pose (R, t), returns diagnostics about the scene configuration.
pub fn analyze_scene(
    corrs: &[Correspondence2D],
    e: &Mat3,
    r: &Mat3,
    t: &Vec3,
) -> SceneDiagnostics {
    let is_pure_rotation = detect_pure_rotation(e);

    // Triangulate to get parallax angles.
    let (pts1, pts2) = Correspondence2D::split(corrs);
    let parallax_angles: Vec<Real> =
        if let Ok(tps) = crate::triangulation::triangulate_two_view(r, t, &pts1, &pts2) {
            tps.iter().map(|tp| tp.parallax_deg).collect()
        } else {
            vec![]
        };

    let median_parallax_deg = if parallax_angles.is_empty() {
        0.0
    } else {
        let mut sorted = parallax_angles.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        sorted[sorted.len() / 2]
    };

    // Estimate baseline ratio: ||t|| / median_depth.
    let baseline_ratio = if let Ok(tps) =
        crate::triangulation::triangulate_two_view(r, t, &pts1, &pts2)
    {
        let mut depths: Vec<Real> = tps.iter().filter(|tp| tp.in_front).map(|tp| tp.point.z).collect();
        if depths.is_empty() {
            0.0
        } else {
            depths.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let median_depth = depths[depths.len() / 2];
            if median_depth.abs() > 1e-12 {
                t.norm() / median_depth
            } else {
                0.0
            }
        }
    } else {
        0.0
    };

    // Planar detection heuristic: check if points lie on a plane.
    // Use the parallax angle distribution — planar scenes have very
    // uniform parallax compared to general 3D scenes.
    let is_planar = if parallax_angles.len() >= 5 {
        let mean: Real = parallax_angles.iter().sum::<Real>() / parallax_angles.len() as Real;
        let var: Real = parallax_angles
            .iter()
            .map(|a| (a - mean).powi(2))
            .sum::<Real>()
            / parallax_angles.len() as Real;
        let cv = if mean.abs() > 1e-12 {
            var.sqrt() / mean
        } else {
            0.0
        };
        cv < 0.1 // very low coefficient of variation suggests planarity
    } else {
        false
    };

    SceneDiagnostics {
        median_parallax_deg,
        is_pure_rotation,
        is_planar,
        baseline_ratio,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Rotation3;
    use vision_calibration_core::{Pt2, Pt3};

    fn skew(v: &Vec3) -> Mat3 {
        Mat3::new(0.0, -v.z, v.y, v.z, 0.0, -v.x, -v.y, v.x, 0.0)
    }

    #[test]
    fn detect_pure_rotation_true_for_zero_translation() {
        // E = [t]x * R with t ≈ 0 gives E ≈ 0.
        let rot = Rotation3::from_euler_angles(0.1, -0.05, 0.2);
        let t = Vec3::new(0.0, 0.0, 0.0);
        let e = skew(&t) * rot.matrix();
        assert!(detect_pure_rotation(&e));
    }

    #[test]
    fn detect_pure_rotation_false_for_good_baseline() {
        let rot = Rotation3::from_euler_angles(0.1, -0.05, 0.2);
        let t = Vec3::new(0.3, 0.01, 0.005);
        let e = skew(&t) * rot.matrix();
        assert!(!detect_pure_rotation(&e));
    }

    #[test]
    fn detect_poor_baseline_true_for_small_parallax() {
        let angles = vec![0.001, 0.002, 0.0015, 0.0008, 0.003];
        assert!(detect_poor_baseline(&angles, 0.5));
    }

    #[test]
    fn detect_poor_baseline_false_for_good_parallax() {
        let angles = vec![2.0, 3.5, 1.8, 4.2, 2.7];
        assert!(!detect_poor_baseline(&angles, 0.5));
    }

    #[test]
    fn detect_planar_scene_by_inlier_ratio() {
        // H explains 90% of points, E explains 95% — scene is planar.
        assert!(detect_planar_scene(90, 95, 100, 0.85));
        // H explains 30% of points, E explains 90% — scene is 3D.
        assert!(!detect_planar_scene(30, 90, 100, 0.85));
    }

    #[test]
    fn analyze_scene_good_stereo() {
        let rot = Rotation3::from_euler_angles(0.05, -0.03, 0.02);
        let r = *rot.matrix();
        let t = Vec3::new(0.3, 0.01, 0.005);
        let e = skew(&t) * r;

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
                let pc2 = r * pw.coords + t;
                Correspondence2D::new(
                    Pt2::new(pc1.x / pc1.z, pc1.y / pc1.z),
                    Pt2::new(pc2.x / pc2.z, pc2.y / pc2.z),
                )
            })
            .collect();

        let diag = analyze_scene(&corrs, &e, &r, &t);
        assert!(!diag.is_pure_rotation);
        assert!(diag.median_parallax_deg > 0.0);
        assert!(diag.baseline_ratio > 0.0);
    }
}
