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

/// Pure-rotation parallax threshold in degrees.
///
/// A median parallax below this value is treated as (approximately) pure
/// rotation — i.e. essentially zero baseline: pure rotation means *no*
/// translation at all, not just a weak one.
const PURE_ROTATION_PARALLAX_DEG: Real = 0.1;

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

/// Detect a pure rotation from triangulated parallax angles.
///
/// ## Why not use the essential matrix norm?
///
/// The old approach (`‖E‖ < threshold`) was scale-dependent: essential
/// matrices are defined only up to an arbitrary scale factor, so `E` and
/// `1e-9 · E` encode the same geometry but have vastly different norms.
/// A valid E scaled down by 1e-9 would be wrongly flagged as pure rotation.
///
/// ## Parallax is the scale-invariant signal
///
/// Pure rotation means the translation vector `t` is exactly zero.  When
/// `t = 0`, all viewing rays from camera 1 and camera 2 to the same scene
/// point are parallel — the parallax angle is zero.  The **median** parallax
/// angle is therefore the correct, scale-invariant test:
///
/// - Returns `true` (pure rotation) when `parallax_angles` is empty (no
///   triangulated points → no baseline evidence) or when the median angle
///   is below `threshold_deg`.
/// - Returns `false` otherwise (there is a non-trivial baseline).
pub(crate) fn detect_pure_rotation(parallax_angles: &[Real], threshold_deg: Real) -> bool {
    if parallax_angles.is_empty() {
        // No triangulated points — we cannot confirm a baseline exists.
        return true;
    }
    let mut sorted = parallax_angles.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let median = sorted[sorted.len() / 2];
    median < threshold_deg
}

/// Analyze a two-view scene for degeneracies.
///
/// Given correspondences and the recovered pose (R, t), returns diagnostics
/// about the scene configuration.  The essential matrix is not required: pure
/// rotation is detected from the scale-invariant parallax signal rather than
/// from `‖E‖`, which an essential matrix fixes only up to scale.
pub fn analyze_scene(corrs: &[Correspondence2D], r: &Mat3, t: &Vec3) -> SceneDiagnostics {
    // Triangulate once, tolerating per-point degeneracies: a single bad
    // correspondence (e.g. an on-baseline / zero-parallax outlier) must not
    // discard the parallax evidence from all the others — that would falsely
    // look like pure rotation with zero baseline.
    let (pts1, pts2) = Correspondence2D::split(corrs);
    let tps = crate::triangulation::triangulate_two_view_partial(r, t, &pts1, &pts2);
    let parallax_angles: Vec<Real> = tps.iter().map(|tp| tp.parallax_deg).collect();

    let median_parallax_deg = if parallax_angles.is_empty() {
        0.0
    } else {
        let mut sorted = parallax_angles.clone();
        sorted.sort_by(|a, b| a.total_cmp(b));
        sorted[sorted.len() / 2]
    };

    // Pure rotation: scale-invariant parallax test.
    let is_pure_rotation = detect_pure_rotation(&parallax_angles, PURE_ROTATION_PARALLAX_DEG);

    // Estimate baseline ratio: ||t|| / median_depth, from the same per-point
    // triangulation (points in front of both cameras).
    let baseline_ratio = {
        let mut depths: Vec<Real> = tps
            .iter()
            .filter(|tp| tp.in_front)
            .map(|tp| tp.point.z)
            .collect();
        if depths.is_empty() {
            0.0
        } else {
            depths.sort_by(|a, b| a.total_cmp(b));
            let median_depth = depths[depths.len() / 2];
            if median_depth.abs() > 1e-12 {
                t.norm() / median_depth
            } else {
                0.0
            }
        }
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
    use vision_calibration_core::{Pt2, Pt3};

    #[test]
    fn detect_pure_rotation_true_for_low_parallax() {
        // Small parallax angles → pure rotation.
        let angles = &[0.001_f64, 0.002, 0.0015];
        assert!(detect_pure_rotation(angles, 0.1));
    }

    #[test]
    fn detect_pure_rotation_false_for_good_parallax() {
        // Large parallax angles → not a pure rotation.
        let angles = &[2.0_f64, 3.5, 1.8];
        assert!(!detect_pure_rotation(angles, 0.1));
    }

    #[test]
    fn detect_pure_rotation_true_for_empty_angles() {
        // No triangulated points — cannot confirm a baseline → treat as pure rotation.
        assert!(detect_pure_rotation(&[], 0.1));
    }

    #[test]
    fn analyze_scene_good_stereo() {
        use nalgebra::Rotation3;
        let rot = Rotation3::from_euler_angles(0.05, -0.03, 0.02);
        let r = *rot.matrix();
        let t = Vec3::new(0.3, 0.01, 0.005);

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

        // `analyze_scene` needs only the correspondences and the relative pose.
        let diag = analyze_scene(&corrs, &r, &t);
        assert!(!diag.is_pure_rotation);
        assert!(diag.median_parallax_deg > 0.0);
        assert!(diag.baseline_ratio > 0.0);
    }

    /// Regression: a single on-baseline (degenerate) correspondence must not
    /// flip an otherwise good-baseline scene to "pure rotation". Before the
    /// per-point-tolerant triangulation, one failing point discarded ALL
    /// parallax evidence, leaving the scene wrongly classified.
    #[test]
    fn analyze_scene_one_degenerate_point_keeps_baseline() {
        use nalgebra::Rotation3;
        let rot = Rotation3::from_euler_angles(0.05, -0.03, 0.02);
        let r = *rot.matrix();
        let t = Vec3::new(0.3, 0.01, 0.005);

        let world = [
            Pt3::new(0.5, 0.3, 3.0),
            Pt3::new(-0.4, 0.2, 4.0),
            Pt3::new(0.6, -0.3, 5.0),
            Pt3::new(-0.3, -0.4, 3.5),
            Pt3::new(0.1, 0.6, 4.5),
            Pt3::new(0.4, -0.5, 6.0),
        ];
        let mut corrs: Vec<Correspondence2D> = world
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

        // A point on the baseline (line through both camera centers): its two
        // viewing rays are collinear → triangulation is rank-deficient and the
        // point is skipped. C1 = origin, C2 = -Rᵀt, so any λ·C2 lies on the line.
        let c2 = -r.transpose() * t;
        let bad = 3.0 * c2;
        let pc2_bad = r * bad + t;
        corrs.insert(
            0,
            Correspondence2D::new(
                Pt2::new(bad.x / bad.z, bad.y / bad.z),
                Pt2::new(pc2_bad.x / pc2_bad.z, pc2_bad.y / pc2_bad.z),
            ),
        );

        let diag = analyze_scene(&corrs, &r, &t);
        assert!(
            !diag.is_pure_rotation,
            "one degenerate point must not flip the scene to pure rotation"
        );
        assert!(diag.median_parallax_deg > 0.0);
        assert!(diag.baseline_ratio > 0.0);
    }
}
