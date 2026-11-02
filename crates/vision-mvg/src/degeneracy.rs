//! Degeneracy detection for two-view geometry.
//!
//! Identifies degenerate or near-degenerate configurations that can cause
//! estimation failures:
//!
//! - **Pure rotation**: no translation baseline, triangulation is impossible
//! - **Planar scene**: homography model is more appropriate than essential
//! - **Poor baseline**: near-parallel rays produce unreliable depth

use crate::types::TriangulatedPoint;
use vision_calibration_core::Real;

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
    /// Ratio of the baseline to the scene depth (baseline ratio). The baseline
    /// is the unit translation of the relative pose, so this is `1` over the
    /// median depth, in baselines. Depth is the distance along the first
    /// camera's bearing, so it is meaningful beyond 90° from the axis, where
    /// `z` is not.
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
/// `points` are the triangulated points in the first camera's frame at the
/// unit baseline of the relative pose; the translation is not needed, so pure
/// rotation is detected from the scale-invariant parallax signal rather than
/// from `‖E‖`, which an essential matrix fixes only up to scale. The caller
/// passes the points that triangulated: a single failed point (e.g. an
/// on-baseline / zero-parallax outlier) must not discard the parallax evidence
/// of the others, which would falsely look like pure rotation.
pub(crate) fn analyze_scene(points: &[TriangulatedPoint]) -> SceneDiagnostics {
    let tps = points;
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

    // Baseline ratio: the unit baseline over the median depth, from the
    // points in front of both cameras. Depth is the distance along the first
    // camera's bearing, i.e. the norm of the point in the first camera's frame.
    let baseline_ratio = {
        let mut depths: Vec<Real> = tps
            .iter()
            .filter(|tp| tp.in_front)
            .map(|tp| tp.point.coords.norm())
            .collect();
        if depths.is_empty() {
            0.0
        } else {
            depths.sort_by(|a, b| a.total_cmp(b));
            let median_depth = depths[depths.len() / 2];
            if median_depth.abs() > 1e-12 {
                1.0 / median_depth
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
    use vision_calibration_core::Pt3;

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

    fn point(p: Pt3, parallax_deg: Real, in_front: bool) -> TriangulatedPoint {
        TriangulatedPoint {
            point: p,
            reprojection_error: Some(0.0),
            parallax_deg,
            in_front,
        }
    }

    #[test]
    fn analyze_scene_good_stereo() {
        let pts: Vec<_> = (0..8)
            .map(|i| {
                point(
                    Pt3::new(0.1 * i as Real, 0.2, 3.0 + i as Real),
                    3.0 + i as Real,
                    true,
                )
            })
            .collect();
        let diag = analyze_scene(&pts);
        assert!(!diag.is_pure_rotation);
        assert!(diag.median_parallax_deg > 0.0);
        // Median depth is the distance along the bearing: ‖(0.4, 0.2, 7)‖.
        let expected = 1.0 / Pt3::new(0.4, 0.2, 7.0).coords.norm();
        assert!(
            (diag.baseline_ratio - expected).abs() < 0.02,
            "{}",
            diag.baseline_ratio
        );
    }

    /// Points that do not triangulate are the caller's to leave out; the
    /// others still carry the baseline, and points behind a camera do not
    /// enter the depth.
    #[test]
    fn analyze_scene_ignores_points_behind_and_keeps_baseline() {
        let mut pts: Vec<_> = (0..6)
            .map(|i| point(Pt3::new(0.0, 0.0, 3.0 + i as Real), 2.0, true))
            .collect();
        pts.push(point(Pt3::new(0.0, 0.0, -50.0), 2.0, false));
        let diag = analyze_scene(&pts);
        assert!(!diag.is_pure_rotation);
        assert!(
            (diag.baseline_ratio - 1.0 / 6.0).abs() < 1e-12,
            "{}",
            diag.baseline_ratio
        );
    }

    /// Negative `z` in front along the bearing still has a depth.
    #[test]
    fn analyze_scene_depth_is_along_the_bearing() {
        let pts: Vec<_> = (0..5)
            .map(|_| point(Pt3::new(0.0, 0.0, -4.0), 2.0, true))
            .collect();
        let diag = analyze_scene(&pts);
        assert!((diag.baseline_ratio - 0.25).abs() < 1e-12);
    }

    #[test]
    fn analyze_scene_without_points_is_pure_rotation() {
        let diag = analyze_scene(&[]);
        assert!(diag.is_pure_rotation);
        assert_eq!(diag.baseline_ratio, 0.0);
    }
}
