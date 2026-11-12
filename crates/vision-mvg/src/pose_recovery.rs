//! Relative pose recovery from calibrated correspondences.
//!
//! Orchestrates the 5-point algorithm, essential matrix decomposition, and
//! cheirality testing to recover the unique relative pose between two views.
//! Pixels become bearings through the rig cameras' own models first, so every
//! camera model works, including a fisheye beyond 90° from its axis.

use crate::bearing::camera_model;
use crate::cheirality;
use crate::robust::bearing_pairs;
use crate::triangulation::{ViewRay, triangulate_views, with_context};
use crate::types::{PixelMatch, TriangulatedPoint};
use crate::{Error, Result};
use nalgebra::{Rotation3, Translation3, UnitQuaternion};
use serde::Serialize;
use vision_calibration_core::{CalibratedRig, Iso3, Mat3, Real, Vec3};

/// Result of relative pose recovery.
#[derive(Debug, Clone, Serialize)]
#[non_exhaustive]
pub struct RelativePose {
    /// Pose of the first camera's frame in the second's (`second_se3_first`).
    ///
    /// Its translation has unit norm: the scale is not observable from two
    /// views.
    pub second_se3_first: Iso3,
    /// The essential matrix used: `b2ᵀ E b1 = 0` on the bearings of matched
    /// pixels.
    pub essential: Mat3,
    /// One triangulated point per match, in the first camera's frame at the
    /// unit baseline, with their pixel reprojection diagnostics through the two
    /// cameras' models. A degenerate match fails the call.
    pub points: Vec<TriangulatedPoint>,
    /// Whether the scene constrains the geometry: parallax, pure rotation,
    /// planarity and baseline-to-depth ratio, from `points`.
    pub scene: SceneDiagnostics,
}

/// Pure-rotation parallax threshold in degrees.
///
/// A median parallax below this value is treated as (approximately) pure
/// rotation — i.e. essentially zero baseline: pure rotation means *no*
/// translation at all, not just a weak one.
const PURE_ROTATION_PARALLAX_DEG: Real = 0.1;

/// Diagnostics about a two-view scene configuration.
#[derive(Debug, Clone, Serialize)]
#[non_exhaustive]
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

/// The rigid transform with rotation `r` and translation `t`.
pub(crate) fn iso_from_rt(r: &Mat3, t: &Vec3) -> Iso3 {
    Iso3::from_parts(
        Translation3::from(*t),
        UnitQuaternion::from_rotation_matrix(&Rotation3::from_matrix_unchecked(*r)),
    )
}

/// Recover relative pose from calibrated correspondences using the 5-point algorithm.
///
/// `matches[i].pt1` is a pixel of rig camera `cameras[0]` in the first
/// capture and `matches[i].pt2` a pixel of rig camera `cameras[1]` in the
/// second (`[0, 0]` for one moving camera). Pixels are unprojected through
/// each camera's own model, so distortion and fisheye need no separate
/// undistortion.
///
/// This is a convenience function that:
/// 1. Samples multiple 5-point subsets from the matches to generate
///    essential matrix candidates
/// 2. For each candidate E, scores it by weighted Sampson error and cheirality
///    over **all** matches
/// 3. Selects the pose with the most points passing cheirality
/// 4. Triangulates all matches with the selected pose
///
/// For outlier-contaminated data, prefer [`crate::robust::recover_relative_pose_robust`]
/// which uses RANSAC.
///
/// # Errors
///
/// [`Error::InsufficientData`] for fewer than 5 matches,
/// [`Error::InvalidInput`] for a camera index out of range or a pixel with
/// no finite viewing ray, [`Error::NoValidPose`] when no candidate passes
/// cheirality, and an error naming the match when it cannot be triangulated.
pub fn recover_relative_pose(
    rig: &CalibratedRig,
    cameras: [usize; 2],
    matches: &[PixelMatch],
) -> Result<RelativePose> {
    let n = matches.len();
    if n < 5 {
        return Err(Error::InsufficientData { need: 5, got: n });
    }

    let pairs = bearing_pairs(rig, cameras, matches)?;
    let b1: Vec<Vec3> = pairs.iter().map(|p| p.b1).collect();
    let b2: Vec<Vec3> = pairs.iter().map(|p| p.b2).collect();

    // Generate essential matrix candidates from multiple 5-point subsets.
    // Use evenly-spaced subsets to avoid dependence on input ordering.
    let mut all_candidates: Vec<Mat3> = Vec::new();

    if n == 5 {
        // Exactly 5 points: single call.
        all_candidates.extend(vision_geometry::epipolar::essential_5point(&b1, &b2)?);
    } else {
        // Sample up to `max_subsets` evenly-spaced 5-point subsets.
        let max_subsets = n.min(20);
        for start in 0..max_subsets {
            let indices: Vec<usize> = (0..5).map(|k| (start + k * n / 5) % n).collect();
            // Skip if indices are not distinct.
            let mut sorted = indices.clone();
            sorted.sort();
            sorted.dedup();
            if sorted.len() < 5 {
                continue;
            }
            let sub1: Vec<_> = indices.iter().map(|&i| b1[i]).collect();
            let sub2: Vec<_> = indices.iter().map(|&i| b2[i]).collect();
            if let Ok(es) = vision_geometry::epipolar::essential_5point(&sub1, &sub2) {
                all_candidates.extend(es);
            }
        }
    }

    if all_candidates.is_empty() {
        return Err(Error::NoValidPose);
    }

    let mut best_r = None;
    let mut best_t = None;
    let mut best_e = None;
    let mut best_count = 0;
    let mut best_residual = Real::INFINITY;

    for e in &all_candidates {
        // Score this E by the mean weighted Sampson error (σ) over all matches.
        let mean_error: Real = pairs.iter().map(|p| p.error(e)).sum::<Real>() / n as Real;

        let decompositions = vision_geometry::epipolar::decompose_essential(e)?;
        if let Some((r, t, count)) = cheirality::select_pose(&decompositions, &b1, &b2) {
            // Prefer more cheirality inliers; break ties by residual.
            let is_better =
                count > best_count || (count == best_count && mean_error < best_residual);
            if is_better {
                best_count = count;
                best_residual = mean_error;
                best_r = Some(r);
                best_t = Some(t);
                best_e = Some(*e);
            }
        }
    }

    let r = best_r.ok_or(Error::NoValidPose)?;
    let t = best_t.ok_or(Error::NoValidPose)?;
    let e = best_e.ok_or(Error::NoValidPose)?;
    let second_se3_first = iso_from_rt(&r, &t);

    // Two views: the first camera at the origin, the second at the pose.
    let (model1, model2) = (
        camera_model(rig, cameras[0])?,
        camera_model(rig, cameras[1])?,
    );
    let points = matches
        .iter()
        .zip(b1.iter().zip(&b2))
        .enumerate()
        .map(|(i, (m, (q1, q2)))| {
            triangulate_views(&[
                ViewRay {
                    cam_se3_world: Iso3::identity(),
                    model: model1,
                    pixel: m.pt1,
                    bearing: *q1,
                },
                ViewRay {
                    cam_se3_world: second_se3_first,
                    model: model2,
                    pixel: m.pt2,
                    bearing: *q2,
                },
            ])
            .map_err(|e| with_context(&format!("match {i}"), e))
        })
        .collect::<Result<Vec<_>>>()?;
    let scene = analyze_scene(&points);

    Ok(RelativePose {
        second_se3_first,
        essential: e,
        points,
        scene,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{fisheye_params, iso, match_of, pinhole_rig, pixel_of};
    use vision_calibration_core::{CalibratedRig, Pt3, Uncertainty};

    /// Eight points well-spread in depth and position.
    fn scene() -> Vec<Pt3> {
        vec![
            Pt3::new(0.5, 0.3, 3.0),
            Pt3::new(-0.4, 0.2, 4.0),
            Pt3::new(0.6, -0.3, 5.0),
            Pt3::new(-0.3, -0.4, 3.5),
            Pt3::new(0.1, 0.6, 4.5),
            Pt3::new(0.4, -0.5, 6.0),
            Pt3::new(-0.2, 0.3, 3.0),
            Pt3::new(0.3, 0.5, 5.5),
        ]
    }

    fn rotation_error_deg(a: &Iso3, b: &Iso3) -> f64 {
        a.rotation.angle_to(&b.rotation).to_degrees()
    }

    fn matches(rig: &CalibratedRig, pose: &Iso3, scene: &[Pt3]) -> Vec<PixelMatch> {
        scene
            .iter()
            .map(|p| {
                match_of(
                    pixel_of(rig, 0, &p.coords),
                    pixel_of(rig, 0, &(pose * p).coords),
                )
            })
            .collect()
    }

    #[test]
    fn recover_relative_pose_noiseless() {
        let rig = pinhole_rig(0.0);
        let gt = iso(0.1, -0.05, 0.2, Vec3::new(0.5, 0.1, 0.05));
        let world = scene();

        let result = recover_relative_pose(&rig, [0, 0], &matches(&rig, &gt, &world)).unwrap();

        assert!(rotation_error_deg(&result.second_se3_first, &gt) < 1e-6);
        let t = result.second_se3_first.translation.vector;
        assert!((t.norm() - 1.0).abs() < 1e-12, "unit translation");
        assert!((t - gt.translation.vector.normalize()).norm() < 1e-6);

        assert_eq!(result.points.len(), world.len());
        for (tp, pw) in result.points.iter().zip(&world) {
            assert!(tp.in_front);
            // Points are in the first camera's frame at the unit baseline.
            let scale = 1.0 / gt.translation.vector.norm();
            assert!((tp.point.coords - pw.coords * scale).norm() < 1e-6 * scale * 6.0);
            assert!(tp.reprojection_error.unwrap() < 1e-6);
        }
        assert!(!result.scene.is_pure_rotation);
        assert!(result.scene.baseline_ratio > 0.0);
    }

    /// Raw distorted pixels recover the truth: the camera model's own
    /// unprojection undistorts them.
    #[test]
    fn recovers_pose_from_distorted_pixels() {
        let rig = pinhole_rig(-0.2);
        let gt = iso(0.05, 0.12, -0.04, Vec3::new(-0.4, 0.05, 0.1));
        let world = scene();
        let result = recover_relative_pose(&rig, [0, 0], &matches(&rig, &gt, &world)).unwrap();
        assert!(rotation_error_deg(&result.second_se3_first, &gt) < 1e-6);
        let t = result.second_se3_first.translation.vector;
        assert!((t - gt.translation.vector.normalize()).norm() < 1e-8);
    }

    /// Two different cameras of a rig (`cameras = [0, 1]`), both
    /// fisheye, with points beyond 90° off the first camera's axis.
    #[test]
    fn recovers_pose_between_two_fisheye_cameras_beyond_90_degrees() {
        let rig = CalibratedRig::new(
            vec![fisheye_params(), fisheye_params()],
            vec![
                Iso3::identity(),
                iso(0.0, 0.0, 0.0, Vec3::new(-0.1, 0.0, 0.0)),
            ],
            Uncertainty::default(),
        )
        .unwrap();
        let gt = iso(0.05, -0.1, 0.02, Vec3::new(-0.3, 0.05, 0.02));
        // Directions all around the first camera, a third behind its image plane.
        let world: Vec<Pt3> = (0..18)
            .map(|i| {
                let a = f64::from(i);
                let theta = 0.3 + 0.12 * a; // 17° .. 2.3 rad (132°)
                let phi = 1.9 * a;
                let d = 2.0 + 0.2 * (a % 5.0);
                Pt3::new(
                    d * theta.sin() * phi.cos(),
                    d * theta.sin() * phi.sin(),
                    d * theta.cos(),
                )
            })
            .collect();
        assert!(world.iter().filter(|p| p.z < 0.0).count() >= 5);
        let matches: Vec<_> = world
            .iter()
            .map(|p| {
                match_of(
                    pixel_of(&rig, 0, &p.coords),
                    pixel_of(&rig, 1, &(gt * p).coords),
                )
            })
            .collect();
        let result = recover_relative_pose(&rig, [0, 1], &matches).unwrap();
        assert!(rotation_error_deg(&result.second_se3_first, &gt) < 1e-6);
        let t = result.second_se3_first.translation.vector;
        assert!((t - gt.translation.vector.normalize()).norm() < 1e-6);
        assert!(result.points.iter().all(|p| p.in_front));
    }

    #[test]
    fn rejects_bad_input() {
        let rig = pinhole_rig(0.0);
        let gt = iso(0.1, 0.0, 0.0, Vec3::new(0.5, 0.0, 0.0));
        let m = matches(&rig, &gt, &scene());
        assert!(matches!(
            recover_relative_pose(&rig, [0, 0], &m[..4]),
            Err(Error::InsufficientData { need: 5, got: 4 })
        ));
        assert!(matches!(
            recover_relative_pose(&rig, [0, 3], &m),
            Err(Error::InvalidInput { .. })
        ));
    }
}

#[cfg(test)]
mod scene_tests {
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
