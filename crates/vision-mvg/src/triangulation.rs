//! Triangulation from a calibration, refined, with the covariance of the points.
//!
//! Pixels become bearings through each rig camera's own model and the linear
//! (DLT) triangulator of `vision-geometry` starts every point. The points are
//! then refined together on the reprojection error of the stated pixel noise,
//! through the public problem builder of `vision-calibration-optim`, with the
//! rig's calibration as consider parameters: the result carries the joint
//! covariance of the points, the calibration's uncertainty included. The
//! diagnostics report the reprojection error in pixels, the parallax and
//! whether the point lies ahead of every observing camera.

use crate::bearing::{bearing, camera_model};
use crate::types::TriangulatedPoint;
use crate::{MvgError, Result};
use nalgebra::SMatrix;
use serde::{Deserialize, Serialize};
use vision_calibration_core::RunControl;
use vision_calibration_core::{
    CalibratedRig, CameraModel, Iso3, Mat3, PointCovariance, Pt2, Pt3, Real, Uncertainty,
    UncertaintyUnavailable, Vec3,
};
use vision_calibration_optim::RobustLoss;
use vision_calibration_optim::problem::{
    BlockId, Manifold, PixelNoise, Problem, SolveOptions, TargetPoint, Use, solve,
};
use vision_geometry::GeometryError;
use vision_geometry::triangulation::triangulate_point_linear;

/// One pixel of a point: seen by rig camera `camera` at capture `capture`,
/// with its noise.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RigObservation {
    /// Index into the capture poses (`rig_se3_world`) the call takes.
    pub capture: usize,
    /// Index of the rig camera that saw the point.
    pub camera: usize,
    /// The observed pixel.
    pub pixel: Pt2,
    /// The pixel's covariance, px²: [`PointCovariance::new`] for a general
    /// one, [`PointCovariance::isotropic`] for one σ, or
    /// [`CalibratedRig::reprojection_noise`] for the calibration's noise
    /// level. It is stated, not estimated: a point seen twice leaves no
    /// redundancy to estimate it from.
    pub noise: PointCovariance,
}

impl RigObservation {
    /// An observation of a point by rig camera `camera` at capture `capture`,
    /// with the covariance `noise` of its pixel.
    pub fn new(capture: usize, camera: usize, pixel: Pt2, noise: PointCovariance) -> Self {
        Self {
            capture,
            camera,
            pixel,
            noise,
        }
    }
}

/// The points of one [`triangulate`] call and their joint uncertainty.
#[derive(Debug, Clone, Serialize)]
#[non_exhaustive]
pub struct Triangulation {
    /// One point per track, in the world frame, refined, with its
    /// diagnostics.
    pub points: Vec<TriangulatedPoint>,
    /// The joint covariance of every refined point's coordinates, m², with
    /// the labels `p{i}.x`, `p{i}.y`, `p{i}.z` for track `i`: from the
    /// observations' stated noise plus the calibration's share. Or why it is
    /// unavailable; a rig without a covariance gives
    /// [`UncertaintyUnavailable::CalibrationWithoutCovariance`].
    pub uncertainty: Uncertainty,
}

impl Triangulation {
    /// Point `i`'s 3×3 covariance, m²; `None` when the uncertainty is
    /// unavailable, `i` is out of range or the point was not refined (see
    /// [`triangulate`]).
    pub fn point_covariance(&self, i: usize) -> Option<Mat3> {
        self.cross_covariance(i, i)
    }

    /// The 3×3 cross-covariance of points `i` and `j` (rows `i`, columns
    /// `j`), m²: the calibration's uncertainty is shared by every point of the
    /// call, so it correlates them. `None` as for
    /// [`point_covariance`](Self::point_covariance).
    pub fn cross_covariance(&self, i: usize, j: usize) -> Option<Mat3> {
        let axes = |p: usize| ["x", "y", "z"].map(|a| format!("p{p}.{a}"));
        sub_covariance(&self.uncertainty, axes(i), axes(j))
    }
}

/// The block of the joint covariance with rows `rows` and columns `cols`
/// (labels); `None` when the uncertainty is unavailable or a label is not in
/// it.
pub(crate) fn sub_covariance<const N: usize>(
    uncertainty: &Uncertainty,
    rows: [String; N],
    cols: [String; N],
) -> Option<SMatrix<Real, N, N>> {
    let c = uncertainty.covariance()?;
    let at = |labels: &[String; N]| -> Option<[usize; N]> {
        let mut index = [0; N];
        for (slot, label) in index.iter_mut().zip(labels) {
            *slot = c.index(label)?;
        }
        Some(index)
    };
    let (rows, cols) = (at(&rows)?, at(&cols)?);
    Some(SMatrix::from_fn(|r, s| c.covariance[rows[r]][cols[s]]))
}

/// Triangulate each track (the observations of one point) in the world frame,
/// refine the points and give their joint covariance.
///
/// `rig_se3_world[k]` is the pose of the rig at capture `k` (world → rig
/// frame), taken as exact. Camera `i` at capture `k` sees the world through
/// `rig.cam_se3_rig()[i] * rig_se3_world[k]`. Every pixel is unprojected
/// through its camera's own model, so distortion, fisheye and tilted sensors
/// need no separate undistortion. The linear (DLT) triangulation of the
/// bearings starts each point; all points are then refined together on the
/// reprojection error, each pixel weighted by its stated noise.
///
/// For one stereo capture pass `&[Iso3::identity()]`: the rig frame is the
/// world frame and the points come out in the rig frame.
///
/// The covariance has two sources: the observations' noise, and the rig's
/// calibration, which enters as consider parameters (its parameters stay at
/// their values and its covariance is propagated into the points'). Every
/// point shares the calibration's uncertainty, so the points are correlated
/// ([`Triangulation::cross_covariance`]) and are solved in one problem. The
/// covariance is dense over the 3N coordinates, so the cost grows as N³: for
/// thousands of points, call per group of points that need a joint
/// covariance.
///
/// A track whose linear estimate lies behind an observing camera, or does not
/// project in it, is not refined: it keeps the linear point, its diagnostics
/// say so (`in_front`, `reprojection_error`), and it has no covariance.
///
/// ```
/// use vision_calibration_core::{
///     BrownConrady5, CalibratedRig, FxFyCxCy, Iso3, PointCovariance, Pt3, RunControl, Uncertainty,
///     make_pinhole_camera, pinhole_camera_params,
/// };
/// use vision_mvg::triangulation::{RigObservation, triangulate};
///
/// // A stereo pair: camera 0 is the rig frame, camera 1 sits 10 cm to its right.
/// let k = FxFyCxCy { fx: 800.0, fy: 800.0, cx: 320.0, cy: 240.0 };
/// let camera = pinhole_camera_params(&make_pinhole_camera(k, BrownConrady5::default()));
/// let cam1_se3_rig = Iso3::translation(-0.1, 0.0, 0.0);
/// let rig = CalibratedRig::new(
///     vec![camera.clone(), camera],
///     vec![Iso3::identity(), cam1_se3_rig],
///     Uncertainty::default(),
/// )?;
///
/// // A point 2 m ahead, and the pixels the two cameras see it at, 0.3 px noise.
/// let point = Pt3::new(0.05, -0.02, 2.0);
/// let noise = PointCovariance::isotropic(0.3)?;
/// let pixel = |camera: usize| {
///     rig.model(camera)
///         .unwrap()
///         .project_point(&(rig.cam_se3_rig()[camera] * point))
///         .unwrap()
/// };
/// let track = vec![
///     RigObservation::new(0, 0, pixel(0), noise),
///     RigObservation::new(0, 1, pixel(1), noise),
/// ];
///
/// let found = triangulate(&rig, &[Iso3::identity()], &[track], &RunControl::default())?;
/// assert!((found.points[0].point - point).norm() < 1e-6);
/// assert!(found.points[0].in_front);
/// // This rig has no covariance, so neither do the points; one with a
/// // covariance gives `found.point_covariance(0)`, a 3×3 matrix in m².
/// assert!(found.point_covariance(0).is_none());
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// # Errors
///
/// All-or-nothing: a track with fewer than 2 observations, a camera or
/// capture index out of range, a pixel without a finite viewing ray or a
/// degenerate (zero-parallax) system is an error whose message names the
/// track by its index in `tracks`; so is a refinement that fails to run. A
/// refinement that stops short of convergence is not an error: the points are
/// returned as they stand and the uncertainty says
/// [`UncertaintyUnavailable::NotConverged`]. [`MvgError::Cancelled`] when
/// `control`'s token is cancelled; the refinement reports each iteration to
/// `control`'s observers, and an observer cannot change the result.
pub fn triangulate(
    rig: &CalibratedRig,
    rig_se3_world: &[Iso3],
    tracks: &[Vec<RigObservation>],
    control: &RunControl,
) -> Result<Triangulation> {
    let mut starts = Vec::with_capacity(tracks.len());
    for (i, track) in tracks.iter().enumerate() {
        let views = track_views(rig, rig_se3_world, track)
            .map_err(|e| with_context(&format!("track {i}"), e))?;
        let linear =
            triangulate_views(&views).map_err(|e| with_context(&format!("track {i}"), e))?;
        starts.push((views, linear));
    }

    // The problem: the calibration (considered), the capture poses (exact),
    // one free point per refinable track.
    let mut problem = Problem::new();
    let blocks = problem
        .add_calibrated_rig(rig, Use::Consider)
        .map_err(refine_error)?;
    let mut captures: Vec<Option<BlockId>> = vec![None; rig_se3_world.len()];
    let mut point_blocks: Vec<Option<BlockId>> = Vec::with_capacity(tracks.len());
    for (i, (track, (_, linear))) in tracks.iter().zip(&starts).enumerate() {
        if !(linear.in_front && linear.reprojection_error.is_some()) {
            point_blocks.push(None);
            continue;
        }
        let at = &linear.point;
        let block = problem
            .add_block(format!("p{i}"), &[at.x, at.y, at.z], Manifold::Euclidean)
            .map_err(refine_error)?;
        problem
            .set_labels(block, &["x", "y", "z"])
            .map_err(refine_error)?;
        for obs in track {
            let capture = match captures[obs.capture] {
                Some(id) => id,
                None => {
                    let id = problem
                        .add_pose(
                            format!("capture{}", obs.capture),
                            &rig_se3_world[obs.capture],
                        )
                        .map_err(refine_error)?;
                    problem.fix(id).map_err(refine_error)?;
                    captures[obs.capture] = Some(id);
                    id
                }
            };
            // The camera and chain exist: `track_views` checked the index.
            let camera = blocks.camera(obs.camera);
            let chain = blocks.chain(obs.camera, capture);
            let (Some(camera), Some(chain)) = (camera, chain) else {
                return Err(MvgError::numerical(format!(
                    "track {i}: camera {} missing from the problem",
                    obs.camera
                )));
            };
            problem
                .add_reprojection(
                    camera,
                    chain,
                    TargetPoint::Block(block),
                    obs.pixel,
                    PixelNoise::Stated(obs.noise),
                    RobustLoss::None,
                )
                .map_err(|e| refine_error_in(&format!("track {i}"), e))?;
        }
        point_blocks.push(Some(block));
    }

    if point_blocks.iter().all(Option::is_none) {
        let points = starts.into_iter().map(|(_, p)| p).collect();
        return Ok(Triangulation {
            points,
            uncertainty: Uncertainty::Unavailable {
                reason: UncertaintyUnavailable::NotComputed,
            },
        });
    }
    // `SolveOptions` is non-exhaustive: start from its default.
    let mut solve_options = SolveOptions::default();
    solve_options.control = control.clone();
    let solution = solve(&problem, &solve_options).map_err(refine_error)?;

    let points = starts
        .into_iter()
        .zip(&point_blocks)
        .map(|((views, linear), block)| {
            let refined = block
                .and_then(|b| solution.value(b))
                .map(|v| Pt3::new(v[0], v[1], v[2]));
            match refined {
                Some(point) => diagnose(&views, point),
                None => linear,
            }
        })
        .collect();
    Ok(Triangulation {
        points,
        uncertainty: solution.uncertainty,
    })
}

fn refine_error(e: vision_calibration_optim::Error) -> MvgError {
    if matches!(e, vision_calibration_optim::Error::Cancelled) {
        return MvgError::Cancelled;
    }
    MvgError::numerical(format!("refining the points: {e}"))
}

fn refine_error_in(context: &str, e: vision_calibration_optim::Error) -> MvgError {
    MvgError::numerical(format!("{context}: refining the points: {e}"))
}

/// The views of one track, with their bearings.
pub(crate) fn track_views<'a>(
    rig: &'a CalibratedRig,
    rig_se3_world: &[Iso3],
    track: &[RigObservation],
) -> Result<Vec<ViewRay<'a>>> {
    if track.len() < 2 {
        return Err(MvgError::invalid_input(format!(
            "has {} observation(s); triangulation needs at least 2",
            track.len()
        )));
    }
    let mut views = Vec::with_capacity(track.len());
    for (j, obs) in track.iter().enumerate() {
        let rig_pose = rig_se3_world.get(obs.capture).ok_or_else(|| {
            MvgError::invalid_input(format!(
                "observation {j}: capture {} out of range: {} rig poses given",
                obs.capture,
                rig_se3_world.len()
            ))
        })?;
        let model = camera_model(rig, obs.camera)
            .map_err(|e| with_context(&format!("observation {j}"), e))?;
        views.push(ViewRay {
            cam_se3_world: rig.cam_se3_rig()[obs.camera] * rig_pose,
            model,
            pixel: obs.pixel,
            bearing: bearing(model, &obs.pixel, j)?,
        });
    }
    Ok(views)
}

/// Prefix the message of an error with the item it concerns.
pub(crate) fn with_context(context: &str, err: MvgError) -> MvgError {
    match err {
        MvgError::InvalidInput { reason } => {
            MvgError::invalid_input(format!("{context}: {reason}"))
        }
        MvgError::Degenerate { reason }
        | MvgError::Geometry(GeometryError::Degenerate { reason }) => {
            MvgError::degenerate(format!("{context}: {reason}"))
        }
        other => MvgError::numerical(format!("{context}: {other}")),
    }
}

/// One observation of a point as the triangulator needs it.
pub(crate) struct ViewRay<'a> {
    /// World → camera pose of the observing camera.
    pub cam_se3_world: Iso3,
    /// The camera's model, for the reprojection diagnostic.
    pub model: &'a CameraModel,
    /// The observed pixel.
    pub pixel: Pt2,
    /// Its unit bearing in the camera frame.
    pub bearing: Vec3,
}

/// Linear triangulation of one point from its views, with diagnostics.
pub(crate) fn triangulate_views(views: &[ViewRay<'_>]) -> Result<TriangulatedPoint> {
    let poses: Vec<Iso3> = views.iter().map(|v| v.cam_se3_world).collect();
    let bearings: Vec<Vec3> = views.iter().map(|v| v.bearing).collect();
    let point = triangulate_point_linear(&poses, &bearings)?;
    Ok(diagnose(views, point))
}

/// The diagnostics of `point` against the views that observed it.
fn diagnose(views: &[ViewRay<'_>], point: Pt3) -> TriangulatedPoint {
    let mut in_front = true;
    let mut sq_sum = 0.0;
    let mut projects = true;
    for v in views {
        let p_cam = v.cam_se3_world * point;
        in_front &= v.bearing.dot(&p_cam.coords) > 0.0;
        match v.model.project_point(&p_cam) {
            Some(px) => sq_sum += (px - v.pixel).norm_squared(),
            None => projects = false,
        }
    }
    let reprojection_error = projects.then(|| (sq_sum / views.len() as Real).sqrt());
    let poses: Vec<Iso3> = views.iter().map(|v| v.cam_se3_world).collect();

    TriangulatedPoint {
        point,
        reprojection_error,
        parallax_deg: max_parallax_deg(&poses, &point),
        in_front,
    }
}

/// The largest angle, in degrees, between two rays from camera centres to the
/// point. A camera centred on the point has no ray and is skipped.
fn max_parallax_deg(cam_se3_world: &[Iso3], point: &Pt3) -> Real {
    let rays: Vec<Vec3> = cam_se3_world
        .iter()
        .filter_map(|pose| {
            let centre = pose.inverse_transform_point(&Pt3::origin());
            let d = point - centre;
            let n = d.norm();
            (n > 0.0).then(|| d / n)
        })
        .collect();
    let mut widest = 0.0_f64;
    for (i, a) in rays.iter().enumerate() {
        for b in &rays[i + 1..] {
            widest = widest.max(a.dot(b).clamp(-1.0, 1.0).acos().to_degrees());
        }
    }
    widest
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{Lcg, fisheye_params, iso, pinhole_params, pixel_of};
    use vision_calibration_core::Uncertainty;

    fn noise() -> PointCovariance {
        PointCovariance::isotropic(0.5).unwrap()
    }

    fn stereo_rig(k1: f64) -> CalibratedRig {
        CalibratedRig::new(
            vec![pinhole_params(k1), pinhole_params(k1)],
            vec![
                Iso3::identity(),
                iso(0.0, 0.05, 0.0, Vec3::new(-0.12, 0.0, 0.01)),
            ],
            Uncertainty::default(),
        )
        .unwrap()
    }

    /// The track of `world` seen by every camera at every capture.
    fn track(rig: &CalibratedRig, rig_se3_world: &[Iso3], world: &Pt3) -> Vec<RigObservation> {
        let mut obs = Vec::new();
        for (k, rig_pose) in rig_se3_world.iter().enumerate() {
            for (c, cam_pose) in rig.cam_se3_rig().iter().enumerate() {
                let p_cam = (cam_pose * rig_pose * world).coords;
                obs.push(RigObservation::new(k, c, pixel_of(rig, c, &p_cam), noise()));
            }
        }
        obs
    }

    #[test]
    fn stereo_capture_is_triangulated_in_the_rig_frame() {
        let rig = stereo_rig(0.0);
        let world = [Pt3::new(0.05, -0.02, 2.0), Pt3::new(-0.3, 0.2, 1.2)];
        let tracks: Vec<_> = world
            .iter()
            .map(|p| track(&rig, &[Iso3::identity()], p))
            .collect();
        let found =
            triangulate(&rig, &[Iso3::identity()], &tracks, &RunControl::default()).unwrap();
        assert_eq!(found.points.len(), 2);
        for (f, p) in found.points.iter().zip(&world) {
            assert!((f.point - p).norm() < 1e-9);
            assert!(f.in_front);
            assert!(f.reprojection_error.unwrap() < 1e-8);
            assert!(f.parallax_deg > 1.0);
        }
    }

    /// Several captures of a moving rig, through lens distortion, from raw
    /// distorted pixels.
    #[test]
    fn moving_rig_with_distortion_recovers_the_point() {
        let rig = stereo_rig(-0.2);
        let captures = [
            iso(0.0, 0.0, 0.0, Vec3::new(0.0, 0.0, 0.0)),
            iso(0.02, -0.03, 0.01, Vec3::new(-0.3, 0.05, 0.02)),
            iso(-0.05, 0.04, 0.0, Vec3::new(0.25, -0.1, 0.1)),
        ];
        let world = Pt3::new(0.15, -0.1, 2.5);
        let found = triangulate(
            &rig,
            &captures,
            &[track(&rig, &captures, &world)],
            &RunControl::default(),
        )
        .unwrap();
        assert!((found.points[0].point - world).norm() < 1e-9);
        assert!(found.points[0].reprojection_error.unwrap() < 1e-7);
    }

    /// With pixel noise the diagnostics report it: the reprojection RMS is of
    /// the order of the noise and the point stays close to the truth.
    #[test]
    fn noisy_pixels_show_in_the_reprojection_error() {
        let rig = stereo_rig(-0.1);
        let captures = [
            Iso3::identity(),
            iso(0.0, 0.1, 0.0, Vec3::new(-0.4, 0.0, 0.05)),
        ];
        let mut rng = Lcg(5);
        let world = Pt3::new(0.1, 0.05, 3.0);
        let mut noisy = track(&rig, &captures, &world);
        for o in &mut noisy {
            o.pixel.x += 0.3 * rng.next();
            o.pixel.y += 0.3 * rng.next();
        }
        let found = triangulate(&rig, &captures, &[noisy], &RunControl::default()).unwrap();
        let err = found.points[0].reprojection_error.unwrap();
        assert!(err > 0.01 && err < 0.5, "reprojection {err}");
        assert!((found.points[0].point - world).norm() < 0.1);
    }

    /// Diverging rays put the point behind both cameras: not in front, and a
    /// pinhole does not project it.
    #[test]
    fn point_behind_the_cameras_is_flagged() {
        let rig = CalibratedRig::single(pinhole_params(0.0), Uncertainty::default()).unwrap();
        let captures = [
            Iso3::identity(),
            iso(0.0, 0.0, 0.0, Vec3::new(-1.0, 0.0, 0.0)),
        ];
        // Camera 0 (at the origin) sees the point left of centre; camera 1,
        // 1 m to its right, sees it right of centre: the rays diverge and
        // meet behind both cameras.
        let track = vec![
            RigObservation::new(0, 0, Pt2::new(600.0, 480.0), noise()),
            RigObservation::new(1, 0, Pt2::new(720.0, 480.0), noise()),
        ];
        let found = triangulate(&rig, &captures, &[track], &RunControl::default()).unwrap();
        assert!(!found.points[0].in_front);
        assert!(found.points[0].reprojection_error.is_none());
    }

    /// A fisheye rig sees points beyond 90° from the axis: the bearing
    /// triangulation recovers them, which normalized coordinates cannot.
    #[test]
    fn fisheye_points_beyond_90_degrees_triangulate() {
        let rig = CalibratedRig::new(
            vec![fisheye_params(), fisheye_params()],
            vec![
                Iso3::identity(),
                iso(0.0, 0.1, 0.0, Vec3::new(-0.2, 0.0, 0.05)),
            ],
            Uncertainty::default(),
        )
        .unwrap();
        let world = [
            Pt3::new(1.5, 0.2, -0.5),  // ~109° off the first axis
            Pt3::new(-2.0, 0.5, -1.0), // ~154°
            Pt3::new(0.3, -0.2, 2.0),
            Pt3::new(1.0, 1.0, -0.3),
        ];
        assert!(world.iter().filter(|p| p.z < 0.0).count() >= 3);
        let tracks: Vec<_> = world
            .iter()
            .map(|p| track(&rig, &[Iso3::identity()], p))
            .collect();
        let found =
            triangulate(&rig, &[Iso3::identity()], &tracks, &RunControl::default()).unwrap();
        for (f, p) in found.points.iter().zip(&world) {
            assert!((f.point - p).norm() < 1e-9, "{:?} vs {p:?}", f.point);
            assert!(f.in_front);
            assert!(f.reprojection_error.unwrap() < 1e-7);
        }
    }

    #[test]
    fn errors_name_the_track() {
        let rig = stereo_rig(0.0);
        let captures = [Iso3::identity()];
        let good = track(&rig, &captures, &Pt3::new(0.0, 0.0, 2.0));
        let message = |tracks: Vec<Vec<RigObservation>>| -> String {
            triangulate(&rig, &captures, &tracks, &RunControl::default())
                .unwrap_err()
                .to_string()
        };

        // Fewer than two observations.
        let m = message(vec![good.clone(), good[..1].to_vec()]);
        assert!(m.contains("track 1") && m.contains("at least 2"), "{m}");

        // Camera out of range.
        let mut bad = good.clone();
        bad[1].camera = 5;
        let m = message(vec![bad]);
        assert!(m.contains("track 0") && m.contains("camera 5"), "{m}");

        // Capture out of range.
        let mut bad = good.clone();
        bad[0].capture = 3;
        let m = message(vec![good.clone(), good.clone(), bad]);
        assert!(m.contains("track 2") && m.contains("capture 3"), "{m}");

        // Zero baseline: the same camera at the same capture twice.
        let same = vec![good[0], good[0]];
        let m = message(vec![same]);
        assert!(m.contains("track 0"), "{m}");
        assert!(matches!(
            triangulate(
                &rig,
                &captures,
                &[vec![good[0], good[0]]],
                &RunControl::default()
            ),
            Err(MvgError::Degenerate { .. })
        ));
    }
}
