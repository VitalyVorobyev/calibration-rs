//! Triangulation from a calibration, with quality diagnostics.
//!
//! Pixels become bearings through each rig camera's own model, the linear
//! (DLT) triangulator of `vision-geometry` solves for the point, and the
//! diagnostics report the reprojection error in pixels, the parallax and
//! whether the point lies ahead of every observing camera.

use crate::bearing::{bearing, camera_model};
use crate::types::TriangulatedPoint;
use crate::{MvgError, Result};
use vision_calibration_core::{CalibratedRig, CameraModel, Iso3, Pt2, Pt3, Real, Vec3};
use vision_geometry::GeometryError;
use vision_geometry::triangulation::triangulate_point_linear;

/// One pixel of a point: seen by rig camera `camera` at capture `capture`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RigObservation {
    /// Index into the capture poses (`rig_se3_world`) the call takes.
    pub capture: usize,
    /// Index of the rig camera that saw the point.
    pub camera: usize,
    /// The observed pixel.
    pub pixel: Pt2,
}

impl RigObservation {
    /// An observation of a point by rig camera `camera` at capture `capture`.
    pub fn new(capture: usize, camera: usize, pixel: Pt2) -> Self {
        Self {
            capture,
            camera,
            pixel,
        }
    }
}

/// Triangulate each track (the observations of one point) in the world frame.
///
/// `rig_se3_world[k]` is the pose of the rig at capture `k` (world → rig
/// frame). Camera `i` at capture `k` sees the world through
/// `rig.cam_se3_rig()[i] * rig_se3_world[k]`. Every pixel is unprojected
/// through its camera's own model, so distortion, fisheye and tilted sensors
/// need no separate undistortion, and the estimate is the linear (DLT)
/// triangulation of the resulting bearings.
///
/// For one stereo capture pass `&[Iso3::identity()]`: the rig frame is the
/// world frame and the points come out in the rig frame.
///
/// ```
/// use vision_calibration_core::{
///     BrownConrady5, CalibratedRig, FxFyCxCy, Iso3, Pt2, Pt3, Uncertainty, Vec3,
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
/// // A point 2 m ahead, and the pixels the two cameras see it at.
/// let point = Pt3::new(0.05, -0.02, 2.0);
/// let pixel = |camera: usize| {
///     rig.model(camera)
///         .unwrap()
///         .project_point(&(rig.cam_se3_rig()[camera] * point))
///         .unwrap()
/// };
/// let track = vec![
///     RigObservation::new(0, 0, pixel(0)),
///     RigObservation::new(0, 1, pixel(1)),
/// ];
///
/// let found = triangulate(&rig, &[Iso3::identity()], &[track])?;
/// assert!((found[0].point - point).norm() < 1e-9);
/// assert!(found[0].in_front);
/// # Ok::<(), vision_mvg::MvgError>(())
/// ```
///
/// # Errors
///
/// All-or-nothing: a track with fewer than 2 observations, a camera or
/// capture index out of range, a pixel without a finite viewing ray or a
/// degenerate (zero-parallax) system is an error whose message names the
/// track by its index in `tracks`.
pub fn triangulate(
    rig: &CalibratedRig,
    rig_se3_world: &[Iso3],
    tracks: &[Vec<RigObservation>],
) -> Result<Vec<TriangulatedPoint>> {
    tracks
        .iter()
        .enumerate()
        .map(|(i, track)| {
            triangulate_track(rig, rig_se3_world, track)
                .map_err(|e| with_context(&format!("track {i}"), e))
        })
        .collect()
}

fn triangulate_track(
    rig: &CalibratedRig,
    rig_se3_world: &[Iso3],
    track: &[RigObservation],
) -> Result<TriangulatedPoint> {
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
    triangulate_views(&views)
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

    Ok(TriangulatedPoint {
        point,
        reprojection_error,
        parallax_deg: max_parallax_deg(&poses, &point),
        in_front,
    })
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
                obs.push(RigObservation::new(k, c, pixel_of(rig, c, &p_cam)));
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
        let found = triangulate(&rig, &[Iso3::identity()], &tracks).unwrap();
        assert_eq!(found.len(), 2);
        for (f, p) in found.iter().zip(&world) {
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
        let found = triangulate(&rig, &captures, &[track(&rig, &captures, &world)]).unwrap();
        assert!((found[0].point - world).norm() < 1e-9);
        assert!(found[0].reprojection_error.unwrap() < 1e-7);
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
        let found = triangulate(&rig, &captures, &[noisy]).unwrap();
        let err = found[0].reprojection_error.unwrap();
        assert!(err > 0.01 && err < 0.5, "reprojection {err}");
        assert!((found[0].point - world).norm() < 0.1);
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
            RigObservation::new(0, 0, Pt2::new(600.0, 480.0)),
            RigObservation::new(1, 0, Pt2::new(720.0, 480.0)),
        ];
        let found = triangulate(&rig, &captures, &[track]).unwrap();
        assert!(!found[0].in_front);
        assert!(found[0].reprojection_error.is_none());
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
        let found = triangulate(&rig, &[Iso3::identity()], &tracks).unwrap();
        for (f, p) in found.iter().zip(&world) {
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
            triangulate(&rig, &captures, &tracks)
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
            triangulate(&rig, &captures, &[vec![good[0], good[0]]]),
            Err(MvgError::Degenerate { .. })
        ));
    }
}
