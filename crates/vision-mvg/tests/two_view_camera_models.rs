//! Two-view geometry through camera models: raw pixels of a distorted pinhole
//! and of a fisheye rig recover the truth, with no separate undistortion.
//!
//! Complements `two_view_matrix.rs` (a distortion-free pinhole across baseline
//! regimes and noise levels): here the camera model carries the lens.

#![allow(missing_docs)]

use nalgebra::{Rotation3, Translation3, UnitQuaternion};
use vision_calibration_core::synthetic::noise::UniformPixelNoise;
use vision_calibration_core::{
    BrownConrady5, CalibratedRig, CameraParams, DistortionParams, FxFyCxCy, Iso3, KannalaBrandt4,
    PointCovariance, ProjectionParams, Pt2, Pt3, SensorParams, Uncertainty, Vec3,
    make_pinhole_camera, pinhole_camera_params,
};
use vision_mvg::triangulation::{RigObservation, triangulate};
use vision_mvg::{PixelMatch, recover_relative_pose};

/// A match of two pixels, each detected to a stated 0.5 px.
fn matched(pt1: Pt2, pt2: Pt2) -> PixelMatch {
    let noise = PointCovariance::isotropic(0.5).unwrap();
    PixelMatch::new(pt1, noise, pt2, noise)
}

fn pose(rx: f64, ry: f64, rz: f64, t: Vec3) -> Iso3 {
    Iso3::from_parts(
        Translation3::from(t),
        UnitQuaternion::from_rotation_matrix(&Rotation3::from_euler_angles(rx, ry, rz)),
    )
}

/// An 800 px pinhole with strong barrel distortion `k1 = -0.2`.
fn distorted_pinhole() -> CameraParams {
    let k = FxFyCxCy {
        fx: 800.0,
        fy: 790.0,
        cx: 640.0,
        cy: 360.0,
    };
    let distortion = BrownConrady5 {
        k1: -0.2,
        k2: 0.03,
        k3: 0.0,
        p1: 0.001,
        p2: -0.001,
        // Enough fixed-point iterations to undistort to machine precision.
        iters: 60,
    };
    pinhole_camera_params(&make_pinhole_camera(k, distortion))
}

/// A 300 px/rad equidistant fisheye with Kannala–Brandt distortion.
fn fisheye() -> CameraParams {
    CameraParams {
        projection: ProjectionParams::Equidistant,
        distortion: DistortionParams::KannalaBrandt4 {
            params: KannalaBrandt4 {
                k1: 0.02,
                k2: -0.002,
                k3: 0.0,
                k4: 0.0,
                iters: 0,
            },
        },
        sensor: SensorParams::Identity,
        intrinsics: FxFyCxCy {
            fx: 300.0,
            fy: 300.0,
            cx: 640.0,
            cy: 480.0,
        },
    }
}

fn stereo(camera: CameraParams, cam1_se3_rig: Iso3) -> CalibratedRig {
    CalibratedRig::new(
        vec![camera.clone(), camera],
        vec![Iso3::identity(), cam1_se3_rig],
        Uncertainty::default(),
    )
    .unwrap()
}

fn pixel(rig: &CalibratedRig, camera: usize, pw: &Pt3) -> Pt2 {
    rig.model(camera)
        .unwrap()
        .project_point(&(rig.cam_se3_rig()[camera] * pw))
        .expect("point projects")
}

/// A non-coplanar cloud spread across the shared frustum (the cloud of the
/// matrix test): depth varies non-linearly so the points lie near no plane.
fn cloud_ahead() -> Vec<Pt3> {
    let mut world = Vec::new();
    for iy in 0..6 {
        for ix in 0..7 {
            let x = -1.2 + 0.4 * ix as f64;
            let y = -0.9 + 0.36 * iy as f64;
            let z = 3.0
                + 0.6 * ((ix as f64) * 0.7).sin()
                + 0.5 * ((iy as f64) * 1.1).cos()
                + 0.12 * (ix + iy) as f64;
            world.push(Pt3::new(x, y, z));
        }
    }
    world
}

/// Rotation error (deg) and translation-direction error (deg).
///
/// Both angles come from `atan2`, which stays exact near zero; an `acos` of
/// a cosine cannot resolve angles below about 1e-6° (one ulp off 1).
fn pose_errors(est: &Iso3, gt: &Iso3) -> (f64, f64) {
    let q = (est.rotation.inverse() * gt.rotation).into_inner();
    let rot = 2.0 * q.imag().norm().atan2(q.scalar().abs());
    let (a, b) = (est.translation.vector, gt.translation.vector);
    let tdir = a.cross(&b).norm().atan2(a.dot(&b));
    (rot.to_degrees(), tdir.to_degrees())
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

#[test]
fn distorted_pinhole_noiseless_recovers_pose_and_points() {
    let gt = pose(0.05, -0.08, 0.03, Vec3::new(0.8, 0.05, 0.10));
    let rig = stereo(distorted_pinhole(), gt);
    let world = cloud_ahead();
    let px_noise = PointCovariance::isotropic(0.5).unwrap();

    let matches: Vec<_> = world
        .iter()
        .map(|p| matched(pixel(&rig, 0, p), pixel(&rig, 1, p)))
        .collect();
    let rel = recover_relative_pose(&rig, [0, 1], &matches).unwrap();
    let (rot, tdir) = pose_errors(&rel.second_se3_first, &gt);
    assert!(rot < 1e-7, "rotation {rot} deg");
    assert!(tdir < 1e-7, "translation direction {tdir} deg");

    // Metric triangulation with the true rig: relative error ~1e-9.
    let tracks: Vec<_> = world
        .iter()
        .map(|p| {
            vec![
                RigObservation::new(0, 0, pixel(&rig, 0, p), px_noise),
                RigObservation::new(0, 1, pixel(&rig, 1, p), px_noise),
            ]
        })
        .collect();
    let found = triangulate(&rig, &[Iso3::identity()], &tracks).unwrap();
    for (f, p) in found.points.iter().zip(&world) {
        let rel_err = (f.point - p).norm() / p.coords.norm();
        assert!(rel_err < 1e-9, "relative error {rel_err}");
        assert!(f.reprojection_error.unwrap() < 1e-6);
    }
}

#[test]
fn distorted_pinhole_with_pixel_noise_stays_within_the_noise_floor() {
    let gt = pose(0.05, -0.08, 0.03, Vec3::new(0.8, 0.05, 0.10));
    let rig = stereo(distorted_pinhole(), gt);
    let world = cloud_ahead();
    let px_noise = PointCovariance::isotropic(0.5).unwrap();
    let noise = UniformPixelNoise {
        seed: 0xB1AB,
        max_abs_px: 0.3,
    };

    let mut matches = Vec::new();
    let mut tracks = Vec::new();
    for (j, p) in world.iter().enumerate() {
        let (n0, n1) = (noise.sample(0, j), noise.sample(1, j));
        let q0 = pixel(&rig, 0, p) + n0;
        let q1 = pixel(&rig, 1, p) + n1;
        matches.push(matched(q0, q1));
        tracks.push(vec![
            RigObservation::new(0, 0, q0, px_noise),
            RigObservation::new(0, 1, q1, px_noise),
        ]);
    }
    let rel = recover_relative_pose(&rig, [0, 1], &matches).unwrap();
    let (rot, tdir) = pose_errors(&rel.second_se3_first, &gt);
    // The wide-baseline cells of the matrix test bound rotation by
    // 0.5 deg/px and translation direction by 1.0 deg/px.
    assert!(rot < 0.5 * 0.3, "rotation {rot} deg");
    assert!(tdir < 1.0 * 0.3, "translation direction {tdir} deg");

    let found = triangulate(&rig, &[Iso3::identity()], &tracks).unwrap();
    let err3d = median(
        found
            .points
            .iter()
            .zip(&world)
            .map(|(f, p)| (f.point - p).norm())
            .collect(),
    );
    assert!(err3d < 0.04 * 0.3, "median 3D error {err3d}");
    let reproj = median(
        found
            .points
            .iter()
            .map(|f| f.reprojection_error.unwrap())
            .collect(),
    );
    assert!(reproj < 0.3 + 0.02, "median reprojection {reproj} px");
}

/// Points in every direction around the first fisheye camera, a third of them
/// more than 90° from its axis (`z < 0`).
fn cloud_all_around() -> Vec<Pt3> {
    (0..30)
        .map(|i| {
            let a = f64::from(i);
            let theta = 0.25 + 0.07 * a; // 14° .. 2.3 rad (132°)
            let phi = 2.1 * a;
            let d = 2.0 + 0.25 * (a % 5.0);
            Pt3::new(
                d * theta.sin() * phi.cos(),
                d * theta.sin() * phi.sin(),
                d * theta.cos(),
            )
        })
        .collect()
}

#[test]
fn fisheye_rig_recovers_points_and_pose_beyond_90_degrees() {
    let cam1_se3_rig = pose(0.02, -0.12, 0.03, Vec3::new(-0.35, 0.04, 0.06));
    let rig = stereo(fisheye(), cam1_se3_rig);
    let world = cloud_all_around();
    let px_noise = PointCovariance::isotropic(0.5).unwrap();
    let behind = world.iter().filter(|p| p.z < 0.0).count();
    assert!(
        behind >= 8,
        "{behind} points behind the first camera's plane"
    );

    // Triangulation with the true rig: relative error ~1e-9, every point
    // ahead along its bearings, including those with z < 0.
    let tracks: Vec<_> = world
        .iter()
        .map(|p| {
            vec![
                RigObservation::new(0, 0, pixel(&rig, 0, p), px_noise),
                RigObservation::new(0, 1, pixel(&rig, 1, p), px_noise),
            ]
        })
        .collect();
    let found = triangulate(&rig, &[Iso3::identity()], &tracks).unwrap();
    for (f, p) in found.points.iter().zip(&world) {
        let rel_err = (f.point - p).norm() / p.coords.norm();
        assert!(rel_err < 1e-9, "{p:?}: relative error {rel_err}");
        assert!(f.in_front, "{p:?}");
        assert!(f.reprojection_error.unwrap() < 1e-6);
    }

    // Relative pose from the same pixels: rotation and translation direction.
    let matches: Vec<_> = world
        .iter()
        .map(|p| matched(pixel(&rig, 0, p), pixel(&rig, 1, p)))
        .collect();
    let rel = recover_relative_pose(&rig, [0, 1], &matches).unwrap();
    let (rot, tdir) = pose_errors(&rel.second_se3_first, &cam1_se3_rig);
    assert!(rot < 1e-7, "rotation {rot} deg");
    assert!(tdir < 1e-7, "translation direction {tdir} deg");
    // The points come out in the first camera's frame at the unit baseline.
    let scale = 1.0 / cam1_se3_rig.translation.vector.norm();
    for (tp, p) in rel.points.iter().zip(&world) {
        assert!((tp.point.coords - p.coords * scale).norm() < 1e-6 * p.coords.norm() * scale);
        assert!(tp.in_front);
    }
}
