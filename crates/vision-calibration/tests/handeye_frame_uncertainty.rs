//! Laser and triangulated points in a robot's base frame carry the
//! hand-eye's and the robot pose's uncertainty.
//!
//! Each trial calibrates a two-camera rig on a robot arm jointly with each
//! camera's laser plane (cameras, extrinsics, hand-eye, the board's pose and
//! the planes, in one solve) from freshly noisy board and laser views with
//! the public rig hand-eye laserline workflow. The robot then moves to a
//! station it never took during the calibration and reports its pose with a
//! fresh error of the stated σ; eight held-out points of the true laser plane
//! are measured from freshly noisy laser pixels in each camera, and eight
//! held-out points ahead of the rig are triangulated from both cameras, all
//! in the base frame. Over the trials the scatter of every coordinate
//! matches the σ that `triangulate_laser` and `triangulate` report: the
//! pixels, the cameras and planes, the hand-eye (correlated with them) and
//! the robot pose. The σ without the robot pose is short of the scatter, and
//! the σ without the cross terms between the hand-eye and the rest (as a
//! frame change applied after the measurement would state it) is far above
//! it: the calibration pins the board's pose in the base frame, so the
//! hand-eye and the planes it was estimated with err together.

#![allow(missing_docs)]

mod common;

use common::laser::{board, laser_intrinsics, laser_pixels_for_view, laser_truth, poses};
use common::{PIXEL_SIGMA, gauss};
use std::time::Instant;
use vision_calibration::common::{CalibratedCameras, RigLaserlineView, RobotPoseMeta};
use vision_calibration::control::RunControl;
use vision_calibration::core::{
    BrownConrady5, CalibratedRig, CorrespondenceView, Iso3, PinholeCamera, PointCovariance, Pt2,
    Pt3, RobotFrame, RobotPose, Uncertainty, Vec3, make_pinhole_camera,
};
use vision_calibration::mvg::laser::{LaserObservation, triangulate_laser};
use vision_calibration::mvg::triangulation::{RigObservation, triangulate};
use vision_calibration::mvg::types::Captures;
use vision_calibration::optim::HandeyeMode;
use vision_calibration::rig_handeye_laserline::{
    RigHandeyeLaserlineConfig, RigHandeyeLaserlineInput, RigHandeyeLaserlineProblem,
};
use vision_calibration::session::CalibrationSession;
use vision_calibration::synthetic::noise::UniformPixelNoise;
use vision_calibration::synthetic::poses::{
    BoardPoseSpec, centered_board_poses, make_iso, rig_layout,
};

/// A scatter ratio from `n` trials has a relative error of `1/sqrt(2n)`: 300
/// trials resolve it to 4 %, so the [0.85, 1.15] band is 3.7 of them.
const TRIALS: u64 = 300;
/// The pixel noise of the held-out pixels, stated and true.
const MEASURE_SIGMA: f64 = 0.1;
/// The robot's pose error at the measurement, stated and true: 0.1 mrad
/// about each gripper axis, 50 µm along each.
const ROBOT_ROT_SIGMA: f64 = 1e-4;
const ROBOT_TRANS_SIGMA: f64 = 5e-5;
const POINTS: usize = 8;
const CAMERAS: usize = 2;

/// The rig: camera 1 10 cm to the right of camera 0, toed in by 3°.
fn cam_se3_rig() -> Vec<Iso3> {
    rig_layout(CAMERAS, 0.10, 0.05)
}

fn cameras() -> Vec<PinholeCamera> {
    vec![make_pinhole_camera(laser_intrinsics(), BrownConrady5::default()); CAMERAS]
}

/// The hand-eye `gripper_se3_rig` and the board's `base_se3_target`.
fn handeye() -> Iso3 {
    make_iso((0.1, -0.05, 0.15), (0.05, 0.03, 0.15))
}

fn base_se3_target() -> Iso3 {
    make_iso((0.0, 0.2, 0.0), (0.4, -0.1, 0.2))
}

/// The board's pose in the rig at the ten board stations (0.45 to 0.6 m
/// ahead, tilted and rolled, for a well-conditioned hand-eye) and the six
/// laser stations (about 0.5 m ahead).
fn calibration_poses() -> Vec<Iso3> {
    let spec = BoardPoseSpec::new;
    let specs = [
        spec(0.00, 0.00, 0.00, 0.00),
        spec(0.35, -0.10, 0.50, -0.05),
        spec(-0.30, 0.20, -0.60, 0.04),
        spec(0.10, 0.35, 0.80, 0.08),
        spec(-0.20, -0.33, -0.30, -0.03),
        spec(0.25, 0.25, 1.20, 0.10),
        spec(-0.35, -0.05, 0.90, 0.06),
        spec(0.05, -0.30, -1.00, -0.04),
        spec(0.30, 0.10, -0.80, 0.02),
        spec(-0.15, 0.30, 0.20, 0.09),
    ];
    centered_board_poses(
        &specs,
        (common::laser::BOARD_CX, common::laser::BOARD_CY),
        0.5,
    )
}

/// The robot station that puts the board at `rig_se3_target`: `T_B_G = T_B_T
/// · T_R_T⁻¹ · T_G_R⁻¹`.
fn station(rig_se3_target: &Iso3) -> Iso3 {
    base_se3_target() * rig_se3_target.inverse() * handeye().inverse()
}

/// The board corners of `camera`, with Gaussian noise of σ [`PIXEL_SIGMA`].
fn corners(
    camera: &PinholeCamera,
    cam_se3_target: &Iso3,
    key: impl Fn(usize, u64) -> u64,
) -> CorrespondenceView {
    let points = board();
    let pixels = points
        .iter()
        .enumerate()
        .map(|(i, p)| {
            let q = camera
                .project_point(&cam_se3_target.transform_point(p))
                .unwrap();
            Pt2::new(
                q.x + PIXEL_SIGMA * gauss(key(i, 0)),
                q.y + PIXEL_SIGMA * gauss(key(i, 1)),
            )
        })
        .collect();
    CorrespondenceView::new(points, pixels).unwrap()
}

/// The joint calibration of trial `seed`: ten board views, then six board
/// views with the laser stripe on the board.
fn calibrated_rig(seed: u64) -> CalibratedRig {
    let cameras = cameras();
    let layout = cam_se3_rig();
    let exact = UniformPixelNoise::default();
    let board_poses = calibration_poses();
    let laser_poses = poses(0.5);
    let views = board_poses
        .iter()
        .map(|t| (t, false))
        .chain(laser_poses.iter().map(|t| (t, true)))
        .enumerate()
        .map(|(v, (target, laser))| {
            let mut obs = RigLaserlineView {
                cameras: Vec::new(),
                laser_pixels: Vec::new(),
            };
            for (c, camera) in cameras.iter().enumerate() {
                let cam_se3_target = layout[c] * target;
                let key = |i: usize, axis: u64| {
                    (seed << 32) + 2 * (c * 1_000_000 + v * 10_000 + i) as u64 + axis
                };
                obs.cameras
                    .push(Some(corners(camera, &cam_se3_target, key)));
                obs.laser_pixels.push(laser.then(|| {
                    let plane = laser_truth().transform_by(&layout[c]);
                    laser_pixels_for_view(camera, &cam_se3_target, &plane, v, &exact)
                        .into_iter()
                        .enumerate()
                        .map(|(i, px)| {
                            Pt2::new(
                                px.x + PIXEL_SIGMA * gauss(key(5_000 + i, 0)),
                                px.y + PIXEL_SIGMA * gauss(key(5_000 + i, 1)),
                            )
                        })
                        .collect()
                }));
            }
            vision_calibration::rig_handeye_laserline::RigHandeyeLaserlineView {
                obs,
                meta: RobotPoseMeta {
                    base_se3_gripper: station(target),
                },
            }
        })
        .collect();
    let mut config = RigHandeyeLaserlineConfig::default();
    config.handeye.handeye_init.handeye_mode = HandeyeMode::EyeInHand;
    // The synthetic robot poses are exact.
    config.handeye.handeye_optimize.robot_poses.refine = false;
    config.joint_optimize.robot_poses.refine = false;
    let mut session = CalibrationSession::<RigHandeyeLaserlineProblem>::new();
    session.set_config(config).unwrap();
    session
        .set_input(RigHandeyeLaserlineInput {
            views,
            num_cameras: CAMERAS,
        })
        .unwrap();
    vision_calibration::rig_handeye_laserline::run_calibration(&mut session).unwrap();
    session.export().unwrap().calibrated_rig().unwrap()
}

/// Points of the true laser plane, 0.4 to 0.6 m ahead, in the rig frame.
fn laser_points() -> Vec<Vec3> {
    let plane = laser_truth();
    let n = plane.normal().into_inner();
    (0..POINTS)
        .map(|i| {
            let x = -0.13 + 0.04 * i as f64;
            let y = -0.12 + 0.034 * ((3 * i) % POINTS) as f64;
            let z = (-plane.distance() - n.x * x - n.y * y) / n.z;
            Vec3::new(x, y, z)
        })
        .collect()
}

/// Points 0.5 to 0.8 m ahead of the rig, seen by both cameras.
fn stereo_points() -> Vec<Vec3> {
    (0..POINTS)
        .map(|i| {
            let a = i as f64;
            Vec3::new(-0.12 + 0.035 * a, 0.1 * (1.3 * a).sin(), 0.5 + 0.04 * a)
        })
        .collect()
}

/// The measurement station, which the calibration never took: the board's
/// first laser pose, moved 5 cm and turned 0.1 rad.
fn measurement_station() -> Iso3 {
    station(&poses(0.5)[0]) * make_iso((0.1, -0.05, 0.08), (0.05, -0.02, 0.03))
}

/// `rig` with its covariance entries kept where `keep(label_i, label_j)`.
fn with_covariance(rig: &CalibratedRig, keep: impl Fn(&str, &str) -> bool) -> CalibratedRig {
    let mut covariance = rig.uncertainty().covariance().expect("covariance").clone();
    let labels = covariance.labels.clone();
    for (i, row) in covariance.covariance.iter_mut().enumerate() {
        for (j, v) in row.iter_mut().enumerate() {
            if !keep(&labels[i], &labels[j]) {
                *v = if i == j { 1e-24 } else { 0.0 };
            }
        }
    }
    CalibratedRig::new(
        rig.cameras().to_vec(),
        rig.cam_se3_rig().to_vec(),
        Uncertainty::Available(covariance),
    )
    .unwrap()
    .with_laser_planes((0..CAMERAS).map(|c| rig.laser_plane(c).cloned()).collect())
    .unwrap()
    .with_handeye(*rig.handeye().unwrap())
    .unwrap()
}

/// The stated covariance of every point of a source mix, and the points.
struct Measured {
    laser: Vec<(Pt3, [f64; 3])>,
    stereo: Vec<(Pt3, [f64; 3])>,
}

fn measure(
    rig: &CalibratedRig,
    robot: &RobotPose,
    laser: &[LaserObservation],
    tracks: &[Vec<RigObservation>],
) -> Measured {
    let robot = [*robot];
    let captures = Captures::Robot {
        frame: RobotFrame::Base,
        robot: &robot,
    };
    let diag = |c: vision_calibration::core::Mat3| [c[(0, 0)], c[(1, 1)], c[(2, 2)]];
    let found = triangulate_laser(rig, captures, laser).unwrap();
    let laser = (0..laser.len())
        .map(|i| {
            (
                found.points[i].point,
                diag(found.point_covariance(i).expect("covariance")),
            )
        })
        .collect();
    let found = triangulate(rig, captures, tracks, &RunControl::default()).unwrap();
    let stereo = (0..tracks.len())
        .map(|i| {
            (
                found.points[i].point,
                diag(found.point_covariance(i).expect("covariance")),
            )
        })
        .collect();
    Measured { laser, stereo }
}

#[test]
#[ignore = "Monte-Carlo gate; run in release with --ignored"]
fn the_handeye_and_robot_shares_are_in_the_base_frame_sigma() {
    let cameras = cameras();
    let layout = cam_se3_rig();
    let noise = PointCovariance::isotropic(MEASURE_SIGMA).unwrap();
    let station = measurement_station();
    let start = Instant::now();

    // Source mixes, in order: pixels only; + cameras, planes and extrinsics
    // (the hand-eye held); + the hand-eye without its cross terms; the full
    // calibration; and the full calibration with the robot's σ.
    const MIXES: usize = 5;
    let names = [
        "pixels only",
        "+ cameras, planes, extrinsics",
        "+ hand-eye, no cross terms",
        "+ hand-eye (calibration)",
        "+ robot pose (all)",
    ];
    // Per kind (laser, stereo), point, axis: the estimates, and per mix the
    // stated variances.
    let axes = || [Vec::new(), Vec::new(), Vec::new()];
    let sized = || vec![vec![axes(); CAMERAS * POINTS], vec![axes(); POINTS]];
    let mut estimates = sized();
    let mut stated = vec![sized(); MIXES];
    for trial in 0..TRIALS {
        let seed = 1_300 + trial;
        let rig = calibrated_rig(seed);
        let labels = rig.handeye_labels().unwrap();
        let covariance = rig.uncertainty().covariance().unwrap();
        assert!(
            labels.iter().all(|l| covariance.index(l).is_some()),
            "the export's covariance lists the hand-eye"
        );
        let is_handeye = |l: &str| l.starts_with("gripper_se3_rig.");
        let rigs = [
            with_covariance(&rig, |_, _| false),
            with_covariance(&rig, |a, b| !is_handeye(a) && !is_handeye(b)),
            with_covariance(&rig, |a, b| is_handeye(a) == is_handeye(b)),
            rig.clone(),
            rig,
        ];

        // The robot's reported pose: the true one with a fresh error.
        let key = |i: u64| (seed << 32) + 8_000_000 + i;
        let xi: Vec<f64> = (0..6)
            .map(|i| {
                let sigma = if i < 3 {
                    ROBOT_ROT_SIGMA
                } else {
                    ROBOT_TRANS_SIGMA
                };
                sigma * gauss(key(i))
            })
            .collect();
        let reported = station
            * Iso3::from_parts(
                nalgebra::Translation3::new(xi[3], xi[4], xi[5]),
                nalgebra::UnitQuaternion::from_scaled_axis(Vec3::new(xi[0], xi[1], xi[2])),
            );
        let exact = RobotPose::exact(reported);
        let with_sigma =
            RobotPose::isotropic(reported, ROBOT_ROT_SIGMA, ROBOT_TRANS_SIGMA).unwrap();

        let pixel = |c: usize, p: &Vec3, k: u64| {
            let clean = cameras[c]
                .project_point(&(layout[c] * Pt3::from(*p)))
                .unwrap();
            Pt2::new(
                clean.x + MEASURE_SIGMA * gauss(2 * key(k)),
                clean.y + MEASURE_SIGMA * gauss(2 * key(k) + 1),
            )
        };
        let laser: Vec<LaserObservation> = (0..CAMERAS)
            .flat_map(|c| {
                laser_points()
                    .into_iter()
                    .enumerate()
                    .map(move |(i, p)| (c, i, p))
            })
            .map(|(c, i, p)| {
                LaserObservation::new(0, c, pixel(c, &p, 100 + (c * POINTS + i) as u64), noise)
            })
            .collect();
        let tracks: Vec<Vec<RigObservation>> = stereo_points()
            .iter()
            .enumerate()
            .map(|(i, p)| {
                (0..CAMERAS)
                    .map(|c| {
                        RigObservation::new(0, c, pixel(c, p, 200 + (c * POINTS + i) as u64), noise)
                    })
                    .collect()
            })
            .collect();

        for (m, rig) in rigs.iter().enumerate() {
            let robot = if m == MIXES - 1 { &with_sigma } else { &exact };
            let measured = measure(rig, robot, &laser, &tracks);
            for (kind, points) in [&measured.laser, &measured.stereo].into_iter().enumerate() {
                for (i, (point, variance)) in points.iter().enumerate() {
                    for axis in 0..3 {
                        if m == 0 {
                            estimates[kind][i][axis].push(point[axis]);
                        }
                        stated[m][kind][i][axis].push(variance[axis]);
                    }
                }
            }
        }
    }

    // Pooled over the points and axes: the mean of the squared ratios, then
    // the root; and the RMS stated σ.
    let pooled = |m: usize, kind: usize| -> (f64, f64) {
        let mut ratio_sq = 0.0;
        let mut var = 0.0;
        let mut count = 0.0;
        for i in 0..stated[m][kind].len() {
            for axis in 0..3 {
                let e = &estimates[kind][i][axis];
                let v = &stated[m][kind][i][axis];
                ratio_sq += common::scatter_over_stated(e, v).powi(2);
                var += v.iter().sum::<f64>() / v.len() as f64;
                count += 1.0;
            }
        }
        ((ratio_sq / count).sqrt(), (var / count).sqrt() * 1e3)
    };
    eprintln!(
        "{TRIALS} joint rig hand-eye laser calibrations, {:.1} s; base frame, robot σ \
         {ROBOT_ROT_SIGMA} rad / {ROBOT_TRANS_SIGMA} m",
        start.elapsed().as_secs_f64()
    );
    for (kind, name) in ["laser", "stereo"].iter().enumerate() {
        for (m, mix) in names.iter().enumerate() {
            let (ratio, sigma_mm) = pooled(m, kind);
            eprintln!("  {name:6} {mix:32}: RMS σ {sigma_mm:.4} mm, scatter/stated {ratio:.3}");
        }
    }
    for kind in 0..2 {
        let (full, _) = pooled(MIXES - 1, kind);
        assert!(
            (0.85..=1.15).contains(&full),
            "kind {kind}: scatter/stated {full:.3}"
        );
        if kind == 1 {
            // The stereo points' σ is mostly their pixels' (a 10 cm baseline).
            continue;
        }
        // The robot's share is part of the answer: without it the σ is short.
        let (calibration, _) = pooled(MIXES - 2, kind);
        assert!(
            calibration > 1.2,
            "kind {kind}: without the robot {calibration:.3}"
        );
        // So are the hand-eye's correlations with the cameras and planes: one
        // solve estimated them, and without them the σ is far too large.
        let (uncorrelated, _) = pooled(2, kind);
        assert!(
            uncorrelated < 0.7,
            "kind {kind}: without the cross terms {uncorrelated:.3}"
        );
    }
}
