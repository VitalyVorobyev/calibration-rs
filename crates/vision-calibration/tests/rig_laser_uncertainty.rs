//! The covariance of a rig laser calibration's laser points carries the
//! upstream rig hand-eye calibration's share.
//!
//! Each trial calibrates a two-camera rig on a robot arm (cameras, their
//! extrinsics, the hand-eye and the board's pose) from freshly noisy board
//! views with the public rig hand-eye workflow, then fits each camera's
//! laser plane against that calibration from freshly noisy laser views at
//! the same robot stations with the rig laserline workflow, and finally
//! triangulates eight points of the true laser plane in each camera from
//! freshly noisy pixels, stated at the true σ. Over the trials the scatter
//! of a point's depth matches the σ that `triangulate_laser` reports on the
//! export's calibration: the pixels, the planes' own fit, and the upstream
//! calibration's share, with the cross terms between the cameras and the
//! planes they were fitted against. The planes' σ from their own fit
//! alone, held conditional on the upstream as before, and the σ from the
//! pixels alone understate it.

#![allow(missing_docs)]

mod common;

use common::laser::{board, laser_intrinsics, laser_pixels_for_view, laser_truth, poses};
use common::{PIXEL_SIGMA, gauss, scatter_over_stated, without_calibration_share};
use std::time::Instant;
use vision_calibration::common::config::RobotPoseConfig;
use vision_calibration::common::{
    CalibratedCameras, HandeyeSolution, RigLaserlineView, RobotPoseMeta,
};
use vision_calibration::core::{
    BrownConrady5, CalibratedRig, CorrespondenceView, Iso3, ParameterCovariance, PinholeCamera,
    PointCovariance, Pt2, Pt3, RigDataset, RigView, RigViewObs, Uncertainty, Vec3,
    make_pinhole_camera,
};
use vision_calibration::mvg::laser::{LaserObservation, triangulate_laser};
use vision_calibration::mvg::types::Captures;
use vision_calibration::rig_handeye::{RigHandeyeConfig, RigHandeyeProblem};
use vision_calibration::rig_laserline_device::{
    RigLaserlineDataset, RigLaserlineDeviceExport, RigLaserlineDeviceInput,
    RigLaserlineDeviceProblem,
};
use vision_calibration::session::CalibrationSession;
use vision_calibration::synthetic::noise::UniformPixelNoise;
use vision_calibration::synthetic::poses::{make_iso, rig_layout};

/// A scatter ratio from `n` trials has a relative error of `1/sqrt(2n)`: 300
/// trials resolve it to 4 %, so the [0.85, 1.15] band is 3.7 of them.
const TRIALS: u64 = 300;
/// The pixel noise of the held-out laser points, stated and true.
const LASER_SIGMA: f64 = 0.1;
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

/// The board's pose in the rig at each calibration station: ten poses
/// 0.45 to 0.6 m ahead, tilted by up to 20° about two axes and rolled, for
/// a well-conditioned hand-eye.
fn calibration_poses() -> Vec<Iso3> {
    let spec = vision_calibration::synthetic::poses::BoardPoseSpec::new;
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
    vision_calibration::synthetic::poses::centered_board_poses(
        &specs,
        (common::laser::BOARD_CX, common::laser::BOARD_CY),
        0.5,
    )
}

/// The board's pose in the rig at each laser station: the six diverse poses
/// of the laser tests, about 0.5 m ahead.
fn rig_se3_target() -> Vec<Iso3> {
    poses(0.5)
}

/// The robot stations that put the board at `rig_se3_target`: `T_B_G = T_B_T ·
/// T_R_T⁻¹ · T_G_R⁻¹`, so `T_R_T = T_G_R⁻¹ · T_B_G⁻¹ · T_B_T`.
fn robot_stations(rig_se3_target: &[Iso3]) -> Vec<Iso3> {
    rig_se3_target
        .iter()
        .map(|t| base_se3_target() * t.inverse() * handeye().inverse())
        .collect()
}

/// The board corners of `camera` at view `view`, with Gaussian noise of σ
/// [`PIXEL_SIGMA`] from `key(i, axis)`.
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

/// The upstream calibration of trial `seed`.
fn rig_handeye_export(seed: u64) -> vision_calibration::rig_handeye::RigHandeyeExport {
    let cameras = cameras();
    let layout = cam_se3_rig();
    let poses = calibration_poses();
    let views = poses
        .iter()
        .zip(robot_stations(&poses))
        .enumerate()
        .map(|(v, (target, station))| RigView {
            meta: RobotPoseMeta {
                base_se3_gripper: station,
            },
            obs: RigViewObs {
                cameras: (0..CAMERAS)
                    .map(|c| {
                        let key = |i: usize, axis: u64| {
                            (seed << 32) + 2 * (c * 1_000_000 + v * 10_000 + i) as u64 + axis
                        };
                        Some(corners(&cameras[c], &(layout[c] * target), key))
                    })
                    .collect(),
            },
        })
        .collect();
    let mut config = RigHandeyeConfig::default();
    // The extrinsics are refined with the hand-eye, so the calibration's
    // covariance holds them; the synthetic robot poses are exact.
    config.handeye_optimize.refine_cam_se3_rig = true;
    config.handeye_optimize.robot_poses = RobotPoseConfig::Exact;
    let mut session = CalibrationSession::<RigHandeyeProblem>::new();
    session.set_config(config).unwrap();
    session
        .set_input(RigDataset::new(views, CAMERAS).unwrap())
        .unwrap();
    vision_calibration::rig_handeye::run_calibration(&mut session).unwrap();
    session.export().unwrap()
}

/// The laser views of trial `seed`: each camera's board corners and laser
/// stripe at the same robot stations, freshly noisy.
fn laser_dataset(seed: u64) -> RigLaserlineDataset {
    let cameras = cameras();
    let layout = cam_se3_rig();
    let exact = UniformPixelNoise::default();
    let views = rig_se3_target()
        .iter()
        .enumerate()
        .map(|(v, target)| {
            let mut view = RigLaserlineView {
                cameras: Vec::new(),
                laser_pixels: Vec::new(),
            };
            for (c, camera) in cameras.iter().enumerate() {
                let cam_se3_target = layout[c] * target;
                let key = |i: usize, axis: u64| {
                    (seed << 32) + 2 * (4_000_000 + c * 1_000_000 + v * 10_000 + i) as u64 + axis
                };
                view.cameras
                    .push(Some(corners(camera, &cam_se3_target, key)));
                let plane = laser_truth().transform_by(&layout[c]);
                let pixels = laser_pixels_for_view(camera, &cam_se3_target, &plane, v, &exact)
                    .into_iter()
                    .enumerate()
                    .map(|(i, px)| {
                        Pt2::new(
                            px.x + PIXEL_SIGMA * gauss(key(5_000 + i, 0)),
                            px.y + PIXEL_SIGMA * gauss(key(5_000 + i, 1)),
                        )
                    })
                    .collect();
                view.laser_pixels.push(Some(pixels));
            }
            view
        })
        .collect();
    RigLaserlineDataset::new(views, CAMERAS).unwrap()
}

fn rig_laser_export(
    seed: u64,
    upstream: vision_calibration::rig_laserline_device::RigUpstreamCalibration,
) -> RigLaserlineDeviceExport {
    let mut session = CalibrationSession::<RigLaserlineDeviceProblem>::new();
    session
        .set_input(RigLaserlineDeviceInput {
            dataset: laser_dataset(seed),
            upstream,
        })
        .unwrap();
    vision_calibration::rig_laserline_device::run_calibration(&mut session).unwrap();
    session.export().unwrap()
}

/// Eight points of the true laser plane, 0.4 to 0.6 m ahead, in the rig
/// frame.
fn truth_points() -> Vec<Vec3> {
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

/// `rig` with its covariance replaced: the cameras' parameters and
/// extrinsics at nothing and the planes at `planes`' block, uncorrelated.
fn with_planes_only(rig: &CalibratedRig, planes: &ParameterCovariance) -> CalibratedRig {
    let full = rig.uncertainty().covariance().expect("covariance");
    let n = full.labels.len();
    let covariance = (0..n)
        .map(|i| {
            (0..n)
                .map(
                    |j| match (planes.index(&full.labels[i]), planes.index(&full.labels[j])) {
                        (Some(a), Some(b)) => planes.covariance[a][b],
                        _ if i == j => 1e-20,
                        _ => 0.0,
                    },
                )
                .collect()
        })
        .collect();
    let mut covariance_only = full.clone();
    covariance_only.covariance = covariance;
    CalibratedRig::new(
        rig.cameras().to_vec(),
        rig.cam_se3_rig().to_vec(),
        Uncertainty::Available(covariance_only),
    )
    .unwrap()
    .with_laser_planes((0..CAMERAS).map(|c| rig.laser_plane(c).cloned()).collect())
    .unwrap()
}

#[test]
#[ignore = "Monte-Carlo gate; run in release with --ignored"]
fn the_upstream_calibrations_share_is_in_the_rig_laser_point_sigma() {
    let truth = truth_points();
    let cameras = cameras();
    let layout = cam_se3_rig();
    let noise = PointCovariance::isotropic(LASER_SIGMA).unwrap();
    let start = Instant::now();

    let observations_of = |seed: u64| -> Vec<LaserObservation> {
        (0..CAMERAS)
            .flat_map(|c| truth.iter().enumerate().map(move |(i, p)| (c, i, p)))
            .map(|(c, i, p)| {
                let clean = cameras[c]
                    .project_point(&(layout[c] * Pt3::from(*p)))
                    .unwrap();
                let key = (seed << 32) + 9_000_000 + (c * POINTS + i) as u64;
                let pixel = Pt2::new(
                    clean.x + LASER_SIGMA * gauss(2 * key),
                    clean.y + LASER_SIGMA * gauss(2 * key + 1),
                );
                LaserObservation::new(0, c, pixel, noise)
            })
            .collect()
    };

    let count = CAMERAS * POINTS;
    // Depth per trial and its stated variance: the export's; the planes'
    // own fit alone, conditional on the upstream; the pixels alone.
    let mut depth = vec![Vec::new(); count];
    let mut consider = vec![Vec::new(); count];
    let mut planes_only = vec![Vec::new(); count];
    let mut bare = vec![Vec::new(); count];
    for trial in 0..TRIALS {
        let seed = 900 + trial;
        let handeye = rig_handeye_export(seed);
        // The laser views' target poses through the calibrated chain.
        let HandeyeSolution::EyeInHand {
            gripper_se3_rig,
            base_se3_target,
        } = handeye.handeye
        else {
            panic!("an eye-in-hand export");
        };
        let chain: Vec<Iso3> = robot_stations(&rig_se3_target())
            .iter()
            .map(|b| gripper_se3_rig.inverse() * b.inverse() * base_se3_target)
            .collect();
        let upstream = handeye.to_upstream_calibration(chain).unwrap();
        let mut conditional = upstream.clone();
        conditional.uncertainty = Uncertainty::default();

        let rig = rig_laser_export(seed, upstream).calibrated_rig().unwrap();
        let conditional = rig_laser_export(seed, conditional);
        let planes = conditional.uncertainty.covariance().expect("planes' own");
        let own = with_planes_only(&rig, planes);
        let pixels = without_calibration_share(&rig);

        let observations = observations_of(seed);
        let with = triangulate_laser(&rig, Captures::Rig, &observations).unwrap();
        let own = triangulate_laser(&own, Captures::Rig, &observations).unwrap();
        let without = triangulate_laser(&pixels, Captures::Rig, &observations).unwrap();
        for i in 0..count {
            depth[i].push(with.points[i].point.z);
            consider[i].push(with.point_covariance(i).expect("covariance")[(2, 2)]);
            planes_only[i].push(own.point_covariance(i).expect("covariance")[(2, 2)]);
            bare[i].push(without.point_covariance(i).expect("covariance")[(2, 2)]);
        }
    }

    // Pooled over the points: the mean of the squared ratios, then the root.
    let pooled = |variances: &[Vec<f64>], cameras: std::ops::Range<usize>| -> f64 {
        let points: Vec<usize> = cameras.flat_map(|c| c * POINTS..(c + 1) * POINTS).collect();
        let mean = points
            .iter()
            .map(|&i| scatter_over_stated(&depth[i], &variances[i]).powi(2))
            .sum::<f64>()
            / points.len() as f64;
        mean.sqrt()
    };
    let rms_mm = |variances: &[Vec<f64>]| -> f64 {
        let all: Vec<f64> = variances.iter().flatten().copied().collect();
        (all.iter().sum::<f64>() / all.len() as f64).sqrt() * 1e3
    };
    let consider_ratio = pooled(&consider, 0..CAMERAS);
    let per_camera: Vec<f64> = (0..CAMERAS).map(|c| pooled(&consider, c..c + 1)).collect();
    let planes_ratio = pooled(&planes_only, 0..CAMERAS);
    let bare_ratio = pooled(&bare, 0..CAMERAS);
    eprintln!(
        "{TRIALS} rig calibrations, {:.1} s; depth sigma: consider {:.3} mm (scatter/stated \
         {consider_ratio:.3}, per camera {per_camera:.3?}), planes' own fit {:.3} mm \
         (scatter/stated {planes_ratio:.2}), pixels only {:.3} mm (scatter/stated \
         {bare_ratio:.2})",
        start.elapsed().as_secs_f64(),
        rms_mm(&consider),
        rms_mm(&planes_only),
        rms_mm(&bare),
    );
    assert!(
        (0.85..=1.15).contains(&consider_ratio),
        "consider scatter/stated {consider_ratio:.3}"
    );
    assert!(
        planes_ratio > 1.3,
        "the planes' own fit should understate: scatter/stated {planes_ratio:.3}"
    );
    assert!(
        bare_ratio > planes_ratio,
        "pixels only should understate more: scatter/stated {bare_ratio:.3}"
    );
}
