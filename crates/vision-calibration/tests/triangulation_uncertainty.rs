//! The covariance of triangulated points carries the calibration's share.
//!
//! Each trial calibrates a stereo rig from freshly noisy board views with the
//! public rig-extrinsics workflow, then triangulates the same ten points of
//! one held-out stereo capture from freshly noisy pixels, stated at the true
//! σ. Over the trials the scatter of a point's depth matches the σ that
//! `triangulate` reports for the calibration it was given (the observations'
//! noise plus the calibration's covariance as consider parameters), and the σ
//! from the observations alone understates it.

#![allow(missing_docs)]

mod common;

use common::{PIXEL_SIGMA, gauss, scatter_over_stated};
use vision_calibration::common::CalibratedCameras;
use vision_calibration::core::{
    BrownConrady5, CalibratedRig, CorrespondenceView, FxFyCxCy, Iso3, NoMeta, PinholeCamera,
    PointCovariance, Pt2, Pt3, RigDataset, RigView, RigViewObs, Uncertainty, make_pinhole_camera,
};
use vision_calibration::mvg::triangulation::{RigObservation, triangulate};
use vision_calibration::rig_extrinsics::{
    RigExtrinsicsConfig, RigExtrinsicsInput, RigExtrinsicsProblem, run_calibration,
};
use vision_calibration::session::CalibrationSession;
use vision_calibration::synthetic::poses::make_iso;
use vision_calibration::synthetic::{planar, poses};

fn cameras() -> [PinholeCamera; 2] {
    let make =
        |fx, fy, cx, cy| make_pinhole_camera(FxFyCxCy { fx, fy, cx, cy }, BrownConrady5::default());
    [
        make(850.0, 845.0, 640.0, 480.0),
        make(820.0, 815.0, 650.0, 470.0),
    ]
}

fn cam1_se3_rig() -> Iso3 {
    make_iso((0.0, 0.05, 0.0), (-0.12, 0.0, 0.0))
}

/// The calibration views: board poses with diverse tilt, 0.6 to 0.8 m away.
fn rig_poses() -> Vec<Iso3> {
    let tilts: [(f64, f64); 8] = [
        (0.00, 0.00),
        (0.16, -0.06),
        (-0.14, 0.10),
        (0.08, 0.18),
        (-0.06, -0.16),
        (0.20, 0.12),
        (-0.18, -0.10),
        (0.11, -0.20),
    ];
    poses::tilted_board_poses(&tilts, (0.105, 0.075), 0.60, 0.03)
}

/// The pixels of `p` (camera 0's frame) in camera `c`, plus Gaussian noise of
/// σ [`PIXEL_SIGMA`] keyed by `key`.
fn noisy_pixel(cameras: &[PinholeCamera; 2], c: usize, p: &Pt3, key: u64) -> Pt2 {
    let cam_se3_rig = [Iso3::identity(), cam1_se3_rig()][c];
    let q = cameras[c].project_point(&(cam_se3_rig * p)).unwrap();
    Pt2::new(
        q.x + PIXEL_SIGMA * gauss(2 * key),
        q.y + PIXEL_SIGMA * gauss(2 * key + 1),
    )
}

/// The noisy calibration dataset of trial `seed`.
fn calibration_input(cameras: &[PinholeCamera; 2], seed: u64) -> RigExtrinsicsInput {
    let board = planar::grid_points(8, 6, 0.03);
    let poses = rig_poses();
    let views: Vec<RigView<NoMeta>> = poses
        .iter()
        .enumerate()
        .map(|(v, rig_se3_target)| RigView {
            meta: NoMeta,
            obs: RigViewObs {
                cameras: (0..2)
                    .map(|c| {
                        // The board point in camera 0's frame.
                        let pixels = board
                            .iter()
                            .enumerate()
                            .map(|(i, p)| {
                                let key = (seed << 32) + (c * 1_000_000 + v * 1000 + i) as u64;
                                noisy_pixel(cameras, c, &(rig_se3_target * p), key)
                            })
                            .collect();
                        Some(CorrespondenceView::new(board.clone(), pixels).unwrap())
                    })
                    .collect(),
            },
        })
        .collect();
    RigDataset::new(views, 2).unwrap()
}

/// Ten points of a held-out capture, 0.6 to 0.8 m ahead of camera 0.
fn test_points() -> Vec<Pt3> {
    (0..10)
        .map(|i| {
            let (a, b) = (i as f64, (i * 7 % 10) as f64);
            Pt3::new(
                -0.09 + 0.02 * a,
                -0.07 + 0.016 * b,
                0.6 + 0.02 * ((i * 3) % 10) as f64,
            )
        })
        .collect()
}

/// The same calibration with its covariance shrunk to nothing: the σ of the
/// observations alone.
fn without_calibration_share(rig: &CalibratedRig) -> CalibratedRig {
    let mut covariance = rig.uncertainty().covariance().expect("covariance").clone();
    let n = covariance.labels.len();
    covariance.covariance = (0..n)
        .map(|i| (0..n).map(|j| if i == j { 1e-20 } else { 0.0 }).collect())
        .collect();
    CalibratedRig::new(
        rig.cameras().to_vec(),
        rig.cam_se3_rig().to_vec(),
        Uncertainty::Available(covariance),
    )
    .unwrap()
}

#[test]
fn the_calibrations_share_is_in_the_triangulated_depth_sigma() {
    let cams = cameras();
    let points = test_points();
    let noise = PointCovariance::isotropic(PIXEL_SIGMA).unwrap();
    let trials = 60_u64;

    // Depth (z) per trial and its stated variance, with and without the share.
    let mut depth = vec![Vec::new(); points.len()];
    let mut consider = vec![Vec::new(); points.len()];
    let mut observations_only = vec![Vec::new(); points.len()];
    for trial in 0..trials {
        let seed = 500 + trial;
        let mut session = CalibrationSession::<RigExtrinsicsProblem>::new();
        session.set_config(RigExtrinsicsConfig::default()).unwrap();
        session.set_input(calibration_input(&cams, seed)).unwrap();
        run_calibration(&mut session).unwrap();
        let rig = session.export().unwrap().calibrated_rig().unwrap();
        let bare = without_calibration_share(&rig);

        let tracks: Vec<Vec<RigObservation>> = points
            .iter()
            .enumerate()
            .map(|(i, p)| {
                (0..2)
                    .map(|c| {
                        let key = (seed << 32) + 9_000_000 + (c * 100 + i) as u64;
                        RigObservation::new(0, c, noisy_pixel(&cams, c, p, key), noise)
                    })
                    .collect()
            })
            .collect();
        let with = triangulate(&rig, &[Iso3::identity()], &tracks).unwrap();
        let without = triangulate(&bare, &[Iso3::identity()], &tracks).unwrap();
        for i in 0..points.len() {
            // The same estimate either way: asking for the share moves no point.
            let d = (with.points[i].point - without.points[i].point).norm();
            assert!(d < 1e-9, "consider moved point {i} by {d}");
            depth[i].push(with.points[i].point.z);
            consider[i].push(with.point_covariance(i).expect("covariance")[(2, 2)]);
            observations_only[i].push(without.point_covariance(i).expect("covariance")[(2, 2)]);
        }
    }

    // Pooled over the points: the mean of the squared ratios, then the root.
    let pooled = |variances: &[Vec<f64>]| -> f64 {
        let mean = (0..points.len())
            .map(|i| scatter_over_stated(&depth[i], &variances[i]).powi(2))
            .sum::<f64>()
            / points.len() as f64;
        mean.sqrt()
    };
    let rms_mm = |variances: &[Vec<f64>]| -> f64 {
        let all: Vec<f64> = variances.iter().flatten().copied().collect();
        (all.iter().sum::<f64>() / all.len() as f64).sqrt() * 1e3
    };
    let with_ratio = pooled(&consider);
    let without_ratio = pooled(&observations_only);
    eprintln!(
        "depth sigma over {trials} calibrations: consider {:.3} mm (scatter/stated \
         {with_ratio:.2}), observations only {:.3} mm (scatter/stated {without_ratio:.2})",
        rms_mm(&consider),
        rms_mm(&observations_only),
    );
    assert!(
        (0.85..=1.15).contains(&with_ratio),
        "consider scatter/stated {with_ratio:.3}"
    );
    assert!(
        without_ratio > 1.5,
        "observations only should understate: scatter/stated {without_ratio:.3}"
    );
}
