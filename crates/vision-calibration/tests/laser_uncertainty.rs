//! The covariance of laser points carries the calibration's share.
//!
//! Each trial calibrates a laserline device (camera, distortion and laser
//! plane) from freshly noisy board and stripe pixels with the public
//! workflow, then triangulates eight points of the true laser plane from
//! freshly noisy pixels, stated at the true σ. Over the trials the scatter
//! of a point's depth matches the σ that `triangulate_laser` reports for the
//! calibration it was given (the pixels' noise plus the calibration's
//! covariance), the length between two points matches the σ that
//! `metrology` derives from the joint covariance, and the σ from the pixels
//! alone understates both.

#![allow(missing_docs)]

mod common;

use common::laser::{laser_intrinsics, laser_truth, noisy_laser_dataset};
use common::{gauss, scatter_over_stated, without_calibration_share};
use std::time::Instant;
use vision_calibration::common::CalibratedCameras;
use vision_calibration::core::{PointCovariance, Pt2, Pt3, Vec3};
use vision_calibration::laserline_device::{
    LaserlineDeviceConfig, LaserlineDeviceProblem, run_calibration,
};
use vision_calibration::mvg::laser::{LaserObservation, triangulate_laser};
use vision_calibration::mvg::metrology::{PointSet, distance};
use vision_calibration::session::CalibrationSession;

/// A scatter ratio from `n` trials has a relative error of `1/sqrt(2n)`: 300
/// trials resolve it to 4 %, which the bounds below need.
const TRIALS: u64 = 300;
/// The pixel noise of the laser points, stated and true.
const LASER_SIGMA: f64 = 0.1;
const POINTS: usize = 8;

/// Eight points of the true laser plane, 0.4 to 0.6 m from the camera.
fn truth_points() -> Vec<Vec3> {
    let plane = laser_truth();
    let n = plane.normal().into_inner();
    (0..POINTS)
        .map(|i| {
            let a = i as f64;
            let x = -0.18 + 0.05 * a;
            let y = -0.12 + 0.034 * ((3 * i) % POINTS) as f64;
            let z = (-plane.distance() - n.x * x - n.y * y) / n.z;
            Vec3::new(x, y, z)
        })
        .collect()
}

fn calibrated_laser_rig(seed: u64) -> vision_calibration::core::CalibratedRig {
    let mut session = CalibrationSession::<LaserlineDeviceProblem>::new();
    session
        .set_config(LaserlineDeviceConfig::default())
        .unwrap();
    session.set_input(noisy_laser_dataset(seed)).unwrap();
    run_calibration(&mut session).unwrap();
    session.export().unwrap().calibrated_rig().unwrap()
}

#[test]
fn the_calibrations_share_is_in_the_laser_point_sigma() {
    let truth = truth_points();
    let camera = vision_calibration::core::make_pinhole_camera(
        laser_intrinsics(),
        vision_calibration::core::BrownConrady5::default(),
    );
    let noise = PointCovariance::isotropic(LASER_SIGMA).unwrap();
    let true_length = (truth[POINTS - 1] - truth[0]).norm();
    let start = Instant::now();

    // Depth per trial and its stated variance, with and without the share.
    let mut depth = vec![Vec::new(); POINTS];
    let mut consider = vec![Vec::new(); POINTS];
    let mut bare_variance = vec![Vec::new(); POINTS];
    // The measured length and its stated σ.
    let mut length_error = Vec::new();
    let (mut length_sigma, mut length_sigma_bare) = (Vec::new(), Vec::new());
    for trial in 0..TRIALS {
        let seed = 500 + trial;
        let rig = calibrated_laser_rig(seed);
        let bare = without_calibration_share(&rig);

        let observations: Vec<LaserObservation> = truth
            .iter()
            .enumerate()
            .map(|(i, p)| {
                let clean = camera.project_point(&Pt3::from(*p)).unwrap();
                let key = (seed << 32) + 9_000_000 + i as u64;
                let pixel = Pt2::new(
                    clean.x + LASER_SIGMA * gauss(2 * key),
                    clean.y + LASER_SIGMA * gauss(2 * key + 1),
                );
                LaserObservation::new(0, pixel, noise)
            })
            .collect();
        let with = triangulate_laser(&rig, &observations).unwrap();
        let without = triangulate_laser(&bare, &observations).unwrap();
        for i in 0..POINTS {
            // The same estimate either way: asking for the share moves no point.
            let d = (with.points[i].point - without.points[i].point).norm();
            assert!(d < 1e-12, "the covariance moved point {i} by {d}");
            depth[i].push(with.points[i].point.z);
            consider[i].push(with.point_covariance(i).expect("covariance")[(2, 2)]);
            bare_variance[i].push(without.point_covariance(i).expect("covariance")[(2, 2)]);
        }
        let set = PointSet::from_laser_triangulation(&with).unwrap();
        let length = distance(&set, 0, POINTS - 1).unwrap();
        length_error.push(length.value - true_length);
        length_sigma.push(length.sigma);
        let set = PointSet::from_laser_triangulation(&without).unwrap();
        length_sigma_bare.push(distance(&set, 0, POINTS - 1).unwrap().sigma);
    }

    // Pooled over the points: the mean of the squared ratios, then the root.
    let pooled = |variances: &[Vec<f64>]| -> f64 {
        let mean = (0..POINTS)
            .map(|i| scatter_over_stated(&depth[i], &variances[i]).powi(2))
            .sum::<f64>()
            / POINTS as f64;
        mean.sqrt()
    };
    let rms =
        |values: &[f64]| (values.iter().map(|v| v * v).sum::<f64>() / values.len() as f64).sqrt();
    let rms_mm = |variances: &[Vec<f64>]| -> f64 {
        let all: Vec<f64> = variances.iter().flatten().copied().collect();
        (all.iter().sum::<f64>() / all.len() as f64).sqrt() * 1e3
    };
    let consider_ratio = pooled(&consider);
    let bare_ratio = pooled(&bare_variance);
    let length_ratio = rms(&length_error) / rms(&length_sigma);
    let length_ratio_bare = rms(&length_error) / rms(&length_sigma_bare);
    eprintln!(
        "{TRIALS} calibrations, {:.1} s; depth sigma: consider {:.3} mm (scatter/stated \
         {consider_ratio:.3}), pixels only {:.3} mm (scatter/stated {bare_ratio:.2}); length \
         {:.2} mm: scatter/stated {length_ratio:.3} (pixels only {length_ratio_bare:.2})",
        start.elapsed().as_secs_f64(),
        rms_mm(&consider),
        rms_mm(&bare_variance),
        rms(&length_error) * 1e3,
    );
    assert!(
        (0.85..=1.15).contains(&consider_ratio),
        "consider scatter/stated {consider_ratio:.3}"
    );
    assert!(
        bare_ratio > 1.5,
        "pixels only should understate: scatter/stated {bare_ratio:.3}"
    );
    assert!(
        (0.8..=1.25).contains(&length_ratio),
        "length scatter/stated {length_ratio:.3}"
    );
}
