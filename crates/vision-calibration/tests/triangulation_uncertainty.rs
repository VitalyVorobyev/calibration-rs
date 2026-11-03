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

use common::{
    PIXEL_SIGMA, calibrated_stereo_rig, scatter_over_stated, stereo_cameras, stereo_noisy_pixel,
    stereo_test_points,
};
use vision_calibration::core::{CalibratedRig, Iso3, PointCovariance, Uncertainty};
use vision_calibration::mvg::triangulation::{RigObservation, triangulate};

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
    let cams = stereo_cameras();
    let points = stereo_test_points();
    let noise = PointCovariance::isotropic(PIXEL_SIGMA).unwrap();
    let trials = 60_u64;

    // Depth (z) per trial and its stated variance, with and without the share.
    let mut depth = vec![Vec::new(); points.len()];
    let mut consider = vec![Vec::new(); points.len()];
    let mut observations_only = vec![Vec::new(); points.len()];
    for trial in 0..trials {
        let seed = 500 + trial;
        let rig = calibrated_stereo_rig(&cams, seed);
        let bare = without_calibration_share(&rig);

        let tracks: Vec<Vec<RigObservation>> = points
            .iter()
            .enumerate()
            .map(|(i, p)| {
                (0..2)
                    .map(|c| {
                        let key = (seed << 32) + 9_000_000 + (c * 100 + i) as u64;
                        RigObservation::new(0, c, stereo_noisy_pixel(&cams, c, p, key), noise)
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
