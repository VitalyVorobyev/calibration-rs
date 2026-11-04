//! The calibration's share is in the depth σ of a stereo measurement.
//!
//! Each trial calibrates a stereo rig from freshly noisy board views with the
//! public rig-extrinsics workflow, rectifies it, takes ten points of one
//! held-out capture from freshly noisy pixels, stated at the true σ, as
//! matches on the rectified pair and measures them with
//! `StereoRectification::triangulate`. Over the trials the scatter of a
//! point's depth matches the σ stated for the calibration it was given, and
//! the σ from the observations alone understates it.

#![allow(missing_docs)]

mod common;

use common::{
    PIXEL_SIGMA, calibrated_stereo_rig, scatter_over_stated, stereo_cameras, stereo_noisy_pixel,
    stereo_test_points, without_calibration_share,
};
use vision_calibration::core::{PointCovariance, Pt2};
use vision_calibration::mvg::rectification::{RectifiedMatch, RectifyOptions, rectify};

#[test]
fn the_calibrations_share_is_in_the_measured_depth_sigma() {
    let cams = stereo_cameras();
    let points = stereo_test_points();
    let noise = PointCovariance::isotropic(PIXEL_SIGMA).unwrap();
    // A pooled scatter ratio from `n` calibrations has a relative error of
    // about `1/sqrt(2n)` (the points share each calibration's error): 200
    // resolve it to 5 %, so the `[0.85, 1.15]` band is 3 standard errors
    // wide, and an understatement above 1.2 is 4 of them from none.
    let trials = 200_u64;

    let mut depth = vec![Vec::new(); points.len()];
    let mut consider = vec![Vec::new(); points.len()];
    let mut observations_only = vec![Vec::new(); points.len()];
    // The disparity's share alone, `disparity_covariance` at σ_d = √2 σ.
    let mut disparity_only = vec![Vec::new(); points.len()];
    for trial in 0..trials {
        let seed = 500 + trial;
        let rig = calibrated_stereo_rig(&cams, seed);
        let bare = without_calibration_share(&rig);
        let rect = rectify(&rig, [0, 1], &RectifyOptions::default()).unwrap();

        // Matches on the rectified pair, from the noisy raw pixels.
        let matches: Vec<RectifiedMatch> = points
            .iter()
            .enumerate()
            .map(|(i, p)| {
                let raw = |c: usize| {
                    let key = (seed << 32) + 9_000_000 + (c * 100 + i) as u64;
                    stereo_noisy_pixel(&cams, c, p, key)
                };
                let left = rect.rectified_pixel(0, &raw(0)).unwrap();
                let right = rect.rectified_pixel(1, &raw(1)).unwrap();
                RectifiedMatch::new(Pt2::new(left.x, left.y), left.x - right.x)
            })
            .collect();
        let with = rect
            .triangulate(
                &rig,
                &matches,
                noise,
                &vision_calibration::control::RunControl::default(),
            )
            .unwrap();
        let without = rect
            .triangulate(
                &bare,
                &matches,
                noise,
                &vision_calibration::control::RunControl::default(),
            )
            .unwrap();
        for (i, m) in matches.iter().enumerate() {
            // Asking for the share moves no point.
            let d = (with.points[i].point - without.points[i].point).norm();
            assert!(d < 1e-9, "consider moved point {i} by {d}");
            depth[i].push(with.points[i].point.z);
            consider[i].push(with.point_covariance(i).expect("covariance")[(2, 2)]);
            observations_only[i].push(without.point_covariance(i).expect("covariance")[(2, 2)]);
            let cov = rect
                .disparity_covariance(&m.left, m.disparity, 2.0_f64.sqrt() * PIXEL_SIGMA)
                .unwrap();
            // The left camera is the rig frame here: z is the rig depth.
            disparity_only[i].push(cov[(2, 2)]);
        }
    }

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
        "stereo depth sigma over {trials} calibrations: with the calibration {:.3} mm \
         (scatter/stated {with_ratio:.2}), observations only {:.3} mm (scatter/stated \
         {without_ratio:.2}), disparity only {:.3} mm (scatter/stated {:.2})",
        rms_mm(&consider),
        rms_mm(&observations_only),
        rms_mm(&disparity_only),
        pooled(&disparity_only),
    );
    assert!(
        (0.85..=1.15).contains(&with_ratio),
        "scatter/stated {with_ratio:.3}"
    );
    assert!(
        without_ratio > 1.2,
        "observations only should understate: scatter/stated {without_ratio:.3}"
    );
}
