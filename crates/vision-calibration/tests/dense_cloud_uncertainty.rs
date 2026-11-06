//! The calibration's share is in the depth σ of a dense stereo cloud.
//!
//! Each trial calibrates a stereo rig from freshly noisy board views with the
//! public rig-extrinsics workflow and rectifies it. A dense match is then
//! simulated as a matcher makes it on images rectified with that
//! calibration: at ten fixed pixels of the rectified left image the true
//! scene point is on the true left camera's ray through the raw pixel the
//! rectification samples there, and the disparity is the column its true
//! right pixel lands on in the rectified right image, plus fresh noise of
//! σ_d. Over the trials the depth error of `StereoRectification::dense_cloud`
//! matches the σ its points state, and the disparity's share alone (the σ
//! of a rig without a covariance) understates it.

#![allow(missing_docs)]

mod common;

use common::{
    calibrated_stereo_rig, gauss, stereo_cam1_se3_rig, stereo_cameras, stereo_test_points,
    without_calibration_share,
};
use vision_calibration::core::{CalibratedRig, Iso3, Pt2, Pt3, Uncertainty, pinhole_camera_params};
use vision_calibration::mvg::dense::DisparityMap;
use vision_calibration::mvg::rectification::{RectifyOptions, rectify};

/// The matcher's disparity noise, pixels.
const SIGMA_D: f64 = 0.2;

#[test]
#[ignore = "Monte-Carlo gate: 200 calibrations; run in release"]
fn the_calibrations_share_is_in_the_dense_depth_sigma() {
    let cams = stereo_cameras();
    // The pixels: where the held-out points of the stereo tests land in the
    // truth's rectified left image, rounded to the grid; each keeps its
    // point's depth as the scene's depth there.
    let truth = CalibratedRig::new(
        cams.iter().map(pinhole_camera_params).collect(),
        vec![Iso3::identity(), stereo_cam1_se3_rig()],
        Uncertainty::default(),
    )
    .unwrap();
    let truth_rect = rectify(&truth, [0, 1], &RectifyOptions::default()).unwrap();
    let scene: Vec<([usize; 2], f64)> = stereo_test_points()
        .iter()
        .map(|p| {
            let raw = cams[0].project_point(p).unwrap();
            let r = truth_rect.rectified_pixel(0, &raw).unwrap();
            ([r.x.round() as usize, r.y.round() as usize], p.z)
        })
        .collect();

    // A pooled ratio from `n` calibrations has a relative error of about
    // `1/sqrt(2n)` (the points share each calibration's error): 200 resolve
    // it to 5 %, so the `[0.85, 1.15]` band is 3 standard errors wide, and
    // an understatement above 1.5 is 10 of them from none. The ratio is 0.91
    // here and 0.93-0.95 on three other seed ranges: the calibration's own
    // covariance states this depth about 10 % high, and `triangulate` on the
    // same matches states the same calibration share (at σ_d = 0 both give
    // 2.36 mm against an error of 0.88 times that).
    let trials = 200_u64;
    let n = scene.len();
    let mut errors = vec![Vec::new(); n];
    let mut stated = vec![Vec::new(); n];
    let mut disparity_only = vec![Vec::new(); n];
    for trial in 0..trials {
        let seed = 700 + trial;
        let rig = calibrated_stereo_rig(&cams, seed);
        let rect = rectify(&rig, [0, 1], &RectifyOptions::default()).unwrap();

        let mut map = DisparityMap::filled(1280, 960, f32::NAN);
        let mut depth = Vec::with_capacity(n);
        for (i, &([x, y], z)) in scene.iter().enumerate() {
            let raw_left = rect.raw_pixel(0, &Pt2::new(x as f64, y as f64)).unwrap();
            // The true scene point the left pixel sees: rig frame = camera 0.
            let ray = cams[0].backproject_pixel(&raw_left).direction;
            let point = Pt3::from(ray * (z / ray.z));
            let raw_right = cams[1]
                .project_point(&(stereo_cam1_se3_rig() * point))
                .unwrap();
            let column = rect.rectified_pixel(1, &raw_right).unwrap().x;
            let key = (seed << 32) + 5_000_000 + i as u64;
            let d = x as f64 - column + SIGMA_D * gauss(key);
            map.set(x, y, d as f32);
            depth.push(point.z);
        }
        let cloud = rect.dense_cloud(&rig, &map, SIGMA_D).unwrap();
        let bare = rect
            .dense_cloud(&without_calibration_share(&rig), &map, SIGMA_D)
            .unwrap();
        assert_eq!(cloud.len(), n);
        for (k, p) in cloud.points().iter().enumerate() {
            let i = scene.iter().position(|(px, _)| *px == p.pixel).unwrap();
            errors[i].push(p.point.z - depth[i]);
            stated[i].push(cloud.point_covariance(k).unwrap()[(2, 2)]);
            disparity_only[i].push(bare.point_covariance(k).unwrap()[(2, 2)]);
        }
    }

    // RMS error over RMS stated σ per point, pooled over the points: the
    // scene point moves with each calibration, so the error is against it,
    // not a scatter about a mean.
    let ratio = |e: &[f64], v: &[f64]| -> f64 {
        let mse = e.iter().map(|x| x * x).sum::<f64>() / e.len() as f64;
        (mse / (v.iter().sum::<f64>() / v.len() as f64)).sqrt()
    };
    let pooled = |variances: &[Vec<f64>]| -> f64 {
        let mean = (0..n)
            .map(|i| ratio(&errors[i], &variances[i]).powi(2))
            .sum::<f64>()
            / n as f64;
        mean.sqrt()
    };
    let rms_mm = |variances: &[Vec<f64>]| -> f64 {
        let all: Vec<f64> = variances.iter().flatten().copied().collect();
        (all.iter().sum::<f64>() / all.len() as f64).sqrt() * 1e3
    };
    let bias_over_sigma = (0..n)
        .map(|i| {
            let mean = errors[i].iter().sum::<f64>() / errors[i].len() as f64;
            let sigma = (stated[i].iter().sum::<f64>() / stated[i].len() as f64).sqrt();
            (mean / sigma).abs()
        })
        .fold(0.0_f64, f64::max);
    let with_ratio = pooled(&stated);
    let without_ratio = pooled(&disparity_only);
    eprintln!(
        "dense depth sigma over {trials} calibrations at sigma_d {SIGMA_D} px: with the \
         calibration {:.3} mm (error/stated {with_ratio:.3}), disparity only {:.3} mm \
         (error/stated {without_ratio:.2}); largest |mean error|/sigma {bias_over_sigma:.2}",
        rms_mm(&stated),
        rms_mm(&disparity_only),
    );
    assert!(
        (0.85..=1.15).contains(&with_ratio),
        "error/stated {with_ratio:.3}"
    );
    assert!(
        without_ratio > 1.5,
        "the disparity's share alone should understate: error/stated {without_ratio:.3}"
    );
}
