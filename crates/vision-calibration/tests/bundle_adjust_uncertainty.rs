//! Bundle adjustment carries the calibration's uncertainty.
//!
//! Each trial calibrates a stereo rig from freshly noisy board views with the
//! public rig-extrinsics workflow, then adjusts three captures of ten points
//! from freshly noisy pixels, stated at the true σ. Over the trials:
//!
//! - with the calibration as consider parameters, the scatter of a point's
//!   depth and of the second capture's translation matches the σ that
//!   `bundle_adjust` reports (observations plus the calibration's share);
//! - with the calibration as a prior, each camera's focal length comes out
//!   no less certain than the calibration had it, and the refined rig's
//!   scatter matches its stated σ.

#![allow(missing_docs)]

mod common;

use common::{
    PIXEL_SIGMA, calibrated_stereo_rig, gauss, scatter_over_stated, stereo_cameras,
    stereo_test_points,
};
use std::sync::OnceLock;
use std::time::Instant;
use vision_calibration::core::{CalibratedRig, Iso3, PinholeCamera, PointCovariance, Pt2, Pt3};
use vision_calibration::mvg::bundle_adjust::{
    BundleAdjustOptions, BundleAdjustment, Scale, bundle_adjust,
};
use vision_calibration::mvg::triangulation::RigObservation;
use vision_calibration::optim::problem::Use;
use vision_calibration::synthetic::poses::make_iso;

/// A scatter ratio from `n` trials has a relative error of `1/sqrt(2n)`: 300
/// trials resolve it to 4 %, which the bounds below need.
const TRIALS: u64 = 300;

/// The rig calibrated in each trial, computed once for both tests.
fn calibrations() -> &'static Vec<CalibratedRig> {
    static RIGS: OnceLock<Vec<CalibratedRig>> = OnceLock::new();
    RIGS.get_or_init(|| {
        let cameras = stereo_cameras();
        (0..TRIALS)
            .map(|trial| calibrated_stereo_rig(&cameras, 500 + trial))
            .collect()
    })
}

/// Three captures of the held-out scene (world → rig); the world is capture
/// 0's rig frame.
fn captures() -> [Iso3; 3] {
    [
        Iso3::identity(),
        make_iso((0.01, -0.03, 0.005), (-0.04, 0.01, 0.01)),
        make_iso((-0.02, 0.02, 0.01), (0.03, -0.02, 0.02)),
    ]
}

/// The tracks of `points` seen by both cameras at every capture, with pixel
/// noise of σ [`PIXEL_SIGMA`] keyed by `seed`.
fn observe(cameras: &[PinholeCamera; 2], points: &[Pt3], seed: u64) -> Vec<Vec<RigObservation>> {
    let noise = PointCovariance::isotropic(PIXEL_SIGMA).unwrap();
    let cam_se3_rig = [Iso3::identity(), common::stereo_cam1_se3_rig()];
    points
        .iter()
        .enumerate()
        .map(|(i, p)| {
            let mut track = Vec::new();
            for (k, pose) in captures().iter().enumerate() {
                for c in 0..2 {
                    let q = cameras[c]
                        .project_point(&(cam_se3_rig[c] * pose * p))
                        .unwrap();
                    let key = (seed << 32) + 9_000_000 + (k * 10_000 + c * 100 + i) as u64;
                    let pixel = Pt2::new(
                        q.x + PIXEL_SIGMA * gauss(2 * key),
                        q.y + PIXEL_SIGMA * gauss(2 * key + 1),
                    );
                    track.push(RigObservation::new(k, c, pixel, noise));
                }
            }
            track
        })
        .collect()
}

/// The start of every adjustment: the truth, moved by a few millimetres and
/// tenths of a degree.
fn start(points: &[Pt3]) -> (Vec<Iso3>, Vec<Pt3>) {
    let captures = captures();
    let poses = vec![
        captures[0],
        make_iso((0.0, 0.0, 0.0), (0.003, -0.002, 0.002)) * captures[1],
        make_iso((0.003, -0.002, 0.0), (-0.002, 0.003, -0.002)) * captures[2],
    ];
    let points = points
        .iter()
        .enumerate()
        .map(|(i, p)| p + nalgebra::Vector3::new(0.004, -0.003, 0.005) * (1.0 + (i % 3) as f64))
        .collect();
    (poses, points)
}

fn adjust(rig: &CalibratedRig, use_: Use, points: &[Pt3], seed: u64) -> BundleAdjustment {
    let tracks = observe(&stereo_cameras(), points, seed);
    let (poses, initial) = start(points);
    let mut opts = BundleAdjustOptions::new(Scale::Rig);
    opts.calibration = use_;
    bundle_adjust(rig, &poses, &initial, &tracks, &opts).unwrap()
}

/// `estimate = truth · Exp(δ)`: the translation part of δ, a step in the
/// truth's own frame.
fn translation_error(truth: &Iso3, estimate: &Iso3) -> [f64; 3] {
    let t = truth.rotation.inverse() * (estimate.translation.vector - truth.translation.vector);
    [t.x, t.y, t.z]
}

#[test]
fn consider_carries_the_calibration_into_points_and_poses() {
    let clock = Instant::now();
    let points = stereo_test_points();
    let truth = captures();
    let mut depth = vec![Vec::new(); points.len()];
    let mut depth_var = vec![Vec::new(); points.len()];
    let mut shift = [Vec::new(), Vec::new(), Vec::new()];
    let mut shift_var = [Vec::new(), Vec::new(), Vec::new()];
    for (trial, rig) in calibrations().iter().enumerate() {
        let found = adjust(rig, Use::Consider, &points, 500 + trial as u64);
        for i in 0..points.len() {
            depth[i].push(found.points[i].z);
            depth_var[i].push(found.point_covariance(i).expect("point covariance")[(2, 2)]);
        }
        let e = translation_error(&truth[1], &found.rig_se3_world[1]);
        let c = found.pose_covariance(1).expect("pose covariance");
        for a in 0..3 {
            shift[a].push(e[a]);
            shift_var[a].push(c[(3 + a, 3 + a)]);
        }
    }
    let depth_ratio = (0..points.len())
        .map(|i| scatter_over_stated(&depth[i], &depth_var[i]).powi(2))
        .sum::<f64>()
        / points.len() as f64;
    let depth_ratio = depth_ratio.sqrt();
    let shift_ratio: Vec<f64> = (0..3)
        .map(|a| scatter_over_stated(&shift[a], &shift_var[a]))
        .collect();
    eprintln!(
        "consider over {TRIALS} calibrations: depth scatter/stated {depth_ratio:.3}, \
         capture 1 translation (tx ty tz) {shift_ratio:.3?}; {:.1} s",
        clock.elapsed().as_secs_f64()
    );
    assert!(
        (0.85..=1.15).contains(&depth_ratio),
        "depth scatter/stated {depth_ratio:.3}"
    );
    for (a, r) in shift_ratio.iter().enumerate() {
        assert!((0.85..=1.15).contains(r), "capture 1 t{a}: {r:.3}");
    }
}

#[test]
fn prior_refines_the_cameras_and_says_how_well() {
    let clock = Instant::now();
    let points = stereo_test_points();
    let mut fx = [Vec::new(), Vec::new()];
    let mut fx_var = [Vec::new(), Vec::new()];
    let (mut before, mut after) = ([0.0_f64; 2], [0.0_f64; 2]);
    for (trial, rig) in calibrations().iter().enumerate() {
        let found = adjust(rig, Use::Prior, &points, 500 + trial as u64);
        for c in 0..2 {
            let label = format!("cam{c}.fx");
            let sigma_before = rig
                .uncertainty()
                .covariance()
                .unwrap()
                .std_dev(&label)
                .unwrap();
            let sigma_after = found
                .rig
                .uncertainty()
                .covariance()
                .unwrap()
                .std_dev(&label)
                .unwrap();
            assert!(
                sigma_after <= sigma_before * (1.0 + 1e-9),
                "trial {trial} camera {c}: σ(fx) {sigma_before} -> {sigma_after}"
            );
            before[c] += sigma_before / TRIALS as f64;
            after[c] += sigma_after / TRIALS as f64;
            fx[c].push(found.rig.cameras()[c].intrinsics.fx);
            fx_var[c].push(sigma_after * sigma_after);
        }
    }
    let ratio: Vec<f64> = (0..2)
        .map(|c| scatter_over_stated(&fx[c], &fx_var[c]))
        .collect();
    eprintln!(
        "prior over {TRIALS} calibrations: mean σ(fx) before {before:.3?} after {after:.3?}; \
         refined fx scatter/stated {ratio:.3?}; {:.1} s",
        clock.elapsed().as_secs_f64()
    );
    for (c, r) in ratio.iter().enumerate() {
        assert!(
            (0.8..=1.25).contains(r),
            "camera {c}: fx scatter/stated {r:.3}"
        );
    }
}
