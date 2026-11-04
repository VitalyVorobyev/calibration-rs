//! Measurements from a calibration carry its uncertainty.
//!
//! Each trial calibrates a stereo rig from freshly noisy board views with the
//! public rig-extrinsics workflow, then triangulates the corners of a
//! held-out board of known square size from freshly noisy pixels (stated at
//! the true σ), and measures with the joint covariance: two distances, the
//! angle at a board corner and the board plane. Over the trials the scatter
//! of each measurement about its true value matches the σ that
//! `metrology` reports (the first-order σ holds while the points' errors are
//! small against the length: the short distance is 7.5 cm, where the
//! second-order bias of a length is 0.16 σ), and the σ from the observations
//! alone (the calibration's share replaced by nothing) understates it.

#![allow(missing_docs)]

mod common;

use common::{
    PIXEL_SIGMA, calibrated_stereo_rig, stereo_cameras, stereo_noisy_pixel,
    without_calibration_share,
};
use std::time::Instant;
use vision_calibration::core::{CalibratedRig, Iso3, PointCovariance, Pt3};
use vision_calibration::mvg::metrology::{PointSet, angle, distance, fit_plane};
use vision_calibration::mvg::triangulation::{RigObservation, triangulate};
use vision_calibration::mvg::types::Captures;
use vision_calibration::synthetic::poses::make_iso;

/// A scatter ratio from `n` trials has a relative error of `1/sqrt(2n)`: 300
/// trials resolve it to 4 %, which the bounds below need.
const TRIALS: u64 = 300;
const COLS: usize = 8;
const ROWS: usize = 6;
const SQUARE: f64 = 0.025;

/// The board to rig pose of the held-out board, 0.7 m ahead and tilted.
fn board_pose() -> Iso3 {
    make_iso((0.15, -0.12, 0.05), (-0.09, -0.06, 0.70))
}

fn corner(i: usize, j: usize) -> Pt3 {
    board_pose() * Pt3::new(i as f64 * SQUARE, j as f64 * SQUARE, 0.0)
}

fn index(i: usize, j: usize) -> usize {
    j * COLS + i
}

/// The corners' tracks seen by both cameras, with pixel noise keyed by `seed`.
fn observe(seed: u64) -> Vec<Vec<RigObservation>> {
    let cameras = stereo_cameras();
    let noise = PointCovariance::isotropic(PIXEL_SIGMA).unwrap();
    (0..ROWS)
        .flat_map(|j| (0..COLS).map(move |i| (i, j)))
        .map(|(i, j)| {
            (0..2)
                .map(|c| {
                    let key = (seed << 32) + 20_000_000 + (c * 1000 + index(i, j)) as u64;
                    let pixel = stereo_noisy_pixel(&cameras, c, &corner(i, j), key);
                    RigObservation::new(0, c, pixel, noise)
                })
                .collect()
        })
        .collect()
}

/// What one trial measures, and the σ stated for each.
struct Trial {
    errors: [f64; 4],
    sigma: [f64; 4],
}

fn trial(rig: &CalibratedRig, tracks: &[Vec<RigObservation>]) -> Trial {
    let set = PointSet::from_triangulation(
        &triangulate(
            rig,
            Captures::Rig,
            tracks,
            &vision_calibration::control::RunControl::default(),
        )
        .unwrap(),
    )
    .unwrap();
    let short = distance(&set, index(0, 0), index(3, 0)).unwrap();
    let long = distance(&set, index(0, 0), index(7, 0)).unwrap();
    // The right angle at the board's corner, arms of 3 squares.
    let right = angle(&set, index(3, 0), index(0, 0), index(0, 3)).unwrap();
    let all: Vec<usize> = (0..COLS * ROWS).collect();
    let plane = fit_plane(&set, &all).unwrap();

    // The truth's plane, oriented like the fit (distance >= 0).
    let n = board_pose().rotation * nalgebra::Vector3::z();
    let d = n.dot(&board_pose().translation.vector);
    let d_true = d.abs();
    Trial {
        errors: [
            short.value - 3.0 * SQUARE,
            long.value - 7.0 * SQUARE,
            right.value - std::f64::consts::FRAC_PI_2,
            plane.distance - d_true,
        ],
        sigma: [
            short.sigma,
            long.sigma,
            right.sigma,
            plane.covariance[(2, 2)].sqrt(),
        ],
    }
}

fn rms(values: impl Iterator<Item = f64>) -> f64 {
    let (sum, n) = values.fold((0.0, 0.0), |(s, n), v| (s + v * v, n + 1.0));
    (sum / n).sqrt()
}

#[test]
fn measurement_sigma_matches_scatter_over_calibrations() {
    let cameras = stereo_cameras();
    let start = Instant::now();
    let (mut with, mut without) = (Vec::new(), Vec::new());
    for t in 0..TRIALS {
        let seed = 500 + t;
        let rig = calibrated_stereo_rig(&cameras, seed);
        let bare = without_calibration_share(&rig);
        let tracks = observe(seed);
        with.push(trial(&rig, &tracks));
        without.push(trial(&bare, &tracks));
    }
    let names = [
        "distance, 3 squares",
        "distance, 7 squares",
        "right angle",
        "plane d",
    ];
    let summarize = |trials: &[Trial], k: usize| {
        let scatter = rms(trials.iter().map(|t| t.errors[k]));
        let stated = rms(trials.iter().map(|t| t.sigma[k]));
        let mean_error = trials.iter().map(|t| t.errors[k]).sum::<f64>() / trials.len() as f64;
        let mean_sigma = trials.iter().map(|t| t.sigma[k]).sum::<f64>() / trials.len() as f64;
        (scatter / stated, mean_error / mean_sigma, stated)
    };
    eprintln!(
        "{TRIALS} calibrations, {:.1} s; scatter / stated sigma (bias = mean error / mean sigma):",
        start.elapsed().as_secs_f64()
    );
    let mut results = Vec::new();
    for (k, name) in names.iter().enumerate() {
        let (ratio, bias, stated) = summarize(&with, k);
        let (observations_only, _, stated_only) = summarize(&without, k);
        eprintln!(
            "  {name:20} ratio {ratio:.3}  bias {bias:+.3}  stated {stated:.3e}; observations only: \
             ratio {observations_only:.3} (stated {stated_only:.3e})"
        );
        results.push((ratio, bias));
    }
    for (k, (ratio, bias)) in results.iter().enumerate() {
        assert!(
            (0.85..=1.15).contains(ratio),
            "{}: scatter/stated {ratio:.3}",
            names[k]
        );
        assert!(bias.abs() < 0.3, "{}: bias {bias:.3}", names[k]);
    }
}
