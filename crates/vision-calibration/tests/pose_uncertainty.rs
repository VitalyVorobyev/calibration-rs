//! The pose of an object carries the calibration's uncertainty.
//!
//! Each trial calibrates a stereo rig from freshly noisy board views with the
//! public rig-extrinsics workflow, then estimates the pose of a held-out
//! board from freshly noisy pixels of camera 0 alone, stated at the true σ.
//! Over the trials, the scatter of each pose component matches the σ that
//! `estimate_pose` reports for the calibration it was given (the pixels' noise
//! plus the calibration's covariance as consider parameters); the σ from the
//! pixels alone understates it.

#![allow(missing_docs)]

mod common;

use common::{
    PIXEL_SIGMA, calibrated_stereo_rig, gauss, scatter_over_stated, stereo_cameras,
    without_calibration_share,
};
use std::sync::OnceLock;
use std::time::Instant;
use vision_calibration::core::{CalibratedRig, Iso3, PointCovariance, Pt2};
use vision_calibration::mvg::pose::{ObjectPose, PoseObservation, PoseOptions, estimate_pose};
use vision_calibration::synthetic::planar;
use vision_calibration::synthetic::poses::make_iso;

/// A scatter ratio from `n` trials has a relative error of `1/sqrt(2n)`: 300
/// trials resolve it to 4 %, so the `[0.85, 1.15]` band is 3.7 standard
/// errors wide.
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

/// The held-out board in the rig frame (camera 0's), 0.7 m ahead, tilted.
fn truth() -> Iso3 {
    make_iso((0.18, -0.22, 0.05), (-0.06, -0.04, 0.7))
}

/// Noisy pixels of the board in camera 0, from trial `seed`, with the true
/// camera.
fn observe(seed: u64) -> Vec<PoseObservation> {
    let camera = &stereo_cameras()[0];
    let noise = PointCovariance::isotropic(PIXEL_SIGMA).unwrap();
    planar::grid_points(8, 6, 0.03)
        .into_iter()
        .enumerate()
        .map(|(i, p)| {
            let q = camera.project_point(&(truth() * p)).unwrap();
            let key = (seed << 32) + 9_500_000 + i as u64;
            let pixel = Pt2::new(
                q.x + PIXEL_SIGMA * gauss(2 * key),
                q.y + PIXEL_SIGMA * gauss(2 * key + 1),
            );
            PoseObservation::new(0, p, pixel, noise)
        })
        .collect()
}

/// `estimate = truth · Exp(δ)`: the rotation is a rotation vector, the
/// translation a step in the truth's own frame.
fn tangent_error(estimate: &Iso3) -> [f64; 6] {
    let truth = truth();
    let r = (truth.rotation.inverse() * estimate.rotation).scaled_axis();
    let t = truth.rotation.inverse() * (estimate.translation.vector - truth.translation.vector);
    [r.x, r.y, r.z, t.x, t.y, t.z]
}

fn variances(pose: &ObjectPose) -> [f64; 6] {
    let c = pose.pose_covariance().expect("pose covariance");
    std::array::from_fn(|a| c[(a, a)])
}

#[test]
fn the_calibrations_share_is_in_the_pose_sigma() {
    let clock = Instant::now();
    let mut error = vec![Vec::new(); 6];
    let mut consider = vec![Vec::new(); 6];
    let mut observations_only = vec![Vec::new(); 6];
    for (trial, rig) in calibrations().iter().enumerate() {
        let obs = observe(500 + trial as u64);
        let opts = PoseOptions::default();
        let with = estimate_pose(rig, &obs, &opts).unwrap();
        let without = estimate_pose(&without_calibration_share(rig), &obs, &opts).unwrap();
        // The same estimate either way: asking for the share moves nothing.
        let moved = (with.rig_se3_object.translation.vector
            - without.rig_se3_object.translation.vector)
            .norm();
        assert!(moved < 1e-9, "consider moved the pose by {moved}");
        let e = tangent_error(&with.rig_se3_object);
        let (v, v0) = (variances(&with), variances(&without));
        for a in 0..6 {
            error[a].push(e[a]);
            consider[a].push(v[a]);
            observations_only[a].push(v0[a]);
        }
    }
    let ratio = |variance: &[Vec<f64>]| -> Vec<f64> {
        (0..6)
            .map(|a| scatter_over_stated(&error[a], &variance[a]))
            .collect()
    };
    let with = ratio(&consider);
    let without = ratio(&observations_only);
    eprintln!(
        "pose over {TRIALS} calibrations, scatter/stated (rx ry rz tx ty tz): \
         with the calibration {with:.3?}; pixels only {without:.3?}; {:.1} s",
        clock.elapsed().as_secs_f64()
    );
    for (a, r) in with.iter().enumerate() {
        assert!((0.85..=1.15).contains(r), "component {a}: {r:.3}");
    }
    // The calibration is a real share of the translation along the axis.
    assert!(
        without[5] > 1.15,
        "the calibration's share should show: pixels-only ratio {:.3}",
        without[5]
    );
}
