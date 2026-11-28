//! A Scheimpflug tilt is determined only through the lens distortion.
//!
//! Over a sweep of view designs and tilts, 0.1 px of noise: with the distortion
//! held at zero the information matrix is singular in the tilt (the
//! covariance reports it unobservable), whatever the views; held at its
//! true, non-zero value, or free, the tilt is determined. The workflows
//! refuse the first combination before solving; this pins the reason, with
//! the public problem builder, which does not refuse it.

#![allow(missing_docs)]

use vision_calibration_core::synthetic::planar::grid_points;
use vision_calibration_core::synthetic::poses::{seeded_tilts, tilted_board_poses};
use vision_calibration_core::{
    BrownConrady5, CameraFix, CameraParams, DistortionParams, FxFyCxCy, Iso3, ProjectionParams,
    ScheimpflugParams, SensorParams, Uncertainty, UncertaintyUnavailable,
};
use vision_calibration_optim::problem::{
    PixelNoise, PoseChain, Problem, SolveOptions, TargetPoint, solve,
};
use vision_calibration_optim::{RobustLoss, SolverBackend};

fn camera(k1: f64, tilt: (f64, f64)) -> CameraParams {
    CameraParams::new(
        ProjectionParams::Pinhole,
        DistortionParams::BrownConrady5 {
            params: BrownConrady5 {
                k1,
                iters: 8,
                ..BrownConrady5::default()
            },
        },
        SensorParams::Scheimpflug {
            params: ScheimpflugParams {
                tilt_x: tilt.0,
                tilt_y: tilt.1,
            },
        },
        FxFyCxCy {
            fx: 1200.0,
            fy: 1190.0,
            cx: 640.0,
            cy: 480.0,
        },
    )
}

/// What the solve from the truth reports about the tilt, with `fix` held.
fn tilt_uncertainty(
    truth: &CameraParams,
    poses: &[Iso3],
    fix: &CameraFix,
    backend: SolverBackend,
) -> Uncertainty {
    let model = truth.build().unwrap();
    let board = grid_points(9, 7, 0.03);
    let mut problem = Problem::new();
    let cam = problem.add_camera(truth, fix).unwrap();
    for (v, pose) in poses.iter().enumerate() {
        let block = problem.add_pose(format!("pose{v}"), pose).unwrap();
        for (i, point) in board.iter().enumerate() {
            let Some(q) = model.project_point(&pose.transform_point(point)) else {
                continue;
            };
            let phase = (v * 101 + i * 37) as f64;
            let q =
                vision_calibration_core::Pt2::new(q.x + 0.1 * phase.sin(), q.y + 0.1 * phase.cos());
            problem
                .add_reprojection(
                    cam,
                    PoseChain::Pose(block),
                    TargetPoint::Known(*point),
                    q,
                    PixelNoise::Estimated,
                    RobustLoss::None,
                )
                .unwrap();
        }
    }
    let mut options = SolveOptions::default();
    options.backend = backend;
    solve(&problem, &options).unwrap().uncertainty
}

#[test]
fn the_tilt_is_determined_only_through_the_distortion() {
    let designs = [(6, 0.3, 1), (10, 0.5, 2), (16, 0.7, 3)];
    let tilts = [(0.05, -0.03), (0.2, 0.1)];
    let held_distortion = |c: &CameraParams| {
        let mut fix = CameraFix::default();
        fix.distortion = Some(
            ["k1", "k2", "k3", "p1", "p2"]
                .into_iter()
                .map(String::from)
                .collect(),
        );
        let _ = c;
        fix
    };
    let free_radial = || {
        let mut fix = CameraFix::default();
        fix.distortion = Some(vec!["k3".into(), "p1".into(), "p2".into()]);
        fix
    };
    for backend in [SolverBackend::TinySolver, SolverBackend::Factrs] {
        for &(views, spread, seed) in &designs {
            let poses =
                tilted_board_poses(&seeded_tilts(views, spread, seed), (0.12, 0.09), 0.5, 0.03);
            for &tilt in &tilts {
                let without = camera(0.0, tilt);
                let with = camera(-0.15, tilt);
                let cell = format!("{backend:?} {views} views ±{spread} rad, tilt {tilt:?}");
                // Distortion held at zero: the tilt is a gauge.
                match tilt_uncertainty(&without, &poses, &held_distortion(&without), backend) {
                    Uncertainty::Unavailable {
                        reason: UncertaintyUnavailable::RankDeficient { unobservable },
                    } => assert!(
                        // The null direction: the tilt with the principal
                        // point and the poses (the tilt's own share of it is
                        // small in radians beside pixels and metres).
                        unobservable.iter().any(|l| l == "cx" || l == "cy"),
                        "{cell}: {unobservable:?}"
                    ),
                    other => panic!("{cell}: held at zero, expected rank deficient: {other:?}"),
                }
                // Held at a known non-zero value, or free: determined.
                for (what, fix) in [
                    ("held non-zero", held_distortion(&with)),
                    ("free k1, k2", free_radial()),
                ] {
                    let Uncertainty::Available(cov) =
                        tilt_uncertainty(&with, &poses, &fix, backend)
                    else {
                        panic!("{cell}, distortion {what}: no covariance");
                    };
                    let sigma = cov.std_dev("tilt_x").unwrap().to_degrees();
                    assert!(
                        sigma < 1.0,
                        "{cell}, distortion {what}: σ(tilt_x) {sigma:.3}°"
                    );
                }
            }
        }
    }
}
