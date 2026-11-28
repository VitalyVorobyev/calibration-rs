//! A solve ends at its minimum: restarted from its own result, on both
//! linearization engines, it moves no camera parameter by more than a
//! hundredth of that parameter's standard deviation.
//!
//! The scene stops short under a rule on the last step's gain: a planar
//! calibration of 24 views under a Huber loss at the pixel noise, whose
//! reweighted Gauss–Newton steps cover only part of the way, so the
//! convergence is linear and a step's gain falls below a fixed fraction of
//! the cost well before the minimum.

#![allow(missing_docs)]

use vision_calibration_core::synthetic::planar::grid_points;
use vision_calibration_core::synthetic::poses::{seeded_tilts, tilted_board_poses};
use vision_calibration_core::{
    BrownConrady5, CameraFix, CameraParams, DistortionParams, FxFyCxCy, Iso3, ProjectionParams,
    Pt2, SensorParams, Uncertainty,
};
use vision_calibration_optim::problem::{
    PixelNoise, PoseChain, Problem, Solution, SolveOptions, TargetPoint, solve,
};
use vision_calibration_optim::{RobustLoss, SolveTermination, SolverBackend};

const VIEWS: usize = 40;

/// A deterministic standard normal draw for `key`.
fn gauss(key: u64) -> f64 {
    let mix = |mut z: u64| {
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    };
    let uniform = |z: u64| ((mix(z) >> 11) as f64 + 0.5) / (1u64 << 53) as f64;
    let (u1, u2) = (uniform(2 * key), uniform(2 * key + 1));
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

fn camera(fx: f64, fy: f64, cx: f64, cy: f64, k1: f64) -> CameraParams {
    CameraParams::new(
        ProjectionParams::Pinhole,
        DistortionParams::BrownConrady5 {
            params: BrownConrady5 {
                k1,
                iters: 8,
                ..BrownConrady5::default()
            },
        },
        SensorParams::Identity,
        FxFyCxCy { fx, fy, cx, cy },
    )
}

fn true_poses() -> Vec<Iso3> {
    tilted_board_poses(&seeded_tilts(VIEWS, 0.4, 11), (0.12, 0.075), 0.45, 0.02)
}

/// The camera's free coordinates, in the covariance's order.
fn coordinates(c: &CameraParams) -> [f64; 8] {
    let DistortionParams::BrownConrady5 { params } = c.distortion else {
        panic!("a Brown-Conrady camera");
    };
    let k = c.intrinsics;
    [
        k.fx, k.fy, k.cx, k.cy, params.k1, params.k2, params.p1, params.p2,
    ]
}
const LABELS: [&str; 8] = ["fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2"];

/// The scene's observations (0.5 px of noise) solved from `start` and
/// `poses`: the solution, its camera and its poses.
fn solve_from(
    start: &CameraParams,
    poses: &[Iso3],
    backend: SolverBackend,
) -> (Solution, CameraParams, Vec<Iso3>) {
    let truth = camera(820.0, 810.0, 640.0, 360.0, -0.12);
    let model = truth.build().unwrap();
    let board = grid_points(9, 6, 0.03);
    let mut problem = Problem::new();
    let cam = problem.add_camera(start, &CameraFix::default()).unwrap();
    let mut blocks = Vec::new();
    for (v, (pose, start_pose)) in true_poses().iter().zip(poses).enumerate() {
        let block = problem.add_pose(format!("pose{v}"), start_pose).unwrap();
        blocks.push(block);
        for (i, point) in board.iter().enumerate() {
            let q = model.project_point(&pose.transform_point(point)).unwrap();
            let key = 1000 * v as u64 + i as u64;
            let pixel = Pt2::new(q.x + 0.5 * gauss(2 * key), q.y + 0.5 * gauss(2 * key + 1));
            problem
                .add_reprojection(
                    cam,
                    PoseChain::Pose(block),
                    TargetPoint::Known(*point),
                    pixel,
                    PixelNoise::Estimated,
                    RobustLoss::Huber { scale: 0.25 },
                )
                .unwrap();
        }
    }
    let mut options = SolveOptions::default();
    options.backend = backend;
    let solution = solve(&problem, &options).unwrap();
    let camera = solution.camera(cam).unwrap();
    let poses = blocks.iter().map(|&b| solution.pose(b).unwrap()).collect();
    (solution, camera, poses)
}

#[test]
fn a_restarted_solve_stays_within_a_hundredth_of_a_sigma() {
    for backend in [SolverBackend::TinySolver, SolverBackend::Factrs] {
        let start = camera(845.0, 795.0, 646.0, 355.0, 0.0);
        let (first, cam, poses) = solve_from(&start, &true_poses(), backend);
        let Uncertainty::Available(cov) = &first.uncertainty else {
            panic!("{backend:?}: no covariance");
        };
        let apart = |other: &CameraParams| {
            let (a, b) = (coordinates(&cam), coordinates(other));
            LABELS
                .iter()
                .enumerate()
                .map(|(i, label)| (b[i] - a[i]).abs() / cov.std_dev(label).unwrap())
                .fold(0.0, f64::max)
        };
        // Restarted from its own result: one Gauss–Newton step, a hundredth
        // of a σ at most.
        let (again, cam_again, _) = solve_from(&cam, &poses, backend);
        let moved = apart(&cam_again);
        // Five restarts in a row: no drift accumulates.
        let (mut chain_cam, mut chain_poses) = (cam_again, poses.clone());
        for _ in 0..4 {
            (_, chain_cam, chain_poses) = solve_from(&chain_cam, &chain_poses, backend);
        }
        let drifted = apart(&chain_cam);
        // Started at the truth instead: the same minimum.
        let (_, from_truth, _) = solve_from(
            &camera(820.0, 810.0, 640.0, 360.0, -0.12),
            &true_poses(),
            backend,
        );
        let other_start = apart(&from_truth);
        println!(
            "{backend:?}: {:?} after {} iterations; restart {:?} after {} moved {moved:.2e} σ, \
             five {drifted:.2e} σ; from the truth {other_start:.2e} σ apart",
            first.report.termination,
            first.report.num_iters,
            again.report.termination,
            again.report.num_iters
        );
        assert_eq!(
            first.report.termination,
            Some(SolveTermination::SigmaStep),
            "{backend:?}"
        );
        assert!(
            again.report.num_iters <= 2,
            "{backend:?}: {:?}",
            again.report
        );
        for (what, z) in [
            ("one restart", moved),
            ("five restarts", drifted),
            ("a start at the truth", other_start),
        ] {
            assert!(z <= 0.01, "{backend:?} {what}: {z:.4} σ");
        }
    }
}
