//! Concise hand-eye calibration using session + pipeline helpers.

#[path = "support/handeye_io.rs"]
mod handeye_io;

use anyhow::{ensure, Result};
use calib::pipeline::handeye_single::{
    init_handeye, optimize_handeye_stage, pinhole_from_camera_params, ransac_planar_poses,
    BackendSolveOptions, HandEyeMode, HandEyeSolveOptions, HandEyeView, PoseRansacOptions,
    RobustLoss,
};
use calib::pipeline::PlanarIntrinsicsConfig;
use calib::session::{
    CalibrationSession, PlanarIntrinsicsInitOptions, PlanarIntrinsicsObservations,
    PlanarIntrinsicsOptimOptions, PlanarIntrinsicsProblem,
};
use chess_corners::ChessConfig;
use handeye_io::{kuka_chessboard_params, load_kuka_dataset};
use std::path::Path;

fn main() -> Result<()> {
    println!("=== Hand-Eye Single Camera Calibration (Session + Pipeline) ===\n");

    let base_path = Path::new("data/kuka_1");
    let chess_config = ChessConfig::default();
    let board_params = kuka_chessboard_params();

    let (samples, summary) = load_kuka_dataset(base_path, &chess_config, &board_params)?;
    println!(
        "Loaded {}/{} views ({} skipped), {} corners, square size {:.4} m",
        summary.used_views,
        summary.total_images,
        summary.skipped_views,
        summary.total_corners,
        summary.square_size_m
    );
    ensure!(
        samples.len() >= 3,
        "need at least 3 valid views, got {}",
        samples.len()
    );

    let views: Vec<HandEyeView> = samples
        .into_iter()
        .map(|s| HandEyeView {
            view: s.view,
            robot_pose: s.robot_pose,
        })
        .collect();

    // Intrinsics via session
    let planar_views = views.iter().map(|v| v.view.clone()).collect();
    let mut session = CalibrationSession::<PlanarIntrinsicsProblem>::new();
    session.set_observations(PlanarIntrinsicsObservations {
        views: planar_views,
    });
    session.initialize(PlanarIntrinsicsInitOptions::default())?;

    let optim_opts = PlanarIntrinsicsOptimOptions {
        config: PlanarIntrinsicsConfig {
            robust_loss: Some(calib::pipeline::RobustLossConfig::Huber { scale: 2.0 }),
            ..Default::default()
        },
    };
    session.optimize(optim_opts)?;
    let optimized = session.export()?;

    let (intrinsics, distortion) = pinhole_from_camera_params(&optimized.report.camera)?;
    println!(
        "Intrinsics optimized: mean reproj error {:.3} px",
        optimized.mean_reproj_error
    );

    // Pose RANSAC + hand-eye
    let pose_ransac = ransac_planar_poses(
        &views,
        &intrinsics,
        &distortion,
        &PoseRansacOptions::default(),
    )?;
    println!(
        "Pose RANSAC: kept {}/{} views, mean reproj error {:.3} px",
        pose_ransac.views.len(),
        views.len(),
        pose_ransac.mean_reproj_error
    );

    let handeye_init = init_handeye(
        &pose_ransac.views,
        &pose_ransac.poses,
        &intrinsics,
        &distortion,
        HandEyeMode::EyeInHand,
    )?;
    println!(
        "Hand-eye init: mean reproj error {:.3} px",
        handeye_init.mean_reproj_error
    );

    let handeye_opts = HandEyeSolveOptions {
        robust_loss: RobustLoss::Huber { scale: 2.0 },
        fix_fx: true,
        fix_fy: true,
        fix_cx: true,
        fix_cy: true,
        fix_k1: true,
        fix_k2: true,
        fix_k3: true,
        fix_p1: true,
        fix_p2: true,
        fix_extrinsics: vec![true],
        fix_target_poses: vec![0],
        ..Default::default()
    };

    let handeye_opt = optimize_handeye_stage(
        &pose_ransac.views,
        &handeye_init,
        &intrinsics,
        &distortion,
        HandEyeMode::EyeInHand,
        &handeye_opts,
        &BackendSolveOptions::default(),
    )?;
    println!(
        "Hand-eye optimized: mean reproj error {:.3} px",
        handeye_opt.mean_reproj_error
    );

    Ok(())
}
