//! Concise hand-eye calibration using a dedicated session problem.

#[path = "support/handeye_io.rs"]
mod handeye_io;

use anyhow::{ensure, Result};
use calib::pipeline::handeye_single::{
    HandEyeStage, HandEyeView, IntrinsicsStage, PoseRansacStage,
};
use calib::session::{
    CalibrationSession, HandEyeModeConfig, HandEyeSingleInitOptions, HandEyeSingleObservations,
    HandEyeSingleOptimOptions, HandEyeSingleProblem,
};
use chess_corners::ChessConfig;
use handeye_io::{kuka_chessboard_params, load_kuka_dataset_with_progress};
use std::io::{self, Write};
use std::path::Path;

fn main() -> Result<()> {
    println!("=== Hand-Eye Single Camera Calibration (Session + Pipeline) ===\n");

    let base_path = Path::new("data/kuka_1");
    let chess_config = ChessConfig::default();
    let board_params = kuka_chessboard_params();

    println!("Extracting calibration features (chessboard corners)...");
    let (samples, summary) =
        load_kuka_dataset_with_progress(base_path, &chess_config, &board_params, |idx, total| {
            print!("\r  processing image {idx}/{total}");
            let _ = io::stdout().flush();
        })?;
    println!();
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

    let mut session = CalibrationSession::<HandEyeSingleProblem>::new();
    session.set_observations(HandEyeSingleObservations {
        views,
        mode: HandEyeModeConfig::EyeInHand,
    });

    let init = session.initialize(HandEyeSingleInitOptions::default())?;
    print_intrinsics_stage("Intrinsics init", &init.intrinsics_init);

    let mut optim_opts = HandEyeSingleOptimOptions::default();
    optim_opts.refine_robot_poses = true;
    optim_opts.robot_rot_sigma = 0.5_f64.to_radians();
    optim_opts.robot_trans_sigma = 1.0e-3;
    session.optimize(optim_opts)?;

    let report = session.export()?;
    print_intrinsics_stage("Intrinsics optimized", &report.intrinsics_optimized);
    print_ransac_stage(&report.pose_ransac);
    print_intrinsics_stage(
        "Intrinsics optimized (inliers)",
        &report.intrinsics_optimized_inliers,
    );
    print_handeye_stage("Hand-eye DLT init", &report.handeye_init);
    print_handeye_stage(
        "Hand-eye optimized (fixed robot poses)",
        &report.handeye_optimized,
    );
    if let Some(refined) = &report.handeye_optimized_refined {
        print_handeye_stage("Hand-eye optimized (refine robot poses)", refined);
    }

    Ok(())
}

fn print_intrinsics_stage(label: &str, stage: &IntrinsicsStage) {
    println!(
        "{label}: mean reproj error {:.3} px",
        stage.mean_reproj_error
    );
}

fn print_ransac_stage(stage: &PoseRansacStage) {
    println!(
        "Pose RANSAC: kept {} views, mean reproj error {:.3} px",
        stage.views.len(),
        stage.mean_reproj_error
    );
}

fn print_handeye_stage(label: &str, stage: &HandEyeStage) {
    println!(
        "{label}: mean reproj error {:.3} px",
        stage.mean_reproj_error
    );
}
