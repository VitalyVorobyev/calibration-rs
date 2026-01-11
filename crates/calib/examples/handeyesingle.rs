//! Step-by-step hand-eye calibration with a single camera.
//!
//! Stages:
//! 1) Intrinsics initialization (mean reprojection error)
//! 2) Intrinsics optimization (mean reprojection error)
//! 3) Per-view pose RANSAC (mean reprojection error on inliers)
//! 4) Intrinsics re-optimization on inliers
//! 5) Hand-eye DLT initialization (mean reprojection error on inliers)
//! 6) Hand-eye optimization (mean reprojection error on inliers)
//! 7) Hand-eye optimization with robot pose refinement

#[path = "support/handeye_io.rs"]
mod handeye_io;

use anyhow::{ensure, Result};
use calib::pipeline::handeye_single::{
    init_handeye, init_intrinsics, optimize_handeye_stage, optimize_intrinsics,
    ransac_planar_poses, BackendSolveOptions, HandEyeMode, HandEyeSolveOptions, HandEyeView,
    IterativeIntrinsicsOptions, PlanarIntrinsicsSolveOptions, PoseRansacOptions, RobustLoss,
};
use chess_corners::ChessConfig;
use handeye_io::{kuka_chessboard_params, load_kuka_dataset_with_progress};
use std::io::{self, Write};
use std::path::Path;

fn main() -> Result<()> {
    println!("=== Hand-Eye Single Camera Calibration (Step-by-Step) ===\n");

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
    ensure!(
        samples.len() >= 3,
        "need at least 3 valid views, got {}",
        samples.len()
    );

    println!(
        "Loaded {}/{} views ({} skipped), {} corners, square size {:.4} m",
        summary.used_views,
        summary.total_images,
        summary.skipped_views,
        summary.total_corners,
        summary.square_size_m
    );

    let views: Vec<HandEyeView> = samples
        .into_iter()
        .map(|s| HandEyeView {
            view: s.view,
            robot_pose: s.robot_pose,
        })
        .collect();

    // 1) Intrinsics init
    let init_opts = IterativeIntrinsicsOptions::default();
    let intr_init = init_intrinsics(&views, &init_opts)?;
    print_intrinsics_stage("Intrinsics init", &intr_init);

    // 2) Intrinsics optimize
    let solve_opts = PlanarIntrinsicsSolveOptions {
        robust_loss: RobustLoss::Huber { scale: 2.0 },
        fix_k3: true,
        ..Default::default()
    };
    let backend_opts = BackendSolveOptions {
        max_iters: 60,
        ..Default::default()
    };
    let intr_opt = optimize_intrinsics(&views, &intr_init, &solve_opts, &backend_opts)?;
    print_intrinsics_stage("Intrinsics optimized", &intr_opt);

    // 3) Pose RANSAC (drop views with too few inliers)
    let ransac_opts = PoseRansacOptions::default();
    println!(
        "Pose RANSAC: thresh={:.2}px, min_inliers={}, max_iters={}",
        ransac_opts.thresh_px, ransac_opts.min_inliers, ransac_opts.max_iters
    );
    let pose_ransac = ransac_planar_poses(
        &views,
        &intr_opt.intrinsics,
        &intr_opt.distortion,
        &ransac_opts,
    )?;
    print_ransac_stage(&pose_ransac, views.len());

    // 4) Intrinsics re-optimization on inliers
    let intr_inliers =
        optimize_intrinsics(&pose_ransac.views, &intr_opt, &solve_opts, &backend_opts)?;
    print_intrinsics_stage("Intrinsics optimized (inliers)", &intr_inliers);

    // 5) Hand-eye DLT init
    let handeye_init = init_handeye(
        &pose_ransac.views,
        &intr_inliers.poses,
        &intr_inliers.intrinsics,
        &intr_inliers.distortion,
        HandEyeMode::EyeInHand,
    )?;
    print_handeye_stage("Hand-eye DLT init", &handeye_init);

    // 6) Hand-eye optimize (intrinsics/distortion fixed)
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
        ..Default::default()
    };

    let handeye_opt = optimize_handeye_stage(
        &pose_ransac.views,
        &handeye_init,
        &intr_inliers.intrinsics,
        &intr_inliers.distortion,
        HandEyeMode::EyeInHand,
        &handeye_opts,
        &BackendSolveOptions::default(),
    )?;
    print_handeye_stage("Hand-eye optimized (fixed robot poses)", &handeye_opt);

    // 7) Hand-eye optimize with robot pose refinement
    let mut handeye_opts_refine = handeye_opts.clone();
    handeye_opts_refine.refine_robot_poses = true;
    handeye_opts_refine.robot_rot_sigma = 0.5_f64.to_radians(); // radians
    handeye_opts_refine.robot_trans_sigma = 1.0e-3; // meters

    let handeye_opt_refine = optimize_handeye_stage(
        &pose_ransac.views,
        &handeye_init,
        &intr_inliers.intrinsics,
        &intr_inliers.distortion,
        HandEyeMode::EyeInHand,
        &handeye_opts_refine,
        &BackendSolveOptions::default(),
    )?;
    print_handeye_stage(
        "Hand-eye optimized (refine robot poses)",
        &handeye_opt_refine,
    );

    Ok(())
}

fn print_intrinsics_stage(label: &str, stage: &calib::pipeline::handeye_single::IntrinsicsStage) {
    let k = &stage.intrinsics;
    let d = &stage.distortion;
    println!(
        "\n{label}\n  mean reproj error: {:.3} px\n  K: fx={:.2}, fy={:.2}, cx={:.2}, cy={:.2}, skew={:.4}\n  dist: k1={:.4}, k2={:.4}, k3={:.4}, p1={:.4}, p2={:.4}",
        stage.mean_reproj_error,
        k.fx,
        k.fy,
        k.cx,
        k.cy,
        k.skew,
        d.k1,
        d.k2,
        d.k3,
        d.p1,
        d.p2
    );
}

fn print_ransac_stage(
    stage: &calib::pipeline::handeye_single::PoseRansacStage,
    total_views: usize,
) {
    let kept = stage.views.len();
    let dropped = stage.dropped_views;
    let avg_inliers = if stage.inliers_per_view.is_empty() {
        0.0
    } else {
        stage.inliers_per_view.iter().sum::<usize>() as f64 / stage.inliers_per_view.len() as f64
    };
    println!(
        "\nPose RANSAC\n  views kept: {kept}/{total_views} (dropped {dropped})\n  mean reproj error (inliers): {:.3} px\n  avg inliers/view: {:.1}",
        stage.mean_reproj_error,
        avg_inliers
    );
}

fn print_handeye_stage(label: &str, stage: &calib::pipeline::handeye_single::HandEyeStage) {
    let t = stage.handeye.translation.vector;
    let r = stage.handeye.rotation.to_rotation_matrix();
    println!(
        "\n{label}\n  mean reproj error (inliers): {:.3} px\n  handeye t = [{:.4}, {:.4}, {:.4}]",
        stage.mean_reproj_error, t.x, t.y, t.z
    );
    println!("  handeye R:\n{}", r.matrix());
}
