//! Single-camera hand-eye calibration with real KUKA robot data.
//!
//! This example demonstrates the full hand-eye calibration workflow:
//! 1. Load images and robot poses from the KUKA dataset
//! 2. Detect chessboard corners in each image
//! 3. Initialize intrinsics (Zhang's method + distortion)
//! 4. Optimize intrinsics (bundle adjustment)
//! 5. Initialize hand-eye transform (Tsai-Lenz)
//! 6. Optimize hand-eye (bundle adjustment with robot pose priors)
//!
//! Dataset: Uses `data/kuka_1/` containing:
//! - 30 images (01.png to 30.png)
//! - Robot poses (RobotPosesVec.txt)
//! - Board square size (squaresize.txt)
//!
//! Run with: `cargo run -p calib --example handeye_session`

#[path = "support/handeye_io.rs"]
mod handeye_io;

use anyhow::Result;
use calib::prelude::*;
use calib::single_cam_handeye::{
    step_handeye_init, step_handeye_optimize, step_intrinsics_init, step_intrinsics_optimize,
    SingleCamHandeyeInput, SingleCamHandeyeProblemV2, SingleCamHandeyeView,
};
use chess_corners::ChessConfig;
use std::io::{self, Write};
use std::path::PathBuf;

fn main() -> Result<()> {
    println!("=== Single-Camera Hand-Eye Calibration (KUKA Dataset) ===\n");

    // Dataset path
    let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let data_path = repo_root.join("data/kuka_1");

    if !data_path.exists() {
        anyhow::bail!(
            "Dataset not found at {}\nPlease ensure the KUKA dataset is available.",
            data_path.display()
        );
    }

    println!("Dataset: {}", data_path.display());

    // Load dataset with corner detection
    println!("\nLoading images and detecting corners...");
    let chess_config = ChessConfig::default();
    let board_params = handeye_io::kuka_chessboard_params();

    let (samples, summary) = handeye_io::load_kuka_dataset_with_progress(
        &data_path,
        &chess_config,
        &board_params,
        |current, total| {
            print!("\r  Processing image {}/{}...", current, total);
            io::stdout().flush().ok();
        },
    )?;

    println!(
        "\r  Processed {} images                ",
        summary.total_images
    );
    println!(
        "  Board: 17x28, square size {:.1}mm",
        summary.square_size_m * 1000.0
    );
    println!(
        "  Valid views: {} (skipped: {})",
        summary.used_views, summary.skipped_views
    );
    println!("  Total corners: {}", summary.total_corners);
    println!();

    if samples.len() < 3 {
        anyhow::bail!("Need at least 3 valid views, got {}", samples.len());
    }

    // Convert to SingleCamHandeyeInput
    let views: Vec<SingleCamHandeyeView> = samples
        .into_iter()
        .map(|s| SingleCamHandeyeView {
            robot_pose: s.robot_pose,
            obs: s.view,
        })
        .collect();

    let input = SingleCamHandeyeInput::new(views)?;
    println!("Created input with {} views\n", input.num_views());

    // Create calibration session
    let mut session = CalibrationSession::<SingleCamHandeyeProblemV2>::new();
    session.set_input(input)?;

    // Step 1: Intrinsics initialization
    println!("--- Step 1: Intrinsics Initialization ---");
    step_intrinsics_init(&mut session, None)?;

    let init_k = session.state.initial_intrinsics.as_ref().unwrap();
    let init_dist = session.state.initial_distortion.as_ref().unwrap();
    println!(
        "  Intrinsics: fx={:.1}, fy={:.1}, cx={:.1}, cy={:.1}",
        init_k.fx, init_k.fy, init_k.cx, init_k.cy
    );
    println!(
        "  Distortion: k1={:.4}, k2={:.4}, p1={:.5}, p2={:.5}",
        init_dist.k1, init_dist.k2, init_dist.p1, init_dist.p2
    );
    println!();

    // Step 2: Intrinsics optimization
    println!("--- Step 2: Intrinsics Optimization ---");
    step_intrinsics_optimize(&mut session, None)?;

    let opt_cam = session.state.optimized_camera.as_ref().unwrap();
    println!(
        "  Intrinsics: fx={:.1}, fy={:.1}, cx={:.1}, cy={:.1}",
        opt_cam.k.fx, opt_cam.k.fy, opt_cam.k.cx, opt_cam.k.cy
    );
    println!(
        "  Distortion: k1={:.4}, k2={:.4}, p1={:.5}, p2={:.5}",
        opt_cam.dist.k1, opt_cam.dist.k2, opt_cam.dist.p1, opt_cam.dist.p2
    );
    let reproj_err = session.state.intrinsics_reproj_error.unwrap_or(f64::NAN);
    println!("  Reprojection error: {:.4} px", reproj_err);
    println!();

    // Step 3: Hand-eye initialization
    println!("--- Step 3: Hand-Eye Initialization (Tsai-Lenz) ---");
    step_handeye_init(&mut session, None)?;

    let init_he = session.state.initial_handeye.as_ref().unwrap();
    let init_target = session.state.initial_target_se3_base.as_ref().unwrap();
    println!("  Hand-eye |t|: {:.4}m", init_he.translation.vector.norm());
    println!(
        "  Target in base |t|: {:.4}m",
        init_target.translation.vector.norm()
    );
    println!();

    // Step 4: Hand-eye optimization
    println!("--- Step 4: Hand-Eye Optimization ---");
    step_handeye_optimize(&mut session, None)?;

    let reproj_err = session.state.handeye_reproj_error.unwrap_or(f64::NAN);
    println!("  Final reprojection error: {:.4} px", reproj_err);
    println!();

    // Export and display final results
    println!("--- Final Results ---");
    let export = session.export()?;

    println!("Camera intrinsics:");
    println!("  fx = {:.2}", export.camera.k.fx);
    println!("  fy = {:.2}", export.camera.k.fy);
    println!("  cx = {:.2}", export.camera.k.cx);
    println!("  cy = {:.2}", export.camera.k.cy);

    println!("Distortion:");
    println!("  k1 = {:.6}", export.camera.dist.k1);
    println!("  k2 = {:.6}", export.camera.dist.k2);
    println!("  k3 = {:.6}", export.camera.dist.k3);
    println!("  p1 = {:.6}", export.camera.dist.p1);
    println!("  p2 = {:.6}", export.camera.dist.p2);

    println!("Hand-eye transform (gripper → camera):");
    let he_t = export.handeye.translation.vector;
    let he_q = export.handeye.rotation;
    println!(
        "  Translation: [{:.4}, {:.4}, {:.4}]m",
        he_t.x, he_t.y, he_t.z
    );
    println!(
        "  Rotation (quat): [{:.4}, {:.4}, {:.4}, {:.4}]",
        he_q.i, he_q.j, he_q.k, he_q.w
    );

    println!("Target pose in base frame:");
    let target_t = export.target_se3_base.translation.vector;
    println!(
        "  Translation: [{:.4}, {:.4}, {:.4}]m",
        target_t.x, target_t.y, target_t.z
    );

    println!("Reprojection error:");
    println!("  Mean: {:.4} px", export.mean_reproj_error);
    println!("  Per-camera: {:?}", export.per_cam_reproj_errors);

    if let Some(deltas) = &export.robot_deltas {
        println!("Robot pose refinements ({} views):", deltas.len());
        let max_rot: f64 = deltas
            .iter()
            .map(|d| (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt())
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or(0.0);
        let max_trans: f64 = deltas
            .iter()
            .map(|d| (d[3] * d[3] + d[4] * d[4] + d[5] * d[5]).sqrt())
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or(0.0);
        println!(
            "  Max rotation delta: {:.4} rad ({:.2}°)",
            max_rot,
            max_rot.to_degrees()
        );
        println!(
            "  Max translation delta: {:.4}m ({:.2}mm)",
            max_trans,
            max_trans * 1000.0
        );
    }

    Ok(())
}
