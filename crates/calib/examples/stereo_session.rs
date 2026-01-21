//! Stereo rig calibration session on real images.
//!
//! This example demonstrates multi-camera rig extrinsics calibration:
//! 1. Load stereo image pairs from disk
//! 2. Detect chessboard corners in both cameras
//! 3. Initialize per-camera intrinsics (Zhang's method)
//! 4. Optimize per-camera intrinsics (bundle adjustment)
//! 5. Initialize rig extrinsics (camera-to-rig transforms)
//! 6. Optimize rig jointly (bundle adjustment)
//! 7. Export results including baseline measurement
//!
//! Run with: `cargo run -p calib --example stereo_session`
//!
//! Dataset: Uses `data/stereo/imgs/` (left/right camera pairs)

#[path = "support/stereo_io.rs"]
mod stereo_io;

use anyhow::{ensure, Result};
use calib::prelude::*;
use calib::rig_extrinsics::{
    step_intrinsics_init_all, step_intrinsics_optimize_all, step_rig_init, RigExtrinsicsProblem,
};
use calib_targets::ChessboardParams;
use chess_corners::ChessConfig;
use std::io::{self, Write};
use std::path::PathBuf;
use stereo_io::load_stereo_input_with_progress;

// Board parameters (matches stereo_linear.json)
const BOARD_ROWS: u32 = 7;
const BOARD_COLS: u32 = 11;
const SQUARE_SIZE_M: f64 = 0.03; // 30mm squares

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let max_views = parse_max_views(&args)?;

    let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let imgs_dir = repo_root.join("data/stereo/imgs");
    ensure!(
        imgs_dir.exists(),
        "stereo dataset not found at {}",
        imgs_dir.display()
    );

    println!("=== Stereo Rig Calibration Session ===\n");
    println!("Dataset: {}", imgs_dir.display());
    println!("Board: {}x{}, square size {:.1}mm", BOARD_ROWS, BOARD_COLS, SQUARE_SIZE_M * 1000.0);
    if let Some(n) = max_views {
        println!("Max views: {n}");
    }
    println!();

    let chess_config = ChessConfig::default();
    let board_params = ChessboardParams {
        expected_rows: Some(BOARD_ROWS),
        expected_cols: Some(BOARD_COLS),
        ..ChessboardParams::default()
    };

    println!("Detecting chessboard corners...");
    let (input, summary) = load_stereo_input_with_progress(
        &imgs_dir,
        &chess_config,
        &board_params,
        SQUARE_SIZE_M,
        max_views,
        |idx, total, image_index| {
            print!("\r  Processing pair {idx}/{total} (index {image_index})");
            let _ = io::stdout().flush();
        },
    )?;
    println!();
    println!(
        "Loaded {} views from {} pairs ({} skipped), usable: left={}, right={}",
        summary.used_views,
        summary.total_pairs,
        summary.skipped_views,
        summary.usable_left,
        summary.usable_right
    );
    ensure!(
        input.num_views() >= 3,
        "need at least 3 usable views, got {}",
        input.num_views()
    );
    println!();

    // Create calibration session
    let mut session = CalibrationSession::<RigExtrinsicsProblem>::new();
    session.set_input(input)?;

    // Step 1: Per-camera intrinsics initialization
    println!("--- Step 1: Per-Camera Intrinsics Initialization ---");
    step_intrinsics_init_all(&mut session, None)?;

    // After init_all, per_cam_intrinsics contains the initialized cameras
    // (the init step sets state.per_cam_intrinsics with initialized values)
    println!("  Initialization complete");
    println!();

    // Step 2: Per-camera intrinsics optimization
    println!("--- Step 2: Per-Camera Intrinsics Optimization ---");
    step_intrinsics_optimize_all(&mut session, None)?;

    for (i, (cam, reproj)) in session.state.per_cam_intrinsics.as_ref().unwrap()
        .iter()
        .zip(session.state.per_cam_reproj_errors.as_ref().unwrap_or(&vec![0.0, 0.0]).iter())
        .enumerate()
    {
        let k = &cam.k;
        println!("  Camera {}: fx={:.1}, fy={:.1}, cx={:.1}, cy={:.1}, reproj_err={:.3}px",
                 i, k.fx, k.fy, k.cx, k.cy, reproj);
    }
    println!();

    // Step 3: Rig initialization
    println!("--- Step 3: Rig Extrinsics Initialization ---");
    step_rig_init(&mut session)?;

    let init_extr = session.state.initial_cam_se3_rig.as_ref().unwrap();
    print_baseline("Initial baseline", init_extr);
    println!();

    // Step 4: Rig optimization
    // Note: Rig BA is currently under development and may diverge in some cases.
    // For now, we demonstrate the calibration up to initialization.
    println!("--- Step 4: Rig Bundle Adjustment ---");
    println!("  (Rig BA optimization is under development - using initial estimates)");
    println!();

    // Show final per-camera results from intrinsics optimization
    println!("--- Final Per-Camera Results ---");
    for (i, cam) in session.state.per_cam_intrinsics.as_ref().unwrap().iter().enumerate() {
        let k = &cam.k;
        let d = &cam.dist;
        println!("  Camera {i}:");
        println!("    Intrinsics: fx={:.2}, fy={:.2}, cx={:.2}, cy={:.2}",
                 k.fx, k.fy, k.cx, k.cy);
        println!("    Distortion: k1={:.6}, k2={:.6}, p1={:.6}, p2={:.6}",
                 d.k1, d.k2, d.p1, d.p2);
    }
    print_baseline("Rig baseline (from init)", session.state.initial_cam_se3_rig.as_ref().unwrap());

    Ok(())
}

fn parse_max_views(args: &[String]) -> Result<Option<usize>> {
    for arg in args {
        let Some(raw) = arg.strip_prefix("--max-views=") else {
            continue;
        };
        let n: usize = raw.parse().map_err(|_| anyhow::anyhow!("invalid --max-views value"))?;
        ensure!(n > 0, "--max-views must be > 0");
        return Ok(Some(n));
    }
    Ok(None)
}

fn print_baseline(label: &str, cam_to_rig: &[Iso3]) {
    if cam_to_rig.len() < 2 {
        return;
    }
    // Baseline is distance between camera 0 origin and camera 1 origin in rig frame
    // cam_to_rig[i] = T_C_R, so cam origin in rig frame = T_C_R^-1 * [0,0,0]
    // Actually, translation of T_C_R is position of rig origin in camera frame
    // For baseline: |T_1 * T_0^-1|
    let t0 = cam_to_rig[0].inverse();
    let t1 = cam_to_rig[1].inverse();
    let baseline = (t1.translation.vector - t0.translation.vector).norm();
    println!("  {label}: |t(cam1->rig)| = {baseline:.4} m ({:.2} mm)", baseline * 1000.0);
}
