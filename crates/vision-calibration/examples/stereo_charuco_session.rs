//! Stereo rig calibration session on real ChArUco images.
//!
//! This example demonstrates multi-camera rig extrinsics calibration:
//! 1. Load stereo image pairs from disk
//! 2. Detect ChArUco corners in both cameras
//! 3. Initialize per-camera intrinsics (Zhang's method)
//! 4. Optimize per-camera intrinsics (bundle adjustment)
//! 5. Initialize rig extrinsics (camera-to-rig transforms)
//! 6. Optimize rig jointly (bundle adjustment)
//! 7. Export results including baseline measurement
//!
//! Run with: `cargo run -p vision-calibration --example stereo_charuco_session`
//!
//! Dataset: Uses `data/stereo_charuco/` (`cam1`/`cam2` camera pairs)

#[path = "support/stereo_charuco_io.rs"]
mod stereo_charuco_io;

use anyhow::{Result, ensure};
use chess_corners::ChessConfig;
use std::io::{self, Write};
use std::path::PathBuf;
use stereo_charuco_io::{
    BOARD_CELL_SIZE_M, BOARD_COLS, BOARD_DICTIONARY_NAME, BOARD_MARKER_SIZE_REL, BOARD_ROWS,
    load_stereo_charuco_input_with_progress, make_charuco_detector_params,
};
use vision_calibration::prelude::*;
use vision_calibration::rig_extrinsics::{
    RigExtrinsicsProblem, run_calibration, step_intrinsics_init_all, step_intrinsics_optimize_all,
    step_rig_init, step_rig_optimize,
};

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let max_views = parse_max_views(&args)?;

    let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let data_dir = repo_root.join("data/stereo_charuco");
    ensure!(
        data_dir.exists(),
        "stereo ChArUco dataset not found at {}",
        data_dir.display()
    );

    println!("=== Stereo Rig Calibration Session (ChArUco) ===\n");
    println!("Dataset: {}", data_dir.display());
    println!(
        "Board: {}x{}, cell size {:.2}mm, marker scale {:.2}, dict {}",
        BOARD_ROWS,
        BOARD_COLS,
        BOARD_CELL_SIZE_M * 1000.0,
        BOARD_MARKER_SIZE_REL,
        BOARD_DICTIONARY_NAME
    );
    if let Some(n) = max_views {
        println!("Max views: {n}");
    }
    println!();

    let chess_config = ChessConfig::default();
    let charuco_params = make_charuco_detector_params();

    println!("Detecting ChArUco corners...");
    let (input, summary) = load_stereo_charuco_input_with_progress(
        &data_dir,
        &chess_config,
        &charuco_params,
        max_views,
        |idx, total, suffix| {
            print!("\r  Processing pair {idx}/{total} ({suffix})");
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

    let mut session = CalibrationSession::<RigExtrinsicsProblem>::new();
    session.set_input(input)?;

    println!("--- Step 1: Per-Camera Intrinsics Initialization ---");
    step_intrinsics_init_all(&mut session, None)?;
    println!("  Initialization complete");
    println!();

    println!("--- Step 2: Per-Camera Intrinsics Optimization ---");
    step_intrinsics_optimize_all(&mut session, None)?;

    for (i, (cam, reproj)) in session
        .state
        .per_cam_intrinsics
        .as_ref()
        .unwrap()
        .iter()
        .zip(
            session
                .state
                .per_cam_reproj_errors
                .as_ref()
                .unwrap_or(&vec![0.0, 0.0])
                .iter(),
        )
        .enumerate()
    {
        let k = &cam.k;
        println!(
            "  Camera {}: fx={:.1}, fy={:.1}, cx={:.1}, cy={:.1}, reproj_err={:.3}px",
            i, k.fx, k.fy, k.cx, k.cy, reproj
        );
    }
    println!();

    println!("--- Step 3: Rig Extrinsics Initialization ---");
    step_rig_init(&mut session)?;
    let init_extr = session.state.initial_cam_se3_rig.as_ref().unwrap();
    print_baseline("Initial baseline", init_extr);
    println!();

    println!("--- Step 4: Rig Bundle Adjustment ---");
    step_rig_optimize(&mut session, None)?;
    let mean_reproj_error = session.state.rig_ba_reproj_error.unwrap_or(f64::NAN);
    println!(
        "  Rig BA mean reprojection error: {:.4} px",
        mean_reproj_error
    );
    if let Some(per_cam) = session.state.rig_ba_per_cam_reproj_errors.as_ref() {
        for (i, err) in per_cam.iter().enumerate() {
            println!("    Camera {}: {:.4} px", i, err);
        }
    }
    println!();

    println!("--- Final Per-Camera Results ---");
    for (i, cam) in session
        .state
        .per_cam_intrinsics
        .as_ref()
        .unwrap()
        .iter()
        .enumerate()
    {
        let k = &cam.k;
        let d = &cam.dist;
        println!("  Camera {i}:");
        println!(
            "    Intrinsics: fx={:.2}, fy={:.2}, cx={:.2}, cy={:.2}",
            k.fx, k.fy, k.cx, k.cy
        );
        println!(
            "    Distortion: k1={:.6}, k2={:.6}, p1={:.6}, p2={:.6}",
            d.k1, d.k2, d.p1, d.p2
        );
    }
    let output = session.require_output()?;
    let cam_se3_rig: Vec<Iso3> = output
        .params
        .cam_to_rig
        .iter()
        .map(|t| t.inverse())
        .collect();
    print_baseline("Rig baseline (after BA)", &cam_se3_rig);

    println!("--- Alternative: single facade function ---");
    let (input2, _summary2) = load_stereo_charuco_input_with_progress(
        &data_dir,
        &chess_config,
        &charuco_params,
        max_views,
        |idx, total, suffix| {
            print!("\r  Processing pair {idx}/{total} ({suffix})");
            let _ = io::stdout().flush();
        },
    )?;

    let mut session2 = CalibrationSession::<RigExtrinsicsProblem>::new();
    session2.set_input(input2)?;

    run_calibration(&mut session2)?;

    let export2 = session2.export()?;
    let mean_reproj_error = session2
        .state
        .rig_ba_reproj_error
        .unwrap_or(export2.mean_reproj_error);
    println!(
        "  Rig BA mean reprojection error: {:.4} px",
        mean_reproj_error
    );
    if let Some(per_cam) = session2.state.rig_ba_per_cam_reproj_errors.as_ref() {
        for (i, err) in per_cam.iter().enumerate() {
            println!("    Camera {}: {:.4} px", i, err);
        }
    }

    Ok(())
}

fn parse_max_views(args: &[String]) -> Result<Option<usize>> {
    let mut value = None;
    let mut i = 1usize;
    while i < args.len() {
        let arg = &args[i];
        if let Some(raw) = arg.strip_prefix("--max-views=") {
            ensure!(value.is_none(), "--max-views provided multiple times");
            let n: usize = raw
                .parse()
                .map_err(|_| anyhow::anyhow!("invalid --max-views value"))?;
            ensure!(n > 0, "--max-views must be > 0");
            value = Some(n);
            i += 1;
            continue;
        }

        if arg == "--max-views" {
            ensure!(value.is_none(), "--max-views provided multiple times");
            ensure!(i + 1 < args.len(), "--max-views requires a value");
            let raw = &args[i + 1];
            let n: usize = raw
                .parse()
                .map_err(|_| anyhow::anyhow!("invalid --max-views value"))?;
            ensure!(n > 0, "--max-views must be > 0");
            value = Some(n);
            i += 2;
            continue;
        }

        i += 1;
    }
    Ok(value)
}

fn print_baseline(label: &str, cam_to_rig: &[Iso3]) {
    if cam_to_rig.len() < 2 {
        return;
    }
    let t0 = cam_to_rig[0].inverse();
    let t1 = cam_to_rig[1].inverse();
    let baseline = (t1.translation.vector - t0.translation.vector).norm();
    println!(
        "  {label}: |t(cam1->rig)| = {baseline:.4} m ({:.2} mm)",
        baseline * 1000.0
    );
}
