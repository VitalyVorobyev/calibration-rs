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
use std::io::{self, Write};
use std::path::PathBuf;
use stereo_charuco_io::{
    BOARD_CELL_SIZE_MM, BOARD_COLS, BOARD_DICTIONARY_NAME, BOARD_MARKER_SIZE_REL, BOARD_ROWS,
    load_stereo_charuco_input_with_progress, make_charuco_detector_params,
};
use vision_calibration::core::Iso3;
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
        BOARD_ROWS, BOARD_COLS, BOARD_CELL_SIZE_MM, BOARD_MARKER_SIZE_REL, BOARD_DICTIONARY_NAME
    );
    if let Some(n) = max_views {
        println!("Max views: {n}");
    }
    println!();

    let charuco_params = make_charuco_detector_params();

    println!("Detecting ChArUco corners...");
    let (input, summary) = load_stereo_charuco_input_with_progress(
        &data_dir,
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
    let _intr_init = step_intrinsics_init_all(&mut session, None)?;
    println!("  Initialization complete");
    println!();

    println!("--- Step 2: Per-Camera Intrinsics Optimization ---");
    let intr_opt = step_intrinsics_optimize_all(&mut session, None)?;

    for (i, (cam, reproj)) in intr_opt
        .per_cam_intrinsics
        .iter()
        .zip(intr_opt.per_cam_reproj_errors.iter())
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
    let rig_init = step_rig_init(&mut session)?;
    print_baseline("Initial baseline", &rig_init.initial_cam_se3_rig);
    println!();

    println!("--- Step 4: Rig Bundle Adjustment ---");
    let rig_opt = step_rig_optimize(&mut session, None)?;
    println!(
        "  Rig BA mean reprojection error: {:.4} px",
        rig_opt.mean_reproj_error
    );
    for (i, err) in rig_opt.per_cam_reproj_errors.iter().enumerate() {
        println!("    Camera {}: {:.4} px", i, err);
    }
    println!();

    println!("--- Final Per-Camera Results ---");
    for (i, cam) in intr_opt.per_cam_intrinsics.iter().enumerate() {
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
    let cam_se3_rig: Vec<Iso3> = output.cam_to_rig().iter().map(|t| t.inverse()).collect();
    print_baseline("Rig baseline (after BA)", &cam_se3_rig);

    println!("--- Alternative: single facade function ---");
    let (input2, _summary2) = load_stereo_charuco_input_with_progress(
        &data_dir,
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
    println!(
        "  Rig BA mean reprojection error: {:.4} px",
        export2.mean_reproj_error
    );
    for (i, err) in export2.per_cam_reproj_errors.iter().enumerate() {
        println!("    Camera {}: {:.4} px", i, err);
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
