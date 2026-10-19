//! Stereo rig calibration session on real data.
//!
//! This example demonstrates multi-camera rig extrinsics calibration:
//! 1. Load the detected chessboard corners of a stereo pair
//! 2. Initialize per-camera intrinsics (Zhang's method)
//! 3. Optimize per-camera intrinsics (bundle adjustment)
//! 4. Initialize rig extrinsics (camera-to-rig transforms)
//! 5. Optimize rig jointly (bundle adjustment)
//! 6. Export results including baseline measurement
//!
//! Run with: `cargo run -p vision-calibration --example stereo_session`
//!
//! Dataset: `regression/observations/stereo_rig.json`, the corners of a
//! 7×11 chessboard with 30 mm squares, detected in 20 image pairs of a
//! 1024×576 stereo camera.

#[path = "support/observations.rs"]
mod observations;

use anyhow::{Result, ensure};
use vision_calibration::core::Iso3;
use vision_calibration::prelude::*;
use vision_calibration::rig_extrinsics::{
    RigExtrinsicsInput, RigExtrinsicsProblem, run_calibration, step_intrinsics_init_all,
    step_intrinsics_optimize_all, step_rig_init, step_rig_optimize,
};

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let max_views = parse_max_views(&args)?;

    println!("=== Stereo Rig Calibration Session ===\n");
    println!("Dataset: {}", observations::path("stereo_rig").display());
    let mut input: RigExtrinsicsInput = observations::load_input("stereo_rig")?;
    if let Some(n) = max_views {
        input.views.truncate(n);
    }
    let seen_by = |cam: usize| {
        input
            .views
            .iter()
            .filter(|v| v.obs.cameras[cam].is_some())
            .count()
    };
    println!(
        "Loaded {} views, usable: left={}, right={}",
        input.num_views(),
        seen_by(0),
        seen_by(1)
    );
    ensure!(
        input.num_views() >= 3,
        "need at least 3 usable views, got {}",
        input.num_views()
    );
    println!();

    // Create calibration session
    let mut session = CalibrationSession::<RigExtrinsicsProblem>::new();
    session.set_input(input.clone())?;

    // Step 1: Per-camera intrinsics initialization
    println!("--- Step 1: Per-Camera Intrinsics Initialization ---");
    step_intrinsics_init_all(&mut session, None)?;
    println!("  Initialization complete");
    println!();

    // Step 2: Per-camera intrinsics optimization
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

    // Step 3: Rig initialization
    println!("--- Step 3: Rig Extrinsics Initialization ---");
    let rig_init = step_rig_init(&mut session)?;

    print_baseline("Initial baseline", &rig_init.initial_cam_se3_rig);
    println!();

    // Step 4: Rig optimization
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

    // Show final per-camera results from intrinsics optimization
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
    println!();

    // The same calibration through the one-call wrapper
    println!("--- Alternative: single facade function ---");
    let mut session2 = CalibrationSession::<RigExtrinsicsProblem>::new();
    session2.set_input(input)?;
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
    for arg in args {
        let Some(raw) = arg.strip_prefix("--max-views=") else {
            continue;
        };
        let n: usize = raw
            .parse()
            .map_err(|_| anyhow::anyhow!("invalid --max-views value"))?;
        ensure!(n > 0, "--max-views must be > 0");
        return Ok(Some(n));
    }
    Ok(None)
}

/// The distance between the two camera centres. `cam_se3_rig[i]` maps rig
/// coordinates into camera `i`, so camera `i`'s centre in the rig frame is
/// the translation of its inverse.
fn print_baseline(label: &str, cam_se3_rig: &[Iso3]) {
    if cam_se3_rig.len() < 2 {
        return;
    }
    let c0 = cam_se3_rig[0].inverse().translation.vector;
    let c1 = cam_se3_rig[1].inverse().translation.vector;
    let baseline = (c1 - c0).norm();
    println!(
        "  {label}: |c1 - c0| = {baseline:.4} m ({:.2} mm)",
        baseline * 1000.0
    );
}
