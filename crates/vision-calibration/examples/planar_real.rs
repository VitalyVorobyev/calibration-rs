//! Planar intrinsics calibration with real images.
//!
//! This example demonstrates the full calibration workflow with real data:
//! 1. Load calibration images from disk
//! 2. Detect chessboard corners
//! 3. Run initialization (Zhang's method with iterative distortion)
//! 4. Run non-linear optimization (bundle adjustment)
//! 5. Export and inspect results, with the uncertainty of the parameters
//!
//! Run with: `cargo run -p vision-calibration --example planar_real --release`
//!
//! Dataset: the 30 images of `data/kuka_1/` (a 17×28 chessboard with 20 mm
//! squares, 1928×1208 px). The images are stored with Git LFS; run
//! `git lfs pull` first if they are pointer files in your checkout.

#[path = "support/handeye_io.rs"]
mod handeye_io;

use anyhow::Result;
use std::io::{self, Write};
use std::path::Path;
use vision_calibration::core::{DistortionParams, NoiseLevel, Uncertainty};
use vision_calibration::planar_intrinsics::{
    IntrinsicsFilterOptions, run_calibration_with_filtering, step_init, step_optimize,
};
use vision_calibration::prelude::*;

fn main() -> Result<()> {
    println!("=== Planar Intrinsics Calibration (Real Images) ===\n");

    let data_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("the facade crate lives two levels below the repository root")
        .join("data/kuka_1");
    println!("Dataset: {}", data_dir.display());

    // Detect chessboard corners in all images
    println!("Detecting chessboard corners...");
    let views = load_views(&data_dir)?;
    println!("Loaded {} views\n", views.len());

    if views.len() < 3 {
        anyhow::bail!("Need at least 3 views for calibration, got {}", views.len());
    }

    // Create dataset
    let dataset = PlanarDataset::new(views.iter().cloned().map(View::without_meta).collect())?;

    // Create calibration session
    let mut session = CalibrationSession::<PlanarIntrinsicsProblem>::new();
    session.set_input(dataset)?;

    // Step-by-step calibration
    println!("--- Step 1: Initialization ---");
    let init = step_init(&mut session, None)?;

    let init_k = &init.intrinsics;
    let init_dist = &init.distortion;
    println!(
        "  Intrinsics: fx={:.1}, fy={:.1}, cx={:.1}, cy={:.1}",
        init_k.fx, init_k.fy, init_k.cx, init_k.cy
    );
    println!(
        "  Distortion: k1={:.4}, k2={:.4}, p1={:.5}, p2={:.5}\n",
        init_dist.k1, init_dist.k2, init_dist.p1, init_dist.p2
    );

    println!("--- Step 2: Optimization ---");
    let opt = step_optimize(&mut session, None)?;

    println!("  Final cost: {:.2e}", opt.final_cost);
    println!("  Mean reprojection error: {:.4} px", opt.mean_reproj_error);
    println!();

    // Export results
    let export = session.export()?;
    let final_k = export.params.camera.intrinsics;
    let final_dist = match export.params.camera.distortion {
        DistortionParams::BrownConrady5 { params } => Some(params),
        _ => None,
    };

    println!("--- Final Results ---");
    println!("  Intrinsics:");
    println!("    fx = {:.2}", final_k.fx);
    println!("    fy = {:.2}", final_k.fy);
    println!("    cx = {:.2}", final_k.cx);
    println!("    cy = {:.2}", final_k.cy);
    println!("  Distortion (Brown-Conrady5):");
    if let Some(d) = final_dist {
        println!("    k1 = {:.6}", d.k1);
        println!("    k2 = {:.6}", d.k2);
        println!("    k3 = {:.6}", d.k3);
        println!("    p1 = {:.6}", d.p1);
        println!("    p2 = {:.6}", d.p2);
    } else {
        println!("    (no Brown-Conrady5 distortion)");
    }
    println!(
        "  Mean reprojection error: {:.4} px",
        export.mean_reproj_error
    );
    match &export.uncertainty {
        Uncertainty::Available(cov) => {
            println!("  Standard deviations:");
            for label in &cov.labels {
                println!(
                    "    σ({label}) = {:.3e}",
                    cov.std_dev(label).unwrap_or(f64::NAN)
                );
            }
            println!(
                "    corr(fx, cx) = {:+.3}",
                cov.correlation("fx", "cx").unwrap_or(f64::NAN)
            );
            for noise in &cov.noise {
                match noise.level {
                    NoiseLevel::Sigma(s) => {
                        println!("    corner noise σ ≈ {s:.3} px ({} residuals)", noise.count)
                    }
                    NoiseLevel::Sigma0(s) => println!(
                        "    corner noise σ̂₀ ≈ {s:.3} × the stated ({} residuals)",
                        noise.count
                    ),
                }
            }
        }
        Uncertainty::Unavailable { reason } => println!("  No uncertainty: {reason:?}"),
    }
    println!();

    // Alternative: run with filtering for outlier removal
    println!("--- Alternative: With Outlier Filtering ---");
    let dataset2 = PlanarDataset::new(views.into_iter().map(View::without_meta).collect())?;

    let mut session2 = CalibrationSession::<PlanarIntrinsicsProblem>::new();
    session2.set_input(dataset2)?;

    let mut filter_opts = IntrinsicsFilterOptions::default();
    filter_opts.max_reproj_error = 1.0; // Stricter threshold
    filter_opts.min_points_per_view = 10;
    filter_opts.remove_sparse_views = true;
    run_calibration_with_filtering(&mut session2, Some(filter_opts))?;

    let export2 = session2.export()?;
    println!(
        "  Mean reprojection error (filtered): {:.4} px",
        export2.mean_reproj_error
    );

    Ok(())
}

/// Detects the chessboard in every image of the dataset, skipping images
/// where it is not found.
fn load_views(data_dir: &Path) -> Result<Vec<CorrespondenceView>> {
    let (samples, summary) = handeye_io::load_kuka_dataset_with_progress(
        data_dir,
        &handeye_io::kuka_chessboard_params(),
        |idx, total| {
            print!("\r  Processing image {idx}/{total}");
            let _ = io::stdout().flush();
        },
    )?;
    println!(
        "\r  Processed {} images ({} skipped), {} corners, {:.0} mm squares",
        summary.total_images,
        summary.skipped_views,
        summary.total_corners,
        summary.square_size_m * 1000.0
    );
    debug_assert_eq!(summary.used_views, samples.len());
    Ok(samples.into_iter().map(|s| s.view).collect())
}
