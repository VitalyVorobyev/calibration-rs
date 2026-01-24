//! Planar intrinsics calibration with real images.
//!
//! This example demonstrates the full calibration workflow with real data:
//! 1. Load calibration images from disk
//! 2. Detect chessboard corners
//! 3. Run initialization (Zhang's method with iterative distortion)
//! 4. Run non-linear optimization (bundle adjustment)
//! 5. Export and inspect results
//!
//! Run with: `cargo run -p calib --example planar_real`
//!
//! Dataset: Uses left camera images from `data/stereo/imgs/leftcamera/`

use anyhow::{Context, Result};
use calib::planar_intrinsics::{
    run_calibration_with_filtering, step_init, step_optimize, FilterOptions,
};
use calib::prelude::*;
use calib_targets::chessboard::ChessboardDetectionResult;
use calib_targets::{detect, ChessboardParams};
use chess_corners::ChessConfig;
use std::io::{self, Write};
use std::path::{Path, PathBuf};

// Board parameters (matches stereo_linear.json)
const BOARD_ROWS: u32 = 7;
const BOARD_COLS: u32 = 11;
const SQUARE_SIZE_M: f64 = 0.03; // 30mm squares

fn main() -> Result<()> {
    println!("=== Planar Intrinsics Calibration (Real Images) ===\n");

    let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let imgs_dir = repo_root.join("data/stereo/imgs/leftcamera");

    if !imgs_dir.exists() {
        anyhow::bail!(
            "Dataset not found at {}\nPlease ensure the stereo dataset is available.",
            imgs_dir.display()
        );
    }

    println!("Dataset: {}", imgs_dir.display());
    println!(
        "Board: {}x{}, square size {:.1}mm\n",
        BOARD_ROWS,
        BOARD_COLS,
        SQUARE_SIZE_M * 1000.0
    );

    // Detect chessboard corners in all images
    println!("Detecting chessboard corners...");
    let views = load_views_with_progress(&imgs_dir)?;
    println!("\nLoaded {} views\n", views.len());

    if views.len() < 3 {
        anyhow::bail!("Need at least 3 views for calibration, got {}", views.len());
    }

    // Create dataset
    let dataset = PlanarDataset::new(views.into_iter().map(View::without_meta).collect())?;

    // Create calibration session
    let mut session = CalibrationSession::<PlanarIntrinsicsProblem>::new();
    session.set_input(dataset)?;

    // Step-by-step calibration
    println!("--- Step 1: Initialization ---");
    step_init(&mut session, None)?;

    let init_k = session.state.initial_intrinsics.as_ref().unwrap();
    let init_dist = session.state.initial_distortion.as_ref().unwrap();
    println!(
        "  Intrinsics: fx={:.1}, fy={:.1}, cx={:.1}, cy={:.1}",
        init_k.fx, init_k.fy, init_k.cx, init_k.cy
    );
    println!(
        "  Distortion: k1={:.4}, k2={:.4}, p1={:.5}, p2={:.5}\n",
        init_dist.k1, init_dist.k2, init_dist.p1, init_dist.p2
    );

    println!("--- Step 2: Optimization ---");
    step_optimize(&mut session, None)?;

    let state = &session.state;
    println!("  Final cost: {:.2e}", state.final_cost.unwrap());
    println!(
        "  Mean reprojection error: {:.4} px",
        state.mean_reproj_error.unwrap()
    );
    println!();

    // Export results
    let export = session.export()?;
    let final_k = export.params.intrinsics();
    let final_dist = export.params.distortion();

    println!("--- Final Results ---");
    println!("  Intrinsics:");
    println!("    fx = {:.2}", final_k.fx);
    println!("    fy = {:.2}", final_k.fy);
    println!("    cx = {:.2}", final_k.cx);
    println!("    cy = {:.2}", final_k.cy);
    println!("  Distortion:");
    println!("    k1 = {:.6}", final_dist.k1);
    println!("    k2 = {:.6}", final_dist.k2);
    println!("    k3 = {:.6}", final_dist.k3);
    println!("    p1 = {:.6}", final_dist.p1);
    println!("    p2 = {:.6}", final_dist.p2);
    println!(
        "  Mean reprojection error: {:.4} px\n",
        export.mean_reproj_error
    );

    // Alternative: run with filtering for outlier removal
    println!("--- Alternative: With Outlier Filtering ---");
    let views2 = load_views_with_progress(&imgs_dir)?;
    let dataset2 = PlanarDataset::new(views2.into_iter().map(View::without_meta).collect())?;

    let mut session2 = CalibrationSession::<PlanarIntrinsicsProblem>::new();
    session2.set_input(dataset2)?;

    run_calibration_with_filtering(
        &mut session2,
        FilterOptions {
            max_reproj_error: 1.0, // Stricter threshold
            min_points_per_view: 10,
            remove_sparse_views: true,
        },
    )?;

    let export2 = session2.export()?;
    println!(
        "  Mean reprojection error (filtered): {:.4} px",
        export2.mean_reproj_error
    );

    Ok(())
}

fn load_views_with_progress(imgs_dir: &Path) -> Result<Vec<CorrespondenceView>> {
    let chess_config = ChessConfig::default();
    let board_params = ChessboardParams {
        expected_rows: Some(BOARD_ROWS),
        expected_cols: Some(BOARD_COLS),
        ..ChessboardParams::default()
    };

    // Find all left camera images
    let mut indices: Vec<usize> = std::fs::read_dir(imgs_dir)?
        .filter_map(|e| e.ok())
        .filter_map(|e| {
            let name = e.file_name();
            let name = name.to_str()?;
            let name = name.strip_prefix("Im_L_")?.strip_suffix(".png")?;
            name.parse().ok()
        })
        .collect();
    indices.sort();

    let total = indices.len();
    let mut views = Vec::new();
    let mut skipped = 0;

    for (i, idx) in indices.iter().enumerate() {
        print!(
            "\r  Processing image {}/{} (Im_L_{}.png)",
            i + 1,
            total,
            idx
        );
        io::stdout().flush()?;

        let path = imgs_dir.join(format!("Im_L_{}.png", idx));
        match detect_chessboard(&path, &chess_config, board_params.clone()) {
            Ok(Some(view)) => views.push(view),
            Ok(None) => skipped += 1,
            Err(e) => {
                eprintln!("\nWarning: Failed to process Im_L_{}.png: {}", idx, e);
                skipped += 1;
            }
        }
    }

    println!(
        "\r  Processed {}/{} images ({} skipped)    ",
        total, total, skipped
    );

    Ok(views)
}

fn detect_chessboard(
    path: &Path,
    chess_config: &ChessConfig,
    board_params: ChessboardParams,
) -> Result<Option<CorrespondenceView>> {
    let img = image::ImageReader::open(path)
        .with_context(|| format!("Failed to open {}", path.display()))?
        .decode()
        .with_context(|| format!("Failed to decode {}", path.display()))?
        .to_luma8();

    let Some(detection) = detect::detect_chessboard(&img, chess_config, board_params) else {
        return Ok(None);
    };

    Ok(Some(detection_to_view(detection)?))
}

fn detection_to_view(detection: ChessboardDetectionResult) -> Result<CorrespondenceView> {
    let mut points_3d = Vec::new();
    let mut points_2d = Vec::new();

    for corner in detection.detection.corners {
        let Some(grid) = corner.grid else {
            continue;
        };
        points_3d.push(Pt3::new(
            grid.i as f64 * SQUARE_SIZE_M,
            grid.j as f64 * SQUARE_SIZE_M,
            0.0,
        ));
        points_2d.push(Pt2::new(corner.position.x as f64, corner.position.y as f64));
    }

    anyhow::ensure!(
        points_3d.len() >= 4,
        "Insufficient corners after grid filtering"
    );

    CorrespondenceView::new(points_3d, points_2d)
}
