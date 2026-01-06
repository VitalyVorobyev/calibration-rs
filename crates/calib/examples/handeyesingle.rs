//! Example of hand-eye calibration with a single camera.
//!

use calib::prelude::*;
use calib::session::{PlanarIntrinsicsInitOptions, PlanarIntrinsicsOptimOptions};

use calib_targets::chessboard::ChessboardDetectionResult;
use calib_targets::{detect, ChessboardParams};
use chess_corners::ChessConfig;

use image::ImageReader;

use anyhow::Result;

fn detect_chessboards(
    chess_config: ChessConfig,
    board_params: ChessboardParams,
) -> Result<Vec<ChessboardDetectionResult>> {
    let base_path = "data/kuka_1";

    let mut results = Vec::with_capacity(30);
    for i in 1..=30 {
        let img_path = format!("{}/{:02}.png", base_path, i);
        let img = ImageReader::open(&img_path)?.decode()?.to_luma8();
        let result = detect::detect_chessboard(&img, &chess_config, board_params.clone()).ok_or(
            anyhow::anyhow!("Failed to detect chessboard in image {}", img_path),
        )?;
        results.push(result);
    }
    Ok(results)
}

fn detection_as_view_data(
    detection: Vec<ChessboardDetectionResult>,
    cell_size: f64,
) -> PlanarIntrinsicsObservations {
    let views = detection
        .into_iter()
        .map(|det| {
            let corn = det.detection.corners;
            let points_3d: Vec<Pt3> = corn
                .iter()
                .map(|c| {
                    Pt3::new(
                        c.grid.unwrap().i as f64 * cell_size,
                        c.grid.unwrap().j as f64 * cell_size,
                        0.0,
                    )
                })
                .collect();
            let points_2d: Vec<Vec2> = corn
                .iter()
                .map(|c| Vec2::new(c.position.x as f64, c.position.y as f64))
                .collect();
            PlanarViewData {
                points_3d,
                points_2d,
                weights: None,
            }
        })
        .collect();
    PlanarIntrinsicsObservations { views }
}

fn main() -> Result<()> {
    println!("=== Hand-Eye Single Camera Calibration Example ===\n");

    let chess_config = ChessConfig::default();
    let board_params = ChessboardParams::default();

    let chessboards = detect_chessboards(chess_config, board_params)?;
    let total_corners: usize = chessboards
        .iter()
        .map(|cb| cb.detection.corners.len())
        .sum();
    println!("Total detected corners: {}", total_corners);

    let cell_size_mm = 20f64;
    let views = detection_as_view_data(chessboards, cell_size_mm);

    let mut session = CalibrationSession::<PlanarIntrinsicsProblem>::new_with_description(
        "Basic planar intrinsics calibration".to_string(),
    );
    println!("✓ Created calibration session");

    session.set_observations(views);
    println!("✓ Set observations");

    session.initialize(PlanarIntrinsicsInitOptions::default())?;
    println!("✓ Initialization complete");

    println!("\nRunning non-linear optimization...");
    session.optimize(PlanarIntrinsicsOptimOptions::default())?;
    println!("✓ Optimization complete");

    let checkpoint_json = session.to_json()?;
    println!(
        "✓ Session state saved to JSON ({} bytes)",
        checkpoint_json.len()
    );
    std::fs::write("checkpoint_init.json", checkpoint_json)?;

    let final_results = session.export()?;
    println!("\n=== Final Results ===");
    println!("Final cost: {:.6}", final_results.report.final_cost);

    let results_json = serde_json::to_string_pretty(&final_results)?;
    std::fs::write("checkpoint_final.json", results_json)?;

    Ok(())
}
