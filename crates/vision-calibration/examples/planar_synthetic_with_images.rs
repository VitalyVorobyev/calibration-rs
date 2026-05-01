//! Planar intrinsics calibration with synthesized PNGs + image manifest.
//!
//! Source of truth for the v0 diagnose-UI fixture (Track B / ADR 0014).
//! All generation logic lives in [`synthetic_images`] so the regression
//! test (`tests/planar_synthetic_with_images.rs`) can call the same
//! function with a tempdir and assert on the output without spawning a
//! second cargo process.
//!
//! Run with:
//! `cargo run -p vision-calibration --example planar_synthetic_with_images`

use anyhow::Result;

#[path = "support/synthetic_images.rs"]
mod synthetic_images;

fn main() -> Result<()> {
    let out_dir = synthetic_images::default_fixture_dir();
    println!(
        "Rendering {} views @ {}×{} with {}×{} corners each.",
        synthetic_images::NUM_VIEWS,
        synthetic_images::IMAGE_W,
        synthetic_images::IMAGE_H,
        synthetic_images::BOARD_COLS,
        synthetic_images::BOARD_ROWS,
    );

    let summary = synthetic_images::write_fixture(&out_dir)?;
    println!(
        "Wrote {} ({} per-feature residuals, mean {:.4} px, max {:.4} px)",
        summary.export_path.display(),
        summary.num_residuals,
        summary.mean_error_px,
        summary.max_error_px,
    );
    println!("Open this in the diagnose UI: {}", out_dir.display());

    Ok(())
}
