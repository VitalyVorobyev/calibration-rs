//! Generate viewer-loadable rig exports for the public test datasets.
//!
//! Walks the public datasets in `data/`, runs the calibration pipeline,
//! and writes a `viewer_export.json` for each one with the
//! `image_manifest` field populated against the actual image paths used
//! by the dataset loader. The B0 / B1 / B2 desktop workspaces (Diagnose
//! / 3D viewer / Epipolar) are then opened against these JSON files.
//!
//! Run with: `cargo run -p vision-calibration --example viewer_fixtures`
//!
//! Datasets covered (rig exports the new B1 + B2 workspaces consume):
//! - `data/stereo/imgs/` — stereo rig (7×11 chessboard, 30 mm)
//! - `data/stereo_charuco/` — stereo rig (22×22 ChArUco, 1.35 mm)
//!
//! `data/kuka_1/` is single-camera hand-eye and the SingleCamHandeye
//! export does not yet carry `image_manifest` (B3 follow-up). `data/DS8/`
//! has no example wired up yet.

#[path = "support/stereo_charuco_io.rs"]
mod stereo_charuco_io;
#[path = "support/stereo_io.rs"]
mod stereo_io;

use anyhow::{Context, Result, ensure};
use calib_targets::chessboard::DetectorParams;
use chess_corners::ChessConfig;
use std::path::{Path, PathBuf};
use stereo_charuco_io::{
    BOARD_CELL_SIZE_MM, BOARD_COLS, BOARD_DICTIONARY_NAME, BOARD_ROWS,
    load_stereo_charuco_input_with_progress, make_charuco_detector_params,
};
use stereo_io::load_stereo_input_with_progress;
use vision_calibration::core::{FrameRef, ImageManifest};
use vision_calibration::prelude::*;
use vision_calibration::rig_extrinsics::{
    RigExtrinsicsExport, RigExtrinsicsProblem, run_calibration,
};

const STEREO_SQUARE_SIZE_M: f64 = 0.03;

fn main() -> Result<()> {
    let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");

    println!("=== Viewer fixture generator ===\n");

    write_stereo_fixture(&repo_root).context("failed to generate stereo viewer fixture")?;
    println!();
    write_stereo_charuco_fixture(&repo_root)
        .context("failed to generate stereo_charuco viewer fixture")?;
    println!();

    println!("Done. Open each viewer_export.json from `bun run tauri dev`.");
    Ok(())
}

fn write_stereo_fixture(repo_root: &Path) -> Result<()> {
    let dataset_dir = repo_root.join("data/stereo");
    let imgs_dir = dataset_dir.join("imgs");
    ensure!(
        imgs_dir.exists(),
        "stereo dataset missing: {}",
        imgs_dir.display()
    );

    println!("[stereo] detecting chessboard corners (7×11, 30 mm)…");
    let chess_config = ChessConfig::default();
    let board_params = DetectorParams::default();
    let (input, summary) = load_stereo_input_with_progress(
        &imgs_dir,
        &chess_config,
        &board_params,
        STEREO_SQUARE_SIZE_M,
        None,
        |_, _, _| {},
    )?;
    println!(
        "[stereo] {} pairs total, {} accepted ({} skipped); usable left={} right={}",
        summary.total_pairs,
        summary.used_views,
        summary.skipped_views,
        summary.usable_left,
        summary.usable_right
    );

    println!("[stereo] running calibration…");
    let mut session = CalibrationSession::<RigExtrinsicsProblem>::new();
    session.set_input(input)?;
    run_calibration(&mut session)?;
    let mut export = session.export()?;
    println!(
        "[stereo] mean reproj error: {:.4} px",
        export.mean_reproj_error
    );

    // Manifest root = imgs/ (relative to data/stereo/), per-camera paths
    // join leftcamera/Im_L_*.png + rightcamera/Im_R_*.png. The dataset
    // loader's view_paths are absolute; convert to manifest-relative.
    let manifest = manifest_from_view_paths(&summary.view_paths, &imgs_dir, "imgs")?;
    export.image_manifest = Some(manifest);

    let out_path = dataset_dir.join("viewer_export.json");
    write_pretty(&out_path, &export)?;
    println!("[stereo] wrote {}", out_path.display());
    Ok(())
}

fn write_stereo_charuco_fixture(repo_root: &Path) -> Result<()> {
    let dataset_dir = repo_root.join("data/stereo_charuco");
    ensure!(
        dataset_dir.exists(),
        "stereo_charuco dataset missing: {}",
        dataset_dir.display()
    );

    println!(
        "[stereo_charuco] detecting ChArUco corners ({BOARD_ROWS}×{BOARD_COLS} board, \
         cell {BOARD_CELL_SIZE_MM:.4} mm, dict {BOARD_DICTIONARY_NAME})…"
    );
    let chess_config = ChessConfig::default();
    let charuco_params = make_charuco_detector_params();
    let (input, summary) = load_stereo_charuco_input_with_progress(
        &dataset_dir,
        &chess_config,
        &charuco_params,
        None,
        |_, _, _| {},
    )?;
    println!(
        "[stereo_charuco] {} pairs total, {} accepted ({} skipped); usable left={} right={}",
        summary.total_pairs,
        summary.used_views,
        summary.skipped_views,
        summary.usable_left,
        summary.usable_right
    );

    println!("[stereo_charuco] running calibration…");
    let mut session = CalibrationSession::<RigExtrinsicsProblem>::new();
    session.set_input(input)?;
    run_calibration(&mut session)?;
    let mut export = session.export()?;
    println!(
        "[stereo_charuco] mean reproj error: {:.4} px",
        export.mean_reproj_error
    );

    // Manifest root = "." (relative to data/stereo_charuco/); per-camera
    // paths join cam1/Cam1_*.png + cam2/Cam2_*.png.
    let manifest = manifest_from_view_paths(&summary.view_paths, &dataset_dir, ".")?;
    export.image_manifest = Some(manifest);

    let out_path = dataset_dir.join("viewer_export.json");
    write_pretty(&out_path, &export)?;
    println!("[stereo_charuco] wrote {}", out_path.display());
    Ok(())
}

/// Build an `ImageManifest` from the loader's per-view absolute paths.
///
/// `image_root_abs` is the directory the manifest's `root` resolves to
/// (the loader's `imgs_dir` for stereo, the dataset dir itself for
/// stereo_charuco); `manifest_root` is the path written into the
/// manifest, relative to the export JSON's directory.
fn manifest_from_view_paths(
    view_paths: &[[PathBuf; 2]],
    image_root_abs: &Path,
    manifest_root: &str,
) -> Result<ImageManifest> {
    let mut frames: Vec<FrameRef> = Vec::with_capacity(view_paths.len() * 2);
    for (pose_idx, pair) in view_paths.iter().enumerate() {
        for (cam_idx, abs) in pair.iter().enumerate() {
            let rel = abs.strip_prefix(image_root_abs).with_context(|| {
                format!(
                    "image path {} not under {}",
                    abs.display(),
                    image_root_abs.display()
                )
            })?;
            frames.push(FrameRef {
                pose: pose_idx,
                camera: cam_idx,
                path: rel.to_path_buf(),
                roi: None,
            });
        }
    }
    Ok(ImageManifest {
        root: PathBuf::from(manifest_root),
        frames,
    })
}

fn write_pretty(path: &Path, export: &RigExtrinsicsExport) -> Result<()> {
    let json = serde_json::to_string_pretty(export)
        .with_context(|| format!("serializing {}", path.display()))?;
    std::fs::write(path, json).with_context(|| format!("writing {}", path.display()))?;
    Ok(())
}
