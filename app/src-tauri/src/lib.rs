//! Tauri 2 backend for the calibration-rs diagnose viewer.
//!
//! Commands exposed to the React frontend:
//!
//! - [`commands::load_export`] — read + parse an `export.json` produced
//!   by the calibration library, return its parsed contents and the
//!   absolute directory it lives in (so the frontend can resolve the
//!   image manifest's relative paths). Also caches the parsed JSON for
//!   downstream math commands.
//! - [`commands::load_image`] — read an image file from disk, return a
//!   `data:` URL the webview can drop into a `<canvas>` via `Image()`.
//! - [`commands::compute_epipolar_overlay`] — server-side epipolar line
//!   computation through the canonical camera models in
//!   `vision_calibration_core`. See ADR 0014 for the architecture
//!   rationale (no TS-side projection).

mod commands;
mod epipolar;
mod export_cache;
mod run;

use export_cache::ExportCache;

/// Entry point invoked from `main.rs`. Wires up the dialog plugin, the
/// shared export cache, and the viewer + math + runner commands.
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .manage(ExportCache::new())
        .invoke_handler(tauri::generate_handler![
            commands::load_export,
            commands::set_active_export,
            commands::load_image,
            commands::load_undistorted_image,
            commands::load_text_file,
            commands::compute_epipolar_overlay,
            commands::compute_epipolar_overlay_undistorted,
            commands::undistort_points,
            run::run_calibration_cmd,
        ])
        .run(tauri::generate_context!())
        .expect("failed to launch calibration-diagnose");
}
