//! Tauri 2 shell of Calibration Studio, the calibration-rs desktop app.
//!
//! What the app computes lives in `studio_core`; this crate exposes it to
//! the React frontend:
//!
//! - [`commands`] — the Tauri commands, thin wrappers over `studio_core`:
//!   projects, exports, text files, folder sniffing, samples, undistorted
//!   images, epipolar overlays, uncertainty, disparity and the calibration
//!   runner (with stage progress and cancellation).
//! - [`viewer`] — the `calib-viewer` binary's localhost HTTP back end over
//!   the same command bodies: one frontend, two back ends.
//!
//! Source images are not sent over IPC: the webview loads them through the
//! asset protocol, scoped to the folders the user opened ([`assets`]).

mod assets;
mod commands;
pub mod viewer;

use studio_core::export_cache::ExportCache;
use studio_core::run::RunRegistry;

/// Entry point invoked from `main.rs`. Wires up the dialog plugin, the
/// shared export cache, and the project + viewer + math + runner commands.
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .manage(ExportCache::new())
        .manage(RunRegistry::new())
        .invoke_handler(tauri::generate_handler![
            commands::create_project,
            commands::open_project,
            commands::save_project_manifest,
            commands::save_project_run,
            commands::load_export,
            commands::set_active_export,
            commands::load_undistorted_image,
            commands::load_text_file,
            commands::fetch_sample_cmd,
            commands::sniff_folder,
            commands::compute_epipolar_overlay,
            commands::compute_epipolar_overlay_undistorted,
            commands::undistort_points,
            commands::camera_parameter_values,
            commands::projection_uncertainty,
            commands::compute_disparity,
            commands::run_calibration_cmd,
            commands::cancel_run_cmd,
            commands::default_config_cmd,
        ])
        .run(tauri::generate_context!())
        .expect("failed to launch Calibration Studio");
}
