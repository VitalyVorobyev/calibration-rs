//! Tauri 2 backend for the calibration-rs desktop app.
//!
//! Commands exposed to the React frontend, by module:
//!
//! - [`project`] — project folders: create, open, save the manifest, save
//!   a run (`runs/<name>/`). See ADR 0014.
//! - [`commands`] — exports (load, set active), text files, folder
//!   sniffing, undistorted images, epipolar overlays, uncertainty.
//! - [`run`] — the calibration runner, with stage progress and
//!   cancellation.
//! - [`disparity`] — dense stereo for the Depth workspace.
//!
//! Source images are not sent over IPC: the webview loads them through the
//! asset protocol, scoped to the folders the user opened ([`assets`]).

mod assets;
mod commands;
mod disparity;
mod epipolar;
mod export_cache;
mod laser;
mod project;
mod run;
mod uncertainty;

use export_cache::ExportCache;
use run::RunRegistry;

/// Tauri command payload/response types re-exported for the `emit_schemas`
/// binary. Public only under `schema-export` so it never widens
/// the app's runtime surface. These are exactly the roots `emit_schemas`
/// calls `subschema_for::<…>()` on directly; `PointCloud` is *not* listed
/// here even though it's public in `disparity` — it only reaches the
/// emitted schema transitively, as a `$defs` entry nested inside
/// `DisparityResult`.
#[cfg(feature = "schema-export")]
pub mod schema_types {
    pub use crate::disparity::DisparityResult;
    pub use crate::epipolar::EpipolarOverlay;
    pub use crate::project::{ProjectError, ProjectInfo};
    pub use crate::run::RunProgress;
    pub use vision_calibration::optim::ProjectionUncertainty;
}

/// Entry point invoked from `main.rs`. Wires up the dialog plugin, the
/// shared export cache, and the project + viewer + math + runner commands.
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .manage(ExportCache::new())
        .manage(RunRegistry::new())
        .invoke_handler(tauri::generate_handler![
            project::create_project,
            project::open_project,
            project::save_project_manifest,
            project::save_project_run,
            commands::load_export,
            commands::set_active_export,
            commands::load_undistorted_image,
            commands::load_text_file,
            commands::repo_root_cmd,
            commands::sniff_folder,
            commands::compute_epipolar_overlay,
            commands::compute_epipolar_overlay_undistorted,
            commands::undistort_points,
            commands::camera_parameter_values,
            commands::projection_uncertainty,
            disparity::compute_disparity,
            run::run_calibration_cmd,
            run::cancel_run_cmd,
            run::default_config_cmd,
        ])
        .run(tauri::generate_context!())
        .expect("failed to launch calibration-diagnose");
}
