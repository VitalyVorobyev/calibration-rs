//! Tauri 2 shell of Calibration Studio, the calibration-rs desktop app.
//!
//! What the app computes lives in `studio_core`, behind its command table
//! (`studio_core::api`). This crate carries the table to the React frontend:
//!
//! - [`commands`] — the Tauri transport: `studio_invoke`, the event
//!   subscription and resources, over one `Api`.
//! - [`viewer`] — the `calib-viewer` binary's localhost HTTP transport over
//!   the same table (its read-only commands): one frontend, two back ends.
//!
//! Source images are not sent over IPC: the webview loads them through the
//! asset protocol, scoped to the folders the user opened ([`assets`]). Every
//! folder the API's scope grants (a project, its dataset, an export's image
//! root, a run's dataset) is granted to the asset protocol too.

mod assets;
mod commands;
pub mod viewer;

use std::sync::Arc;

use studio_core::api::Api;
use studio_core::scope::RootScope;
use tauri::Manager;

/// Entry point invoked from `main.rs`. Wires up the dialog plugin, the API
/// (its scope mirrored into the asset protocol's) and the transport.
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .setup(|app| {
            let handle = app.handle().clone();
            let scope = RootScope::growable(Some(Box::new(move |dir| {
                if let Err(e) = assets::allow_dir(&handle, dir) {
                    eprintln!("calibration-studio: {e}");
                }
            })));
            app.manage::<commands::SharedApi>(Arc::new(Api::new(scope)));
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            commands::studio_invoke,
            commands::studio_subscribe,
            commands::studio_unsubscribe,
            commands::studio_resource,
        ])
        .run(tauri::generate_context!())
        .expect("failed to launch Calibration Studio");
}
