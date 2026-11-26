//! Tauri 2 shell of Calibration Studio, the calibration-rs desktop app.
//!
//! What the app computes lives in `studio_core`, behind its command table
//! (`studio_core::api`). This crate carries the table to the React frontend:
//!
//! - [`commands`] — the Tauri transport: `studio_invoke` and the event
//!   subscription, over one `Api`.
//! - [`scheme`] — the `studio:` URI scheme: the table's resources (source
//!   images, undistorted views, blobs) by URL, as the browser gets them from
//!   `/api/r/…`.
//! - [`viewer`] — the `calib-viewer` binary's localhost HTTP transport over
//!   the same table (its read-only commands): one frontend, two back ends.

mod commands;
mod scheme;
pub mod viewer;

use std::sync::Arc;

use studio_core::api::{Api, Dirs};
use studio_core::scope::RootScope;
use tauri::Manager;

/// Entry point invoked from `main.rs`. Wires up the dialog plugin, the API,
/// the `studio:` scheme and the transport.
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .setup(|app| {
            let scope = RootScope::growable(None);
            app.manage::<commands::SharedApi>(Arc::new(Api::new(scope, Dirs::platform())));
            Ok(())
        })
        .register_asynchronous_uri_scheme_protocol(scheme::SCHEME, |ctx, request, responder| {
            let api = Arc::clone(ctx.app_handle().state::<commands::SharedApi>().inner());
            tauri::async_runtime::spawn_blocking(move || {
                responder.respond(scheme::respond(&api, &request));
            });
        })
        .invoke_handler(tauri::generate_handler![
            commands::studio_invoke,
            commands::studio_subscribe,
            commands::studio_unsubscribe,
        ])
        .run(tauri::generate_context!())
        .expect("failed to launch Calibration Studio");
}
