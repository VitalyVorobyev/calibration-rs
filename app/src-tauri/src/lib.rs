//! Tauri 2 backend for the calibration-rs diagnose viewer (B0).
//!
//! See `app/README.md` and `docs/adrs/0014-tauri-desktop-app.md` for the
//! v0 scope. The backend exposes exactly two commands:
//!
//! - [`commands::load_export`] — read + parse an `export.json` produced
//!   by the calibration library, return its parsed contents and the
//!   absolute directory it lives in (so the frontend can resolve the
//!   image manifest's relative paths).
//! - [`commands::load_image`] — read a PNG file from disk, return a
//!   `data:` URL the webview can drop into a `<canvas>` via
//!   `Image()`.
//!
//! No long-lived state, no calibration mutation, no IPC events — the
//! viewer is a one-way passive consumer of an export bundle.

mod commands;

/// Entry point invoked from `main.rs`. Wires up the dialog plugin and
/// the two viewer commands and starts the Tauri runtime.
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .invoke_handler(tauri::generate_handler![
            commands::load_export,
            commands::load_image,
        ])
        .run(tauri::generate_context!())
        .expect("failed to launch calibration-diagnose");
}
