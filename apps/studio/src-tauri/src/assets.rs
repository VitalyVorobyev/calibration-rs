//! The asset protocol's scope. The webview loads image files straight from
//! disk through Tauri's asset protocol (`convertFileSrc`), and only from the
//! folders the user opened: a project's dataset, an export's image root, the
//! dataset folder of a run. `tauri.conf.json` starts the scope empty.

use std::path::Path;

use tauri::{AppHandle, Manager, Runtime};

/// Let the webview load files under `dir` (recursively). A folder that does
/// not exist is skipped: there is nothing in it to load.
pub fn allow_dir<R: Runtime>(app: &AppHandle<R>, dir: &Path) -> Result<(), String> {
    let Ok(dir) = dunce::canonicalize(dir) else {
        return Ok(());
    };
    app.asset_protocol_scope()
        .allow_directory(&dir, true)
        .map_err(|e| format!("asset scope {}: {e}", dir.display()))
}
