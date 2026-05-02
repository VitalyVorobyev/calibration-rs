//! Tauri commands exposed to the React frontend.
//!
//! The wire format is intentionally untyped (`serde_json::Value`) for
//! `load_export` so we do not have to mirror every Rust field as a TS
//! interface for the v0 viewer — the frontend narrows to the small subset
//! it actually consumes (residuals + manifest + mean error). Trade-off
//! recorded in ADR 0014.

use base64::Engine;
use serde::Serialize;
use std::path::PathBuf;
use tauri::State;

use crate::epipolar::{self, EpipolarOverlay};
use crate::export_cache::ExportCache;

/// Successful response for [`load_export`].
#[derive(Serialize)]
pub struct LoadExportResult {
    /// Raw parsed export JSON. The frontend treats this as `AnyExport`
    /// (subset interface) and ignores fields it does not render.
    pub export: serde_json::Value,
    /// Absolute path to the directory containing `export.json`. The
    /// frontend joins this with the manifest's `root` and per-frame
    /// `path` to obtain absolute image filenames it can hand back to
    /// [`load_image`].
    pub export_dir: String,
}

/// Read and parse a calibration export JSON file. Does NOT cache the
/// result — the frontend validates the export (e.g. requires
/// `image_manifest`) before the user can interact with it, and a
/// rejected export must not displace the cache the previous session is
/// still using. The frontend calls [`set_active_export`] explicitly
/// once it accepts the file.
#[tauri::command]
pub async fn load_export(path: String) -> Result<LoadExportResult, String> {
    let p = PathBuf::from(&path);
    let parent = p
        .parent()
        .ok_or_else(|| format!("export path has no parent directory: {path}"))?
        .to_path_buf();
    let bytes = std::fs::read(&p).map_err(|e| format!("read {}: {e}", p.display()))?;
    let export: serde_json::Value =
        serde_json::from_slice(&bytes).map_err(|e| format!("parse {}: {e}", p.display()))?;
    let export_dir = parent
        .canonicalize()
        .map_err(|e| format!("canonicalize {}: {e}", parent.display()))?
        .to_string_lossy()
        .into_owned();

    Ok(LoadExportResult { export, export_dir })
}

/// Commit the export the frontend has accepted as the active dataset.
///
/// Called by the frontend after [`load_export`] returned and the
/// frontend's validation (image_manifest required, frames non-empty,
/// …) passed. Writes `path` + the parsed `export` into the
/// [`ExportCache`] so subsequent math commands operate on exactly the
/// dataset the user is looking at.
#[tauri::command]
pub async fn set_active_export(
    path: String,
    export: serde_json::Value,
    cache: State<'_, ExportCache>,
) -> Result<(), String> {
    cache.set(path, export);
    Ok(())
}

/// Read an image file and return it as a `data:` URL the webview can use
/// directly. PNG is the only format the v0 fixture writes; the MIME type
/// is inferred from the extension.
#[tauri::command]
pub async fn load_image(path: String) -> Result<String, String> {
    let bytes = std::fs::read(&path).map_err(|e| format!("read {path}: {e}"))?;
    let mime = mime_for(&path);
    let b64 = base64::engine::general_purpose::STANDARD.encode(&bytes);
    Ok(format!("data:{mime};base64,{b64}"))
}

/// Compute the epipolar polyline + epipole for a click in pane A.
///
/// Reads the most recently loaded export from [`ExportCache`] (no disk
/// I/O), backprojects `point_px` through cam-A's full distortion +
/// (optional) Scheimpflug chain, transforms the resulting ray into
/// cam-B's frame via the rig extrinsics, and projects 64 logarithmically
/// spaced depth samples through cam-B's full chain. The returned line
/// is in distorted pane-B pixel coordinates and can be rendered as an
/// SVG `<polyline>` directly.
#[tauri::command]
pub async fn compute_epipolar_overlay(
    cam_a: usize,
    cam_b: usize,
    point_px: [f64; 2],
    cache: State<'_, ExportCache>,
) -> Result<EpipolarOverlay, String> {
    cache
        .read(|cached| epipolar::compute_overlay(&cached.value, cam_a, cam_b, point_px))
        .ok_or_else(|| "no export loaded yet".to_string())?
}

fn mime_for(path: &str) -> &'static str {
    let lower = path.to_ascii_lowercase();
    if lower.ends_with(".png") {
        "image/png"
    } else if lower.ends_with(".jpg") || lower.ends_with(".jpeg") {
        "image/jpeg"
    } else if lower.ends_with(".webp") {
        "image/webp"
    } else {
        // Default to octet-stream so the browser refuses to render
        // something we don't recognise rather than mis-decode it.
        "application/octet-stream"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mime_detection() {
        assert_eq!(mime_for("foo.png"), "image/png");
        assert_eq!(mime_for("FOO.PNG"), "image/png");
        assert_eq!(mime_for("foo.jpg"), "image/jpeg");
        assert_eq!(mime_for("foo.jpeg"), "image/jpeg");
        assert_eq!(mime_for("foo.bin"), "application/octet-stream");
    }
}
