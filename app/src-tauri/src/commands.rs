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

/// Successful response for [`load_export`].
#[derive(Serialize)]
pub struct LoadExportResult {
    /// Raw parsed export JSON. The frontend treats this as `PlanarExport`
    /// (subset interface) and ignores fields it does not render.
    pub export: serde_json::Value,
    /// Absolute path to the directory containing `export.json`. The
    /// frontend joins this with the manifest's `root` and per-frame
    /// `path` to obtain absolute image filenames it can hand back to
    /// [`load_image`].
    pub export_dir: String,
}

/// Read and parse a calibration export JSON file.
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
