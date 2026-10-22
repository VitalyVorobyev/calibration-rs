//! Tauri commands exposed to the React frontend.
//!
//! The wire format is intentionally untyped (`serde_json::Value`) for
//! `load_export` so we do not have to mirror every Rust field as a TS
//! interface for the v0 viewer — the frontend narrows to the small subset
//! it actually consumes (residuals + manifest + mean error). Trade-off
//! recorded in ADR 0014.

use serde::Serialize;
use std::path::{Path, PathBuf};
use tauri::ipc::Response;
use tauri::{AppHandle, Runtime, State};
use vision_calibration::core::PixelRect;

use crate::epipolar::{self, EpipolarOverlay};
use crate::export_cache::ExportCache;
use crate::uncertainty;
use vision_calibration::optim::ProjectionUncertainty;

/// Successful response for [`load_export`].
#[derive(Serialize)]
pub struct LoadExportResult {
    /// Raw parsed export JSON. The frontend treats this as `AnyExport`
    /// (subset interface) and ignores fields it does not render.
    pub export: serde_json::Value,
    /// Absolute path to the directory containing `export.json`. The
    /// frontend joins this with the manifest's `root` and per-frame
    /// `path` to obtain absolute image filenames, which it loads through
    /// the asset protocol.
    pub export_dir: String,
}

/// Read and parse a calibration export JSON file. Does NOT cache the
/// result — the frontend validates the export (e.g. requires
/// `image_manifest`) before the user can interact with it, and a
/// rejected export must not displace the cache the previous session is
/// still using. The frontend calls [`set_active_export`] explicitly
/// once it accepts the file. The export's image root becomes loadable
/// through the asset protocol.
#[tauri::command]
pub async fn load_export<R: Runtime>(
    app: AppHandle<R>,
    path: String,
) -> Result<LoadExportResult, String> {
    let p = PathBuf::from(&path);
    let parent = p
        .parent()
        .ok_or_else(|| format!("export path has no parent directory: {path}"))?
        .to_path_buf();
    let bytes = std::fs::read(&p).map_err(|e| format!("read {}: {e}", p.display()))?;
    let export: serde_json::Value =
        serde_json::from_slice(&bytes).map_err(|e| format!("parse {}: {e}", p.display()))?;
    let export_dir = dunce::canonicalize(&parent)
        .map_err(|e| format!("canonicalize {}: {e}", parent.display()))?;
    if let Some(images) = image_root(&export, &export_dir) {
        crate::assets::allow_dir(&app, &images)?;
    }

    Ok(LoadExportResult {
        export,
        export_dir: export_dir.to_string_lossy().replace('\\', "/"),
    })
}

/// The folder an export's image manifest resolves against: its `root`
/// joined onto the export's own folder. `None` without a manifest.
fn image_root(export: &serde_json::Value, export_dir: &Path) -> Option<PathBuf> {
    let root = export.get("image_manifest")?.get("root")?.as_str()?;
    Some(export_dir.join(root))
}

/// Commit the export the frontend has accepted as the active dataset.
///
/// Called by the frontend after [`load_export`] returned and the
/// frontend's validation (image_manifest required, frames non-empty,
/// …) passed. Writes the parsed `export` into the
/// [`ExportCache`] so subsequent math commands operate on exactly the
/// dataset the user is looking at.
#[tauri::command]
pub async fn set_active_export(
    export: serde_json::Value,
    cache: State<'_, ExportCache>,
) -> Result<(), String> {
    cache.set(export);
    Ok(())
}

/// Read a UTF-8 text file (TOML / JSON / arbitrary text) and return
/// its contents. Used by the Run workspace's preset loader to fetch
/// `data/<dataset>/dataset.toml` files at runtime so they round-trip
/// to the form state via the same TOML→Manifest decoder the Rust
/// runner uses. Errors carry the path so the UI can surface them
/// directly.
#[tauri::command]
pub async fn load_text_file(path: String) -> Result<String, String> {
    std::fs::read_to_string(&path).map_err(|e| format!("read {path}: {e}"))
}

/// Absolute path to the `calibration-rs` workspace root, resolved via
/// `CARGO_MANIFEST_DIR` (this crate always lives at
/// `<repo-root>/app/src-tauri`, a structural invariant of the layout —
/// see `app/README.md`'s dev notes). `CARGO_MANIFEST_DIR` is a
/// compile-time constant, but since `app/src-tauri` is never shipped as
/// a prebuilt binary (it's rebuilt locally by every `bun run tauri
/// dev`/`build`), it always reflects *this developer's own checkout*.
///
/// Frontend preset manifests (`RunWorkspace/presets.ts`) store paths
/// relative to this root instead of a hard-coded personal absolute
/// path; `lib/tauri.ts`'s `repoRoot()` calls this command once per
/// session and joins it with each preset's relative path.
#[tauri::command]
pub fn repo_root_cmd() -> Result<String, String> {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest_dir
        // app/src-tauri -> app
        .parent()
        // app -> <repo root>
        .and_then(|p| p.parent())
        .map(|p| p.to_string_lossy().into_owned())
        .ok_or_else(|| {
            format!(
                "CARGO_MANIFEST_DIR has no grandparent: {}",
                manifest_dir.display()
            )
        })
}

/// Heuristically infer a `DatasetSpec` from a dataset folder.
///
/// Walks `folder`, groups images into cameras, detects a robot-pose file,
/// and guesses the topology — leaving every field it cannot determine at a
/// placeholder and recording the dotted path in the manifest's `_unresolved`
/// array. The frontend loads the returned JSON straight into the Run
/// workspace's manifest form; the non-empty `_unresolved` list drives the red
/// badges and blocks Run until the user fills and clears them (ADR 0016).
/// This is the same inference the `generate-manifest` CLI uses.
#[tauri::command]
pub async fn sniff_folder(folder: String) -> Result<serde_json::Value, String> {
    let spec = vision_calibration::dataset::sniff_folder(std::path::Path::new(&folder))
        .map_err(|e| e.to_string())?;
    serde_json::to_value(&spec).map_err(|e| format!("serialize manifest: {e}"))
}

/// Read an image file and render it into a camera's undistorted pixel
/// frame, returned as PNG bytes (a binary IPC response, not text). Only
/// the Epipolar workspace uses it; the source images themselves are loaded
/// by the webview through the asset protocol.
#[tauri::command]
pub async fn load_undistorted_image(
    path: String,
    camera: usize,
    roi: Option<PixelRect>,
    cache: State<'_, ExportCache>,
) -> Result<Response, String> {
    let path_buf = PathBuf::from(&path);
    let bytes = cache
        .read(|cached| epipolar::undistort_image_png(&cached.value, &path_buf, camera, roi))
        .ok_or_else(|| "no export loaded yet".to_string())??;
    Ok(Response::new(bytes))
}

/// Map raw/distorted camera pixels into the undistorted pixel frame
/// used by the Epipolar workspace.
#[tauri::command]
pub async fn undistort_points(
    camera: usize,
    points_px: Vec<[f64; 2]>,
    cache: State<'_, ExportCache>,
) -> Result<Vec<[f64; 2]>, String> {
    cache
        .read(|cached| epipolar::undistort_points(&cached.value, camera, points_px))
        .ok_or_else(|| "no export loaded yet".to_string())?
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

/// Compute a clipped straight epipolar segment in cam-B's undistorted
/// image frame. `point_px` is in cam-A's undistorted pixel frame.
#[tauri::command]
pub async fn compute_epipolar_overlay_undistorted(
    cam_a: usize,
    cam_b: usize,
    point_px: [f64; 2],
    image_width_b: f64,
    image_height_b: f64,
    cache: State<'_, ExportCache>,
) -> Result<EpipolarOverlay, String> {
    cache
        .read(|cached| {
            epipolar::compute_overlay_undistorted(
                &cached.value,
                cam_a,
                cam_b,
                point_px,
                image_width_b,
                image_height_b,
            )
        })
        .ok_or_else(|| "no export loaded yet".to_string())?
}

/// The value of each labelled parameter of the cached export that is a
/// camera parameter, aligned with `labels`; `null` for poses and planes.
#[tauri::command]
pub async fn camera_parameter_values(
    labels: Vec<String>,
    cache: State<'_, ExportCache>,
) -> Result<Vec<Option<f64>>, String> {
    cache
        .read(|cached| uncertainty::camera_parameter_values(&cached.value, &labels))
        .ok_or_else(|| "no export loaded yet".to_string())?
}

/// The projection uncertainty of one camera of the cached export over an
/// image of `image_width` × `image_height` pixels.
#[tauri::command]
pub async fn projection_uncertainty(
    camera: usize,
    image_width: u32,
    image_height: u32,
    cache: State<'_, ExportCache>,
) -> Result<ProjectionUncertainty, String> {
    cache
        .read(|cached| {
            uncertainty::projection_uncertainty(&cached.value, camera, [image_width, image_height])
        })
        .ok_or_else(|| "no export loaded yet".to_string())?
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn image_root_joins_the_manifest_root_onto_the_export_folder() {
        let export = json!({ "image_manifest": { "root": "../..", "frames": [] } });
        assert_eq!(
            image_root(&export, Path::new("/p/runs/a")),
            Some(PathBuf::from("/p/runs/a/../.."))
        );
        assert_eq!(image_root(&json!({}), Path::new("/p")), None);
    }
}
