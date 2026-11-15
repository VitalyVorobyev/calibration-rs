//! Tauri commands exposed to the React frontend.
//!
//! The viewer commands are thin wrappers over [`crate::ops`], whose bodies
//! the calib-viewer's HTTP server calls too; what is desktop-only here is
//! the asset scope (`load_export` lets the webview load the export's
//! images) and the Run workspace's helpers (samples, folder sniffing).
//!
//! The wire format is intentionally untyped (`serde_json::Value`) for
//! `load_export` so we do not have to mirror every Rust field as a TS
//! interface — the frontend narrows to the subset it consumes. Trade-off
//! recorded in ADR 0014.

use std::path::Path;
use tauri::ipc::Response;
use tauri::{AppHandle, Runtime, State};
use vision_calibration::core::PixelRect;
use vision_calibration::dataset::samples;
use vision_calibration::optim::ProjectionUncertainty;

use crate::epipolar::EpipolarOverlay;
use crate::export_cache::ExportCache;
use crate::ops::{self, LoadExportResult};

/// Read and parse a calibration export JSON file. Does NOT cache the
/// result — the frontend validates the export before the user can
/// interact with it, and a rejected export must not displace the cache
/// the previous session is still using. The frontend calls
/// [`set_active_export`] explicitly once it accepts the file. The export's
/// image root becomes loadable through the asset protocol.
#[tauri::command]
pub async fn load_export<R: Runtime>(
    app: AppHandle<R>,
    path: String,
) -> Result<LoadExportResult, String> {
    let loaded = ops::read_export(Path::new(&path))?;
    if let Some(images) = &loaded.image_root {
        crate::assets::allow_dir(&app, images)?;
    }
    Ok(loaded.result)
}

/// Commit the export the frontend has accepted as the active dataset.
///
/// Called by the frontend after [`load_export`] returned and the
/// frontend's validation passed. Writes the parsed `export` into the
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
    ops::read_text_file(Path::new(&path))
}

/// Make the sample dataset `id` (one of
/// `vision_calibration::dataset::samples::catalogue`) available locally and
/// return the absolute path of its `dataset.toml`.
///
/// The files are downloaded from their publisher into the platform cache
/// (`VISION_CALIBRATION_DATA` overrides it), checked against their SHA-256
/// and reused on later calls, so the installed app carries no dataset.
/// Runs on a blocking thread: the first call downloads tens of megabytes.
#[tauri::command]
pub async fn fetch_sample_cmd(id: String) -> Result<String, String> {
    tauri::async_runtime::spawn_blocking(move || {
        let cache = samples::default_cache_dir().map_err(|e| e.to_string())?;
        fetch_sample(&id, &cache)
    })
    .await
    .map_err(|e| format!("sample fetch task: {e}"))?
}

/// [`fetch_sample_cmd`] against an explicit cache directory.
fn fetch_sample(id: &str, cache: &Path) -> Result<String, String> {
    samples::fetch(id, cache)
        .map(|manifest| manifest.to_string_lossy().into_owned())
        .map_err(|e| format!("sample {id}: {e}"))
}

/// Heuristically infer a `DatasetSpec` from a dataset folder.
///
/// Walks `folder`, groups images into cameras, detects a robot-pose file,
/// and guesses the topology — leaving every field it cannot determine at a
/// placeholder and recording the dotted path in the manifest's `_unresolved`
/// array. The frontend loads the returned JSON straight into the Run
/// workspace's manifest form; the non-empty `_unresolved` list drives the red
/// badges and blocks Run until the user fills and clears them (ADR 0016).
/// This is the same inference `vcal init` uses.
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
    ops::load_undistorted_image(&cache, Path::new(&path), camera, roi).map(Response::new)
}

/// Map raw/distorted camera pixels into the undistorted pixel frame
/// used by the Epipolar workspace.
#[tauri::command]
pub async fn undistort_points(
    camera: usize,
    points_px: Vec<[f64; 2]>,
    cache: State<'_, ExportCache>,
) -> Result<Vec<[f64; 2]>, String> {
    ops::undistort_points(&cache, camera, points_px)
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
    ops::compute_epipolar_overlay(&cache, cam_a, cam_b, point_px)
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
    ops::compute_epipolar_overlay_undistorted(
        &cache,
        cam_a,
        cam_b,
        point_px,
        [image_width_b, image_height_b],
    )
}

/// The value of each labelled parameter of the cached export that is a
/// camera parameter, aligned with `labels`; `null` for poses and planes.
#[tauri::command]
pub async fn camera_parameter_values(
    labels: Vec<String>,
    cache: State<'_, ExportCache>,
) -> Result<Vec<Option<f64>>, String> {
    ops::camera_parameter_values(&cache, &labels)
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
    ops::projection_uncertainty(&cache, camera, [image_width, image_height])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fetch_sample_names_the_unknown_sample_and_the_known_ones() {
        let cache = tempfile::tempdir().unwrap();
        let err = fetch_sample("no_such_sample", cache.path()).unwrap_err();
        assert!(err.contains("no_such_sample"), "{err}");
        assert!(err.contains("kuka_1") && err.contains("ds8"), "{err}");
    }

    #[test]
    fn the_quick_start_samples_are_in_the_catalogue() {
        for id in ["kuka_1", "ds8"] {
            assert!(samples::sample(id).is_ok(), "{id}");
        }
    }
}
