//! Tauri commands exposed to the React frontend.
//!
//! Every command is a thin wrapper over `studio_core`: the viewer commands
//! over [`studio_core::ops`], whose bodies the calib-viewer's HTTP server
//! calls too, the project commands over [`studio_core::project`] and the
//! runner over [`studio_core::run`]. What is desktop-only here is the asset
//! scope (opening a project, an export or running a calibration lets the
//! webview load its images), the run's progress channel, and the Run
//! workspace's helpers (samples, folder sniffing).
//!
//! The wire format is intentionally untyped (`serde_json::Value`) for
//! `load_export` so we do not have to mirror every Rust field as a TS
//! interface — the frontend narrows to the subset it consumes. Trade-off
//! recorded in ADR 0014.

use std::path::{Path, PathBuf};
use tauri::ipc::{Channel, Response};
use tauri::{AppHandle, Runtime, State};
use vision_calibration::core::PixelRect;
use vision_calibration::dataset::samples;
use vision_calibration::optim::ProjectionUncertainty;

use studio_core::disparity::{DisparityParams, DisparityResult};
use studio_core::epipolar::EpipolarOverlay;
use studio_core::export_cache::ExportCache;
use studio_core::ops::{self, LoadExportResult};
use studio_core::project::{
    self, ProjectError, ProjectErrorKind, ProjectInfo, RunRecord, RunSummary,
};
use studio_core::run::{self as runner, RunProgress, RunRegistry, RunResponse};

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
    let loaded = tauri::async_runtime::spawn_blocking(move || ops::read_export(Path::new(&path)))
        .await
        .map_err(|e| format!("load export task panicked: {e}"))??;
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
    tauri::async_runtime::spawn_blocking(move || ops::read_text_file(Path::new(&path)))
        .await
        .map_err(|e| format!("load text file task panicked: {e}"))?
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
    tauri::async_runtime::spawn_blocking(move || {
        let spec = vision_calibration::dataset::sniff_folder(std::path::Path::new(&folder))
            .map_err(|e| e.to_string())?;
        serde_json::to_value(&spec).map_err(|e| format!("serialize manifest: {e}"))
    })
    .await
    .map_err(|e| format!("sniff folder task panicked: {e}"))?
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
    let cached = cache
        .snapshot()
        .ok_or_else(|| "no export loaded yet".to_string())?;
    let png = tauri::async_runtime::spawn_blocking(move || {
        studio_core::epipolar::undistort_image_png(&cached.value, Path::new(&path), camera, roi)
    })
    .await
    .map_err(|e| format!("undistort image task panicked: {e}"))??;
    Ok(Response::new(png))
}

/// Map raw/distorted camera pixels into the undistorted pixel frame
/// used by the Epipolar workspace (`None` where back-projection has no
/// finite unit-depth ray).
#[tauri::command]
pub async fn undistort_points(
    camera: usize,
    points_px: Vec<[f64; 2]>,
    cache: State<'_, ExportCache>,
) -> Result<Vec<Option<[f64; 2]>>, String> {
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
    let cached = cache
        .snapshot()
        .ok_or_else(|| "no export loaded yet".to_string())?;
    tauri::async_runtime::spawn_blocking(move || {
        studio_core::uncertainty::projection_uncertainty(
            &cached.value,
            camera,
            [image_width, image_height],
        )
    })
    .await
    .map_err(|e| format!("projection uncertainty task panicked: {e}"))?
}

/// Tauri command: compute a disparity map for the `(cam_a, cam_b)` pair at
/// `pose` and return colormapped PNGs + metrics.
#[tauri::command]
pub async fn compute_disparity(
    path_a: String,
    path_b: String,
    cam_a: usize,
    cam_b: usize,
    pose: usize,
    params: DisparityParams,
    cache: State<'_, ExportCache>,
) -> Result<DisparityResult, String> {
    let cached = cache
        .snapshot()
        .ok_or_else(|| "no export loaded yet".to_string())?;
    // Rectification and SGM matching are the heaviest compute in the app:
    // keep them off the async runtime.
    tauri::async_runtime::spawn_blocking(move || {
        studio_core::disparity::compute(
            &cached.value,
            Path::new(&path_a),
            Path::new(&path_b),
            cam_a,
            cam_b,
            pose,
            &params,
        )
    })
    .await
    .map_err(|e| format!("disparity task panicked: {e}"))?
}

/// Tauri command: make `dir` a project for the dataset in `dataset_dir`.
#[tauri::command]
pub fn create_project<R: Runtime>(
    app: AppHandle<R>,
    dir: PathBuf,
    dataset_dir: PathBuf,
    manifest: serde_json::Value,
) -> Result<ProjectInfo, ProjectError> {
    let info = project::create(&dir, &dataset_dir, &manifest)?;
    allow_project_assets(&app, &info)?;
    Ok(info)
}

/// Tauri command: open the project in `dir`.
#[tauri::command]
pub fn open_project<R: Runtime>(
    app: AppHandle<R>,
    dir: PathBuf,
) -> Result<ProjectInfo, ProjectError> {
    let info = project::open(&dir)?;
    allow_project_assets(&app, &info)?;
    Ok(info)
}

/// Tauri command: replace the project's dataset manifest.
#[tauri::command]
pub fn save_project_manifest(
    dir: PathBuf,
    manifest: serde_json::Value,
) -> Result<(), ProjectError> {
    project::save_manifest(&dir, &manifest)
}

/// Tauri command: save a run as `runs/<name>/`.
#[tauri::command]
pub fn save_project_run(
    dir: PathBuf,
    name: String,
    record: RunRecord,
) -> Result<RunSummary, ProjectError> {
    project::save_run(&dir, &name, record)
}

/// The webview may load the project's dataset images.
fn allow_project_assets<R: Runtime>(
    app: &AppHandle<R>,
    info: &ProjectInfo,
) -> Result<(), ProjectError> {
    crate::assets::allow_dir(app, Path::new(&info.dataset_dir))
        .map_err(|e| ProjectError::new(ProjectErrorKind::Io, e))
}

/// Tauri command: run a calibration end-to-end.
///
/// `run_id` is a caller-generated correlation id (used to route
/// cancellation via [`cancel_run_cmd`]). `manifest_json` is the raw JSON
/// of a `DatasetSpec` (typically emitted by the schema-driven form in
/// the Run workspace). `config_json` is the matching per-problem
/// `*Config` JSON. `manifest_dir` is the directory whose globs are
/// resolved against — usually the parent of the `dataset.toml` file the
/// user picked. `on_progress` streams a [`RunProgress`] per stage.
#[tauri::command]
#[allow(clippy::too_many_arguments)] // Tauri injects state and the app handle as arguments.
pub async fn run_calibration_cmd<R: Runtime>(
    app: AppHandle<R>,
    run_id: String,
    manifest_json: serde_json::Value,
    config_json: serde_json::Value,
    manifest_dir: String,
    on_progress: Channel<RunProgress>,
    cache: State<'_, ExportCache>,
    registry: State<'_, RunRegistry>,
) -> Result<RunResponse, String> {
    let token = registry.register(&run_id)?;
    let images_dir = PathBuf::from(&manifest_dir);

    // Long-running solve: keep IPC responsive by running on a blocking
    // task. The State can't cross the spawn_blocking boundary directly,
    // so we do detection + calibration on the worker thread and pop
    // the produced export into the cache here, in the async caller.
    let result = tauri::async_runtime::spawn_blocking(move || {
        let emit = move |progress: RunProgress| {
            let _ = on_progress.send(progress);
        };
        runner::run_calibration(manifest_json, config_json, &manifest_dir, token, emit)
    })
    .await;

    registry.unregister(&run_id);

    let response = result.map_err(|e| format!("runner task panicked: {e}"))?;
    if let RunResponse::Ok(success) = &response {
        cache.set(success.export.clone());
        // The export's image manifest is relative to the dataset folder,
        // whose images the webview now shows.
        crate::assets::allow_dir(&app, &images_dir)?;
    }
    Ok(response)
}

/// Tauri command: request cancellation of an in-flight run.
///
/// Cancellation is cooperative: the library stops at its next image or
/// solver iteration, and the runner at its next stage boundary. Returns
/// `true` when a matching in-flight run was flagged, `false` if the run
/// already finished (a benign race).
#[tauri::command]
pub fn cancel_run_cmd(run_id: String, registry: State<'_, RunRegistry>) -> bool {
    registry.cancel(&run_id)
}

/// Tauri command: default `*Config` JSON for a topology.
#[tauri::command]
pub fn default_config_cmd(topology: String) -> Result<serde_json::Value, String> {
    runner::default_config(&topology)
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
