//! The bodies of the viewer commands, shared by both back ends: the Tauri
//! commands in [`crate::commands`] and [`crate::disparity`] wrap them for
//! the desktop app, and [`crate::viewer`]'s HTTP handlers call them for the
//! browser viewer. Each is a plain function of its arguments and the
//! [`ExportCache`]; what differs between the back ends (the asset scope of
//! the desktop app, the file scope of the viewer) stays in the wrappers.

use std::path::{Path, PathBuf};

use serde::Serialize;
use vision_calibration::core::PixelRect;
use vision_calibration::optim::ProjectionUncertainty;

use crate::disparity::{self, DisparityParams, DisparityResult};
use crate::epipolar::{self, EpipolarOverlay};
use crate::export_cache::ExportCache;
use crate::uncertainty;

/// Successful response of `load_export`.
#[derive(Serialize)]
pub struct LoadExportResult {
    /// Raw parsed export JSON. The frontend treats this as `AnyExport`
    /// (subset interface) and ignores fields it does not render.
    pub export: serde_json::Value,
    /// Absolute path to the directory containing `export.json`. The
    /// frontend joins this with the manifest's `root` and per-frame
    /// `path` to obtain absolute image filenames.
    pub export_dir: String,
}

/// An export read from disk, with the folder its image manifest resolves
/// against (`None` without a manifest).
pub struct LoadedExport {
    pub result: LoadExportResult,
    pub image_root: Option<PathBuf>,
}

/// Read and parse a calibration export JSON file and check its schema
/// version. Does not cache it: the frontend validates the export first and
/// then calls `set_active_export`.
pub fn read_export(path: &Path) -> Result<LoadedExport, String> {
    let parent = path
        .parent()
        .ok_or_else(|| format!("export path has no parent directory: {}", path.display()))?
        .to_path_buf();
    let bytes = std::fs::read(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    let export: serde_json::Value =
        serde_json::from_slice(&bytes).map_err(|e| format!("parse {}: {e}", path.display()))?;
    vision_calibration::common::check_export_version(&export.to_string())
        .map_err(|e| format!("{}: {e}", path.display()))?;
    let export_dir = dunce::canonicalize(&parent)
        .map_err(|e| format!("canonicalize {}: {e}", parent.display()))?;
    let image_root = image_root(&export, &export_dir);
    Ok(LoadedExport {
        result: LoadExportResult {
            export,
            export_dir: export_dir.to_string_lossy().replace('\\', "/"),
        },
        image_root,
    })
}

/// The folder an export's image manifest resolves against: its
/// `diagnostics.image_manifest.root` joined onto the export's own folder.
/// `None` without a manifest.
pub fn image_root(export: &serde_json::Value, export_dir: &Path) -> Option<PathBuf> {
    let root = export
        .get("diagnostics")?
        .get("image_manifest")?
        .get("root")?
        .as_str()?;
    Some(export_dir.join(root))
}

/// Read a UTF-8 text file.
pub fn read_text_file(path: &Path) -> Result<String, String> {
    std::fs::read_to_string(path).map_err(|e| format!("read {}: {e}", path.display()))
}

/// The image at `path` rendered into `camera`'s undistorted pixel frame, as
/// PNG bytes.
pub fn load_undistorted_image(
    cache: &ExportCache,
    path: &Path,
    camera: usize,
    roi: Option<PixelRect>,
) -> Result<Vec<u8>, String> {
    cache.with(|export| epipolar::undistort_image_png(export, path, camera, roi))
}

/// Raw camera pixels mapped into `camera`'s undistorted pixel frame.
pub fn undistort_points(
    cache: &ExportCache,
    camera: usize,
    points_px: Vec<[f64; 2]>,
) -> Result<Vec<[f64; 2]>, String> {
    cache.with(|export| epipolar::undistort_points(export, camera, points_px))
}

/// The epipolar polyline and epipole in camera B (distorted pixels) of a
/// pixel of camera A.
pub fn compute_epipolar_overlay(
    cache: &ExportCache,
    cam_a: usize,
    cam_b: usize,
    point_px: [f64; 2],
) -> Result<EpipolarOverlay, String> {
    cache.with(|export| epipolar::compute_overlay(export, cam_a, cam_b, point_px))
}

/// A clipped straight epipolar segment in camera B's undistorted frame of a
/// pixel of camera A's undistorted frame.
pub fn compute_epipolar_overlay_undistorted(
    cache: &ExportCache,
    cam_a: usize,
    cam_b: usize,
    point_px: [f64; 2],
    image_size_b: [f64; 2],
) -> Result<EpipolarOverlay, String> {
    cache.with(|export| {
        epipolar::compute_overlay_undistorted(
            export,
            cam_a,
            cam_b,
            point_px,
            image_size_b[0],
            image_size_b[1],
        )
    })
}

/// The value of each labelled camera parameter, aligned with `labels`.
pub fn camera_parameter_values(
    cache: &ExportCache,
    labels: &[String],
) -> Result<Vec<Option<f64>>, String> {
    cache.with(|export| uncertainty::camera_parameter_values(export, labels))
}

/// The projection uncertainty of `camera` over an image of `image_size`.
pub fn projection_uncertainty(
    cache: &ExportCache,
    camera: usize,
    image_size: [u32; 2],
) -> Result<ProjectionUncertainty, String> {
    cache.with(|export| uncertainty::projection_uncertainty(export, camera, image_size))
}

/// A disparity map for the `(cam_a, cam_b)` pair at `pose`.
pub fn compute_disparity(
    cache: &ExportCache,
    paths: [&Path; 2],
    cam_a: usize,
    cam_b: usize,
    pose: usize,
    params: &DisparityParams,
) -> Result<DisparityResult, String> {
    cache.with(|export| disparity::compute(export, paths[0], paths[1], cam_a, cam_b, pose, params))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn image_root_joins_the_manifest_root_onto_the_export_folder() {
        let export = json!({
            "diagnostics": { "image_manifest": { "root": "../..", "frames": [] } }
        });
        assert_eq!(
            image_root(&export, Path::new("/p/runs/a")),
            Some(PathBuf::from("/p/runs/a/../.."))
        );
        assert_eq!(image_root(&json!({}), Path::new("/p")), None);
    }

    #[test]
    fn commands_on_the_cache_need_an_active_export() {
        let cache = ExportCache::new();
        let err = undistort_points(&cache, 0, vec![[0.0, 0.0]]).unwrap_err();
        assert_eq!(err, "no export loaded yet");
    }
}
