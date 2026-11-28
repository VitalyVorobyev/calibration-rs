//! What the map commands share (`maps`, `compare`): a run's camera checked
//! and its image size, the grid of the requested columns, and the memo of
//! a reply in the blob store.

use std::sync::Arc;

use serde::Serialize;
use serde::de::DeserializeOwned;
use vision_calibration::analysis::ImageGrid;
use vision_calibration::common::CalibratedCameras;
use vision_calibration::dataset_runner::camera_image_sizes;

use crate::api::{Ctx, ErrorKind, StudioError};
use crate::blobs::{BlobRef, BlobStore};
use crate::maps::DEFAULT_COLUMNS;
use crate::runs::{CachedRun, RunRef, open_project};
use crate::scope::UserPath;

/// `camera` checked against the run's cameras.
pub(super) fn check_camera(
    cameras: &dyn CalibratedCameras,
    camera: usize,
) -> Result<(), StudioError> {
    let count = cameras.camera_params().len();
    if camera < count {
        Ok(())
    } else {
        Err(StudioError::invalid(format!(
            "the run has {count} camera(s); there is no camera {camera}"
        )))
    }
}

/// The run's export and camera `camera`'s image size: the project's
/// dataset as it is now (its region of interest for a tiled camera), read
/// from the images' headers after every path the project names is judged.
pub(super) fn run_and_size(
    ctx: &Ctx,
    run: &RunRef,
    camera: usize,
) -> Result<(Arc<CachedRun>, [u32; 2]), StudioError> {
    let cached = ctx.runs.get(ctx, run)?;
    check_camera(cached.cameras()?, camera)?;
    let size = size_of(&dataset_sizes(ctx, &run.dir)?, camera)?;
    Ok((cached, size))
}

/// Each camera's image size in the project's dataset as it is now (its
/// region of interest for a tiled camera), read from the images' headers
/// after every path the project names is judged.
pub(super) fn dataset_sizes(ctx: &Ctx, dir: &UserPath) -> Result<Vec<[u32; 2]>, StudioError> {
    let project = open_project(ctx, dir)?;
    ctx.scope.check_project(&project)?;
    Ok(camera_image_sizes(
        &project.dataset()?,
        &project.data_root()?,
    )?)
}

/// Camera `camera`'s size among the dataset's `sizes`.
pub(super) fn size_of(sizes: &[[u32; 2]], camera: usize) -> Result<[u32; 2], StudioError> {
    sizes.get(camera).copied().ok_or_else(|| {
        StudioError::invalid(format!(
            "the project's dataset has {} camera(s); there is no camera {camera}",
            sizes.len()
        ))
    })
}

/// The maps' grid over `size` of `columns` (default 64) columns.
pub(super) fn grid_of(size: [u32; 2], columns: Option<u32>) -> Result<ImageGrid, StudioError> {
    crate::maps::grid(size, columns.unwrap_or(DEFAULT_COLUMNS))
        .map_err(|e| StudioError::new(ErrorKind::InvalidArgs, e))
}

/// The reply `make` builds, or the one the blob store kept for the same
/// `key` while it still holds every blob the reply names. `make` gets the
/// id of each part.
pub(super) fn memoized<R: Serialize + DeserializeOwned>(
    ctx: &Ctx,
    command: &str,
    key: &serde_json::Value,
    blobs: impl Fn(&R) -> Vec<&BlobRef>,
    make: impl FnOnce(&dyn Fn(&str) -> String) -> Result<R, StudioError>,
) -> Result<R, StudioError> {
    let key = key.to_string();
    let id = |part: &str| BlobStore::id(&[command.as_bytes(), key.as_bytes(), part.as_bytes()]);
    if let Some((_, memo)) = ctx.blobs.get(&id("reply"))
        && let Ok(reply) = serde_json::from_slice::<R>(&memo)
        && ctx.blobs.has_all(blobs(&reply))
    {
        return Ok(reply);
    }
    let reply = make(&id)?;
    let memo = serde_json::to_vec(&reply)
        .map_err(|e| StudioError::internal(format!("{command}: serialize the reply: {e}")))?;
    ctx.blobs.put(id("reply"), "application/json", memo);
    Ok(reply)
}
