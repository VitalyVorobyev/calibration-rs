//! A run's diagnostic maps, camera parameters and per-feature residuals.
//!
//! The maps of one camera share one grid ([`MapGrid`]): the same
//! `columns` give the distortion map, the residual map and the projection
//! σ map node for node. Their planes are `VAM1` blobs (the `blob`
//! resource); a reply is memoized under a content address of what made it
//! (the command, its resolved options, the camera's image size and the
//! export's fingerprint), so the same request answers from the blob store
//! with the same blob ids. A blob the store dropped answers
//! `blob_expired`: run the command again (it makes the same ids).
//!
//! Every displacement is `actual − reference` in pixels; a residual is
//! `observed − projected`.

use schemars::JsonSchema;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use vision_calibration::analysis::{
    BandwidthSelection, ImageGrid, ResidualMapOptions, distortion_map as library_distortion_map,
    residual_map as library_residual_map,
};
use vision_calibration::common::CalibratedCameras;
use vision_calibration::core::SensorKind;
use vision_calibration::dataset_runner::camera_image_sizes;

use crate::api::{Ctx, ErrorKind, StudioError};
use crate::blobs::{BlobRef, BlobStore};
use crate::maps::{
    DEFAULT_COLUMNS, MapGrid, MapPlane, ResidualRecords, ResidualStats, displacement_channels,
    major_axis_sigma, put_plane, residual_records as records_of, residual_stats as stats_of,
};
use crate::runs::{CachedRun, RunRef, open_project};
use crate::uncertainty::{
    CameraParameters, SigmaUnavailable, camera_parameters_of, projection_sigma_unavailable,
};

/// `camera` checked against the run's cameras.
fn check_camera(cameras: &dyn CalibratedCameras, camera: usize) -> Result<(), StudioError> {
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
fn run_and_size(
    ctx: &Ctx,
    run: &RunRef,
    camera: usize,
) -> Result<(std::sync::Arc<CachedRun>, [u32; 2]), StudioError> {
    let cached = ctx.runs.get(ctx, run)?;
    check_camera(cached.cameras()?, camera)?;
    let project = open_project(ctx, &run.dir)?;
    ctx.scope.check_project(&project)?;
    let sizes = camera_image_sizes(&project.dataset()?, &project.data_root()?)?;
    let size = sizes.get(camera).copied().ok_or_else(|| {
        StudioError::invalid(format!(
            "the project's dataset has {} camera(s); there is no camera {camera}",
            sizes.len()
        ))
    })?;
    Ok((cached, size))
}

fn grid_of(size: [u32; 2], columns: Option<u32>) -> Result<ImageGrid, StudioError> {
    crate::maps::grid(size, columns.unwrap_or(DEFAULT_COLUMNS))
        .map_err(|e| StudioError::new(ErrorKind::InvalidArgs, e))
}

/// The reply `make` builds, or the one the blob store kept for the same
/// `key` while it still holds every blob the reply names. `make` gets the
/// id of each part.
fn memoized<R: Serialize + DeserializeOwned>(
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

/// The lens and sensor displacement of camera `camera` of the run over its
/// image: three layers, each a plane of `dx`, `dy` and `magnitude` in
/// pixels, `actual − reference` at the same ray (where the ray lands minus
/// where it would land without the stage), indexed by the camera's own
/// pixel. `distortion` is the lens alone (the camera without its sensor
/// tilt minus the camera without distortion), `sensor` what the sensor
/// model adds (zero for an untilted sensor), `total` their sum. `NaN` where
/// no ray reaches a node. `columns` (default 64, at most 1024) sets the
/// grid every map of the camera shares.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct DistortionMapArgs {
    pub run: RunRef,
    pub camera: usize,
    #[serde(default)]
    pub columns: Option<u32>,
}

/// A camera's distortion map (see `distortion_map`).
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct DistortionMap {
    pub camera: usize,
    pub grid: MapGrid,
    /// Nodes with a value (the others are `NaN` in every layer).
    pub valid_nodes: usize,
    /// Whether the sensor model can add a layer (a tilted or homography
    /// sensor); without one `sensor` is zero.
    pub has_sensor_layer: bool,
    pub distortion: MapPlane,
    pub sensor: MapPlane,
    pub total: MapPlane,
}

pub fn distortion_map(ctx: &Ctx, args: DistortionMapArgs) -> Result<DistortionMap, StudioError> {
    let (run, size) = run_and_size(ctx, &args.run, args.camera)?;
    let grid = grid_of(size, args.columns)?;
    let key = serde_json::json!({
        "camera": args.camera, "image_size": size, "stride": grid.stride(),
        "calibration": run.fingerprint,
    });
    memoized(
        ctx,
        "distortion_map",
        &key,
        |r: &DistortionMap| vec![&r.distortion.blob, &r.sensor.blob, &r.total.blob],
        |id| {
            let params = run.cameras()?.camera_params().swap_remove(args.camera);
            let map = library_distortion_map(&params, &grid)
                .map_err(|e| StudioError::invalid(e.to_string()))?;
            let layer = |name: &str, plane| {
                let channels = displacement_channels(plane);
                let channels: Vec<(&str, &str, &[f64])> = channels
                    .iter()
                    .map(|(n, v)| (*n, "px", v.as_slice()))
                    .collect();
                put_plane(&ctx.blobs, id(name), &grid, &channels).map_err(StudioError::internal)
            };
            Ok(DistortionMap {
                camera: args.camera,
                grid: MapGrid::of(&grid),
                valid_nodes: map.valid_nodes,
                has_sensor_layer: params.sensor.kind() != Some(SensorKind::Identity),
                distortion: layer("distortion", &map.distortion)?,
                sensor: layer("sensor", &map.sensor)?,
                total: layer("total", &map.total)?,
            })
        },
    )
}

/// The systematic part of camera `camera`'s reprojection residuals over its
/// image, the library's local-linear residual map: one plane of eight
/// channels, `mean_dx`, `mean_dy`, `mean_magnitude` (the smoothed mean of
/// `observed − projected`, px), `rms` (px), `std_error` (px, view-clustered),
/// `significance` (χ: above 3.03 is p < 0.01), `effective_samples` and
/// `effective_views`. Records sit at their projection. `NaN` where the
/// support is too thin (`effective_samples` below
/// `options.min_effective_samples`) and, for `std_error` and
/// `significance`, with too few effective views. `options.views` maps a
/// subset of the views (their indices in the export); one view, or too few
/// effective views, gives a mean that includes the views' pose error and no
/// significance. `columns` as `distortion_map`'s. `invalid` for a run
/// without per-feature residuals of the camera.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ResidualMapArgs {
    pub run: RunRef,
    pub camera: usize,
    #[serde(default)]
    pub columns: Option<u32>,
    #[serde(default)]
    pub options: ResidualMapOptions,
}

/// A camera's residual map (see `residual_map`). The residual is
/// `observed − projected`, the sign of a distortion map's layer.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ResidualMap {
    pub camera: usize,
    pub grid: MapGrid,
    pub plane: MapPlane,
    /// The Gaussian kernel's σ the map used, pixels.
    pub bandwidth_px: f64,
    pub bandwidth_selection: BandwidthSelection,
    /// The options as used (`views` sorted): `min_effective_samples` is
    /// the support boundary, the `effective_samples` level where the map
    /// ends; `min_effective_views` where `std_error` and `significance`
    /// end.
    pub options: ResidualMapOptions,
    /// Whether the run's board poses come through a rig or robot chain
    /// (rig and hand-eye workflows): the residuals then include the chain's
    /// pose error, with a smooth pattern per view, besides the camera
    /// model's misfit.
    pub chain_level: bool,
    /// The records mapped.
    pub used: usize,
    /// The selected records without a projection, left out.
    pub unprojected: usize,
    /// The records left out as outliers, as indices into the export's
    /// `per_feature_residuals.target` (and `residual_records`' `index`).
    pub outliers: Vec<usize>,
    /// The number of views mapped.
    pub views: usize,
}

/// Whether an export of `kind` places its boards through a chain.
fn chain_level(export: &serde_json::Value) -> bool {
    !matches!(
        export.get("kind").and_then(serde_json::Value::as_str),
        Some("planar_intrinsics" | "laserline_device")
    )
}

pub fn residual_map(ctx: &Ctx, args: ResidualMapArgs) -> Result<ResidualMap, StudioError> {
    let (run, size) = run_and_size(ctx, &args.run, args.camera)?;
    let grid = grid_of(size, args.columns)?;
    let mut options = args.options;
    if let Some(views) = options.views.as_mut() {
        views.sort_unstable();
        views.dedup();
    }
    let key = serde_json::json!({
        "camera": args.camera, "image_size": size, "stride": grid.stride(),
        "options": options, "calibration": run.fingerprint,
    });
    memoized(
        ctx,
        "residual_map",
        &key,
        |r: &ResidualMap| vec![&r.plane.blob],
        |id| {
            let records = &run.cameras()?.diagnostics().per_feature_residuals.target;
            if !records.iter().any(|r| r.camera == args.camera) {
                return Err(StudioError::invalid(format!(
                    "the run has no per-feature residuals of camera {}",
                    args.camera
                )));
            }
            let map = library_residual_map(records, args.camera, &grid, &options)
                .map_err(|e| StudioError::invalid(e.to_string()).on_field("options"))?;
            let magnitude: Vec<f64> = map
                .mean_dx
                .iter()
                .zip(&map.mean_dy)
                .map(|(x, y)| x.hypot(*y))
                .collect();
            let plane = put_plane(
                &ctx.blobs,
                id("plane"),
                &grid,
                &[
                    ("mean_dx", "px", &map.mean_dx),
                    ("mean_dy", "px", &map.mean_dy),
                    ("mean_magnitude", "px", &magnitude),
                    ("rms", "px", &map.rms),
                    ("std_error", "px", &map.std_error),
                    ("significance", "chi", &map.significance),
                    ("effective_samples", "samples", &map.effective_samples),
                    ("effective_views", "views", &map.effective_views),
                ],
            )
            .map_err(StudioError::internal)?;
            Ok(ResidualMap {
                camera: args.camera,
                grid: MapGrid::of(&grid),
                plane,
                bandwidth_px: map.bandwidth_px,
                bandwidth_selection: map.bandwidth_selection,
                options: options.clone(),
                chain_level: chain_level(&run.export),
                used: map.used,
                unprojected: map.unprojected,
                outliers: map.outliers,
                views: map.views,
            })
        },
    )
}

/// The projection uncertainty of camera `camera` on the maps' grid: one
/// plane, `sigma`, the major-axis σ in pixels of where the camera-frame ray
/// through each node lands, from the covariance of the camera's
/// parameters; `NaN` where no ray reaches a node. `unavailable`, with the
/// reason, for a run without a covariance or whose camera parameters were
/// all held fixed. `columns` as `distortion_map`'s.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ProjectionSigmaMapArgs {
    pub run: RunRef,
    pub camera: usize,
    #[serde(default)]
    pub columns: Option<u32>,
}

/// A camera's projection σ on the maps' grid, or why it has none.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum ProjectionSigmaMap {
    Available {
        camera: usize,
        grid: MapGrid,
        plane: MapPlane,
    },
    Unavailable {
        camera: usize,
        reason: SigmaUnavailable,
    },
}

pub fn projection_sigma_map(
    ctx: &Ctx,
    args: ProjectionSigmaMapArgs,
) -> Result<ProjectionSigmaMap, StudioError> {
    let (run, size) = run_and_size(ctx, &args.run, args.camera)?;
    let grid = grid_of(size, args.columns)?;
    let cameras = run.cameras()?;
    if let Some(reason) = projection_sigma_unavailable(cameras, args.camera) {
        return Ok(ProjectionSigmaMap::Unavailable {
            camera: args.camera,
            reason,
        });
    }
    let key = serde_json::json!({
        "camera": args.camera, "image_size": size, "stride": grid.stride(),
        "calibration": run.fingerprint,
    });
    memoized(
        ctx,
        "projection_sigma_map",
        &key,
        |r: &ProjectionSigmaMap| match r {
            ProjectionSigmaMap::Available { plane, .. } => vec![&plane.blob],
            ProjectionSigmaMap::Unavailable { .. } => Vec::new(),
        },
        |id| {
            let nodes: Vec<[f64; 2]> = grid.nodes().collect();
            let covariances = cameras
                .projection_covariance(args.camera, &nodes)
                .map_err(|e| StudioError::invalid(e.to_string()))?;
            let sigma: Vec<f64> = covariances
                .iter()
                .map(|c| {
                    c.map_or(f64::NAN, |c| {
                        major_axis_sigma(c[(0, 0)], 0.5 * (c[(0, 1)] + c[(1, 0)]), c[(1, 1)])
                    })
                })
                .collect();
            let plane = put_plane(&ctx.blobs, id("plane"), &grid, &[("sigma", "px", &sigma)])
                .map_err(StudioError::internal)?;
            Ok(ProjectionSigmaMap::Available {
                camera: args.camera,
                grid: MapGrid::of(&grid),
                plane,
            })
        },
    )
}

/// Every parameter of every camera of the run, whatever its model: its
/// label in the export's uncertainty, name, group, value and unit, its σ
/// or why it has none (no covariance and the library's reason, held fixed,
/// not listed), and whether the solve ended with it on a bound.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct CameraParameterValuesArgs {
    pub run: RunRef,
}

pub fn camera_parameter_values(
    ctx: &Ctx,
    args: CameraParameterValuesArgs,
) -> Result<CameraParameters, StudioError> {
    let run = ctx.runs.get(ctx, &args.run)?;
    Ok(camera_parameters_of(run.cameras()?))
}

/// Camera `camera`'s per-feature residual records (of one `view`, when
/// given), one column per field, in the export's order: each record's
/// index in the export, view, feature, target point, observed and
/// projected pixel and `observed − projected`. The target point is the one
/// the calibration used: an exclusion names a feature by its detection's
/// `world_xyz` (`detection_detail`), which a rig of plain chessboards may
/// have relabelled.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ResidualRecordsArgs {
    pub run: RunRef,
    pub camera: usize,
    #[serde(default)]
    pub view: Option<usize>,
}

pub fn residual_records(
    ctx: &Ctx,
    args: ResidualRecordsArgs,
) -> Result<ResidualRecords, StudioError> {
    let run = ctx.runs.get(ctx, &args.run)?;
    let cameras = run.cameras()?;
    check_camera(cameras, args.camera)?;
    Ok(records_of(
        &cameras.diagnostics().per_feature_residuals.target,
        args.camera,
        args.view,
    ))
}

/// The run's reprojection statistics from its per-feature residuals (the
/// library's mean, RMS, largest error and count of `|observed −
/// projected|`, px), per camera, per view over every camera, and per
/// camera and view, with the records whose projection diverged.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ResidualStatsArgs {
    pub run: RunRef,
}

pub fn residual_stats(ctx: &Ctx, args: ResidualStatsArgs) -> Result<ResidualStats, StudioError> {
    let run = ctx.runs.get(ctx, &args.run)?;
    Ok(stats_of(
        &run.cameras()?.diagnostics().per_feature_residuals.target,
    ))
}
