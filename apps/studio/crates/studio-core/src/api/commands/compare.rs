//! Runs side by side (`compare_runs`, the library's `compare_recorded_runs`)
//! and, for one camera of two of them, where their calibrations project the
//! same rays (`projection_difference_map`, the library's
//! `run_projection_difference` on the maps' grid, as a `VAM1` plane).
//!
//! The comparison is the library's reply as it is: what each run says of
//! itself, each difference with its materiality and, where the data relation
//! the records establish supports one, its test, and every test or
//! projection the library could not make as its typed reason.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use vision_calibration::analysis::{
    CompareRunsOptions, ComparedRun, Compensation, ProjectionUnavailable, RunComparison,
    run_projection_difference,
};
use vision_calibration::project::{Project, RunInfo, compare_recorded_runs};

use super::map_support::{dataset_sizes, grid_of, memoized, size_of};
use crate::api::{Ctx, ErrorKind, StudioError};
use crate::maps::{
    DEFAULT_COLUMNS, MAX_COLUMNS, MapGrid, MapPlane, displacement_channels, put_plane,
};
use crate::runs::{RunRef, open_project};
use crate::scope::UserPath;

/// The project's runs `runs` side by side against the first, the reference:
/// each run's own evidence (its verdict, its covariance, each camera's fit),
/// and for each later run its data relation to the reference (read from the
/// records: independent, identical, the same data, nested, overlapping or
/// unknown), its configuration and exclusion changes, and per camera (matched
/// by id) its parameters with `Δ`, `Δ / σ_ref` and the test the relation
/// supports, and the projection difference's summary; the rig poses, the
/// hand-eye and the laser planes likewise. A test the relation does not
/// support is `not_available` with the reason. `options` are the library's
/// (`grid_columns` at most 1024). At least two runs; an unknown id is
/// `run_not_found`, runs the library cannot compare (other workflows, no
/// shared camera) are `invalid`.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct CompareRunsArgs {
    /// The project folder.
    pub dir: UserPath,
    /// The run ids, the reference first.
    pub runs: Vec<String>,
    #[serde(default)]
    pub options: CompareRunsOptions,
}

pub fn compare_runs(ctx: &Ctx, args: CompareRunsArgs) -> Result<RunComparison, StudioError> {
    if args.runs.len() < 2 {
        return Err(StudioError::new(
            ErrorKind::InvalidArgs,
            format!(
                "compare_runs: compare at least two runs, not {}",
                args.runs.len()
            ),
        )
        .on_field("runs"));
    }
    let columns = args.options.grid_columns;
    if columns == 0 || columns > MAX_COLUMNS {
        return Err(StudioError::new(
            ErrorKind::InvalidArgs,
            format!("compare_runs: a grid has 1 to {MAX_COLUMNS} columns, not {columns}"),
        )
        .on_field("options.grid_columns"));
    }
    let project = open_project(ctx, &args.dir)?;
    let runs = args
        .runs
        .iter()
        .map(|id| checked_run(ctx, &project, id))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(compare_recorded_runs(&runs, &args.options)?)
}

/// Run `id` of `project`, its export inside a root and every image the
/// export's manifest names judged: the library reads the export, and the
/// images' headers when the record does not state their size.
fn checked_run(ctx: &Ctx, project: &Project, id: &str) -> Result<RunInfo, StudioError> {
    let info = project.run(id)?;
    let export = ctx.scope.check(&info.export_path())?;
    ctx.scope.check_export_images(&export)?;
    Ok(info)
}

/// Where run `other`'s calibration of camera `camera` (its id: cameras are
/// matched by id, as `compare_runs` matches them) projects the rays run
/// `reference`'s calibration of it sees, on the maps' grid over the
/// reference's image: one plane of `dx`, `dy` and `magnitude` (px, `other −
/// reference` at the same ray, after `compensation`, by default the
/// rotation of the other camera's frame that brings the two closest) and
/// `fitted` (1 where the node is in the region either run's features cover,
/// which the rotation is fitted on and the `*_fit_px` statistics are taken
/// over; 0 outside it, where both models extrapolate). `NaN` where either
/// camera has no value. With the same `columns` (`grid_columns`) and
/// `compensation` its numbers are `compare_runs`' projection summary of the
/// camera. `unavailable`, with the library's reason, when the difference
/// cannot be made (too few covered nodes with a value, a singular fit, a
/// camera that is not central). `columns` as `distortion_map`'s; a run whose
/// record states another image size for the camera than the dataset's
/// images have now is `invalid`.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ProjectionDifferenceMapArgs {
    /// The project folder.
    pub dir: UserPath,
    /// The reference run's id: the grid is its camera's image.
    pub reference: String,
    /// The id of the run compared with it.
    pub other: String,
    /// The camera's id.
    pub camera: String,
    #[serde(default)]
    pub columns: Option<u32>,
    #[serde(default)]
    pub compensation: Compensation,
}

/// Two runs' calibrations of a camera compared over its image, or why they
/// are not (see `projection_difference_map`).
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum ProjectionDifferenceMap {
    Compared {
        /// The camera's id.
        camera: String,
        grid: MapGrid,
        plane: MapPlane,
        /// How the other camera's frame was aligned.
        compensation: Compensation,
        /// The angle of the compensating rotation, degrees.
        rotation_deg: f64,
        /// Nodes in the fit (`fitted` is 1).
        fit_nodes: usize,
        /// Nodes where both cameras have a value.
        valid_nodes: usize,
        /// The root mean square of `magnitude` over the fitted nodes, px.
        rms_fit_px: f64,
        /// The largest `magnitude` over the fitted nodes, px.
        max_fit_px: f64,
        /// The root mean square over every node with a value, px.
        rms_valid_px: f64,
        /// The largest `magnitude` over every node with a value, px.
        max_valid_px: f64,
    },
    Unavailable {
        camera: String,
        reason: ProjectionUnavailable,
    },
}

/// One camera of a run as the run's record names it.
struct RecordedCamera {
    /// The ids of the run's cameras, in its camera order.
    ids: Vec<String>,
    /// The camera's index among them.
    index: usize,
    /// The image size the record states for it.
    image_size: Option<[u32; 2]>,
}

/// Camera `id` of run `run`, as its record names it.
fn camera_of(project: &Project, run: &str, id: &str) -> Result<RecordedCamera, StudioError> {
    let info = project.run(run)?;
    let summary = info
        .summary()
        .ok_or_else(|| StudioError::invalid(format!("run {run}: its record names no cameras")))?;
    let ids: Vec<String> = summary.cameras.iter().map(|c| c.id.clone()).collect();
    let index = ids.iter().position(|c| c == id).ok_or_else(|| {
        StudioError::invalid(format!("run {run} has no camera {id}")).on_field("camera")
    })?;
    Ok(RecordedCamera {
        ids,
        index,
        image_size: summary.cameras[index].image_size,
    })
}

pub fn projection_difference_map(
    ctx: &Ctx,
    args: ProjectionDifferenceMapArgs,
) -> Result<ProjectionDifferenceMap, StudioError> {
    let project = open_project(ctx, &args.dir)?;
    let RecordedCamera {
        ids: ids_a,
        index: i,
        image_size: size_a,
    } = camera_of(&project, &args.reference, &args.camera)?;
    let RecordedCamera {
        ids: ids_b,
        index: j,
        image_size: size_b,
    } = camera_of(&project, &args.other, &args.camera)?;
    let run = |id: &str| RunRef {
        dir: args.dir.clone(),
        id: id.to_owned(),
    };
    let reference = ctx.runs.get(ctx, &run(&args.reference))?;
    let other = ctx.runs.get(ctx, &run(&args.other))?;
    let sizes = dataset_sizes(ctx, &args.dir)?;
    let size = size_of(&sizes, i)?;
    // The grid is the camera's other maps' (the dataset's images as they
    // are); `compare_runs` reads the size each record states.
    for (id, recorded) in [(&args.reference, size_a), (&args.other, size_b)] {
        if let Some(recorded) = recorded
            && recorded != size
        {
            return Err(StudioError::invalid(format!(
                "camera {} is {}×{} in run {id}'s record and {}×{} in the dataset now",
                args.camera, recorded[0], recorded[1], size[0], size[1]
            )));
        }
    }
    let columns = args.columns.unwrap_or(DEFAULT_COLUMNS);
    let grid = grid_of(size, Some(columns))?;
    let key = serde_json::json!({
        "cameras": [i, j], "image_size": size, "stride": grid.stride(),
        "compensation": args.compensation,
        "calibrations": [reference.fingerprint, other.fingerprint],
    });
    memoized(
        ctx,
        "projection_difference_map",
        &key,
        |r: &ProjectionDifferenceMap| match r {
            ProjectionDifferenceMap::Compared { plane, .. } => vec![&plane.blob],
            ProjectionDifferenceMap::Unavailable { .. } => Vec::new(),
        },
        |id| {
            let a = ComparedRun::new(
                args.reference.clone(),
                reference.cameras()?,
                ids_a,
                sizes.clone(),
            );
            let b = ComparedRun::new(args.other.clone(), other.cameras()?, ids_b, sizes);
            let mut options = CompareRunsOptions::new();
            options.compensation = args.compensation;
            options.grid_columns = columns;
            let difference = match run_projection_difference(&a, &b, &args.camera, &options)
                .map_err(|e| StudioError::invalid(e.to_string()))?
            {
                Ok(d) => d,
                Err(reason) => {
                    return Ok(ProjectionDifferenceMap::Unavailable {
                        camera: args.camera.clone(),
                        reason,
                    });
                }
            };
            let fitted: Vec<f64> = difference
                .fitted
                .iter()
                .zip(&difference.difference.dx)
                .map(|(&fit, dx)| match (fit, dx.is_finite()) {
                    (_, false) => f64::NAN,
                    (true, true) => 1.0,
                    (false, true) => 0.0,
                })
                .collect();
            let channels = displacement_channels(&difference.difference);
            let mut planes: Vec<(&str, &str, &[f64])> = channels
                .iter()
                .map(|(n, v)| (*n, "px", v.as_slice()))
                .collect();
            planes.push(("fitted", "flag", &fitted));
            let plane = put_plane(&ctx.blobs, id("plane"), &difference.grid, &planes)
                .map_err(StudioError::internal)?;
            Ok(ProjectionDifferenceMap::Compared {
                camera: args.camera.clone(),
                grid: MapGrid::of(&difference.grid),
                plane,
                compensation: args.compensation,
                rotation_deg: difference.rotation_angle().to_degrees(),
                fit_nodes: difference.fit_nodes,
                valid_nodes: difference.valid_nodes,
                rms_fit_px: difference.rms_fit_px,
                max_fit_px: difference.max_fit_px,
                rms_valid_px: difference.rms_valid_px,
                max_valid_px: difference.max_valid_px,
            })
        },
    )
}
