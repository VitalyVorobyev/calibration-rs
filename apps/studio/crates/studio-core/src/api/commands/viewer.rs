//! The viewers' commands: loading an export and making it active, and the
//! math over the active export ([`crate::ops`]): undistortion, epipolar
//! overlays, uncertainty, disparity.

use std::path::Path;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use vision_calibration::core::PixelRect;
use vision_calibration::optim::ProjectionUncertainty;

use crate::api::{Ctx, Resource, StudioError};
use crate::disparity::{DisparityParams, DisparityResult};
use crate::epipolar::EpipolarOverlay;
use crate::ops;
use crate::scope::{RootMode, UserPath};

/// Read and parse the calibration export at `path` and check its schema
/// version. It does not become active: the frontend validates it first and
/// then calls `set_active_export`. Its folder and image root become
/// folders the app may read.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct LoadExportArgs {
    pub path: UserPath,
}

/// A loaded export.
#[derive(Debug, Serialize, JsonSchema)]
pub struct LoadExportResult {
    /// The parsed export JSON; the frontend narrows it by its `kind`.
    pub export: serde_json::Value,
    /// The folder holding the export (absolute, `/`-separated): its image
    /// manifest's `root` resolves against it.
    pub export_dir: String,
}

pub fn load_export(ctx: &Ctx, args: LoadExportArgs) -> Result<LoadExportResult, StudioError> {
    let file = ctx.scope.open_file(&args.path)?;
    let loaded =
        ops::read_export(&file).map_err(|e| StudioError::invalid(e).at(args.path.as_str()))?;
    // The export's images: a folder that does not resolve has none to show.
    if let Some(root) = &loaded.image_root
        && ctx.scope.mode() == RootMode::Growable
    {
        let _ = ctx.scope.open_path(root);
    }
    Ok(LoadExportResult {
        export: loaded.result.export,
        export_dir: loaded.result.export_dir,
    })
}

/// Make `export` the active export, the one every viewer command reads.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SetActiveExportArgs {
    pub export: serde_json::Value,
}

pub fn set_active_export(ctx: &Ctx, args: SetActiveExportArgs) -> Result<(), StudioError> {
    ctx.exports.set(args.export);
    Ok(())
}

/// Map raw (distorted) pixels of `camera` into its undistorted pixel frame;
/// `null` where back-projection has no finite ray.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct UndistortPointsArgs {
    pub camera: usize,
    pub points_px: Vec<[f64; 2]>,
}

pub fn undistort_points(
    ctx: &Ctx,
    args: UndistortPointsArgs,
) -> Result<Vec<Option<[f64; 2]>>, StudioError> {
    Ok(ops::undistort_points(
        &ctx.exports,
        args.camera,
        args.points_px,
    )?)
}

/// The epipolar polyline and epipole in camera `cam_b` (distorted pixels)
/// of the pixel `point_px` of camera `cam_a`, through each camera's full
/// model.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct EpipolarOverlayArgs {
    pub cam_a: usize,
    pub cam_b: usize,
    pub point_px: [f64; 2],
}

pub fn epipolar_overlay(
    ctx: &Ctx,
    args: EpipolarOverlayArgs,
) -> Result<EpipolarOverlay, StudioError> {
    Ok(ops::compute_epipolar_overlay(
        &ctx.exports,
        args.cam_a,
        args.cam_b,
        args.point_px,
    )?)
}

/// A straight epipolar segment in camera `cam_b`'s undistorted frame,
/// clipped to its `image_width_b` × `image_height_b` image, of the pixel
/// `point_px` of camera `cam_a`'s undistorted frame.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct EpipolarOverlayUndistortedArgs {
    pub cam_a: usize,
    pub cam_b: usize,
    pub point_px: [f64; 2],
    pub image_width_b: f64,
    pub image_height_b: f64,
}

pub fn epipolar_overlay_undistorted(
    ctx: &Ctx,
    args: EpipolarOverlayUndistortedArgs,
) -> Result<EpipolarOverlay, StudioError> {
    Ok(ops::compute_epipolar_overlay_undistorted(
        &ctx.exports,
        args.cam_a,
        args.cam_b,
        args.point_px,
        [args.image_width_b, args.image_height_b],
    )?)
}

/// The value of each labelled parameter of the active export that is a
/// camera parameter, aligned with `labels`; `null` for poses and planes.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct CameraParameterValuesArgs {
    pub labels: Vec<String>,
}

pub fn camera_parameter_values(
    ctx: &Ctx,
    args: CameraParameterValuesArgs,
) -> Result<Vec<Option<f64>>, StudioError> {
    Ok(ops::camera_parameter_values(&ctx.exports, &args.labels)?)
}

/// The projection uncertainty of `camera` of the active export over an
/// image of `image_width` × `image_height` pixels.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ProjectionUncertaintyArgs {
    pub camera: usize,
    pub image_width: u32,
    pub image_height: u32,
}

pub fn projection_uncertainty(
    ctx: &Ctx,
    args: ProjectionUncertaintyArgs,
) -> Result<ProjectionUncertainty, StudioError> {
    Ok(ops::projection_uncertainty(
        &ctx.exports,
        args.camera,
        [args.image_width, args.image_height],
    )?)
}

/// A disparity map of the stereo pair `path_a` (camera `cam_a`) and
/// `path_b` (camera `cam_b`) at `pose`: rectified from the active export,
/// dense-matched, with colour-mapped images and quality metrics.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct DisparityArgs {
    pub path_a: UserPath,
    pub path_b: UserPath,
    pub cam_a: usize,
    pub cam_b: usize,
    pub pose: usize,
    pub params: DisparityParams,
}

pub fn disparity(ctx: &Ctx, args: DisparityArgs) -> Result<DisparityResult, StudioError> {
    let path_a = ctx.scope.existing_file(&args.path_a)?;
    let path_b = ctx.scope.existing_file(&args.path_b)?;
    Ok(ops::compute_disparity(
        &ctx.exports,
        [&path_a, &path_b],
        args.cam_a,
        args.cam_b,
        args.pose,
        &args.params,
    )?)
}

/// The image at `path` rendered into `camera`'s undistorted pixel frame (a
/// PNG), cropped to `roi` when given.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct UndistortedImageQuery {
    pub path: UserPath,
    pub camera: usize,
    #[serde(default)]
    pub roi: Option<PixelRect>,
}

pub fn undistorted_image(ctx: &Ctx, query: UndistortedImageQuery) -> Result<Resource, StudioError> {
    let file = ctx.scope.existing_file(&query.path)?;
    let png = ops::load_undistorted_image(&ctx.exports, Path::new(&file), query.camera, query.roi)?;
    Ok(Resource {
        media_type: "image/png",
        body: png,
    })
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use serde_json::json;

    use crate::api::{Api, ErrorKind};
    use crate::run_source::{find_run, slash};
    use crate::scope::RootScope;

    /// The committed fixture run (a two-camera rig with images), behind the
    /// fixed scope the browser viewer gives it.
    fn viewer() -> (Api, crate::run_source::RunSource) {
        let fixture = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../e2e/fixtures/viewer-run");
        let run = find_run(&fixture).unwrap();
        let api = Api::new(RootScope::fixed([run.dir.clone()]));
        (api, run)
    }

    #[test]
    fn the_viewer_commands_answer_over_the_active_export() {
        let (api, run) = viewer();
        let early = api
            .invoke(
                "undistort_points",
                json!({"camera": 0, "points_px": [[1.0, 2.0]]}),
            )
            .unwrap_err();
        assert_eq!(early.kind, ErrorKind::Invalid);
        assert_eq!(early.message, "no export loaded yet");

        let loaded = api
            .invoke("load_export", json!({"path": slash(&run.export)}))
            .unwrap();
        assert_eq!(loaded["export_dir"], slash(&run.dir));
        assert_eq!(loaded["export"]["kind"], "rig_extrinsics");
        api.invoke("set_active_export", json!({"export": loaded["export"]}))
            .unwrap();

        // A distortion-free camera maps pixels to themselves.
        let px = api
            .invoke(
                "undistort_points",
                json!({"camera": 1, "points_px": [[10.0, 20.0]]}),
            )
            .unwrap();
        assert!((px[0][0].as_f64().unwrap() - 10.0).abs() < 1e-6, "{px}");
        let values = api
            .invoke(
                "camera_parameter_values",
                json!({"labels": ["cam1.fx", "cam0.cy"]}),
            )
            .unwrap();
        assert_eq!(values, json!([200.0, 59.5]));
        api.invoke(
            "projection_uncertainty",
            json!({"camera": 0, "image_width": 160, "image_height": 120}),
        )
        .unwrap();
        api.invoke(
            "epipolar_overlay_undistorted",
            json!({"cam_a": 0, "cam_b": 1, "point_px": [80.0, 60.0],
                   "image_width_b": 160.0, "image_height_b": 120.0}),
        )
        .unwrap();

        let image = slash(&run.dir.join("images/cam0_view0.png"));
        let png = api
            .resource(
                "undistorted_image",
                json!({"path": image, "camera": 0, "roi": null}),
            )
            .unwrap();
        assert_eq!(png.media_type, "image/png");
        assert!(png.body.starts_with(b"\x89PNG"));

        let bad = api
            .invoke("undistort_points", json!({"camera": "zero"}))
            .unwrap_err();
        assert_eq!(bad.kind, ErrorKind::InvalidArgs);
        // The old camelCase names are refused.
        let camel = api
            .invoke("undistort_points", json!({"camera": 0, "pointsPx": []}))
            .unwrap_err();
        assert_eq!(camel.kind, ErrorKind::InvalidArgs);
    }

    #[test]
    fn paths_outside_the_viewer_scope_are_refused() {
        let (api, run) = viewer();
        let outside = Path::new(env!("CARGO_MANIFEST_DIR")).join("Cargo.toml");
        let err = api
            .invoke("read_manifest", json!({"path": slash(&outside)}))
            .unwrap_err();
        assert_eq!(err.kind, ErrorKind::ForbiddenPath);
        let err = api
            .resource(
                "undistorted_image",
                json!({"path": slash(&run.dir.join("../viewer-run-vcal/calibration.json")), "camera": 0}),
            )
            .unwrap_err();
        assert_eq!(err.kind, ErrorKind::ForbiddenPath);
        let err = api
            .invoke("load_export", json!({"path": "export.json"}))
            .unwrap_err();
        assert_eq!(err.kind, ErrorKind::ForbiddenPath);
    }
}
