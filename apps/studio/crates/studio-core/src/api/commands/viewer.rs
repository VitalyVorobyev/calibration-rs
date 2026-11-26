//! The viewers' commands: loading an export and making it active, and the
//! math over the active export ([`crate::ops`]): undistortion, epipolar
//! overlays, uncertainty, and the disparity job.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use vision_calibration::optim::ProjectionUncertainty;

use crate::api::{Ctx, JobCx, StudioError};
use crate::blobs::{BlobRef, BlobStore};
use crate::disparity::{DisparityMetrics, DisparityParams};
use crate::epipolar::EpipolarOverlay;
use crate::images::ImageCache;
use crate::ops;
use crate::planes;
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
/// `path_b` (camera `cam_b`) at `pose`: rectified from the active export and
/// dense-matched. The output names its fields as blobs: a value plane of
/// disparity, depth and the rectified left image, the point cloud, and the
/// rectified pair. The same request again answers from the blob store while
/// it still holds them.
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

/// A dense match, its fields as blobs (`blob` resource).
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct DisparityResult {
    /// A `VAM1` value plane of the matched size with three channels:
    /// disparity (px) and metric depth (m), both `NaN` where there is no
    /// match, and the rectified left image's luminance (0–255).
    pub planes: BlobRef,
    /// Rectified left | right side by side with shared horizontal epipolar
    /// lines (a PNG).
    pub rectified_pair: BlobRef,
    /// The point cloud's positions: `3 · point_count` little-endian `f32`
    /// (x, y, z in the reference camera's frame, metres).
    pub point_positions: BlobRef,
    /// Its colours: `3 · point_count` little-endian `f32` in `[0, 1]`.
    pub point_colors: BlobRef,
    pub point_count: usize,
    /// Matched (downscaled) image width.
    pub width: usize,
    /// Matched (downscaled) image height.
    pub height: usize,
    pub metrics: DisparityMetrics,
}

pub fn disparity(
    ctx: &Ctx,
    args: DisparityArgs,
    _: &JobCx<()>,
) -> Result<DisparityResult, StudioError> {
    let path_a = ctx.scope.existing_file(&args.path_a)?;
    let path_b = ctx.scope.existing_file(&args.path_b)?;
    let export = ctx
        .exports
        .snapshot()
        .ok_or_else(|| StudioError::invalid("no export loaded yet"))?;
    // What makes the result: the request, the images as they are on disk
    // and the calibration.
    let request = serde_json::json!({
        "images": [ImageCache::key(&path_a, "")?, ImageCache::key(&path_b, "")?],
        "cam_a": args.cam_a,
        "cam_b": args.cam_b,
        "pose": args.pose,
        "block_size": args.params.block_size,
        "semi_global": args.params.semi_global,
        "downscale": args.params.downscale,
        "calibration": export.fingerprint,
    })
    .to_string();
    let id = |part: &str| BlobStore::id(&[b"disparity", request.as_bytes(), part.as_bytes()]);
    if let Some((_, memo)) = ctx.blobs.get(&id("result"))
        && let Ok(result) = serde_json::from_slice::<DisparityResult>(&memo)
        && ctx.blobs.has_all([
            &result.planes,
            &result.rectified_pair,
            &result.point_positions,
            &result.point_colors,
        ])
    {
        return Ok(result);
    }

    let data = crate::disparity::compute(
        &export.value,
        &path_a,
        &path_b,
        args.cam_a,
        args.cam_b,
        args.pose,
        &args.params,
    )
    .map_err(StudioError::invalid)?;
    let plane = planes::encode(
        data.width,
        data.height,
        1,
        &[&data.disparity, &data.depth, &data.left],
    )
    .map_err(|e| StudioError::internal(format!("encode the disparity plane: {e}")))?;
    let cloud = &data.point_cloud;
    let result = DisparityResult {
        planes: ctx.blobs.put(id("planes"), planes::MEDIA_TYPE, plane),
        rectified_pair: ctx.blobs.put(id("pair"), "image/png", data.pair_png),
        point_positions: ctx.blobs.put(
            id("positions"),
            planes::F32_MEDIA_TYPE,
            planes::f32_bytes(&cloud.positions),
        ),
        point_colors: ctx.blobs.put(
            id("colors"),
            planes::F32_MEDIA_TYPE,
            planes::f32_bytes(&cloud.colors),
        ),
        point_count: cloud.count,
        width: data.width,
        height: data.height,
        metrics: data.metrics,
    };
    let memo = serde_json::to_vec(&result)
        .map_err(|e| StudioError::internal(format!("serialize the disparity result: {e}")))?;
    ctx.blobs.put(id("result"), "application/json", memo);
    Ok(result)
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use serde_json::json;

    use crate::api::{Api, Conditions, Dirs, ErrorKind};
    use crate::run_source::{find_run, slash};
    use crate::scope::RootScope;

    /// The committed fixture run (a two-camera rig with images), behind the
    /// fixed scope the browser viewer gives it.
    fn viewer() -> (Api, crate::run_source::RunSource, tempfile::TempDir) {
        let fixture = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../e2e/fixtures/viewer-run");
        let run = find_run(&fixture).unwrap();
        let cache = tempfile::tempdir().unwrap();
        let dirs = Dirs {
            cache: cache.path().to_path_buf(),
        };
        let api = Api::new(RootScope::fixed([run.dir.clone()]), dirs);
        (api, run, cache)
    }

    #[test]
    fn the_viewer_commands_answer_over_the_active_export() {
        let (api, run, _cache) = viewer();
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
                &Conditions::default(),
            )
            .unwrap();
        assert_eq!(png.media_type, "image/png");
        assert!(png.read_body().unwrap().starts_with(b"\x89PNG"));

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

    /// Wait for job `id` to end; its terminal state.
    fn finish(api: &Api, id: &serde_json::Value) -> serde_json::Value {
        for _ in 0..600 {
            let info = api.invoke("job_status", json!({ "id": id })).unwrap();
            if !matches!(info["state"]["kind"].as_str(), Some("queued" | "running")) {
                return info["state"].clone();
            }
            std::thread::sleep(std::time::Duration::from_millis(50));
        }
        panic!("job {id} did not end");
    }

    #[test]
    fn a_disparity_job_answers_blobs_and_memoizes_them() {
        let (api, run, _cache) = viewer();
        let loaded = api
            .invoke("load_export", json!({"path": slash(&run.export)}))
            .unwrap();
        api.invoke("set_active_export", json!({"export": loaded["export"]}))
            .unwrap();
        let args = json!({
            "path_a": slash(&run.dir.join("images/cam0_view0.png")),
            "path_b": slash(&run.dir.join("images/cam1_view0.png")),
            "cam_a": 0, "cam_b": 1, "pose": 0,
            "params": {"block_size": 5, "semi_global": true, "downscale": 1}
        });
        let job = api.invoke("disparity", args.clone()).unwrap();
        let state = finish(&api, &job["id"]);
        assert_eq!(state["kind"], "done", "{state}");
        let output = &state["output"];
        let (w, h) = (
            output["width"].as_u64().unwrap(),
            output["height"].as_u64().unwrap(),
        );
        let plane = api
            .resource(
                "blob",
                json!({"id": output["planes"]["id"]}),
                &Conditions::default(),
            )
            .unwrap();
        assert_eq!(plane.media_type, "application/x-vam1");
        let bytes = plane.read_body().unwrap();
        assert_eq!(&bytes[..4], b"VAM1");
        assert_eq!(u32::from_le_bytes(bytes[16..20].try_into().unwrap()), 3);
        assert_eq!(bytes.len() as u64, 24 + w * h * 3 * 4);
        let pair = api
            .resource(
                "blob",
                json!({"id": output["rectified_pair"]["id"]}),
                &Conditions::default(),
            )
            .unwrap();
        assert!(pair.read_body().unwrap().starts_with(b"\x89PNG"));
        let count = output["point_count"].as_u64().unwrap();
        assert_eq!(
            output["point_positions"]["len"].as_u64().unwrap(),
            count * 12
        );

        // The same request answers the same blobs, from the store.
        let again = api.invoke("disparity", args).unwrap();
        let state = finish(&api, &again["id"]);
        assert_eq!(&state["output"], output);
    }

    #[test]
    fn paths_outside_the_viewer_scope_are_refused() {
        let (api, run, _cache) = viewer();
        let outside = Path::new(env!("CARGO_MANIFEST_DIR")).join("Cargo.toml");
        let err = api
            .invoke("read_manifest", json!({"path": slash(&outside)}))
            .unwrap_err();
        assert_eq!(err.kind, ErrorKind::ForbiddenPath);
        let err = api
            .resource(
                "undistorted_image",
                json!({"path": slash(&run.dir.join("../viewer-run-vcal/calibration.json")), "camera": 0}),
                &Conditions::default(),
            )
            .unwrap_err();
        assert_eq!(err.kind, ErrorKind::ForbiddenPath);
        let err = api
            .invoke("load_export", json!({"path": "export.json"}))
            .unwrap_err();
        assert_eq!(err.kind, ErrorKind::ForbiddenPath);
    }
}
