//! The viewers' commands over a run's export ([`RunRef`]): undistortion,
//! epipolar overlays, uncertainty, and the disparity job.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use vision_calibration::optim::ProjectionUncertainty;

use crate::api::{Ctx, JobCx, StudioError};
use crate::blobs::{BlobRef, BlobStore};
use crate::disparity::{DisparityMetrics, DisparityParams};
use crate::epipolar::{self, EpipolarOverlay};
use crate::images::ImageCache;
use crate::planes;
use crate::runs::RunRef;
use crate::scope::UserPath;
use crate::uncertainty;

/// Map raw (distorted) pixels of `camera` of the run into its undistorted
/// pixel frame; `null` where back-projection has no finite ray.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct UndistortPointsArgs {
    pub run: RunRef,
    pub camera: usize,
    pub points_px: Vec<[f64; 2]>,
}

pub fn undistort_points(
    ctx: &Ctx,
    args: UndistortPointsArgs,
) -> Result<Vec<Option<[f64; 2]>>, StudioError> {
    let run = ctx.runs.get(ctx, &args.run)?;
    Ok(epipolar::undistort_points(
        &run.export,
        args.camera,
        args.points_px,
    )?)
}

/// The epipolar polyline and epipole in camera `cam_b` (distorted pixels)
/// of the pixel `point_px` of camera `cam_a`, through each camera's full
/// model of the run.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct EpipolarOverlayArgs {
    pub run: RunRef,
    pub cam_a: usize,
    pub cam_b: usize,
    pub point_px: [f64; 2],
}

pub fn epipolar_overlay(
    ctx: &Ctx,
    args: EpipolarOverlayArgs,
) -> Result<EpipolarOverlay, StudioError> {
    let run = ctx.runs.get(ctx, &args.run)?;
    Ok(epipolar::compute_overlay(
        &run.export,
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
    pub run: RunRef,
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
    let run = ctx.runs.get(ctx, &args.run)?;
    Ok(epipolar::compute_overlay_undistorted(
        &run.export,
        args.cam_a,
        args.cam_b,
        args.point_px,
        args.image_width_b,
        args.image_height_b,
    )?)
}

/// The projection uncertainty of `camera` of the run over an image of
/// `image_width` × `image_height` pixels.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ProjectionUncertaintyArgs {
    pub run: RunRef,
    pub camera: usize,
    pub image_width: u32,
    pub image_height: u32,
}

pub fn projection_uncertainty(
    ctx: &Ctx,
    args: ProjectionUncertaintyArgs,
) -> Result<ProjectionUncertainty, StudioError> {
    let run = ctx.runs.get(ctx, &args.run)?;
    Ok(uncertainty::projection_uncertainty(
        run.cameras()?,
        args.camera,
        [args.image_width, args.image_height],
    )?)
}

/// A disparity map of the stereo pair `path_a` (camera `cam_a`) and
/// `path_b` (camera `cam_b`) at `pose`: rectified from the run's export and
/// dense-matched. The output names its fields as blobs: a value plane of
/// disparity, depth and the rectified left image, the point cloud, and the
/// rectified pair. The same request again answers from the blob store while
/// it still holds them.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct DisparityArgs {
    pub run: RunRef,
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
    let run = ctx.runs.get(ctx, &args.run)?;
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
        "calibration": run.fingerprint,
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
        &run.export,
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
    use std::path::{Path, PathBuf};

    use serde_json::json;

    use crate::api::{Api, Conditions, ErrorKind, Setup};
    use crate::scope::RootScope;

    /// The committed fixture project and its legacy run (a two-camera rig
    /// with images).
    struct Fixture {
        project: PathBuf,
        images: PathBuf,
    }

    fn slash(path: &Path) -> String {
        path.to_string_lossy().replace('\\', "/")
    }

    /// The fixture, behind a scope of the fixture project alone.
    fn viewer() -> (Api, Fixture, tempfile::TempDir) {
        let project = dunce::canonicalize(
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../../e2e/fixtures/library/rig-demo"),
        )
        .unwrap();
        let fixture = Fixture {
            images: project.join("images"),
            project: project.clone(),
        };
        let cache = tempfile::tempdir().unwrap();
        let setup = Setup::for_tests(Some(cache.path().to_path_buf()));
        let api = Api::with_scope(RootScope::fixed([project]), setup);
        (api, fixture, cache)
    }

    fn run(fixture: &Fixture) -> serde_json::Value {
        json!({"dir": slash(&fixture.project), "id": "run-001"})
    }

    #[test]
    fn the_viewer_commands_answer_over_a_named_run() {
        let (api, fixture, _cache) = viewer();
        let run = run(&fixture);
        let export = api.invoke("run_export", json!({"run": run})).unwrap();
        assert_eq!(export["export"]["kind"], "rig_extrinsics");
        // A run of the app's earlier format: its frames resolve from the
        // run folder's image root (`../../images`).
        assert_eq!(export["image_root"], slash(&fixture.images));

        // A distortion-free camera maps pixels to themselves.
        let px = api
            .invoke(
                "undistort_points",
                json!({"run": run, "camera": 1, "points_px": [[10.0, 20.0]]}),
            )
            .unwrap();
        assert!((px[0][0].as_f64().unwrap() - 10.0).abs() < 1e-6, "{px}");
        let values = api
            .invoke("camera_parameter_values", json!({"run": run}))
            .unwrap();
        let value = |camera: usize, name: &str| {
            values["cameras"][camera]["parameters"]
                .as_array()
                .unwrap()
                .iter()
                .find(|p| p["name"] == name)
                .map(|p| (p["label"].clone(), p["value"].clone()))
                .unwrap()
        };
        assert_eq!(value(1, "fx"), (json!("cam1.fx"), json!(200.0)));
        assert_eq!(value(0, "cy"), (json!("cam0.cy"), json!(59.5)));
        api.invoke(
            "projection_uncertainty",
            json!({"run": run, "camera": 0, "image_width": 160, "image_height": 120}),
        )
        .unwrap();
        api.invoke(
            "epipolar_overlay_undistorted",
            json!({"run": run, "cam_a": 0, "cam_b": 1, "point_px": [80.0, 60.0],
                   "image_width_b": 160.0, "image_height_b": 120.0}),
        )
        .unwrap();

        let image = slash(&fixture.images.join("cam0_view0.png"));
        let png = api
            .resource(
                "undistorted_image",
                json!({"run": run, "path": image, "camera": 0, "roi": null}),
                &Conditions::default(),
            )
            .unwrap();
        assert_eq!(png.media_type, "image/png");
        assert!(png.read_body().unwrap().starts_with(b"\x89PNG"));

        // A run the project does not have.
        let missing = api
            .invoke(
                "undistort_points",
                json!({"run": {"dir": run["dir"], "id": "run-404"}, "camera": 0, "points_px": []}),
            )
            .unwrap_err();
        assert_eq!(missing.kind, ErrorKind::RunNotFound);
        let bad = api
            .invoke("undistort_points", json!({"run": run, "camera": "zero"}))
            .unwrap_err();
        assert_eq!(bad.kind, ErrorKind::InvalidArgs);
        // The old camelCase names are refused.
        let camel = api
            .invoke(
                "undistort_points",
                json!({"run": run, "camera": 0, "pointsPx": []}),
            )
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
        let (api, fixture, _cache) = viewer();
        let args = json!({
            "run": run(&fixture),
            "path_a": slash(&fixture.images.join("cam0_view0.png")),
            "path_b": slash(&fixture.images.join("cam1_view0.png")),
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
        let (api, fixture, _cache) = viewer();
        let outside = Path::new(env!("CARGO_MANIFEST_DIR")).join("Cargo.toml");
        let err = api
            .invoke("read_manifest", json!({"path": slash(&outside)}))
            .unwrap_err();
        assert_eq!(err.kind, ErrorKind::ForbiddenPath);
        let err = api
            .resource(
                "undistorted_image",
                json!({"run": run(&fixture),
                       "path": slash(&fixture.images.join("../../vcal-run/calibration.json")),
                       "camera": 0}),
                &Conditions::default(),
            )
            .unwrap_err();
        assert_eq!(err.kind, ErrorKind::ForbiddenPath);
        let err = api
            .invoke(
                "run_export",
                json!({"run": {"dir": "rig-demo", "id": "run-001"}}),
            )
            .unwrap_err();
        assert_eq!(err.kind, ErrorKind::ForbiddenPath);
    }
}
