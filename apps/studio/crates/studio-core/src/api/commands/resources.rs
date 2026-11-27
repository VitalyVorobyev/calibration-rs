//! The resources: what the frontend's URLs (`api.urls.*`) load. A source
//! image, an image undistorted through a run's export, a derived blob, and
//! a run's report page. Every path goes through the scope; derived images are cached on
//! disk ([`crate::images::ImageCache`]), blobs in memory
//! ([`crate::blobs::BlobStore`]).

use std::sync::Arc;

use schemars::JsonSchema;
use serde::Deserialize;
use vision_calibration::core::PixelRect;

use crate::api::resource::{IMMUTABLE, REVALIDATE};
use crate::api::{Ctx, ErrorKind, Resource, StudioError};
use crate::images::{self, ImageCache};
use crate::runs::RunRef;
use crate::scope::UserPath;

/// The image file at `path`, as a browser can decode it: JPEG, PNG, WebP,
/// GIF and BMP are the file itself; TIFF, PNM and the other formats the
/// back end reads become an 8-bit PNG (a 16-bit image reduced as the
/// detectors reduce it). With `max`, a JPEG thumbnail whose longest side is
/// at most `max`, snapped up to 160, 320, 640 or 1280.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ImageQuery {
    pub path: UserPath,
    #[serde(default)]
    pub max: Option<u32>,
}

pub fn image(ctx: &Ctx, query: ImageQuery) -> Result<Resource, StudioError> {
    let file = ctx.scope.existing_file(&query.path)?;
    let cached = |variant: &str, ext: &str, make: &dyn Fn() -> Result<Vec<u8>, StudioError>| {
        let key = ImageCache::key(&file, variant)?;
        ctx.images.get_or_make(&key, ext, make)
    };
    if let Some(max) = query.max {
        let size = images::thumbnail_size(max);
        let path = cached(&format!("thumbnail:{size}"), "jpg", &|| {
            images::thumbnail(&images::decode(&file)?, size)
        })?;
        return Ok(file_resource(path, "image/jpeg"));
    }
    if let Some(media_type) = images::browser_media_type(&file)? {
        return Ok(file_resource(file, media_type));
    }
    let path = cached("png8", "png", &|| images::png8(&images::decode(&file)?))?;
    Ok(file_resource(path, "image/png"))
}

fn file_resource(path: std::path::PathBuf, media_type: &'static str) -> Resource {
    Resource::File {
        path,
        media_type,
        cache_control: REVALIDATE,
    }
}

/// The image at `path` rendered into `camera`'s undistorted pixel frame of
/// the run's export (a PNG), cropped to `roi` when given.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct UndistortedImageQuery {
    pub run: RunRef,
    pub path: UserPath,
    pub camera: usize,
    #[serde(default)]
    pub roi: Option<PixelRect>,
}

pub fn undistorted_image(ctx: &Ctx, query: UndistortedImageQuery) -> Result<Resource, StudioError> {
    let file = ctx.scope.existing_file(&query.path)?;
    let run = ctx.runs.get(ctx, &query.run)?;
    let roi = query
        .roi
        .map(|r| format!("{},{},{},{}", r.x, r.y, r.w, r.h))
        .unwrap_or_default();
    let variant = format!("undistorted:{}:{}:{roi}", run.fingerprint, query.camera);
    let key = ImageCache::key(&file, &variant)?;
    let path = ctx.images.get_or_make(&key, "png", || {
        crate::epipolar::undistort_image_png(&run.export, &file, query.camera, query.roi)
            .map_err(StudioError::invalid)
    })?;
    Ok(file_resource(path, "image/png"))
}

/// The run's page, `report.html` (written with every calibration this app
/// runs), served sandboxed: its scripts run without access to the app.
/// `io` when the run has none.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ReportQuery {
    pub run: RunRef,
}

pub fn report(ctx: &Ctx, query: ReportQuery) -> Result<Resource, StudioError> {
    let run = crate::runs::resolve(ctx, &query.run)?;
    let path = ctx.scope.check(&run.info.report_path())?;
    Ok(Resource::Page { path })
}

/// The blob `id` a command made (a value plane, raw floats, a PNG).
/// `blob_expired` once the store has evicted it: run the command again.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct BlobQuery {
    pub id: String,
}

pub fn blob(ctx: &Ctx, query: BlobQuery) -> Result<Resource, StudioError> {
    let (media_type, bytes) = ctx.blobs.get(&query.id).ok_or_else(|| {
        StudioError::new(
            ErrorKind::BlobExpired,
            format!("blob {} is no longer kept; run its command again", query.id),
        )
    })?;
    Ok(Resource::Bytes {
        etag: format!("\"{}\"", query.id),
        bytes: Arc::clone(&bytes),
        media_type,
        cache_control: IMMUTABLE,
    })
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use serde_json::json;

    use crate::api::{Api, Body, Conditions, ErrorKind, Setup};
    use crate::scope::RootScope;

    fn api(root: &Path, cache: &Path) -> Api {
        Api::with_scope(
            RootScope::fixed([root.to_path_buf()]),
            Setup::for_tests(Some(cache.to_path_buf())),
        )
    }

    fn get(api: &Api, name: &str, query: serde_json::Value) -> crate::api::ResourceReply {
        api.resource(name, query, &Conditions::default()).unwrap()
    }

    #[test]
    fn images_pass_through_transcode_and_shrink() {
        let tmp = tempfile::tempdir().unwrap();
        let root = dunce::canonicalize(tmp.path()).unwrap().join("data");
        let cache = tmp.path().join("cache");
        std::fs::create_dir(&root).unwrap();
        let frame =
            image::DynamicImage::ImageLuma16(image::ImageBuffer::from_fn(400, 200, |x, y| {
                image::Luma([(x * 100 + y) as u16])
            }));
        frame.save(root.join("a.png")).unwrap();
        frame.save(root.join("b.tif")).unwrap();
        let api = api(&root, &cache);
        let path = |name: &str| root.join(name).to_string_lossy().into_owned();

        let png = get(&api, "image", json!({ "path": path("a.png") }));
        assert_eq!((png.status, png.media_type), (200, "image/png"));
        assert!(matches!(&png.body, Body::File { path: p, .. } if p == &root.join("a.png")));

        let tiff = get(&api, "image", json!({ "path": path("b.tif") }));
        assert_eq!(tiff.media_type, "image/png");
        let decoded = image::load_from_memory(&tiff.read_body().unwrap()).unwrap();
        assert_eq!(decoded.to_luma8(), frame.to_luma8());
        // Cached under the cache folder, never beside the dataset.
        assert!(matches!(&tiff.body, Body::File { path: p, .. } if p.starts_with(&cache)));

        let thumb = get(&api, "image", json!({ "path": path("b.tif"), "max": 200 }));
        assert_eq!(thumb.media_type, "image/jpeg");
        let small = image::load_from_memory(&thumb.read_body().unwrap()).unwrap();
        assert_eq!((small.width(), small.height()), (320, 160));

        // A range and a revalidation of the same image.
        let conditions = Conditions {
            range: Some("bytes=0-7".into()),
            if_none_match: None,
        };
        let head = api
            .resource("image", json!({ "path": path("a.png") }), &conditions)
            .unwrap();
        assert_eq!(head.status, 206);
        assert_eq!(head.read_body().unwrap(), b"\x89PNG\r\n\x1a\n");
        let conditions = Conditions {
            range: None,
            if_none_match: Some(png.etag.clone()),
        };
        let same = api
            .resource("image", json!({ "path": path("a.png") }), &conditions)
            .unwrap();
        assert_eq!(same.status, 304);
    }

    #[test]
    fn images_outside_the_roots_are_refused() {
        let tmp = tempfile::tempdir().unwrap();
        let base = dunce::canonicalize(tmp.path()).unwrap();
        let root = base.join("data");
        std::fs::create_dir(&root).unwrap();
        std::fs::write(base.join("secret.png"), b"x").unwrap();
        let api = api(&root, &base.join("cache"));
        for path in [
            base.join("secret.png"),
            root.join("../secret.png"),
            Path::new("secret.png").to_path_buf(),
        ] {
            let err = api
                .resource(
                    "image",
                    json!({ "path": path.to_string_lossy() }),
                    &Conditions::default(),
                )
                .unwrap_err();
            assert_eq!(err.kind, ErrorKind::ForbiddenPath, "{}", path.display());
        }
        let err = api
            .resource(
                "blob",
                json!({ "id": "0".repeat(64) }),
                &Conditions::default(),
            )
            .unwrap_err();
        assert_eq!(err.kind, ErrorKind::BlobExpired);
        // A resource is not a command, nor a command a resource.
        let err = api.invoke("image", json!({})).unwrap_err();
        assert_eq!(err.kind, ErrorKind::UnknownCommand);
        let err = api
            .resource("list_jobs", json!({}), &Conditions::default())
            .unwrap_err();
        assert_eq!(err.kind, ErrorKind::UnknownCommand);
    }
}
