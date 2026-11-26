//! The Run workspace: loading a setup (a manifest file, a sniffed folder, a
//! Quick Start sample, a default config) and running the calibration
//! ([`crate::run`]).

use std::path::Path;

use schemars::JsonSchema;
use serde::Deserialize;
use vision_calibration::dataset::{DatasetSpec, ManifestError, samples};

use crate::api::{Ctx, ErrorKind, JobCx, StudioError};
use crate::run::{self, RunProgress, RunResponse};
use crate::scope::UserPath;

/// Read the dataset manifest at `path`: TOML for a `.toml` extension, JSON
/// otherwise. Its folder becomes one the app may read.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ReadManifestArgs {
    pub path: UserPath,
}

pub fn read_manifest(ctx: &Ctx, args: ReadManifestArgs) -> Result<DatasetSpec, StudioError> {
    let file = ctx.scope.open_file(&args.path)?;
    DatasetSpec::from_path(&file).map_err(|e| {
        let kind = match e {
            ManifestError::Io { .. } => ErrorKind::Io,
            _ => ErrorKind::Manifest,
        };
        StudioError::new(kind, e.to_string()).at(args.path.as_str())
    })
}

/// Infer a dataset manifest from the folder `folder`: images grouped into
/// cameras, a robot-pose file, a guessed topology. What cannot be
/// determined is left at a placeholder and listed in the manifest's
/// `_unresolved` (the same inference as `vcal init`).
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SniffFolderArgs {
    pub folder: UserPath,
}

pub fn sniff_folder(ctx: &Ctx, args: SniffFolderArgs) -> Result<DatasetSpec, StudioError> {
    let folder = ctx.scope.open_dir(&args.folder)?;
    vision_calibration::dataset::sniff_folder(&folder)
        .map_err(|e| StudioError::new(ErrorKind::Io, e.to_string()).at(args.folder.as_str()))
}

/// Make the sample dataset `id` (one of the library's sample catalogue)
/// available locally and answer the absolute path of its manifest. The
/// files are downloaded into the platform cache (`VISION_CALIBRATION_DATA`
/// overrides it), checked against their SHA-256 and reused later, so the
/// app carries no dataset.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct FetchSampleArgs {
    pub id: String,
}

pub fn fetch_sample(_: &Ctx, args: FetchSampleArgs, _: &JobCx<()>) -> Result<String, StudioError> {
    let cache =
        samples::default_cache_dir().map_err(|e| StudioError::new(ErrorKind::Io, e.to_string()))?;
    fetch_sample_into(&args.id, &cache).map_err(|e| StudioError::new(ErrorKind::Io, e))
}

/// [`fetch_sample`] against an explicit cache directory.
fn fetch_sample_into(id: &str, cache: &Path) -> Result<String, String> {
    samples::fetch(id, cache)
        .map(|manifest| manifest.to_string_lossy().into_owned())
        .map_err(|e| format!("sample {id}: {e}"))
}

/// Run a calibration end to end: detection (cached), the solve and the
/// export, reporting each stage and the library's reports within it. The
/// folder `manifest_dir` is what the manifest's relative paths resolve
/// against; it becomes one the app may read. A successful run's export
/// becomes the active export.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct RunCalibrationArgs {
    /// The dataset manifest (`DatasetSpec` JSON).
    pub manifest: serde_json::Value,
    /// The topology's `*Config` JSON.
    pub config: serde_json::Value,
    pub manifest_dir: UserPath,
}

/// The run's outcome is its output; a cancelled run ends the job as
/// `cancelled`.
pub fn run_calibration(
    ctx: &Ctx,
    args: RunCalibrationArgs,
    cx: &JobCx<RunProgress>,
) -> Result<RunResponse, StudioError> {
    let dir = ctx.scope.open_dir(&args.manifest_dir)?;
    let reports = cx.clone();
    let response = run::run_calibration(
        args.manifest,
        args.config,
        &dir.to_string_lossy(),
        cx.token(),
        move |progress| reports.emit(&progress),
    );
    match response {
        RunResponse::Cancelled => Err(StudioError::cancelled()),
        RunResponse::Ok(success) => {
            ctx.exports.set(success.export.clone());
            Ok(RunResponse::Ok(success))
        }
        other => Ok(other),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fetch_sample_names_the_unknown_sample_and_the_known_ones() {
        let cache = tempfile::tempdir().unwrap();
        let err = fetch_sample_into("no_such_sample", cache.path()).unwrap_err();
        assert!(err.contains("no_such_sample"), "{err}");
        assert!(err.contains("kuka_1") && err.contains("ds8"), "{err}");
    }

    #[test]
    fn the_quick_start_samples_are_in_the_catalogue() {
        for id in ["kuka_1", "ds8"] {
            assert!(samples::sample(id).is_ok(), "{id}");
        }
    }

    #[test]
    fn a_manifest_reads_as_a_dataset_spec_and_a_broken_one_says_why() {
        use crate::api::Api;
        use crate::scope::RootScope;
        use serde_json::json;

        let tmp = tempfile::tempdir().unwrap();
        let base = dunce::canonicalize(tmp.path()).unwrap();
        let toml = base.join("dataset.toml");
        std::fs::write(
            &toml,
            r#"
version = 1
topology = "planar_intrinsics"

[target]
kind = "chessboard"
rows = 9
cols = 6
square_size_m = 0.025

[[cameras]]
id = "cam0"

[cameras.images]
kind = "glob"
pattern = "*.png"
"#,
        )
        .unwrap();
        let api = Api::for_tests(RootScope::growable(None));
        let spec = api
            .invoke("read_manifest", json!({"path": toml.to_string_lossy()}))
            .unwrap();
        assert_eq!(spec["topology"], "planar_intrinsics");
        assert_eq!(spec["cameras"][0]["id"], "cam0");
        // Reading it opened its folder.
        assert_eq!(api.ctx().scope.roots(), std::slice::from_ref(&base));

        let broken = base.join("broken.json");
        std::fs::write(&broken, r#"{"version": 1}"#).unwrap();
        let err = api
            .invoke("read_manifest", json!({"path": broken.to_string_lossy()}))
            .unwrap_err();
        assert_eq!(err.kind, ErrorKind::Manifest);
        assert_eq!(err.path.as_deref(), Some(&*broken.to_string_lossy()));
    }
}
