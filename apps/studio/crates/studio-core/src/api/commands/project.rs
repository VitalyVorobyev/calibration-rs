//! Project folders: create, open, save the manifest, save a run
//! ([`crate::project`]).

use std::path::Path;

use schemars::JsonSchema;
use serde::Deserialize;

use crate::api::{Ctx, StudioError};
use crate::project::{self, ProjectInfo, RunDiagnostics, RunSummary};
use crate::scope::UserPath;

/// Make the folder `dir` a project for the dataset in `dataset_dir`,
/// described by `manifest`. Fails with `project_exists` when `dir` already
/// holds one.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct CreateProjectArgs {
    pub dir: UserPath,
    pub dataset_dir: UserPath,
    /// The dataset manifest (`DatasetSpec` JSON), stored as `dataset.json`.
    pub manifest: serde_json::Value,
}

pub fn create_project(ctx: &Ctx, args: CreateProjectArgs) -> Result<ProjectInfo, StudioError> {
    let dir = ctx.scope.open_dir(&args.dir)?;
    let dataset_dir = ctx.scope.open_dir(&args.dataset_dir)?;
    Ok(project::create(&dir, &dataset_dir, &args.manifest)?)
}

/// Open the project in `dir`; `not_a_project` for any other folder.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct OpenProjectArgs {
    pub dir: UserPath,
}

pub fn open_project(ctx: &Ctx, args: OpenProjectArgs) -> Result<ProjectInfo, StudioError> {
    let dir = ctx.scope.open_dir(&args.dir)?;
    let info = project::open(&dir)?;
    // The dataset may lie outside the project; opening the project opens it.
    ctx.scope.open_path(Path::new(&info.dataset_dir))?;
    Ok(info)
}

/// Replace the project's dataset manifest (`dataset.json`).
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SaveProjectManifestArgs {
    pub dir: UserPath,
    pub manifest: serde_json::Value,
}

pub fn save_project_manifest(ctx: &Ctx, args: SaveProjectManifestArgs) -> Result<(), StudioError> {
    let dir = ctx.scope.existing_dir(&args.dir)?;
    Ok(project::save_manifest(&dir, &args.manifest)?)
}

/// Everything a saved run records (`runs/<name>/*.json`).
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct RunRecord {
    pub manifest: serde_json::Value,
    pub config: serde_json::Value,
    /// The export as the runner returned it: its image manifest's `root` is
    /// relative to `dataset_dir`.
    pub export: serde_json::Value,
    pub diagnostics: RunDiagnostics,
    /// The folder the run resolved the manifest's paths against.
    pub dataset_dir: UserPath,
}

/// Save a run as `runs/<name>/`; `run_exists` rather than overwrite one.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SaveProjectRunArgs {
    pub dir: UserPath,
    pub name: String,
    pub record: RunRecord,
}

pub fn save_project_run(ctx: &Ctx, args: SaveProjectRunArgs) -> Result<RunSummary, StudioError> {
    let dir = ctx.scope.existing_dir(&args.dir)?;
    let record = args.record;
    let dataset_dir = ctx.scope.existing_dir(&record.dataset_dir)?;
    let record = project::RunRecord {
        manifest: record.manifest,
        config: record.config,
        export: record.export,
        diagnostics: record.diagnostics,
        dataset_dir,
    };
    Ok(project::save_run(&dir, &args.name, record)?)
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use crate::api::{Api, ErrorKind};
    use crate::scope::RootScope;

    fn manifest() -> serde_json::Value {
        json!({ "version": 1, "topology": "planar_intrinsics", "cameras": [] })
    }

    #[test]
    fn opening_a_project_opens_its_folder_and_saving_stays_inside_it() {
        let tmp = tempfile::tempdir().unwrap();
        let base = dunce::canonicalize(tmp.path()).unwrap();
        let project = base.join("p");
        std::fs::create_dir(&project).unwrap();
        let api = Api::new(RootScope::growable(None));
        let dir = project.to_string_lossy().into_owned();
        let info = api
            .invoke(
                "create_project",
                json!({"dir": dir, "dataset_dir": dir, "manifest": manifest()}),
            )
            .unwrap();
        assert_eq!(info["runs"], json!([]));
        assert_eq!(api.ctx().scope.roots(), std::slice::from_ref(&project));
        api.invoke(
            "save_project_manifest",
            json!({"dir": dir, "manifest": manifest()}),
        )
        .unwrap();

        // A second folder the user never opened is out of reach.
        let other = base.join("other");
        std::fs::create_dir(&other).unwrap();
        let err = api
            .invoke(
                "save_project_manifest",
                json!({"dir": other.to_string_lossy(), "manifest": manifest()}),
            )
            .unwrap_err();
        assert_eq!(err.kind, ErrorKind::ForbiddenPath);
        // Opening a folder that is no project grants nothing useful but is
        // reported as what it is.
        let err = api
            .invoke("open_project", json!({"dir": other.to_string_lossy()}))
            .unwrap_err();
        assert_eq!(err.kind, ErrorKind::NotAProject);
        // Unknown argument names are refused, not ignored.
        let err = api
            .invoke("open_project", json!({"dir": dir, "dataSetDir": dir}))
            .unwrap_err();
        assert_eq!(err.kind, ErrorKind::InvalidArgs);
    }
}
