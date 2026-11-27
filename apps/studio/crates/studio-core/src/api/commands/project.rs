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
/// holds one, and with `forbidden_path` when the manifest names a path
/// outside the roots.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct CreateProjectArgs {
    pub dir: UserPath,
    pub dataset_dir: UserPath,
    /// The dataset manifest (`DatasetSpec` JSON), stored as `dataset.json`.
    pub manifest: serde_json::Value,
}

pub fn create_project(ctx: &Ctx, args: CreateProjectArgs) -> Result<ProjectInfo, StudioError> {
    let dir = ctx.scope.existing_dir(&args.dir)?;
    let dataset_dir = ctx.scope.existing_dir(&args.dataset_dir)?;
    crate::scope::check_manifest(&ctx.scope, &args.manifest, &dataset_dir)?;
    let info = project::create(&dir, &dataset_dir, &args.manifest)?;
    ctx.settings.touch_recent(&dir);
    Ok(info)
}

/// Open the project in `dir`; `not_a_project` for any other folder,
/// `forbidden_path` when its dataset lies outside the roots. It becomes
/// the most recent project.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct OpenProjectArgs {
    pub dir: UserPath,
}

pub fn open_project(ctx: &Ctx, args: OpenProjectArgs) -> Result<ProjectInfo, StudioError> {
    let dir = ctx.scope.existing_dir(&args.dir)?;
    let info = project::open(&dir)?;
    // The project file may name any folder; only one inside a root opens.
    ctx.scope.check(Path::new(&info.dataset_dir))?;
    ctx.settings.touch_recent(&dir);
    Ok(info)
}

/// Replace the project's dataset manifest (`dataset.json`); every path it
/// names must lie inside a root.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SaveProjectManifestArgs {
    pub dir: UserPath,
    pub manifest: serde_json::Value,
}

pub fn save_project_manifest(ctx: &Ctx, args: SaveProjectManifestArgs) -> Result<(), StudioError> {
    let dir = ctx.scope.existing_dir(&args.dir)?;
    let info = project::open(&dir)?;
    let dataset_dir = ctx.scope.check(Path::new(&info.dataset_dir))?;
    crate::scope::check_manifest(&ctx.scope, &args.manifest, &dataset_dir)?;
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
    fn projects_open_and_save_only_inside_the_roots() {
        let tmp = tempfile::tempdir().unwrap();
        let base = dunce::canonicalize(tmp.path()).unwrap();
        let root = base.join("root");
        let project = root.join("p");
        std::fs::create_dir_all(&project).unwrap();
        let api = Api::for_tests(RootScope::fixed([root.clone()]));
        let dir = project.to_string_lossy().into_owned();
        let info = api
            .invoke(
                "create_project",
                json!({"dir": dir, "dataset_dir": dir, "manifest": manifest()}),
            )
            .unwrap();
        assert_eq!(info["runs"], json!([]));
        api.invoke("open_project", json!({"dir": dir})).unwrap();
        assert_eq!(api.ctx().settings.recent()[0], dir);
        api.invoke(
            "save_project_manifest",
            json!({"dir": dir, "manifest": manifest()}),
        )
        .unwrap();

        // A manifest that names a path outside the roots is refused.
        let escape = json!({ "version": 1, "topology": "planar_intrinsics", "cameras": [
            { "id": "c", "images": { "kind": "glob", "pattern": "../../*.png" } }
        ]});
        let err = api
            .invoke(
                "save_project_manifest",
                json!({"dir": dir, "manifest": escape}),
            )
            .unwrap_err();
        assert_eq!(err.kind, ErrorKind::ForbiddenPath);

        // A folder outside the roots is out of reach.
        let other = base.join("other");
        std::fs::create_dir(&other).unwrap();
        let err = api
            .invoke(
                "save_project_manifest",
                json!({"dir": other.to_string_lossy(), "manifest": manifest()}),
            )
            .unwrap_err();
        assert_eq!(err.kind, ErrorKind::ForbiddenPath);
        // A project whose file names a dataset outside the roots does not
        // open.
        std::fs::write(
            project.join("calibration-project.json"),
            json!({"version": 1, "dataset": other.to_string_lossy()}).to_string(),
        )
        .unwrap();
        let err = api.invoke("open_project", json!({"dir": dir})).unwrap_err();
        assert_eq!(err.kind, ErrorKind::ForbiddenPath);
        // A folder that is no project is reported as what it is.
        let plain = root.join("plain");
        std::fs::create_dir(&plain).unwrap();
        let err = api
            .invoke("open_project", json!({"dir": plain.to_string_lossy()}))
            .unwrap_err();
        assert_eq!(err.kind, ErrorKind::NotAProject);
        // Unknown argument names are refused, not ignored.
        let err = api
            .invoke("open_project", json!({"dir": dir, "dataSetDir": dir}))
            .unwrap_err();
        assert_eq!(err.kind, ErrorKind::InvalidArgs);
    }
}
