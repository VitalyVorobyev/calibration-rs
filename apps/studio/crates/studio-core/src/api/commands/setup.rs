//! Where a project's dataset comes from before there is a project: a
//! manifest file read, a folder sniffed into a draft manifest, a Quick
//! Start sample fetched as a project of its own.

use std::path::Path;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use vision_calibration::dataset::{DatasetSpec, samples};

use crate::api::{Ctx, ErrorKind, JobCx, StudioError};
use crate::scope::UserPath;

/// Read the dataset manifest at `path`: TOML for a `.toml` extension, JSON
/// otherwise.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ReadManifestArgs {
    pub path: UserPath,
}

pub fn read_manifest(ctx: &Ctx, args: ReadManifestArgs) -> Result<DatasetSpec, StudioError> {
    let file = ctx.scope.existing_file(&args.path)?;
    DatasetSpec::from_path(&file).map_err(|e| StudioError::from(e).at(args.path.as_str()))
}

/// Infer a draft dataset manifest from the folder `folder` (the same
/// inference as `vcal project init`): images grouped into cameras, a
/// robot-pose file, a guessed topology. What cannot be determined is left
/// at a placeholder and listed in `unresolved`; a run refuses the manifest
/// until each is set and cleared from its `_unresolved`.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SniffFolderArgs {
    pub folder: UserPath,
}

/// A sniffed folder.
#[derive(Debug, Serialize, JsonSchema)]
pub struct SniffedFolder {
    /// The draft manifest, its relative paths against the folder.
    pub spec: DatasetSpec,
    /// The fields the sniffer could not determine, as dotted paths into the
    /// manifest (`target`, `topology`, `pose_pairing`, …): the draft's
    /// `_unresolved`.
    pub unresolved: Vec<String>,
}

pub fn sniff_folder(ctx: &Ctx, args: SniffFolderArgs) -> Result<SniffedFolder, StudioError> {
    let folder = ctx.scope.existing_dir(&args.folder)?;
    let spec = vision_calibration::dataset::sniff_folder(&folder)
        .map_err(|e| StudioError::new(ErrorKind::Io, e.to_string()).at(args.folder.as_str()))?;
    Ok(SniffedFolder {
        unresolved: spec.unresolved.clone(),
        spec,
    })
}

/// Make the sample dataset `id` (one of `server_info.samples`) available
/// as a project and answer its folder: the files are downloaded into the
/// back end's samples folder (always one of its roots), checked against
/// their SHA-256 and reused later, and the folder becomes a project
/// referencing the sample's manifest (once; an existing project is kept
/// with its runs). A back end built without samples answers `invalid`.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct FetchSampleArgs {
    pub id: String,
}

/// A Quick Start sample, ready to open.
#[derive(Debug, Serialize, JsonSchema)]
pub struct SampleProject {
    /// The sample's project folder.
    pub dir: String,
}

pub fn fetch_sample(
    ctx: &Ctx,
    args: FetchSampleArgs,
    _: &JobCx<()>,
) -> Result<SampleProject, StudioError> {
    let manifest = fetch_sample_into(&args.id, &ctx.dirs.samples)
        .map_err(|e| StudioError::new(ErrorKind::Io, e))?;
    let manifest = ctx.scope.check(&manifest)?;
    let dir = manifest
        .parent()
        .ok_or_else(|| StudioError::internal("a sample's manifest is in its folder"))?;
    sample_project(&args.id, dir, &manifest)?;
    Ok(SampleProject {
        dir: crate::runs::slash(dir),
    })
}

/// The sample `id` in `dir` as a project referencing its `manifest`.
fn sample_project(id: &str, dir: &Path, manifest: &Path) -> Result<(), StudioError> {
    use vision_calibration::project::{Project, ProjectError, ProjectSpec};
    match Project::open(dir) {
        Ok(_) => return Ok(()),
        Err(ProjectError::NotAProject { .. }) => {}
        Err(e) => return Err(e.into()),
    }
    let name = manifest.file_name().map(Path::new).unwrap_or(manifest);
    let mut spec = ProjectSpec::for_manifest(name);
    spec.name = Some(id.to_owned());
    spec.description = samples::sample(id).ok().map(|s| s.description.clone());
    match Project::create(dir, spec) {
        Ok(_) | Err(ProjectError::ProjectExists { .. }) => Ok(()),
        Err(e) => Err(e.into()),
    }
}

/// The sample `id` fetched into `cache`: its manifest's path.
#[cfg(feature = "samples")]
fn fetch_sample_into(id: &str, cache: &Path) -> Result<std::path::PathBuf, String> {
    samples::fetch(id, cache).map_err(|e| format!("sample {id}: {e}"))
}

#[cfg(not(feature = "samples"))]
fn fetch_sample_into(id: &str, _: &Path) -> Result<std::path::PathBuf, String> {
    Err(format!(
        "sample {id}: this back end was built without the Quick Start samples"
    ))
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
    fn a_fetched_sample_becomes_a_project_once() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("kuka_1");
        std::fs::create_dir_all(&dir).unwrap();
        let manifest = dir.join("dataset.toml");
        std::fs::write(
            &manifest,
            include_str!("../../../../../../../data/kuka_1/dataset.toml"),
        )
        .unwrap();
        sample_project("kuka_1", &dir, &manifest).unwrap();
        let project = vision_calibration::project::Project::open(&dir).unwrap();
        assert_eq!(project.spec().name.as_deref(), Some("kuka_1"));
        assert!(project.dataset().is_ok());
        let revision = project.revision().to_owned();
        // Again: the project is kept as it is.
        sample_project("kuka_1", &dir, &manifest).unwrap();
        let again = vision_calibration::project::Project::open(&dir).unwrap();
        assert_eq!(again.revision(), revision);
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
        let api = Api::for_tests(RootScope::fixed([base.clone()]));
        let spec = api
            .invoke("read_manifest", json!({"path": toml.to_string_lossy()}))
            .unwrap();
        assert_eq!(spec["topology"], "planar_intrinsics");
        assert_eq!(spec["cameras"][0]["id"], "cam0");

        let broken = base.join("broken.json");
        std::fs::write(&broken, r#"{"version": 1}"#).unwrap();
        let err = api
            .invoke("read_manifest", json!({"path": broken.to_string_lossy()}))
            .unwrap_err();
        assert_eq!(err.kind, ErrorKind::Manifest);
        assert_eq!(err.path.as_deref(), Some(&*broken.to_string_lossy()));

        // A sniffed folder: a draft and what it leaves open.
        std::fs::write(base.join("a.png"), b"").unwrap();
        let sniffed = api
            .invoke("sniff_folder", json!({"folder": base.to_string_lossy()}))
            .unwrap();
        assert_eq!(sniffed["unresolved"], sniffed["spec"]["_unresolved"]);
        assert!(
            sniffed["unresolved"]
                .as_array()
                .unwrap()
                .contains(&json!("target"))
        );
    }
}
