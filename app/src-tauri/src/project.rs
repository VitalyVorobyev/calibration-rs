//! Project folders: a dataset manifest and the runs calibrated from it.
//!
//! ```text
//! <project>/
//!   calibration-project.json   {"version": 1, "dataset": "<dir>"}
//!   dataset.json               the dataset manifest (`DatasetSpec` JSON)
//!   runs/<name>/
//!     manifest.json            the manifest the run used
//!     config.json              the calibration config
//!     export.json              the export; its image paths resolve from here
//!     diagnostics.json         views used and run time
//! ```
//!
//! `dataset` is the folder the manifest's relative paths (image globs, pose
//! files) resolve against: relative to the project when it lies inside it
//! (`"."` when the images sit next to the project file), absolute otherwise,
//! so a project can also describe a dataset it does not own. A folder without
//! `calibration-project.json` is not a project, and opening it is a
//! [`ProjectErrorKind::NotAProject`] error: nothing is guessed from a folder's
//! contents.

use std::path::{Component, Path, PathBuf};

use serde::{Deserialize, Serialize};

/// The file that makes a folder a project.
pub const PROJECT_FILE: &str = "calibration-project.json";
/// The project's dataset manifest.
pub const MANIFEST_FILE: &str = "dataset.json";
/// The folder holding one sub-folder per saved run.
pub const RUNS_DIR: &str = "runs";

const PROJECT_VERSION: u32 = 1;

/// Why a project command failed. `kind` is what the UI branches on;
/// `message` is what it shows.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[cfg_attr(feature = "schema-export", derive(schemars::JsonSchema))]
pub struct ProjectError {
    pub kind: ProjectErrorKind,
    pub message: String,
}

/// The kinds of [`ProjectError`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[cfg_attr(feature = "schema-export", derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case")]
pub enum ProjectErrorKind {
    /// The folder has no `calibration-project.json`.
    NotAProject,
    /// The folder already holds a project; creating one would overwrite it.
    ProjectExists,
    /// The project file has a version this app does not read.
    UnsupportedVersion,
    /// A run name that is empty or not a plain folder name.
    InvalidRunName,
    /// A run of that name is already saved; runs are never overwritten.
    RunExists,
    /// A file or folder could not be read or written.
    Io,
    /// A project file did not parse.
    Parse,
}

impl ProjectError {
    fn new(kind: ProjectErrorKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
        }
    }

    fn io(path: &Path, err: impl std::fmt::Display) -> Self {
        Self::new(ProjectErrorKind::Io, format!("{}: {err}", path.display()))
    }

    fn parse(path: &Path, err: impl std::fmt::Display) -> Self {
        Self::new(
            ProjectErrorKind::Parse,
            format!("{}: {err}", path.display()),
        )
    }
}

impl std::fmt::Display for ProjectError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

/// `calibration-project.json`.
#[derive(Debug, Serialize, Deserialize)]
struct ProjectFile {
    version: u32,
    /// The dataset folder: relative to the project, or absolute.
    dataset: PathBuf,
}

/// An opened project, as the UI shows it.
#[derive(Debug, Clone, Serialize)]
#[cfg_attr(feature = "schema-export", derive(schemars::JsonSchema))]
pub struct ProjectInfo {
    /// The project folder (absolute).
    pub dir: String,
    /// The folder the manifest's relative paths resolve against (absolute).
    pub dataset_dir: String,
    /// The dataset manifest (`dataset.json`), as stored.
    pub manifest: serde_json::Value,
    /// The saved runs, sorted by name.
    pub runs: Vec<RunSummary>,
}

/// One saved run: its folder name and diagnostics.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[cfg_attr(feature = "schema-export", derive(schemars::JsonSchema))]
pub struct RunSummary {
    /// The run's folder name under `runs/`.
    pub name: String,
    /// `diagnostics.json`, or `None` when it is missing or does not parse
    /// (the run is still listed, so its export can be opened).
    pub diagnostics: Option<RunDiagnostics>,
}

/// `diagnostics.json`: how a run went.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema-export", derive(schemars::JsonSchema))]
pub struct RunDiagnostics {
    /// The manifest topology the run calibrated.
    pub topology: String,
    /// Views with enough detected features to use.
    pub usable_views: usize,
    /// Views attempted.
    pub total_views: usize,
    /// Detection and calibration time, in milliseconds.
    pub duration_ms: u64,
    /// Whether detections were served from the detection cache.
    pub cache_used: bool,
}

/// A run to save: everything the runner was given and returned.
#[derive(Debug, Deserialize)]
pub struct RunRecord {
    pub manifest: serde_json::Value,
    pub config: serde_json::Value,
    /// The export as the runner produced it: its image manifest's `root` is
    /// relative to `dataset_dir`.
    pub export: serde_json::Value,
    pub diagnostics: RunDiagnostics,
    /// The folder the run resolved the manifest's paths against — usually,
    /// but not necessarily, the project's dataset folder.
    pub dataset_dir: PathBuf,
}

/// Make `dir` a project for the dataset in `dataset_dir`, described by
/// `manifest`. Fails with [`ProjectErrorKind::ProjectExists`] when `dir`
/// already holds one.
pub fn create(
    dir: &Path,
    dataset_dir: &Path,
    manifest: &serde_json::Value,
) -> Result<ProjectInfo, ProjectError> {
    let dir = canonical_dir(dir)?;
    let dataset_dir = canonical_dir(dataset_dir)?;
    let project_file = dir.join(PROJECT_FILE);
    if project_file.exists() {
        return Err(ProjectError::new(
            ProjectErrorKind::ProjectExists,
            format!("{} already holds a project", dir.display()),
        ));
    }
    let dataset = match dataset_dir.strip_prefix(&dir) {
        Ok(rel) if rel.as_os_str().is_empty() => PathBuf::from("."),
        Ok(rel) => rel.to_path_buf(),
        Err(_) => dataset_dir.clone(),
    };
    write_json(
        &project_file,
        &ProjectFile {
            version: PROJECT_VERSION,
            dataset,
        },
    )?;
    write_json(&dir.join(MANIFEST_FILE), manifest)?;
    let runs = dir.join(RUNS_DIR);
    std::fs::create_dir_all(&runs).map_err(|e| ProjectError::io(&runs, e))?;
    open(&dir)
}

/// Open the project in `dir`.
pub fn open(dir: &Path) -> Result<ProjectInfo, ProjectError> {
    let dir = canonical_dir(dir)?;
    let project_file = dir.join(PROJECT_FILE);
    if !project_file.is_file() {
        return Err(ProjectError::new(
            ProjectErrorKind::NotAProject,
            format!(
                "{} is not a calibration project (it has no {PROJECT_FILE})",
                dir.display()
            ),
        ));
    }
    let file: ProjectFile = read_json(&project_file)?;
    if file.version != PROJECT_VERSION {
        return Err(ProjectError::new(
            ProjectErrorKind::UnsupportedVersion,
            format!(
                "{}: project version {}; this app reads version {PROJECT_VERSION}",
                project_file.display(),
                file.version
            ),
        ));
    }
    let dataset_dir = canonical_dir(&dir.join(&file.dataset))?;
    let manifest = read_json(&dir.join(MANIFEST_FILE))?;
    let runs = list_runs(&dir)?;
    Ok(ProjectInfo {
        dir: path_string(&dir),
        dataset_dir: path_string(&dataset_dir),
        manifest,
        runs,
    })
}

/// Replace the project's dataset manifest.
pub fn save_manifest(dir: &Path, manifest: &serde_json::Value) -> Result<(), ProjectError> {
    let info = open(dir)?;
    write_json(&Path::new(&info.dir).join(MANIFEST_FILE), manifest)
}

/// Save `record` as `runs/<name>/`. The export's image root is rebased onto
/// the run folder, so the saved `export.json` opens on its own.
pub fn save_run(dir: &Path, name: &str, record: RunRecord) -> Result<RunSummary, ProjectError> {
    validate_run_name(name)?;
    let info = open(dir)?;
    let project = PathBuf::from(&info.dir);
    let run_dir = project.join(RUNS_DIR).join(name);
    if run_dir.exists() {
        return Err(ProjectError::new(
            ProjectErrorKind::RunExists,
            format!("run {name:?} is already saved in {}", project.display()),
        ));
    }
    let dataset_dir = canonical_dir(&record.dataset_dir)?;
    let mut export = record.export;
    rebase_image_root(&mut export, &dataset_dir, &project);
    std::fs::create_dir_all(&run_dir).map_err(|e| ProjectError::io(&run_dir, e))?;
    write_json(&run_dir.join("manifest.json"), &record.manifest)?;
    write_json(&run_dir.join("config.json"), &record.config)?;
    write_json(&run_dir.join("export.json"), &export)?;
    write_json(&run_dir.join("diagnostics.json"), &record.diagnostics)?;
    Ok(RunSummary {
        name: name.to_owned(),
        diagnostics: Some(record.diagnostics),
    })
}

/// The saved runs of the project in `dir`, sorted by name. A folder under
/// `runs/` without an `export.json` is not a run.
fn list_runs(dir: &Path) -> Result<Vec<RunSummary>, ProjectError> {
    let runs_dir = dir.join(RUNS_DIR);
    if !runs_dir.is_dir() {
        return Ok(Vec::new());
    }
    let entries = std::fs::read_dir(&runs_dir).map_err(|e| ProjectError::io(&runs_dir, e))?;
    let mut runs: Vec<RunSummary> = entries
        .filter_map(Result::ok)
        .filter(|entry| entry.path().join("export.json").is_file())
        .map(|entry| RunSummary {
            name: entry.file_name().to_string_lossy().into_owned(),
            diagnostics: read_json(&entry.path().join("diagnostics.json")).ok(),
        })
        .collect();
    runs.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(runs)
}

/// A run name is one plain folder name: ASCII letters, digits, `-`, `_` and
/// `.`, not starting with `.`.
fn validate_run_name(name: &str) -> Result<(), ProjectError> {
    let ok = !name.is_empty()
        && !name.starts_with('.')
        && name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'));
    if ok {
        Ok(())
    } else {
        Err(ProjectError::new(
            ProjectErrorKind::InvalidRunName,
            format!("{name:?} is not a run name (use letters, digits, '-', '_' and '.')"),
        ))
    }
}

/// Point the export's `image_manifest.root` (relative to `dataset_dir`) at
/// the same folder as seen from `<project>/runs/<name>/`: a relative path
/// when the images lie inside the project, the absolute folder otherwise.
fn rebase_image_root(export: &mut serde_json::Value, dataset_dir: &Path, project: &Path) {
    let Some(manifest) = export
        .get_mut("image_manifest")
        .and_then(|m| m.as_object_mut())
    else {
        return;
    };
    let old_root = manifest
        .get("root")
        .and_then(|r| r.as_str())
        .unwrap_or(".")
        .to_owned();
    let images = normalize(&dataset_dir.join(old_root));
    let new_root = match images.strip_prefix(project) {
        Ok(rel) if rel.as_os_str().is_empty() => PathBuf::from("../.."),
        Ok(rel) => Path::new("../..").join(rel),
        Err(_) => images,
    };
    manifest.insert(
        "root".into(),
        serde_json::Value::String(path_string(&new_root)),
    );
}

/// `path` with its `.` and `..` components folded (lexically).
fn normalize(path: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for component in path.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                out.pop();
            }
            other => out.push(other),
        }
    }
    out
}

/// `dir` made absolute and canonical; an error unless it is a folder.
fn canonical_dir(dir: &Path) -> Result<PathBuf, ProjectError> {
    let canonical = dunce::canonicalize(dir).map_err(|e| ProjectError::io(dir, e))?;
    if canonical.is_dir() {
        Ok(canonical)
    } else {
        Err(ProjectError::io(dir, "not a folder"))
    }
}

/// A path for the webview: `/`-separated on every platform.
fn path_string(path: &Path) -> String {
    path.to_string_lossy().replace('\\', "/")
}

fn read_json<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T, ProjectError> {
    let text = std::fs::read_to_string(path).map_err(|e| ProjectError::io(path, e))?;
    serde_json::from_str(&text).map_err(|e| ProjectError::parse(path, e))
}

fn write_json(path: &Path, value: &impl Serialize) -> Result<(), ProjectError> {
    let mut text = serde_json::to_string_pretty(value).map_err(|e| ProjectError::parse(path, e))?;
    text.push('\n');
    std::fs::write(path, text).map_err(|e| ProjectError::io(path, e))
}

/// Tauri command: make `dir` a project for the dataset in `dataset_dir`.
#[tauri::command]
pub fn create_project<R: tauri::Runtime>(
    app: tauri::AppHandle<R>,
    dir: PathBuf,
    dataset_dir: PathBuf,
    manifest: serde_json::Value,
) -> Result<ProjectInfo, ProjectError> {
    let info = create(&dir, &dataset_dir, &manifest)?;
    allow_project_assets(&app, &info)?;
    Ok(info)
}

/// Tauri command: open the project in `dir`.
#[tauri::command]
pub fn open_project<R: tauri::Runtime>(
    app: tauri::AppHandle<R>,
    dir: PathBuf,
) -> Result<ProjectInfo, ProjectError> {
    let info = open(&dir)?;
    allow_project_assets(&app, &info)?;
    Ok(info)
}

/// Tauri command: replace the project's dataset manifest.
#[tauri::command]
pub fn save_project_manifest(
    dir: PathBuf,
    manifest: serde_json::Value,
) -> Result<(), ProjectError> {
    save_manifest(&dir, &manifest)
}

/// Tauri command: save a run as `runs/<name>/`.
#[tauri::command]
pub fn save_project_run(
    dir: PathBuf,
    name: String,
    record: RunRecord,
) -> Result<RunSummary, ProjectError> {
    save_run(&dir, &name, record)
}

/// The webview may load the project's dataset images.
fn allow_project_assets<R: tauri::Runtime>(
    app: &tauri::AppHandle<R>,
    info: &ProjectInfo,
) -> Result<(), ProjectError> {
    crate::assets::allow_dir(app, Path::new(&info.dataset_dir))
        .map_err(|e| ProjectError::new(ProjectErrorKind::Io, e))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn manifest() -> serde_json::Value {
        json!({ "version": 1, "topology": "planar_intrinsics", "cameras": [] })
    }

    fn diagnostics() -> RunDiagnostics {
        RunDiagnostics {
            topology: "planar_intrinsics".into(),
            usable_views: 9,
            total_views: 10,
            duration_ms: 1234,
            cache_used: false,
        }
    }

    /// A run of the dataset in `dataset_dir`.
    fn record(dataset_dir: &Path) -> RunRecord {
        RunRecord {
            manifest: manifest(),
            config: json!({ "max_iters": 50 }),
            export: json!({
                "kind": "planar_intrinsics",
                "image_manifest": { "root": ".", "frames": [{ "path": "img/01.png" }] }
            }),
            diagnostics: diagnostics(),
            dataset_dir: dataset_dir.to_path_buf(),
        }
    }

    #[test]
    fn create_then_open_round_trips_the_manifest() {
        let tmp = tempfile::tempdir().unwrap();
        let created = create(tmp.path(), tmp.path(), &manifest()).unwrap();
        assert_eq!(created.manifest, manifest());
        assert!(created.runs.is_empty());

        let opened = open(tmp.path()).unwrap();
        assert_eq!(opened.dir, created.dir);
        assert_eq!(opened.dataset_dir, opened.dir);
        assert_eq!(opened.manifest, manifest());

        let file: serde_json::Value = read_json(&tmp.path().join(PROJECT_FILE)).unwrap();
        assert_eq!(file, json!({ "version": 1, "dataset": "." }));
    }

    #[test]
    fn a_dataset_outside_the_project_is_stored_absolute() {
        let project = tempfile::tempdir().unwrap();
        let data = tempfile::tempdir().unwrap();
        let info = create(project.path(), data.path(), &manifest()).unwrap();
        let data_abs = path_string(&dunce::canonicalize(data.path()).unwrap());
        assert_eq!(info.dataset_dir, data_abs);
    }

    #[test]
    fn a_folder_without_a_project_file_is_not_a_project() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(tmp.path().join("dataset.toml"), "version = 1").unwrap();
        let err = open(tmp.path()).unwrap_err();
        assert_eq!(err.kind, ProjectErrorKind::NotAProject);
    }

    #[test]
    fn a_missing_folder_is_an_io_error() {
        let tmp = tempfile::tempdir().unwrap();
        let err = open(&tmp.path().join("nope")).unwrap_err();
        assert_eq!(err.kind, ProjectErrorKind::Io);
    }

    #[test]
    fn creating_over_a_project_is_refused() {
        let tmp = tempfile::tempdir().unwrap();
        create(tmp.path(), tmp.path(), &manifest()).unwrap();
        let err = create(tmp.path(), tmp.path(), &json!({})).unwrap_err();
        assert_eq!(err.kind, ProjectErrorKind::ProjectExists);
        // The first project's manifest is untouched.
        assert_eq!(open(tmp.path()).unwrap().manifest, manifest());
    }

    #[test]
    fn an_unknown_version_is_refused() {
        let tmp = tempfile::tempdir().unwrap();
        create(tmp.path(), tmp.path(), &manifest()).unwrap();
        std::fs::write(
            tmp.path().join(PROJECT_FILE),
            r#"{ "version": 2, "dataset": "." }"#,
        )
        .unwrap();
        let err = open(tmp.path()).unwrap_err();
        assert_eq!(err.kind, ProjectErrorKind::UnsupportedVersion);
    }

    #[test]
    fn a_broken_manifest_is_a_parse_error() {
        let tmp = tempfile::tempdir().unwrap();
        create(tmp.path(), tmp.path(), &manifest()).unwrap();
        std::fs::write(tmp.path().join(MANIFEST_FILE), "{ not json").unwrap();
        let err = open(tmp.path()).unwrap_err();
        assert_eq!(err.kind, ProjectErrorKind::Parse);
    }

    #[test]
    fn save_manifest_replaces_the_manifest() {
        let tmp = tempfile::tempdir().unwrap();
        create(tmp.path(), tmp.path(), &manifest()).unwrap();
        let edited = json!({ "version": 1, "topology": "rig_extrinsics" });
        save_manifest(tmp.path(), &edited).unwrap();
        assert_eq!(open(tmp.path()).unwrap().manifest, edited);
    }

    #[test]
    fn saved_runs_are_listed_and_rebased() {
        let tmp = tempfile::tempdir().unwrap();
        create(tmp.path(), tmp.path(), &manifest()).unwrap();
        save_run(tmp.path(), "run-002", record(tmp.path())).unwrap();
        let summary = save_run(tmp.path(), "run-001", record(tmp.path())).unwrap();
        assert_eq!(summary.name, "run-001");

        let info = open(tmp.path()).unwrap();
        let names: Vec<&str> = info.runs.iter().map(|r| r.name.as_str()).collect();
        assert_eq!(names, ["run-001", "run-002"]);
        assert_eq!(info.runs[0].diagnostics, Some(diagnostics()));

        let run_dir = tmp.path().join(RUNS_DIR).join("run-001");
        let export: serde_json::Value = read_json(&run_dir.join("export.json")).unwrap();
        assert_eq!(export["image_manifest"]["root"], "../..");
        // The rebased root resolves to the dataset folder.
        let images = dunce::canonicalize(run_dir.join("../..")).unwrap();
        assert_eq!(path_string(&images), info.dataset_dir);
        let config: serde_json::Value = read_json(&run_dir.join("config.json")).unwrap();
        assert_eq!(config, json!({ "max_iters": 50 }));
    }

    #[test]
    fn a_run_of_a_dataset_outside_the_project_keeps_an_absolute_root() {
        let project = tempfile::tempdir().unwrap();
        let data = tempfile::tempdir().unwrap();
        let info = create(project.path(), data.path(), &manifest()).unwrap();
        save_run(project.path(), "a", record(data.path())).unwrap();
        let export: serde_json::Value =
            read_json(&project.path().join(RUNS_DIR).join("a").join("export.json")).unwrap();
        assert_eq!(export["image_manifest"]["root"], info.dataset_dir.as_str());
    }

    #[test]
    fn a_run_is_rebased_from_the_folder_it_ran_on() {
        // The run used a sub-folder of the project, not the project's dataset.
        let tmp = tempfile::tempdir().unwrap();
        create(tmp.path(), tmp.path(), &manifest()).unwrap();
        std::fs::create_dir_all(tmp.path().join("cam_left")).unwrap();
        save_run(tmp.path(), "a", record(&tmp.path().join("cam_left"))).unwrap();
        let export: serde_json::Value =
            read_json(&tmp.path().join(RUNS_DIR).join("a").join("export.json")).unwrap();
        assert_eq!(export["image_manifest"]["root"], "../../cam_left");
    }

    #[test]
    fn runs_are_never_overwritten() {
        let tmp = tempfile::tempdir().unwrap();
        create(tmp.path(), tmp.path(), &manifest()).unwrap();
        save_run(tmp.path(), "run-001", record(tmp.path())).unwrap();
        let err = save_run(tmp.path(), "run-001", record(tmp.path())).unwrap_err();
        assert_eq!(err.kind, ProjectErrorKind::RunExists);
    }

    #[test]
    fn run_names_are_plain_folder_names() {
        let tmp = tempfile::tempdir().unwrap();
        create(tmp.path(), tmp.path(), &manifest()).unwrap();
        for bad in ["", "..", ".hidden", "a/b", "a\\b", "x y"] {
            let err = save_run(tmp.path(), bad, record(tmp.path())).unwrap_err();
            assert_eq!(err.kind, ProjectErrorKind::InvalidRunName, "{bad:?}");
        }
    }

    #[test]
    fn a_runs_folder_without_an_export_is_not_a_run() {
        let tmp = tempfile::tempdir().unwrap();
        create(tmp.path(), tmp.path(), &manifest()).unwrap();
        std::fs::create_dir_all(tmp.path().join(RUNS_DIR).join("scratch")).unwrap();
        assert!(open(tmp.path()).unwrap().runs.is_empty());
    }

    #[test]
    fn errors_serialize_with_a_snake_case_kind() {
        let err = ProjectError::new(ProjectErrorKind::NotAProject, "nope");
        assert_eq!(
            serde_json::to_value(&err).unwrap(),
            json!({ "kind": "not_a_project", "message": "nope" })
        );
    }
}
