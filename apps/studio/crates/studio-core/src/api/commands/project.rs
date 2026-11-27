//! Projects through the library's `vision_calibration::project`: the
//! library scan, open, create and save (refused on a stale revision), the
//! views a dataset pairs into, and the runs' listing, notes, deletion and
//! imports. A project is named by its folder in every command; the back
//! end keeps nothing open between calls.

use std::path::{Path, PathBuf};

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use vision_calibration::dataset::DatasetSpec;
use vision_calibration::dataset_runner::CalibrationConfig;
use vision_calibration::project::{
    DatasetSource, LibraryScan, LibraryScanOptions, Project, ProjectDigest, ProjectSpec, RunInfo,
    RunNotes, scan_library as scan,
};

use super::forms::config_fields;
use crate::api::{Ctx, ErrorKind, StudioError};
use crate::runs::{open_project as open_checked, slash};
use crate::scope::UserPath;

/// Every project under the root folders, with its runs; a folder or
/// project that cannot be read is listed as a failure, never an error.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ScanLibraryArgs {}

pub fn scan_library(ctx: &Ctx, _: ScanLibraryArgs) -> Result<LibraryScan, StudioError> {
    Ok(scan(&ctx.scope.roots(), &LibraryScanOptions::default()))
}

/// An open project: its definition as on disk, what it resolves to, and its
/// runs.
#[derive(Debug, Serialize, JsonSchema)]
pub struct ProjectView {
    /// The project folder.
    pub dir: String,
    /// Its name, or its folder's.
    pub name: String,
    /// The project file's format version on disk: 1 for a project of the
    /// app's earlier format until it is saved (then 2).
    pub file_version: u32,
    /// The project file's content revision as read: `save_project` sends it
    /// back, and is refused with `changed_on_disk` when the file changed
    /// since.
    pub revision: String,
    /// The project file's definition.
    pub spec: ProjectSpec,
    /// The folder the manifest's relative paths resolve against.
    pub data_root: String,
    /// The dataset's manifest, or why it cannot be read: a broken
    /// referenced manifest still opens the project.
    pub dataset: DatasetState,
    /// Whether the manifest is the project file's (and edited with it); a
    /// referenced manifest file is never written.
    pub dataset_editable: bool,
    /// The workflow's configuration: its defaults with the project's
    /// changes, or why they do not make a valid one.
    pub config: ConfigState,
    /// The digests of the project's inputs as they stand (no image read),
    /// to compare with each run's: absent when the dataset or the
    /// configuration does not resolve.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_digest: Option<ProjectDigest>,
    /// The runs: numbered runs by number, then others by name.
    pub runs: Vec<RunInfo>,
}

/// A project's dataset manifest as it resolves.
#[derive(Debug, Serialize, JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum DatasetState {
    Ok { spec: Box<DatasetSpec> },
    Unreadable { error: StudioError },
}

/// A project's configuration as it resolves.
#[derive(Debug, Serialize, JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ConfigState {
    /// The workflow's config object (as `config_form` serves it).
    Ok {
        config: serde_json::Value,
    },
    Invalid {
        error: StudioError,
    },
}

impl ProjectView {
    /// The view of `project`, whose paths the scope has judged
    /// (`check_project_paths`); the manifest's are judged here, before
    /// anything they name is read.
    pub(crate) fn of(ctx: &Ctx, project: &Project) -> Result<Self, StudioError> {
        let dataset = project
            .dataset()
            .map_err(StudioError::from)
            .and_then(|dataset| {
                ctx.scope
                    .check_dataset(&dataset, &project.data_root()?)
                    .map(|()| dataset)
            });
        let (dataset, config, input_digest) = match dataset {
            Ok(dataset) => {
                let config = CalibrationConfig::patched(dataset.topology, &project.spec().config)
                    .map_err(|e| {
                        StudioError::new(ErrorKind::Config, e.to_string()).on_field("config")
                    })
                    .and_then(|config| config_fields(&config));
                let config = match config {
                    Ok(config) => ConfigState::Ok { config },
                    Err(error) => ConfigState::Invalid { error },
                };
                let digest = match config {
                    ConfigState::Ok { .. } => project.input_digest().ok(),
                    ConfigState::Invalid { .. } => None,
                };
                (
                    DatasetState::Ok {
                        spec: Box::new(dataset),
                    },
                    config,
                    digest,
                )
            }
            Err(error) => (
                DatasetState::Unreadable {
                    error: error.clone(),
                },
                ConfigState::Invalid { error },
                None,
            ),
        };
        Ok(Self {
            dir: slash(project.dir()),
            name: display_name(project),
            file_version: project.file_version(),
            revision: project.revision().to_owned(),
            spec: project.spec().clone(),
            data_root: slash(&project.data_root()?),
            dataset,
            dataset_editable: matches!(project.spec().dataset, DatasetSource::Inline { .. }),
            config,
            input_digest,
            runs: project.runs()?,
        })
    }
}

fn display_name(project: &Project) -> String {
    project.spec().name.clone().unwrap_or_else(|| {
        project
            .dir()
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default()
    })
}

/// Open the project in `dir`: its definition, the resolved dataset and
/// configuration, the digests of its inputs and its runs. A version 1
/// project of the app's earlier format opens as it is (it is migrated when
/// saved). `not_a_project` for a folder without a project file,
/// `forbidden_path` when the project names a data folder or manifest
/// outside the roots. It becomes the most recent project. Opening again is
/// how a client reloads a project: the back end keeps none open.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct OpenProjectArgs {
    pub dir: UserPath,
}

pub fn open_project(ctx: &Ctx, args: OpenProjectArgs) -> Result<ProjectView, StudioError> {
    let project = open_checked(ctx, &args.dir)?;
    let view = ProjectView::of(ctx, &project)?;
    ctx.settings.touch_recent(project.dir());
    Ok(view)
}

/// Make the folder `dir` (created when missing, inside a root) a project of
/// `spec`, written as `calibration-project.toml`. Every path the spec names
/// must lie inside a root; `project_exists` when the folder already holds a
/// project. It becomes the most recent project.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct CreateProjectArgs {
    pub dir: UserPath,
    pub spec: ProjectSpec,
}

pub fn create_project(ctx: &Ctx, args: CreateProjectArgs) -> Result<ProjectView, StudioError> {
    let dir = ctx.scope.check_lexical(Path::new(args.dir.as_str()))?;
    check_spec_paths(ctx, &dir, &args.spec)?;
    let project = Project::create(&dir, args.spec)?;
    let view = ProjectView::of(ctx, &project)?;
    ctx.settings.touch_recent(project.dir());
    ctx.events.project_changed(project.dir());
    Ok(view)
}

/// The paths a spec for a project in `dir` names, judged before the
/// project file is written: the data folder (the inline manifest's `root`
/// against the project folder, or the referenced manifest's folder), the
/// referenced manifest, and every file the manifest names.
fn check_spec_paths(ctx: &Ctx, dir: &Path, spec: &ProjectSpec) -> Result<(), StudioError> {
    let outside = |path: &Path| {
        StudioError::new(
            ErrorKind::ForbiddenPath,
            format!(
                "the project names {}, which is outside the folders this back end serves",
                path.display()
            ),
        )
        .at(path.to_string_lossy())
    };
    let (data_root, dataset) = match &spec.dataset {
        DatasetSource::Inline { root, spec, .. } => (dir.join(root), Some(spec.as_ref().clone())),
        DatasetSource::File { path, .. } => {
            let file = dir.join(path);
            let file = ctx.scope.check_lexical(&file).map_err(|_| outside(&file))?;
            let root = file
                .parent()
                .map_or_else(|| dir.to_path_buf(), Path::to_path_buf);
            // Read only once it is known to lie inside a root.
            (root, DatasetSpec::from_path(&file).ok())
        }
        _ => {
            return Err(StudioError::invalid(
                "a dataset source this app does not know",
            ));
        }
    };
    let data_root = ctx
        .scope
        .check_lexical(&data_root)
        .map_err(|_| outside(&data_root))?;
    if let Some(dataset) = dataset {
        ctx.scope.check_dataset(&dataset, &data_root)?;
    }
    Ok(())
}

/// Replace the project's definition with `spec` and write it, in its
/// format (a version 1 project becomes version 2). `revision` is the one
/// `open_project` answered: when the file changed on disk since, nothing is
/// written and the answer is `changed_on_disk` (open it again and apply the
/// edit). Every path the spec names must lie inside a root.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SaveProjectArgs {
    pub dir: UserPath,
    pub revision: String,
    pub spec: ProjectSpec,
}

pub fn save_project(ctx: &Ctx, args: SaveProjectArgs) -> Result<ProjectView, StudioError> {
    let mut project = open_checked(ctx, &args.dir)?;
    if project.revision() != args.revision {
        return Err(StudioError::from(
            vision_calibration::project::ProjectError::ChangedOnDisk {
                path: project.file().to_path_buf(),
            },
        ));
    }
    check_spec_paths(ctx, project.dir(), &args.spec)?;
    *project.spec_mut() = args.spec;
    project.save()?;
    ctx.events.project_changed(project.dir());
    ProjectView::of(ctx, &project)
}

/// The views the project's images pair into, without reading an image, and
/// what is wrong with the project as a run would find it before detection:
/// of the saved project, or of the draft `spec` when given (nothing is
/// written). Each problem is a message with the field it is about.
///
/// With a `spec`, `dir` may also be a folder that is not a project yet (a
/// new project's data folder): the draft is then judged as a project in
/// that folder, its inline manifest's `root` read against it.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ProjectViewsArgs {
    pub dir: UserPath,
    #[serde(default)]
    pub spec: Option<ProjectSpec>,
}

/// The dataset's views and the project's problems.
#[derive(Debug, Serialize, JsonSchema)]
pub struct ProjectViews {
    /// The cameras' ids, in manifest order.
    pub cameras: Vec<String>,
    /// The views, in pairing order; empty when the images cannot be
    /// listed or paired (a message says why).
    pub views: Vec<ViewRow>,
    /// What a run would refuse, each with the field it is about when there
    /// is one; empty when the project would run.
    pub messages: Vec<StudioError>,
}

/// One view of the dataset.
#[derive(Debug, Serialize, JsonSchema)]
pub struct ViewRow {
    /// Its pairing token (the view's index under index pairing).
    pub token: String,
    /// Each camera's image, relative to the data folder (absolute outside
    /// it), `/`-separated; `null` where a camera has none.
    pub images: Vec<Option<String>>,
}

pub fn project_views(ctx: &Ctx, args: ProjectViewsArgs) -> Result<ProjectViews, StudioError> {
    // The scratch project of a draft in a folder that is no project yet;
    // removed when this call ends.
    let mut _scratch = None;
    let project = match (open_checked(ctx, &args.dir), &args.spec) {
        (Err(e), Some(spec)) if e.kind == ErrorKind::NotAProject => {
            let (project, guard) = draft_project(ctx, &args.dir, spec)?;
            _scratch = Some(guard);
            project
        }
        (opened, _) => {
            let mut project = opened?;
            if let Some(spec) = args.spec {
                check_spec_paths(ctx, project.dir(), &spec)?;
                *project.spec_mut() = spec;
            }
            project
        }
    };
    let mut messages: Vec<StudioError> = Vec::new();
    let push = |error: StudioError, messages: &mut Vec<StudioError>| {
        if !messages.iter().any(|m| m.message == error.message) {
            messages.push(error);
        }
    };
    let dataset = match project.dataset() {
        Ok(dataset) => dataset,
        Err(e) => {
            return Ok(ProjectViews {
                cameras: Vec::new(),
                views: Vec::new(),
                messages: vec![e.into()],
            });
        }
    };
    let data_root = project.data_root()?;
    ctx.scope.check_dataset(&dataset, &data_root)?;
    if let Err(e) = project.check() {
        push(e.into(), &mut messages);
    }
    let views = match project.views() {
        Ok(views) => views
            .tokens
            .iter()
            .zip(&views.paths)
            .map(|(token, paths)| ViewRow {
                token: token.clone(),
                images: paths
                    .iter()
                    .map(|p| p.as_deref().map(|p| relative(&data_root, p)))
                    .collect(),
            })
            .collect(),
        Err(e) => {
            push(e.into(), &mut messages);
            Vec::new()
        }
    };
    Ok(ProjectViews {
        cameras: dataset.cameras.iter().map(|c| c.id.clone()).collect(),
        views,
        messages,
    })
}

/// A scratch folder for a draft project, removed on drop.
struct Scratch(PathBuf);

impl Drop for Scratch {
    fn drop(&mut self) {
        // Best effort: a leftover scratch folder holds one small file.
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// The draft `spec` as a project whose data folder is `folder` (a folder
/// that holds no project yet), so a new project's layout can be previewed
/// before anything is written to it: the project file goes to a scratch
/// folder under the cache, its manifest's `root` made the absolute data
/// folder. Only an inline manifest has a data folder of its own.
fn draft_project(
    ctx: &Ctx,
    folder: &UserPath,
    spec: &ProjectSpec,
) -> Result<(Project, Scratch), StudioError> {
    static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let folder = ctx.scope.existing_dir(folder)?;
    let DatasetSource::Inline {
        root,
        spec: manifest,
        ..
    } = &spec.dataset
    else {
        return Err(StudioError::invalid(
            "a draft in a folder that holds no project needs its manifest written in the spec",
        ));
    };
    // What the draft names must lie inside the roots, as a saved project's
    // does (judged as if the project were the folder itself).
    check_spec_paths(ctx, &folder, spec)?;
    let data_root = folder.join(root);
    let mut draft = spec.clone();
    draft.dataset = DatasetSource::inline(data_root, manifest.as_ref().clone());
    let dir = ctx.dirs.cache.join("drafts").join(format!(
        "{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    ));
    let guard = Scratch(dir.clone());
    let project = Project::create(&dir, draft)?;
    Ok((project, guard))
}

/// `path` relative to `root` when it lies under it, `/`-separated.
fn relative(root: &Path, path: &Path) -> String {
    slash(path.strip_prefix(root).unwrap_or(path))
}

/// The project's runs (as `open_project` lists them).
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ListRunsArgs {
    pub dir: UserPath,
}

pub fn list_runs(ctx: &Ctx, args: ListRunsArgs) -> Result<Vec<RunInfo>, StudioError> {
    Ok(open_checked(ctx, &args.dir)?.runs()?)
}

/// Replace the notes (label, notes, star) of run `id`; empty notes remove
/// them. Answers the run as listed after.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SetRunNotesArgs {
    pub dir: UserPath,
    pub id: String,
    pub notes: RunNotes,
}

pub fn set_run_notes(ctx: &Ctx, args: SetRunNotesArgs) -> Result<RunInfo, StudioError> {
    let project = open_checked(ctx, &args.dir)?;
    project.set_run_notes(&args.id, &args.notes)?;
    ctx.events.project_changed(project.dir());
    Ok(project.run(&args.id)?)
}

/// Delete run `id` with every file in it. Answers the runs left.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct DeleteRunArgs {
    pub dir: UserPath,
    pub id: String,
}

pub fn delete_run(ctx: &Ctx, args: DeleteRunArgs) -> Result<Vec<RunInfo>, StudioError> {
    let project = open_checked(ctx, &args.dir)?;
    project.delete_run(&args.id)?;
    ctx.events.project_changed(project.dir());
    Ok(project.runs()?)
}

/// Add an export calibrated elsewhere to the project as its next run: an
/// export file, a `vcal run -o` folder (its `calibration.json`) or a run
/// folder (its `export.json`). `import_mismatch` for another workflow or
/// camera count than the project's; the export, and every image its image
/// manifest names, must lie inside a root.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ImportExportArgs {
    pub dir: UserPath,
    pub export: UserPath,
    #[serde(default)]
    pub notes: RunNotes,
}

pub fn import_export(ctx: &Ctx, args: ImportExportArgs) -> Result<RunInfo, StudioError> {
    let project = open_checked(ctx, &args.dir)?;
    let source = ctx.scope.existing(&args.export)?;
    let file = if source.is_dir() {
        let vcal = source.join("calibration.json");
        if vcal.is_file() {
            vcal
        } else {
            source.join(vision_calibration::project::EXPORT_FILE)
        }
    } else {
        source.clone()
    };
    let file = ctx.scope.check(&file)?;
    check_export_images(ctx, &file)?;
    let run = project.import_export(&file, args.notes)?;
    ctx.events.project_changed(project.dir());
    Ok(run)
}

/// The images an export's image manifest names (the import reads their
/// sizes), judged against the roots: an export may name any folder.
fn check_export_images(ctx: &Ctx, file: &Path) -> Result<(), StudioError> {
    let Ok(text) = std::fs::read_to_string(file) else {
        // The library reports what it cannot read.
        return Ok(());
    };
    let Ok(export) = serde_json::from_str::<serde_json::Value>(&text) else {
        return Ok(());
    };
    let Some(manifest) = export.pointer("/diagnostics/image_manifest") else {
        return Ok(());
    };
    let base = file.parent().map(Path::to_path_buf).unwrap_or_default();
    let root = base.join(manifest["root"].as_str().unwrap_or("."));
    let mut paths: Vec<PathBuf> = vec![root.clone()];
    for frame in manifest["frames"].as_array().into_iter().flatten() {
        if let Some(path) = frame["path"].as_str() {
            paths.push(root.join(path));
        }
    }
    for path in paths {
        ctx.scope.check_lexical(&path).map_err(|_| {
            StudioError::new(
                ErrorKind::ForbiddenPath,
                format!(
                    "the export names {}, which is outside the folders this back end serves",
                    path.display()
                ),
            )
            .at(path.to_string_lossy())
        })?;
    }
    Ok(())
}
