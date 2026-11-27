//! Runs as the commands address them: a project folder and a run id
//! ([`RunRef`]), never a global "active export". [`RunCache`] keeps the
//! last few exports parsed, keyed by their file as it is on disk, so the
//! viewers' per-click commands do not re-read a multi-megabyte export and
//! two tabs on two runs do not evict each other's run on every call.

use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use schemars::JsonSchema;
use serde::Deserialize;
use sha2::Digest;
use vision_calibration::project::{Project, RunInfo, RunSource, RunState};

use crate::api::{Ctx, StudioError};
use crate::scope::UserPath;

/// A run of a project: the project folder and the run's id (its folder
/// under `runs/`).
#[derive(Debug, Clone, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct RunRef {
    /// The project folder.
    pub dir: UserPath,
    /// The run's id, as `run-007`.
    pub id: String,
}

/// A run's export, parsed, with what derived results of it are keyed by.
#[derive(Debug)]
pub struct CachedRun {
    /// The export JSON.
    pub export: serde_json::Value,
    /// The SHA-256 of the export file: what derived images and blobs of the
    /// run are keyed by.
    pub fingerprint: String,
    /// The folder the export's image manifest frames resolve against: the
    /// project's data root for a run the project calibrated (so a relinked
    /// dataset fixes every run), else the manifest's own root restated from
    /// the run folder. `None` without an image manifest.
    pub image_root: Option<PathBuf>,
}

/// How many parsed exports the cache keeps.
const CAPACITY: usize = 8;

/// The key of an export as it is on disk: its canonical path, length and
/// modification time. A rewritten file is a new entry.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Key {
    path: PathBuf,
    len: u64,
    modified: Option<std::time::SystemTime>,
}

/// The last [`CAPACITY`] exports read, most recently used last.
#[derive(Default)]
pub struct RunCache {
    entries: Mutex<VecDeque<(Key, Arc<CachedRun>)>>,
}

/// A run resolved through the scope: its project, listing entry and export
/// file (canonical, inside a root).
pub struct ResolvedRun {
    pub project: Project,
    pub info: RunInfo,
    pub export: PathBuf,
}

impl RunCache {
    pub fn new() -> Self {
        Self::default()
    }

    /// The export of `run`, parsed (from the cache when its file has not
    /// changed).
    pub fn get(&self, ctx: &Ctx, run: &RunRef) -> Result<Arc<CachedRun>, StudioError> {
        let resolved = resolve(ctx, run)?;
        let meta = std::fs::metadata(&resolved.export).map_err(|e| {
            StudioError::new(
                crate::api::ErrorKind::Io,
                format!("{}: {e}", resolved.export.display()),
            )
        })?;
        let key = Key {
            path: resolved.export.clone(),
            len: meta.len(),
            modified: meta.modified().ok(),
        };
        {
            let mut entries = self.lock();
            if let Some(at) = entries.iter().position(|(k, _)| *k == key) {
                let entry = entries.remove(at).expect("the position is in range");
                let cached = Arc::clone(&entry.1);
                entries.push_back(entry);
                return Ok(cached);
            }
        }
        let cached = Arc::new(read(&resolved)?);
        let mut entries = self.lock();
        entries.retain(|(k, _)| k.path != key.path);
        if entries.len() == CAPACITY {
            entries.pop_front();
        }
        entries.push_back((key, Arc::clone(&cached)));
        Ok(cached)
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, VecDeque<(Key, Arc<CachedRun>)>> {
        // A list of parsed files: a panic elsewhere leaves it whole.
        self.entries.lock().unwrap_or_else(|e| e.into_inner())
    }
}

/// The project of `dir`, opened through the scope: the folder inside a
/// root, and the paths its file names checked before anything reads them.
pub fn open_project(ctx: &Ctx, dir: &UserPath) -> Result<Project, StudioError> {
    let dir = ctx.scope.existing_dir(dir)?;
    let project = Project::open(&dir)?;
    ctx.scope.check_project_paths(&project)?;
    Ok(project)
}

/// `run` resolved: the project opened through the scope, the run listed,
/// its export a file inside a root.
pub fn resolve(ctx: &Ctx, run: &RunRef) -> Result<ResolvedRun, StudioError> {
    let project = open_project(ctx, &run.dir)?;
    let info = project.run(&run.id)?;
    let export = ctx.scope.check(&info.export_path())?;
    Ok(ResolvedRun {
        project,
        info,
        export,
    })
}

/// Read and check a run's export.
fn read(run: &ResolvedRun) -> Result<CachedRun, StudioError> {
    let bytes = std::fs::read(&run.export).map_err(|e| {
        StudioError::new(
            crate::api::ErrorKind::Io,
            format!("{}: {e}", run.export.display()),
        )
        .at(slash(&run.export))
    })?;
    let export: serde_json::Value = serde_json::from_slice(&bytes).map_err(|e| {
        StudioError::new(
            crate::api::ErrorKind::Parse,
            format!("{}: {e}", run.export.display()),
        )
        .at(slash(&run.export))
    })?;
    vision_calibration::common::check_export_version(&export.to_string())
        .map_err(|e| StudioError::invalid(format!("{}: {e}", run.export.display())))?;
    let fingerprint = crate::api::resource::hex(&sha2::Sha256::digest(&bytes));
    Ok(CachedRun {
        image_root: image_root(run, &export),
        export,
        fingerprint,
    })
}

/// Where a run's frames resolve (see [`CachedRun::image_root`]).
fn image_root(run: &ResolvedRun, export: &serde_json::Value) -> Option<PathBuf> {
    let manifest_root = export
        .get("diagnostics")?
        .get("image_manifest")?
        .get("root")?
        .as_str()?;
    let calibrated = matches!(
        &run.info.state,
        RunState::Recorded(record) if matches!(record.source, RunSource::Calibrated(_))
    );
    if calibrated && let Ok(root) = run.project.data_root() {
        return Some(root);
    }
    Some(folded(&run.info.dir.join(manifest_root)))
}

/// `path` with its `.` and `..` components folded away, lexically.
fn folded(path: &Path) -> PathBuf {
    use std::path::Component;
    let mut out = PathBuf::new();
    for part in path.components() {
        match part {
            Component::CurDir => {}
            Component::ParentDir => {
                out.pop();
            }
            other => out.push(other),
        }
    }
    out
}

/// `path` with `/` separators, as the frontend joins paths.
pub fn slash(path: &Path) -> String {
    path.to_string_lossy().replace('\\', "/")
}
