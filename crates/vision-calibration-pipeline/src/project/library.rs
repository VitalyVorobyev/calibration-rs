//! The projects under a set of folders, and the calibration history of
//! each device across them: [`scan_library`].
//!
//! There is no index: the folders are the library, read afresh by every
//! scan, so nothing can drift from them.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use vision_calibration_dataset::ProblemKind;

use super::{
    DeviceIdentity, PROJECT_FILE, PROJECT_FILE_JSON, Project, RunInfo, RunSource, RunState,
};
use crate::analysis::Verdict;

/// The options of [`scan_library`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[serde(default, deny_unknown_fields)]
#[non_exhaustive]
pub struct LibraryScanOptions {
    /// How many folders deep below a root the scan looks for projects.
    /// Default 8: a library sorted by site, device and date stays well
    /// within it, while a root set to a home folder does not walk the
    /// whole disk.
    pub max_depth: usize,
}

impl Default for LibraryScanOptions {
    fn default() -> Self {
        Self { max_depth: 8 }
    }
}

impl LibraryScanOptions {
    /// The default options.
    pub fn new() -> Self {
        Self::default()
    }
}

/// What a scan found: the projects, and what it could not read.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[non_exhaustive]
pub struct LibraryScan {
    /// The projects, sorted by folder.
    pub projects: Vec<ProjectEntry>,
    /// The folders and projects that could not be read, sorted by path.
    pub failures: Vec<ScanFailure>,
}

/// One project a scan found.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[non_exhaustive]
pub struct ProjectEntry {
    /// The project folder, absolute.
    pub dir: PathBuf,
    /// The project's name, or its folder's.
    pub name: String,
    /// The device it calibrates.
    pub device: Option<DeviceIdentity>,
    /// Its dataset's workflow; `None` when a referenced manifest cannot be
    /// read.
    pub topology: Option<ProblemKind>,
    /// Its runs, in listing order.
    pub runs: Vec<RunInfo>,
}

/// A folder or project a scan could not read. A scan never fails on one.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[non_exhaustive]
pub struct ScanFailure {
    /// The folder.
    pub path: PathBuf,
    /// Why it could not be read.
    pub message: String,
}

/// The calibration history of one device: every run of every project of
/// it.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[non_exhaustive]
pub struct DeviceHistory {
    /// The device, as the first of its projects (by folder) states it.
    pub device: DeviceIdentity,
    /// Its runs: those without a timestamp (the app's earlier format)
    /// first, then by timestamp, then by project folder and listing order.
    pub runs: Vec<DeviceRun>,
}

/// One run of a device's history.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[non_exhaustive]
pub struct DeviceRun {
    /// The project folder.
    pub project: PathBuf,
    /// The run.
    pub run: RunInfo,
}

impl LibraryScan {
    /// The calibration history of each device, by `product` and `id`; a
    /// project without a device is in no history. Sorted by product (none
    /// first), then id.
    pub fn devices(&self) -> Vec<DeviceHistory> {
        let mut groups: BTreeMap<(Option<&str>, &str), DeviceHistory> = BTreeMap::new();
        for project in &self.projects {
            let Some(device) = &project.device else {
                continue;
            };
            let history = groups
                .entry((device.product.as_deref(), device.id.as_str()))
                .or_insert_with(|| DeviceHistory {
                    device: device.clone(),
                    runs: Vec::new(),
                });
            history
                .runs
                .extend(project.runs.iter().map(|run| DeviceRun {
                    project: project.dir.clone(),
                    run: run.clone(),
                }));
        }
        let mut histories: Vec<DeviceHistory> = groups.into_values().collect();
        for history in &mut histories {
            // A stable sort keeps each project's listing order; the projects
            // were added in folder order.
            history.runs.sort_by_key(|r| timestamp(&r.run));
        }
        histories
    }
}

impl DeviceHistory {
    /// The run that stands for the device, to seed its next calibration or
    /// a seed of its type from: its latest starred run, else its latest run
    /// calibrated in a project (not imported) whose quality verdict is
    /// pass; `None` when it has neither. "Latest" is the history's order
    /// ([`runs`](Self::runs)).
    pub fn representative_run(&self) -> Option<&DeviceRun> {
        self.runs
            .iter()
            .rev()
            .find(|entry| entry.run.notes.starred)
            .or_else(|| {
                self.runs.iter().rev().find(|entry| {
                    entry.run.record().is_some_and(|record| {
                        matches!(record.source, RunSource::Calibrated(_))
                            && record.summary.verdict == Some(Verdict::Pass)
                    })
                })
            })
    }
}

/// A run's timestamp, `None` for one without a record.
fn timestamp(run: &RunInfo) -> Option<u64> {
    match &run.state {
        RunState::Recorded(record) => Some(record.timestamp),
        _ => None,
    }
}

/// The projects under the folders `roots`.
///
/// Each root is walked without following symbolic links and without
/// entering folders whose name starts with `.`, at most
/// `options.max_depth` folders deep. A folder holding a project file is a
/// project: the scan reads it with its runs and does not look inside it
/// (projects do not nest). A project or folder that cannot be read is a
/// [`ScanFailure`], never an error of the scan.
pub fn scan_library(roots: &[PathBuf], options: &LibraryScanOptions) -> LibraryScan {
    let mut scan = LibraryScan::default();
    for root in roots {
        let root = vision_calibration_core::lexical_absolute(root);
        if root.is_dir() {
            walk(&root, 0, options.max_depth, &mut scan);
        } else {
            scan.failures.push(ScanFailure {
                path: root,
                message: "not a folder".into(),
            });
        }
    }
    scan.projects.sort_by(|a, b| a.dir.cmp(&b.dir));
    scan.projects.dedup_by(|a, b| a.dir == b.dir);
    scan.failures.sort_by(|a, b| a.path.cmp(&b.path));
    scan.failures.dedup_by(|a, b| a.path == b.path);
    scan
}

fn walk(dir: &Path, depth: usize, max_depth: usize, scan: &mut LibraryScan) {
    if dir.join(PROJECT_FILE).is_file() || dir.join(PROJECT_FILE_JSON).is_file() {
        match entry(dir) {
            Ok(entry) => scan.projects.push(entry),
            Err(err) => scan.failures.push(ScanFailure {
                path: dir.to_path_buf(),
                message: err.to_string(),
            }),
        }
        return;
    }
    if depth >= max_depth {
        return;
    }
    let entries = match std::fs::read_dir(dir) {
        Ok(entries) => entries,
        Err(err) => {
            scan.failures.push(ScanFailure {
                path: dir.to_path_buf(),
                message: err.to_string(),
            });
            return;
        }
    };
    let mut folders: Vec<PathBuf> = entries
        .filter_map(Result::ok)
        // `file_type` does not follow a symbolic link: a link is not a dir.
        .filter(|e| e.file_type().is_ok_and(|t| t.is_dir()))
        .filter(|e| !e.file_name().to_string_lossy().starts_with('.'))
        .map(|e| e.path())
        .collect();
    folders.sort();
    for folder in folders {
        walk(&folder, depth + 1, max_depth, scan);
    }
}

fn entry(dir: &Path) -> Result<ProjectEntry, super::ProjectError> {
    let project = Project::open(dir)?;
    Ok(ProjectEntry {
        dir: project.dir().to_path_buf(),
        name: project.display_name(),
        device: project.spec().device.clone(),
        topology: project.dataset().ok().map(|d| d.topology),
        runs: project.runs()?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::project::{DatasetSource, ProjectSpec, RUNS_DIR};

    fn project(dir: &Path, device: Option<(&str, &str)>) {
        let mut spec = ProjectSpec::new(DatasetSource::file("dataset.toml"));
        spec.device = device.map(|(product, id)| {
            let mut device = DeviceIdentity::new(id);
            device.product = Some(product.into());
            device
        });
        Project::create(dir, spec).unwrap();
    }

    /// A run of the app's earlier format: listed without a record.
    fn unrecorded_run(project: &Path, id: &str) {
        let dir = project.join(RUNS_DIR).join(id);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("export.json"), "{}").unwrap();
    }

    #[test]
    fn projects_are_found_and_grouped_by_device() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        project(&root.join("site-a/2026-01"), Some(("scanner", "SN-1")));
        unrecorded_run(&root.join("site-a/2026-01"), "run-001");
        project(&root.join("site-b/2026-02"), Some(("scanner", "SN-1")));
        unrecorded_run(&root.join("site-b/2026-02"), "run-001");
        unrecorded_run(&root.join("site-b/2026-02"), "run-002");
        project(&root.join("site-b/other"), Some(("camera", "SN-1")));
        project(&root.join("loose"), None);
        // A project inside a project is not looked for; neither are
        // hidden folders, nor folders past the depth limit.
        project(&root.join("loose/inner"), Some(("scanner", "SN-9")));
        project(&root.join(".hidden/p"), Some(("scanner", "SN-8")));
        project(&root.join("a/b/c/deep"), Some(("scanner", "SN-7")));
        // A broken project is a failure, not an error.
        std::fs::create_dir_all(root.join("broken")).unwrap();
        std::fs::write(root.join("broken/calibration-project.toml"), "version = ").unwrap();

        let mut options = LibraryScanOptions::new();
        options.max_depth = 3;
        let scan = scan_library(&[root.to_path_buf(), root.join("missing")], &options);
        let names: Vec<&str> = scan.projects.iter().map(|p| p.name.as_str()).collect();
        assert_eq!(names, ["loose", "2026-01", "2026-02", "other"]);
        let failures: Vec<&Path> = scan.failures.iter().map(|f| f.path.as_path()).collect();
        assert_eq!(failures.len(), 2, "{failures:?}");
        assert!(failures[0].ends_with("broken"));
        assert!(failures[1].ends_with("missing"));
        // The referenced manifest does not exist: no topology, still listed.
        assert_eq!(scan.projects[1].topology, None);

        let devices = scan.devices();
        let keys: Vec<(Option<&str>, &str, usize)> = devices
            .iter()
            .map(|d| {
                (
                    d.device.product.as_deref(),
                    d.device.id.as_str(),
                    d.runs.len(),
                )
            })
            .collect();
        assert_eq!(
            keys,
            [(Some("camera"), "SN-1", 0), (Some("scanner"), "SN-1", 3)]
        );
        let history: Vec<(String, &str)> = devices[1]
            .runs
            .iter()
            .map(|r| {
                (
                    r.project
                        .file_name()
                        .unwrap()
                        .to_string_lossy()
                        .into_owned(),
                    r.run.id.as_str(),
                )
            })
            .collect();
        assert_eq!(
            history,
            [
                ("2026-01".to_string(), "run-001"),
                ("2026-02".to_string(), "run-001"),
                ("2026-02".to_string(), "run-002"),
            ]
        );

        let deeper = scan_library(&[root.to_path_buf()], &LibraryScanOptions::default());
        assert_eq!(deeper.projects.len(), 5);
    }

    #[cfg(unix)]
    #[test]
    fn symbolic_links_are_not_followed() {
        let tmp = tempfile::tempdir().unwrap();
        project(&tmp.path().join("real/p"), None);
        std::os::unix::fs::symlink(tmp.path().join("real"), tmp.path().join("link")).unwrap();
        let scan = scan_library(&[tmp.path().to_path_buf()], &LibraryScanOptions::default());
        assert_eq!(scan.projects.len(), 1);
    }
}
