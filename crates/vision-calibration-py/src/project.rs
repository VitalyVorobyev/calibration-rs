//! Calibration projects: the `Project` class over the library's
//! `project::Project`, and the library scan.

use std::path::PathBuf;
use std::sync::Mutex;

use pyo3::prelude::*;
use serde::Serialize;
use vision_calibration::analysis::CompareRunsOptions;
use vision_calibration::dataset_runner::{
    ConfigPatch, LaserPixelExtractor, PoseRow, UnpairedLaserFrame, ViewLaserFrames, ViewPoses,
};
use vision_calibration::project::{
    DeviceHistory, LibraryScan, LibraryScanOptions, Project, ProjectError, ProjectRunOptions,
    ProjectSpec, RunInfo, RunNotes, SeedSource, compare_recorded_runs, scan_library,
};

use crate::control::PyRunControl;
use crate::dataset::PyLaserExtractor;
use crate::errors::project_err;
use crate::parse_payload;
use crate::session::{parse_checked, to_py};
use crate::validation::value_err;

/// A calibration project: a folder with its project file, runs and
/// detection cache.
#[pyclass(module = "vision_calibration._vision_calibration", name = "Project")]
pub(crate) struct PyProject {
    inner: Mutex<Project>,
}

impl PyProject {
    fn new(project: Project) -> Self {
        Self {
            inner: Mutex::new(project),
        }
    }

    /// The project, for a call that only reads it.
    fn snapshot(&self) -> Project {
        match self.inner.lock() {
            Ok(guard) => guard.clone(),
            Err(poisoned) => poisoned.into_inner().clone(),
        }
    }

    fn with_mut<T>(&self, f: impl FnOnce(&mut Project) -> T) -> T {
        match self.inner.lock() {
            Ok(mut guard) => f(&mut guard),
            Err(poisoned) => f(&mut poisoned.into_inner()),
        }
    }
}

/// Notes from their keyword arguments.
fn notes(label: Option<String>, text: Option<String>, starred: bool) -> RunNotes {
    let mut notes = RunNotes::new();
    notes.label = label;
    notes.notes = text;
    notes.starred = starred;
    notes
}

#[pymethods]
impl PyProject {
    /// Open the project in the folder `path`, or of the project file `path`.
    #[staticmethod]
    fn open(py: Python<'_>, path: PathBuf) -> PyResult<Self> {
        Project::open(&path)
            .map(Self::new)
            .map_err(|e| project_err(py, e))
    }

    /// Make the folder `dir` a project of the `ProjectSpec` payload `spec`.
    #[staticmethod]
    fn create(py: Python<'_>, dir: PathBuf, spec: &Bound<'_, PyAny>) -> PyResult<Self> {
        let spec: ProjectSpec = parse_checked(spec, "spec")?;
        Project::create(&dir, spec)
            .map(Self::new)
            .map_err(|e| project_err(py, e))
    }

    /// Write the spec back to the project file.
    fn save(&self, py: Python<'_>) -> PyResult<()> {
        self.with_mut(Project::save).map_err(|e| project_err(py, e))
    }

    /// Read the project file again, dropping unsaved edits.
    fn reload(&self, py: Python<'_>) -> PyResult<()> {
        self.with_mut(Project::reload)
            .map_err(|e| project_err(py, e))
    }

    /// The project folder.
    #[getter]
    fn dir(&self) -> PathBuf {
        self.snapshot().dir().to_path_buf()
    }

    /// The project file.
    #[getter]
    fn file(&self) -> PathBuf {
        self.snapshot().file().to_path_buf()
    }

    /// The version of the project file on disk.
    #[getter]
    fn file_version(&self) -> u32 {
        self.snapshot().file_version()
    }

    /// The content revision last read or written.
    #[getter]
    fn revision(&self) -> String {
        self.snapshot().revision().to_string()
    }

    /// The `ProjectSpec` payload, with unsaved edits.
    #[getter]
    fn spec(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        to_py(py, self.snapshot().spec())
    }

    /// Replace the spec (saved by `save`).
    #[setter]
    fn set_spec(&self, spec: &Bound<'_, PyAny>) -> PyResult<()> {
        let spec: ProjectSpec = parse_checked(spec, "spec")?;
        self.with_mut(|p| *p.spec_mut() = spec);
        Ok(())
    }

    /// The folder the manifest's relative paths resolve against.
    fn data_root(&self, py: Python<'_>) -> PyResult<PathBuf> {
        self.snapshot().data_root().map_err(|e| project_err(py, e))
    }

    /// The `DatasetSpec` payload of the dataset.
    fn dataset(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let dataset = self.snapshot().dataset().map_err(|e| project_err(py, e))?;
        to_py(py, &dataset)
    }

    /// The configuration payload of the dataset's workflow with the
    /// project's changes.
    fn config(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let config = self.snapshot().config().map_err(|e| project_err(py, e))?;
        let tagged = serde_json::to_value(&config)
            .map_err(|e| value_err(format!("config serialization failed: {e}")))?;
        to_py(py, &tagged["config"])
    }

    /// Check what a run checks before it reads an image.
    fn check(&self, py: Python<'_>) -> PyResult<()> {
        let project = self.snapshot();
        py.detach(|| project.check())
            .map_err(|e| project_err(py, e))
    }

    /// The digests of the inputs as they stand, without reading an image:
    /// `{"dataset", "exclusions", "config", "quality"}`.
    fn input_digest(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let project = self.snapshot();
        let digest = py
            .detach(|| project.input_digest())
            .map_err(|e| project_err(py, e))?;
        to_py(py, &digest)
    }

    /// The views the images pair into: `{"paths", "tokens", "robot_poses",
    /// "laser_frames"}`; the last two are `None` when the manifest declares
    /// no robot poses or laser frames, and `{"error", "field"}` when they
    /// cannot be listed.
    fn views(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        #[derive(Serialize)]
        struct Row {
            index: usize,
            id: Option<String>,
        }
        #[derive(Serialize)]
        struct Poses {
            views: Vec<Option<Row>>,
            unpaired: Vec<Row>,
        }
        #[derive(Serialize)]
        struct Unpaired {
            camera: usize,
            path: PathBuf,
            token: Option<String>,
        }
        #[derive(Serialize)]
        struct Lasers {
            paths: Vec<Vec<Option<PathBuf>>>,
            unpaired: Vec<Unpaired>,
        }
        /// A listing, or why it could not be made.
        #[derive(Serialize)]
        #[serde(untagged)]
        enum Listed<T> {
            Ok(T),
            Err {
                error: String,
                field: Option<String>,
            },
        }
        fn listed<A, T>(
            result: Result<Option<A>, ProjectError>,
            convert: impl FnOnce(A) -> T,
        ) -> Option<Listed<T>> {
            result.transpose().map(|result| match result {
                Ok(value) => Listed::Ok(convert(value)),
                Err(e) => Listed::Err {
                    error: e.to_string(),
                    field: e.field(),
                },
            })
        }
        fn row(row: PoseRow) -> Row {
            Row {
                index: row.index,
                id: row.id,
            }
        }
        #[derive(Serialize)]
        struct Views {
            paths: Vec<Vec<Option<PathBuf>>>,
            tokens: Vec<String>,
            robot_poses: Option<Listed<Poses>>,
            laser_frames: Option<Listed<Lasers>>,
        }
        let project = self.snapshot();
        let (views, poses, frames) = py
            .detach(|| {
                let views = project.views()?;
                let poses = project.view_poses(&views);
                let frames = project.view_laser_frames(&views);
                Ok((views, poses, frames))
            })
            .map_err(|e| project_err(py, e))?;
        let robot_poses = listed(poses, |poses: ViewPoses| Poses {
            views: poses.views.into_iter().map(|r| r.map(row)).collect(),
            unpaired: poses.unpaired.into_iter().map(row).collect(),
        });
        let laser_frames = listed(frames, |frames: ViewLaserFrames| Lasers {
            paths: frames.paths,
            unpaired: frames
                .unpaired
                .into_iter()
                .map(|frame: UnpairedLaserFrame| Unpaired {
                    camera: frame.camera,
                    path: frame.path,
                    token: frame.token,
                })
                .collect(),
        });
        to_py(
            py,
            &Views {
                paths: views.paths,
                tokens: views.tokens,
                robot_poses,
                laser_frames,
            },
        )
    }

    /// The detection cache folder.
    fn detection_cache_dir(&self) -> PathBuf {
        self.snapshot().detection_cache_dir()
    }

    /// The runs, as `RunInfo` payloads.
    fn runs(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let runs = self.snapshot().runs().map_err(|e| project_err(py, e))?;
        to_py(py, &runs)
    }

    /// The run `id`, as a `RunInfo` payload.
    fn run(&self, py: Python<'_>, id: &str) -> PyResult<Py<PyAny>> {
        let run = self.snapshot().run(id).map_err(|e| project_err(py, e))?;
        to_py(py, &run)
    }

    /// The runs `ids` side by side against the first; the relations come
    /// from the records. Returns a `RunComparison` payload.
    #[pyo3(signature = (ids, options=None))]
    fn compare_runs(
        &self,
        py: Python<'_>,
        ids: Vec<String>,
        options: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        let options: CompareRunsOptions =
            crate::parse_optional_payload(options, "options")?.unwrap_or_default();
        let project = self.snapshot();
        let ids: Vec<&str> = ids.iter().map(String::as_str).collect();
        let comparison = py
            .detach(|| project.compare_runs(&ids, &options))
            .map_err(|e| project_err(py, e))?;
        to_py(py, &comparison)
    }

    /// Calibrate the dataset into the next run; returns its `RunInfo`.
    #[pyo3(signature = (tool, label=None, notes_text=None, starred=false, config_patch=None, redetect=false, report_html=false, laser_extractor=None, progress=None, seed=None))]
    #[allow(clippy::too_many_arguments)] // The Python keyword arguments.
    fn calibrate(
        &self,
        py: Python<'_>,
        tool: String,
        label: Option<String>,
        notes_text: Option<String>,
        starred: bool,
        config_patch: Option<&Bound<'_, PyAny>>,
        redetect: bool,
        report_html: bool,
        laser_extractor: Option<&Bound<'_, PyAny>>,
        progress: Option<Py<PyAny>>,
        seed: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        let mut options = ProjectRunOptions::new(tool);
        options.notes = notes(label, notes_text, starred);
        if let Some(patch) = config_patch.filter(|p| !p.is_none()) {
            options.config_patch = parse_checked::<ConfigPatch>(patch, "config_patch")?;
        }
        if let Some(seed) = seed.filter(|s| !s.is_none()) {
            options.seed = Some(parse_checked::<SeedSource>(seed, "seed")?);
        }
        options.redetect = redetect;
        options.report_html = report_html;
        let extractor = laser_extractor
            .filter(|object| !object.is_none())
            .map(PyLaserExtractor::new)
            .transpose()?;
        let project = self.snapshot();
        let control = PyRunControl::new(progress);
        let result = py.detach(|| {
            project.calibrate(
                &options,
                extractor.as_ref().map(|e| e as &dyn LaserPixelExtractor),
                control.control(),
            )
        });
        // A cancelled run is a KeyboardInterrupt, normally replaced by the
        // exception the progress callable raised or the Ctrl-C that
        // cancelled the token.
        let run = control.finish(result, |e| project_err(py, e))?;
        to_py(py, &run)
    }

    /// Import an export calibrated elsewhere as a run; returns its `RunInfo`.
    #[pyo3(signature = (export, label=None, notes_text=None, starred=false))]
    fn import_export(
        &self,
        py: Python<'_>,
        export: PathBuf,
        label: Option<String>,
        notes_text: Option<String>,
        starred: bool,
    ) -> PyResult<Py<PyAny>> {
        let project = self.snapshot();
        let notes = notes(label, notes_text, starred);
        let run = py
            .detach(|| project.import_export(&export, notes))
            .map_err(|e| project_err(py, e))?;
        to_py(py, &run)
    }

    /// Replace a run's notes with the `RunNotes` payload `notes`.
    fn set_run_notes(&self, py: Python<'_>, id: &str, notes: &Bound<'_, PyAny>) -> PyResult<()> {
        let notes: RunNotes = parse_payload(notes, "notes")?;
        self.snapshot()
            .set_run_notes(id, &notes)
            .map_err(|e| project_err(py, e))
    }

    /// Delete a run with its files.
    fn delete_run(&self, py: Python<'_>, id: &str) -> PyResult<()> {
        self.snapshot()
            .delete_run(id)
            .map_err(|e| project_err(py, e))
    }
}

/// `project::scan_library`: the projects under `roots`, the folders that
/// could not be read, and each device's history.
#[pyfunction]
#[pyo3(signature = (roots, max_depth=8))]
pub(crate) fn project_scan_library(
    py: Python<'_>,
    roots: Vec<PathBuf>,
    max_depth: usize,
) -> PyResult<Py<PyAny>> {
    #[derive(Serialize)]
    struct Output<'a> {
        #[serde(flatten)]
        scan: &'a LibraryScan,
        devices: Vec<DeviceHistory>,
    }
    let mut options = LibraryScanOptions::new();
    options.max_depth = max_depth;
    let scan = py.detach(|| scan_library(&roots, &options));
    to_py(
        py,
        &Output {
            devices: scan.devices(),
            scan: &scan,
        },
    )
}

/// `DeviceHistory::representative_run`: the `DeviceRun` payload that stands
/// for the device of the `DeviceHistory` payload `device`, or `None`.
#[pyfunction]
pub(crate) fn project_representative_run(
    py: Python<'_>,
    device: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    let history: DeviceHistory = parse_payload(device, "device")?;
    to_py(py, &history.representative_run())
}

/// Recorded runs (`RunInfo` payloads, of any projects) side by side against
/// the first, the relations read from their records; a `RunComparison`
/// payload.
#[pyfunction]
#[pyo3(signature = (runs, options=None))]
pub(crate) fn project_compare_recorded_runs(
    py: Python<'_>,
    runs: &Bound<'_, PyAny>,
    options: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    let runs: Vec<RunInfo> = parse_payload(runs, "runs")?;
    let options: CompareRunsOptions =
        crate::parse_optional_payload(options, "options")?.unwrap_or_default();
    let comparison = py
        .detach(|| compare_recorded_runs(&runs, &options))
        .map_err(|e| project_err(py, e))?;
    to_py(py, &comparison)
}

/// Register the project class and functions.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyProject>()?;
    m.add_function(wrap_pyfunction!(project_scan_library, m)?)?;
    m.add_function(wrap_pyfunction!(project_representative_run, m)?)?;
    m.add_function(wrap_pyfunction!(project_compare_recorded_runs, m)?)?;
    m.add(
        "ProjectError",
        m.py().get_type::<crate::errors::ProjectError>(),
    )?;
    Ok(())
}
