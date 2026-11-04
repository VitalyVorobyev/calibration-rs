//! Manifest-driven runs: the dataset sniffer, the manifest loader and the
//! `build_*_input` converters of the dataset runner, plus the bridge that
//! lets a Python object serve as the runner's laser-line extractor.

use std::path::{Path, PathBuf};

use numpy::{PyArray1, PyArrayMethods, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::prelude::*;
use serde::Serialize;
use vision_calibration::dataset::{
    DatasetSpec, LaserExtractionSpec, samples, sniff_folder as rust_sniff,
};
use vision_calibration::dataset_runner::{self as runner, Detection, LaserPixelExtractor};
use vision_calibration::detect::GrayImageView;

use crate::control::PyRunControl;
use crate::detect::PyFsDetectionCache;
use crate::errors::{fetch_err, run_err, sniff_err};
use crate::parse_payload;
use crate::session::to_py;
use crate::validation::value_err;
use vision_calibration::control::RunControl;

fn load_manifest_file(path: &Path) -> PyResult<DatasetSpec> {
    DatasetSpec::from_path(path).map_err(|err| value_err(err.to_string()))
}

/// A manifest path (its directory is the base directory unless `base_dir`
/// says otherwise) or a `DatasetSpec` payload (then `base_dir` is required).
fn resolve_manifest(
    manifest: &Bound<'_, PyAny>,
    base_dir: Option<PathBuf>,
) -> PyResult<(DatasetSpec, PathBuf)> {
    if let Ok(path) = manifest.extract::<PathBuf>() {
        let spec = load_manifest_file(&path)?;
        let base = base_dir.unwrap_or_else(|| match path.parent() {
            Some(parent) if !parent.as_os_str().is_empty() => parent.to_path_buf(),
            _ => PathBuf::from("."),
        });
        return Ok((spec, base));
    }
    let spec: DatasetSpec = parse_payload(manifest, "manifest")?;
    let base = base_dir.ok_or_else(|| {
        value_err("base_dir is required when the manifest is a mapping rather than a file")
    })?;
    Ok((spec, base))
}

/// A Python object with a `name` and an `extract(image, spec)` method, as the
/// runner's laser-line extractor. `extract` receives the (ROI-cropped)
/// grayscale image as a `(H, W)` `uint8` array and the manifest's `laser`
/// section, and returns `(N, 2)` `float64` pixels.
struct PyLaserExtractor {
    name: String,
    object: Py<PyAny>,
}

impl PyLaserExtractor {
    fn new(object: &Bound<'_, PyAny>) -> PyResult<Self> {
        let name: String = object.getattr("name")?.extract()?;
        Ok(Self {
            name,
            object: object.clone().unbind(),
        })
    }

    fn call(
        &self,
        image: GrayImageView<'_>,
        spec: &LaserExtractionSpec,
    ) -> PyResult<Vec<[f64; 2]>> {
        Python::attach(|py| {
            let (w, h) = (image.width() as usize, image.height() as usize);
            let array = PyArray1::from_slice(py, image.pixels()).reshape([h, w])?;
            let spec = to_py(py, spec)?;
            let out = self.object.call_method1(py, "extract", (array, spec))?;
            let points: PyReadonlyArray2<'_, f64> = out.extract(py)?;
            if points.shape()[1] != 2 {
                return Err(value_err(format!(
                    "laser extractor {:?} must return an (N, 2) array, got {:?}",
                    self.name,
                    points.shape()
                )));
            }
            Ok(points.as_slice()?.as_chunks::<2>().0.to_vec())
        })
    }
}

impl LaserPixelExtractor for PyLaserExtractor {
    fn name(&self) -> &str {
        &self.name
    }

    fn extract(
        &self,
        image: GrayImageView<'_>,
        spec: &LaserExtractionSpec,
    ) -> Result<Vec<[f64; 2]>, Box<dyn std::error::Error + Send + Sync>> {
        self.call(image, spec).map_err(|err| err.to_string().into())
    }
}

/// Run a converter with the GIL released: against the given cache, refreshing
/// it when `force_redetect` is set, or with no cache at all; reporting to
/// `progress` and stopping on Ctrl-C (see [`PyRunControl`]).
fn convert<T: Send>(
    py: Python<'_>,
    cache: Option<PyRef<'_, PyFsDetectionCache>>,
    force_redetect: bool,
    progress: Option<Py<PyAny>>,
    build: impl FnOnce(Detection<'_>, &RunControl) -> Result<T, runner::RunError> + Send,
) -> PyResult<T> {
    let control = PyRunControl::new(progress);
    let run = control.control();
    let result = match &cache {
        Some(cache) => {
            let cache = cache.inner();
            let detection = if force_redetect {
                Detection::Refresh(cache)
            } else {
                Detection::Cached(cache)
            };
            py.detach(|| build(detection, run))
        }
        None => py.detach(|| build(Detection::Uncached, run)),
    };
    control.finish(result, |err| run_err(py, err))
}

/// Inspect a dataset folder and return a best-effort `DatasetSpec` payload.
#[pyfunction]
pub(crate) fn sniff_folder(py: Python<'_>, root: PathBuf) -> PyResult<Py<PyAny>> {
    let spec = rust_sniff(&root).map_err(|err| sniff_err(py, err))?;
    to_py(py, &spec)
}

/// The sample datasets: one dict per sample (`id`, `description`, `licence`,
/// `citation`, `source_url`).
#[pyfunction]
pub(crate) fn sample_catalogue(py: Python<'_>) -> PyResult<Py<PyAny>> {
    #[derive(Serialize)]
    struct Entry<'a> {
        id: &'a str,
        description: &'a str,
        licence: &'a str,
        citation: &'a str,
        source_url: &'a str,
    }
    let entries: Vec<Entry<'_>> = samples::catalogue()
        .iter()
        .map(|s| Entry {
            id: &s.id,
            description: &s.description,
            licence: &s.licence,
            citation: &s.citation,
            source_url: &s.source_url,
        })
        .collect();
    to_py(py, &entries)
}

/// Download the sample `id` into `cache` (the default cache directory when
/// `None`), verify it and return the path of its manifest.
#[pyfunction]
#[pyo3(signature = (id, cache=None))]
pub(crate) fn fetch_sample(
    py: Python<'_>,
    id: String,
    cache: Option<PathBuf>,
) -> PyResult<PathBuf> {
    let cache = match cache {
        Some(dir) => dir,
        None => samples::default_cache_dir().map_err(fetch_err)?,
    };
    py.detach(|| samples::fetch(&id, &cache)).map_err(fetch_err)
}

/// The directory samples are fetched into when no cache is given.
#[pyfunction]
pub(crate) fn sample_cache_dir() -> PyResult<PathBuf> {
    samples::default_cache_dir().map_err(fetch_err)
}

/// Each camera's image size `[width, height]` in pixels, in camera order.
#[pyfunction]
#[pyo3(signature = (manifest, base_dir=None))]
pub(crate) fn camera_image_sizes(
    py: Python<'_>,
    manifest: &Bound<'_, PyAny>,
    base_dir: Option<PathBuf>,
) -> PyResult<Vec<[u32; 2]>> {
    let (spec, base) = resolve_manifest(manifest, base_dir)?;
    py.detach(|| runner::camera_image_sizes(&spec, &base))
        .map_err(|err| run_err(py, err))
}

/// Run the `vcal` command line on `argv` (the program name first) and
/// return its exit status. Output goes to the process's standard output and
/// error; Ctrl-C stops a run, which returns the interrupt status (130).
#[pyfunction]
pub(crate) fn vcal_main(py: Python<'_>, argv: Vec<std::ffi::OsString>) -> u8 {
    let control = PyRunControl::new(None);
    let status = py.detach(|| vision_calibration_cli::run(argv, control.control()));
    // The interrupt is the exit status here, not an exception.
    let _ = control.finish(Ok::<(), PyErr>(()), |err| err);
    status
}

/// Read a manifest file (TOML or JSON) and return its `DatasetSpec` payload.
#[pyfunction]
pub(crate) fn load_manifest(py: Python<'_>, path: PathBuf) -> PyResult<Py<PyAny>> {
    to_py(py, &load_manifest_file(&path)?)
}

/// The payload of a converter's run result. `laser_paths` is present for the
/// laser converters only.
#[derive(Serialize)]
struct RunOutput<'a, I, P> {
    input: &'a I,
    view_paths: &'a P,
    #[serde(skip_serializing_if = "Option::is_none")]
    laser_paths: Option<&'a P>,
    usable_views: usize,
    total_views: usize,
    rejected: Vec<RejectedOutput<'a>>,
}

/// A detection the runner left out, with its reason in snake case.
#[derive(Serialize)]
struct RejectedOutput<'a> {
    camera: usize,
    path: &'a Path,
    reason: &'static str,
}

fn rejected_output(rejected: &[runner::RejectedDetection]) -> Vec<RejectedOutput<'_>> {
    rejected
        .iter()
        .map(|r| RejectedOutput {
            camera: r.camera,
            path: &r.path,
            reason: reject_reason_name(r.reason),
        })
        .collect()
}

/// The Python name of a `RejectReason`: its Rust variant in snake case.
fn reject_reason_name(reason: runner::RejectReason) -> &'static str {
    match reason {
        runner::RejectReason::TooFewFeatures => "too_few_features",
        runner::RejectReason::PartialChessboard => "partial_chessboard",
        runner::RejectReason::TurnedChessboard => "turned_chessboard",
        runner::RejectReason::AmbiguousChessboardTurn => "ambiguous_chessboard_turn",
        // `RejectReason` is `#[non_exhaustive]`; a reason added upstream
        // shows as "other" until it is named here.
        _ => "other",
    }
}

/// Bind a converter whose run result carries `view_paths` and no laser
/// images.
macro_rules! target_converter {
    ($name:ident) => {
        #[doc = concat!("`dataset_runner::", stringify!($name), "`.")]
        #[pyfunction]
        #[pyo3(signature = (manifest, base_dir=None, cache=None, force_redetect=false, progress=None))]
        pub(crate) fn $name(
            py: Python<'_>,
            manifest: &Bound<'_, PyAny>,
            base_dir: Option<PathBuf>,
            cache: Option<PyRef<'_, PyFsDetectionCache>>,
            force_redetect: bool,
            progress: Option<Py<PyAny>>,
        ) -> PyResult<Py<PyAny>> {
            let (spec, base) = resolve_manifest(manifest, base_dir)?;
            let run = convert(py, cache, force_redetect, progress, |detection, control| {
                runner::$name(&spec, &base, detection, control)
            })?;
            to_py(
                py,
                &RunOutput {
                    input: &run.input,
                    view_paths: &run.view_paths,
                    laser_paths: None,
                    usable_views: run.usable_views,
                    total_views: run.total_views,
                    rejected: rejected_output(&run.rejected),
                },
            )
        }
    };
}

/// Bind a laser converter, whose run result also carries `laser_paths`.
macro_rules! laser_converter {
    ($name:ident) => {
        #[doc = concat!("`dataset_runner::", stringify!($name), "`.")]
        #[pyfunction]
        #[pyo3(signature = (manifest, laser_extractor, base_dir=None, cache=None, force_redetect=false, progress=None))]
        #[allow(clippy::too_many_arguments)] // The Python keyword arguments.
        pub(crate) fn $name(
            py: Python<'_>,
            manifest: &Bound<'_, PyAny>,
            laser_extractor: &Bound<'_, PyAny>,
            base_dir: Option<PathBuf>,
            cache: Option<PyRef<'_, PyFsDetectionCache>>,
            force_redetect: bool,
            progress: Option<Py<PyAny>>,
        ) -> PyResult<Py<PyAny>> {
            let (spec, base) = resolve_manifest(manifest, base_dir)?;
            let extractor = PyLaserExtractor::new(laser_extractor)?;
            let run = convert(py, cache, force_redetect, progress, |detection, control| {
                runner::$name(&spec, &base, detection, &extractor, control)
            })?;
            to_py(
                py,
                &RunOutput {
                    input: &run.input,
                    view_paths: &run.view_paths,
                    laser_paths: Some(&run.laser_paths),
                    usable_views: run.usable_views,
                    total_views: run.total_views,
                    rejected: rejected_output(&run.rejected),
                },
            )
        }
    };
}

target_converter!(build_planar_input);
target_converter!(build_single_cam_handeye_input);
target_converter!(build_rig_extrinsics_input);
target_converter!(build_rig_handeye_input);
laser_converter!(build_laserline_device_input);
laser_converter!(build_rig_laserline_device_input);
laser_converter!(build_rig_handeye_laserline_input);

/// Register the dataset functions on the extension module.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sniff_folder, m)?)?;
    m.add_function(wrap_pyfunction!(load_manifest, m)?)?;
    m.add_function(wrap_pyfunction!(camera_image_sizes, m)?)?;
    m.add_function(wrap_pyfunction!(vcal_main, m)?)?;
    m.add_function(wrap_pyfunction!(sample_catalogue, m)?)?;
    m.add_function(wrap_pyfunction!(fetch_sample, m)?)?;
    m.add_function(wrap_pyfunction!(sample_cache_dir, m)?)?;
    m.add_function(wrap_pyfunction!(build_planar_input, m)?)?;
    m.add_function(wrap_pyfunction!(build_single_cam_handeye_input, m)?)?;
    m.add_function(wrap_pyfunction!(build_rig_extrinsics_input, m)?)?;
    m.add_function(wrap_pyfunction!(build_rig_handeye_input, m)?)?;
    m.add_function(wrap_pyfunction!(build_laserline_device_input, m)?)?;
    m.add_function(wrap_pyfunction!(build_rig_laserline_device_input, m)?)?;
    m.add_function(wrap_pyfunction!(build_rig_handeye_laserline_input, m)?)?;
    Ok(())
}
