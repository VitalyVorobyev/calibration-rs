//! Manifest-driven runs: the dataset sniffer, the manifest loader and the
//! `build_*_input` converters of the dataset runner, plus the bridge that
//! lets a Python object serve as the runner's laser-line extractor.

use std::path::{Path, PathBuf};

use numpy::{PyArray1, PyArrayMethods, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::prelude::*;
use serde::Serialize;
use vision_calibration::dataset::{DatasetSpec, LaserExtractionSpec, sniff_folder as rust_sniff};
use vision_calibration::dataset_runner::{self as runner, LaserPixelExtractor};
use vision_calibration::detect::{CacheError, CacheKey, CachedFeatures, DetectionCache};

use crate::detect::PyFsDetectionCache;
use crate::errors::{run_err, sniff_err};
use crate::parse_payload;
use crate::session::to_py;
use crate::validation::value_err;

/// Read a manifest file: TOML for a `.toml` extension, JSON otherwise. The
/// error is the message to raise.
pub(crate) fn read_manifest(path: &Path) -> Result<DatasetSpec, String> {
    let text = std::fs::read_to_string(path)
        .map_err(|err| format!("cannot read manifest {}: {err}", path.display()))?;
    let is_toml = path
        .extension()
        .is_some_and(|ext| ext.eq_ignore_ascii_case("toml"));
    let spec = if is_toml {
        toml::from_str(&text).map_err(|err| err.to_string())
    } else {
        serde_json::from_str(&text).map_err(|err| err.to_string())
    };
    spec.map_err(|err| format!("invalid manifest {}: {err}", path.display()))
}

fn load_manifest_file(path: &Path) -> PyResult<DatasetSpec> {
    read_manifest(path).map_err(value_err)
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

/// A cache that never hits: a run without a cache detects every image and
/// writes nothing.
struct NoCache;

impl DetectionCache for NoCache {
    fn get(&self, _key: &CacheKey) -> Result<Option<CachedFeatures>, CacheError> {
        Ok(None)
    }

    fn put(&self, _key: &CacheKey, _value: &CachedFeatures) -> Result<(), CacheError> {
        Ok(())
    }
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
        image: &image::DynamicImage,
        spec: &LaserExtractionSpec,
    ) -> PyResult<Vec<[f64; 2]>> {
        Python::attach(|py| {
            let gray = image.to_luma8();
            let (w, h) = (gray.width() as usize, gray.height() as usize);
            let array = PyArray1::from_vec(py, gray.into_raw()).reshape([h, w])?;
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
        image: &image::DynamicImage,
        spec: &LaserExtractionSpec,
    ) -> Result<Vec<[f64; 2]>, Box<dyn std::error::Error + Send + Sync>> {
        self.call(image, spec).map_err(|err| err.to_string().into())
    }
}

/// Run a converter with the GIL released, against the given cache (or none).
fn convert<T: Send>(
    py: Python<'_>,
    cache: Option<PyRef<'_, PyFsDetectionCache>>,
    build: impl FnOnce(&dyn DetectionCache) -> Result<T, runner::RunError> + Send,
) -> PyResult<T> {
    let result = match &cache {
        Some(cache) => {
            let cache = cache.inner();
            py.detach(|| build(cache))
        }
        None => py.detach(|| build(&NoCache)),
    };
    result.map_err(|err| run_err(py, err))
}

/// Inspect a dataset folder and return a best-effort `DatasetSpec` payload.
#[pyfunction]
pub(crate) fn sniff_folder(py: Python<'_>, root: PathBuf) -> PyResult<Py<PyAny>> {
    let spec = rust_sniff(&root).map_err(|err| sniff_err(py, err))?;
    to_py(py, &spec)
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
        #[pyo3(signature = (manifest, base_dir=None, cache=None, force_redetect=false))]
        pub(crate) fn $name(
            py: Python<'_>,
            manifest: &Bound<'_, PyAny>,
            base_dir: Option<PathBuf>,
            cache: Option<PyRef<'_, PyFsDetectionCache>>,
            force_redetect: bool,
        ) -> PyResult<Py<PyAny>> {
            let (spec, base) = resolve_manifest(manifest, base_dir)?;
            let run = convert(py, cache, |cache| {
                runner::$name(&spec, &base, cache, force_redetect)
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
        #[pyo3(signature = (manifest, laser_extractor, base_dir=None, cache=None, force_redetect=false))]
        pub(crate) fn $name(
            py: Python<'_>,
            manifest: &Bound<'_, PyAny>,
            laser_extractor: &Bound<'_, PyAny>,
            base_dir: Option<PathBuf>,
            cache: Option<PyRef<'_, PyFsDetectionCache>>,
            force_redetect: bool,
        ) -> PyResult<Py<PyAny>> {
            let (spec, base) = resolve_manifest(manifest, base_dir)?;
            let extractor = PyLaserExtractor::new(laser_extractor)?;
            let run = convert(py, cache, |cache| {
                runner::$name(&spec, &base, cache, &extractor, force_redetect)
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
    m.add_function(wrap_pyfunction!(build_planar_input, m)?)?;
    m.add_function(wrap_pyfunction!(build_single_cam_handeye_input, m)?)?;
    m.add_function(wrap_pyfunction!(build_rig_extrinsics_input, m)?)?;
    m.add_function(wrap_pyfunction!(build_rig_handeye_input, m)?)?;
    m.add_function(wrap_pyfunction!(build_laserline_device_input, m)?)?;
    m.add_function(wrap_pyfunction!(build_rig_laserline_device_input, m)?)?;
    m.add_function(wrap_pyfunction!(build_rig_handeye_laserline_input, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loads_toml_and_json_manifests() {
        let dir = std::env::temp_dir().join(format!("vc-py-manifest-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let toml_text = r#"
version = 1
topology = "planar_intrinsics"

[[cameras]]
id = "cam0"
images = { kind = "glob", pattern = "*.png" }

[target]
kind = "chessboard"
rows = 5
cols = 7
square_size_m = 0.03
"#;
        let toml_path = dir.join("dataset.toml");
        std::fs::write(&toml_path, toml_text).unwrap();
        let from_toml = read_manifest(&toml_path).unwrap();
        let json_path = dir.join("dataset.json");
        std::fs::write(&json_path, serde_json::to_string(&from_toml).unwrap()).unwrap();
        let from_json = read_manifest(&json_path).unwrap();
        assert_eq!(
            serde_json::to_value(&from_toml).unwrap(),
            serde_json::to_value(&from_json).unwrap()
        );
        assert!(read_manifest(&dir.join("missing.toml")).is_err());
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
