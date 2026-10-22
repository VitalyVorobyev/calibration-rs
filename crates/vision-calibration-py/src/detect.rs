//! Target detection on numpy images and the filesystem detection cache.
//!
//! Features cross as two arrays: `image_xy` `(N, 2)` pixels and `world_xyz`
//! `(N, 3)` target points in metres.

use std::path::PathBuf;

use numpy::{PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use vision_calibration::dataset::{DetectorSpec, TargetSpec};
use vision_calibration::detect::{
    CacheKey, CachedFeatures, DetectionCache, Feature, FsDetectionCache,
    reject_ambiguous_detection as rust_reject,
};

use crate::errors::run_err;
use crate::session::{parse_checked, parse_optional_checked, to_py};
use crate::validation::value_err;

type FeatureArrays<'py> = (Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<f64>>);

/// Features as `(image_xy, world_xyz)` arrays.
fn to_arrays<'py>(py: Python<'py>, features: &[Feature]) -> PyResult<FeatureArrays<'py>> {
    let n = features.len();
    let image_xy: Vec<f64> = features.iter().flat_map(|f| f.image_xy).collect();
    let world_xyz: Vec<f64> = features.iter().flat_map(|f| f.world_xyz).collect();
    Ok((
        PyArray1::from_vec(py, image_xy).reshape([n, 2])?,
        PyArray1::from_vec(py, world_xyz).reshape([n, 3])?,
    ))
}

/// Features from `(N, 2)` pixel and `(N, 3)` target-point arrays.
fn from_arrays(
    image_xy: &PyReadonlyArray2<'_, f64>,
    world_xyz: &PyReadonlyArray2<'_, f64>,
) -> PyResult<Vec<Feature>> {
    let (a, b) = (image_xy.shape(), world_xyz.shape());
    if a[1] != 2 || b[1] != 3 || a[0] != b[0] {
        return Err(value_err(format!(
            "features need image_xy (N, 2) and world_xyz (N, 3), got {a:?} and {b:?}"
        )));
    }
    Ok(image_xy
        .as_slice()?
        .as_chunks::<2>()
        .0
        .iter()
        .zip(world_xyz.as_slice()?.as_chunks::<3>().0)
        .map(|(&image_xy, &world_xyz)| Feature {
            image_xy,
            world_xyz,
        })
        .collect())
}

fn cache_err(err: impl std::fmt::Display) -> PyErr {
    PyRuntimeError::new_err(format!("detection cache: {err}"))
}

/// Detect a manifest target in a `(H, W)` `uint8` grayscale image.
#[pyfunction]
#[pyo3(signature = (image, target, detector=None))]
pub(crate) fn detect_target<'py>(
    py: Python<'py>,
    image: PyReadonlyArray2<'py, u8>,
    target: &Bound<'py, PyAny>,
    detector: Option<&Bound<'py, PyAny>>,
) -> PyResult<FeatureArrays<'py>> {
    let target: TargetSpec = parse_checked(target, "target")?;
    let detector: Option<DetectorSpec> = parse_optional_checked(detector, "detector")?;
    let shape = image.shape();
    let (h, w) = (shape[0], shape[1]);
    let pixels = image.as_slice()?.to_vec();
    let gray = image::GrayImage::from_raw(
        u32::try_from(w).map_err(|_| value_err("image is too wide"))?,
        u32::try_from(h).map_err(|_| value_err("image is too tall"))?,
        pixels,
    )
    .ok_or_else(|| value_err("image buffer does not match its shape"))?;
    let image = image::DynamicImage::ImageLuma8(gray);
    let features = py
        .detach(|| {
            vision_calibration::dataset_runner::detect_target(&image, &target, detector.as_ref())
        })
        .map_err(|err| run_err(py, err))?;
    to_arrays(py, &features)
}

/// Drop a detection whose image points are not distinct (all or nothing).
#[pyfunction]
pub(crate) fn reject_ambiguous_detection<'py>(
    py: Python<'py>,
    image_xy: PyReadonlyArray2<'py, f64>,
    world_xyz: PyReadonlyArray2<'py, f64>,
) -> PyResult<FeatureArrays<'py>> {
    let features = from_arrays(&image_xy, &world_xyz)?;
    to_arrays(py, &rust_reject(features))
}

/// The cache key of an image under a detector and its config.
#[pyfunction]
pub(crate) fn cache_key_from_inputs(
    py: Python<'_>,
    image_bytes: &[u8],
    detector: &str,
    config: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    let config: serde_json::Value = parse_checked(config, "config")?;
    to_py(py, &CacheKey::from_inputs(image_bytes, detector, &config))
}

/// The file name of a cache key's entry.
#[pyfunction]
pub(crate) fn cache_key_file_name(key: &Bound<'_, PyAny>) -> PyResult<String> {
    let key: CacheKey = parse_checked(key, "key")?;
    Ok(key.file_name())
}

/// Filesystem detection cache: one JSON file per entry under `root`.
#[pyclass(
    module = "vision_calibration._vision_calibration",
    name = "FsDetectionCache",
    frozen
)]
pub(crate) struct PyFsDetectionCache {
    inner: FsDetectionCache,
}

impl PyFsDetectionCache {
    pub(crate) fn inner(&self) -> &FsDetectionCache {
        &self.inner
    }
}

#[pymethods]
impl PyFsDetectionCache {
    #[new]
    fn new(root: PathBuf) -> Self {
        Self {
            inner: FsDetectionCache::new(root),
        }
    }

    /// The cache's directory.
    #[getter]
    fn root(&self) -> PathBuf {
        self.inner.root().to_path_buf()
    }

    /// The features cached under `key`, or `None` on a miss.
    fn get<'py>(
        &self,
        py: Python<'py>,
        key: &Bound<'py, PyAny>,
    ) -> PyResult<Option<FeatureArrays<'py>>> {
        let key: CacheKey = parse_checked(key, "key")?;
        match self.inner.get(&key).map_err(cache_err)? {
            Some(entry) => Ok(Some(to_arrays(py, &entry.features)?)),
            None => Ok(None),
        }
    }

    /// Store features under `key`.
    fn put(
        &self,
        key: &Bound<'_, PyAny>,
        image_xy: PyReadonlyArray2<'_, f64>,
        world_xyz: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<()> {
        let key: CacheKey = parse_checked(key, "key")?;
        let features = from_arrays(&image_xy, &world_xyz)?;
        self.inner
            .put(&key, &CachedFeatures { features })
            .map_err(cache_err)
    }
}

/// Register the detection functions and the cache class.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(detect_target, m)?)?;
    m.add_function(wrap_pyfunction!(reject_ambiguous_detection, m)?)?;
    m.add_function(wrap_pyfunction!(cache_key_from_inputs, m)?)?;
    m.add_function(wrap_pyfunction!(cache_key_file_name, m)?)?;
    m.add_class::<PyFsDetectionCache>()?;
    Ok(())
}
