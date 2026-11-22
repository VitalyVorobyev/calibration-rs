//! Target detection on numpy images and the filesystem detection cache.
//!
//! Features cross as arrays: `image_xy` `(N, 2)` pixels, `world_xyz` `(N, 3)`
//! target points in metres and, when the detector reported them, `cov_px`
//! `(N, 3)` corner covariances `[xx, xy, yy]` in px² (`None` otherwise), and
//! the measured blur as `blur_psf_px` `(N, 3)` PSF covariances `[xx, xy, yy]`
//! in px² with `blur_cov` `(N, 9)` their row-major 3×3 covariances in px⁴
//! (`None` when no feature has a blur; a feature without one is a row of
//! NaN in both).

use std::path::PathBuf;

use numpy::{PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use vision_calibration::core::PointBlur;
use vision_calibration::dataset::{DetectorSpec, TargetSpec};
use vision_calibration::detect::{
    CacheKey, CachedFeatures, DetectionCache, Feature, FeatureBlur, FsDetectionCache,
    GrayImageView, reject_ambiguous_detection as rust_reject,
};

use crate::errors::run_err;
use crate::session::{parse_checked, parse_optional_checked, to_py};
use crate::validation::{array_rows, value_err};

type FeatureArrays<'py> = (
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<f64>>,
    Option<Bound<'py, PyArray2<f64>>>,
    Option<Bound<'py, PyArray2<f64>>>,
    Option<Bound<'py, PyArray2<f64>>>,
);

/// Features as `(image_xy, world_xyz, cov_px, blur_psf_px, blur_cov)`
/// arrays. `cov_px` is present only when every feature carries a covariance,
/// the blur arrays when any feature carries a blur (NaN rows for the others).
fn to_arrays<'py>(py: Python<'py>, features: &[Feature]) -> PyResult<FeatureArrays<'py>> {
    let n = features.len();
    let image_xy: Vec<f64> = features.iter().flat_map(|f| f.image_xy).collect();
    let world_xyz: Vec<f64> = features.iter().flat_map(|f| f.world_xyz).collect();
    let cov_px = if n > 0 && features.iter().all(|f| f.cov_px.is_some()) {
        let cov: Vec<f64> = features
            .iter()
            .flat_map(|f| f.cov_px.unwrap_or_default())
            .collect();
        Some(PyArray1::from_vec(py, cov).reshape([n, 3])?)
    } else {
        None
    };
    let (blur_psf_px, blur_cov) = if features.iter().any(|f| f.blur.is_some()) {
        let psf: Vec<f64> = features
            .iter()
            .flat_map(|f| f.blur.map_or([f64::NAN; 3], |b| b.psf_px))
            .collect();
        let cov: Vec<f64> = features
            .iter()
            .flat_map(|f| {
                f.blur.map_or([f64::NAN; 9], |b| {
                    std::array::from_fn(|k| b.cov[k / 3][k % 3])
                })
            })
            .collect();
        (
            Some(PyArray1::from_vec(py, psf).reshape([n, 3])?),
            Some(PyArray1::from_vec(py, cov).reshape([n, 9])?),
        )
    } else {
        (None, None)
    };
    Ok((
        PyArray1::from_vec(py, image_xy).reshape([n, 2])?,
        PyArray1::from_vec(py, world_xyz).reshape([n, 3])?,
        cov_px,
        blur_psf_px,
        blur_cov,
    ))
}

/// The measured blur of each feature from `(N, 3)` PSF-covariance and
/// `(N, 9)` covariance arrays given together: a row of NaN in both is a
/// feature without one, any other row must be a valid blur.
fn blurs_from_arrays(
    n: usize,
    psf_px: Option<&PyReadonlyArray2<'_, f64>>,
    cov: Option<&PyReadonlyArray2<'_, f64>>,
) -> PyResult<Vec<Option<FeatureBlur>>> {
    let (psf_px, cov) = match (psf_px, cov) {
        (None, None) => return Ok(vec![None; n]),
        (Some(psf_px), Some(cov)) => (psf_px, cov),
        _ => {
            return Err(value_err(
                "blur_psf_px and blur_cov are given together or not at all",
            ));
        }
    };
    for (rows, name) in [
        (array_rows(psf_px, 3, "blur_psf_px")?, "blur_psf_px"),
        (array_rows(cov, 9, "blur_cov")?, "blur_cov"),
    ] {
        if rows != n {
            return Err(value_err(format!(
                "{name} must have one row per image_xy row ({n}), got {rows}"
            )));
        }
    }
    psf_px
        .as_slice()?
        .as_chunks::<3>()
        .0
        .iter()
        .zip(cov.as_slice()?.as_chunks::<9>().0)
        .enumerate()
        .map(|(i, (&psf, c))| {
            if psf.iter().chain(c).all(|v| v.is_nan()) {
                return Ok(None);
            }
            let c = [[c[0], c[1], c[2]], [c[3], c[4], c[5]], [c[6], c[7], c[8]]];
            PointBlur::new(psf, c)
                .map(|_| Some(FeatureBlur::new(psf, c)))
                .map_err(|e| value_err(format!("blur of feature {i}: {e}")))
        })
        .collect()
}

/// Features from `(N, 2)` pixel, `(N, 3)` target-point, optional `(N, 3)`
/// covariance and optional blur arrays.
fn from_arrays(
    image_xy: &PyReadonlyArray2<'_, f64>,
    world_xyz: &PyReadonlyArray2<'_, f64>,
    cov_px: Option<&PyReadonlyArray2<'_, f64>>,
    blur_psf_px: Option<&PyReadonlyArray2<'_, f64>>,
    blur_cov: Option<&PyReadonlyArray2<'_, f64>>,
) -> PyResult<Vec<Feature>> {
    let n = array_rows(image_xy, 2, "image_xy")?;
    let same_rows = |rows: usize, name: &str| {
        if rows == n {
            Ok(())
        } else {
            Err(value_err(format!(
                "{name} must have one row per image_xy row ({n}), got {rows}"
            )))
        }
    };
    same_rows(array_rows(world_xyz, 3, "world_xyz")?, "world_xyz")?;
    let covariances: Vec<Option<[f64; 3]>> = match cov_px {
        None => vec![None; n],
        Some(cov) => {
            same_rows(array_rows(cov, 3, "cov_px")?, "cov_px")?;
            cov.as_slice()?
                .as_chunks::<3>()
                .0
                .iter()
                .map(|&c| Some(c))
                .collect()
        }
    };
    let blurs = blurs_from_arrays(n, blur_psf_px, blur_cov)?;
    Ok(image_xy
        .as_slice()?
        .as_chunks::<2>()
        .0
        .iter()
        .zip(world_xyz.as_slice()?.as_chunks::<3>().0)
        .zip(covariances)
        .zip(blurs)
        .map(|(((&image_xy, &world_xyz), cov_px), blur)| {
            Feature::new(image_xy, world_xyz)
                .with_cov_px(cov_px)
                .with_blur(blur)
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
    let width = u32::try_from(w).map_err(|_| value_err("image is too wide"))?;
    let height = u32::try_from(h).map_err(|_| value_err("image is too tall"))?;
    let pixels = image.as_slice()?.to_vec();
    let view = GrayImageView::new(width, height, &pixels).map_err(|e| value_err(e.to_string()))?;
    let features = py
        .detach(|| {
            vision_calibration::dataset_runner::detect_target(view, &target, detector.as_ref())
        })
        .map_err(|err| run_err(py, err))?;
    to_arrays(py, &features)
}

/// Drop a detection whose image points are not distinct (all or nothing).
#[pyfunction]
#[pyo3(signature = (image_xy, world_xyz, cov_px=None, blur_psf_px=None, blur_cov=None))]
pub(crate) fn reject_ambiguous_detection<'py>(
    py: Python<'py>,
    image_xy: PyReadonlyArray2<'py, f64>,
    world_xyz: PyReadonlyArray2<'py, f64>,
    cov_px: Option<PyReadonlyArray2<'py, f64>>,
    blur_psf_px: Option<PyReadonlyArray2<'py, f64>>,
    blur_cov: Option<PyReadonlyArray2<'py, f64>>,
) -> PyResult<FeatureArrays<'py>> {
    let features = from_arrays(
        &image_xy,
        &world_xyz,
        cov_px.as_ref(),
        blur_psf_px.as_ref(),
        blur_cov.as_ref(),
    )?;
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
    #[pyo3(signature = (key, image_xy, world_xyz, cov_px=None, blur_psf_px=None, blur_cov=None))]
    fn put(
        &self,
        key: &Bound<'_, PyAny>,
        image_xy: PyReadonlyArray2<'_, f64>,
        world_xyz: PyReadonlyArray2<'_, f64>,
        cov_px: Option<PyReadonlyArray2<'_, f64>>,
        blur_psf_px: Option<PyReadonlyArray2<'_, f64>>,
        blur_cov: Option<PyReadonlyArray2<'_, f64>>,
    ) -> PyResult<()> {
        let key: CacheKey = parse_checked(key, "key")?;
        let features = from_arrays(
            &image_xy,
            &world_xyz,
            cov_px.as_ref(),
            blur_psf_px.as_ref(),
            blur_cov.as_ref(),
        )?;
        self.inner
            .put(&key, &CachedFeatures::new(features))
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
