//! Conversion utilities between nalgebra types and numpy arrays.

use numpy::{PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use vision_calibration_core::{Mat3, Pt2, Pt3, Real, Vec3};
use vision_geometry::camera_matrix::Mat34;

/// Extract a list of `Pt2` from an (N, 2) numpy array.
pub fn numpy_to_pt2_list(arr: PyReadonlyArray2<'_, Real>) -> PyResult<Vec<Pt2>> {
    let shape = arr.shape();
    if shape[1] != 2 {
        return Err(PyValueError::new_err(format!(
            "expected (N, 2) array, got shape ({}, {})",
            shape[0], shape[1]
        )));
    }
    let n = shape[0];
    let mut pts = Vec::with_capacity(n);
    for i in 0..n {
        let x = *arr.get([i, 0]).unwrap();
        let y = *arr.get([i, 1]).unwrap();
        pts.push(Pt2::new(x, y));
    }
    Ok(pts)
}

/// Extract a list of `Pt3` from an (N, 3) numpy array.
pub fn numpy_to_pt3_list(arr: PyReadonlyArray2<'_, Real>) -> PyResult<Vec<Pt3>> {
    let shape = arr.shape();
    if shape[1] != 3 {
        return Err(PyValueError::new_err(format!(
            "expected (N, 3) array, got shape ({}, {})",
            shape[0], shape[1]
        )));
    }
    let n = shape[0];
    let mut pts = Vec::with_capacity(n);
    for i in 0..n {
        let x = *arr.get([i, 0]).unwrap();
        let y = *arr.get([i, 1]).unwrap();
        let z = *arr.get([i, 2]).unwrap();
        pts.push(Pt3::new(x, y, z));
    }
    Ok(pts)
}

/// Convert a `Mat3` to a (3, 3) numpy array.
pub fn mat3_to_numpy<'py>(py: Python<'py>, m: &Mat3) -> Bound<'py, PyArray2<Real>> {
    // nalgebra is column-major; emit row-major for numpy
    let mut data = Vec::with_capacity(9);
    for i in 0..3 {
        for j in 0..3 {
            data.push(m[(i, j)]);
        }
    }
    let arr = PyArray1::from_vec(py, data);
    arr.reshape([3, 3])
        .expect("reshape to (3, 3) should not fail")
}

/// Extract a `Mat3` from a (3, 3) numpy array.
pub fn numpy_to_mat3(arr: PyReadonlyArray2<'_, Real>) -> PyResult<Mat3> {
    let shape = arr.shape();
    if shape != [3, 3] {
        return Err(PyValueError::new_err(format!(
            "expected (3, 3) array, got shape ({}, {})",
            shape[0], shape[1]
        )));
    }
    let mut m = Mat3::zeros();
    for i in 0..3 {
        for j in 0..3 {
            m[(i, j)] = *arr.get([i, j]).unwrap();
        }
    }
    Ok(m)
}

/// Convert a `Mat34` to a (3, 4) numpy array.
pub fn mat34_to_numpy<'py>(py: Python<'py>, m: &Mat34) -> Bound<'py, PyArray2<Real>> {
    // nalgebra is column-major; emit row-major for numpy
    let mut data = Vec::with_capacity(12);
    for i in 0..3 {
        for j in 0..4 {
            data.push(m[(i, j)]);
        }
    }
    let arr = PyArray1::from_vec(py, data);
    arr.reshape([3, 4])
        .expect("reshape to (3, 4) should not fail")
}

/// Extract a `Mat34` from a (3, 4) numpy array.
pub fn numpy_to_mat34(arr: PyReadonlyArray2<'_, Real>) -> PyResult<Mat34> {
    let shape = arr.shape();
    if shape != [3, 4] {
        return Err(PyValueError::new_err(format!(
            "expected (3, 4) array, got shape ({}, {})",
            shape[0], shape[1]
        )));
    }
    let mut m = Mat34::zeros();
    for i in 0..3 {
        for j in 0..4 {
            m[(i, j)] = *arr.get([i, j]).unwrap();
        }
    }
    Ok(m)
}

/// Convert a `Vec3` to a (3,) numpy array.
pub fn vec3_to_numpy<'py>(py: Python<'py>, v: &Vec3) -> Bound<'py, PyArray1<Real>> {
    PyArray1::from_vec(py, vec![v.x, v.y, v.z])
}

/// Extract a `Vec3` from a (3,) numpy array.
pub fn numpy_to_vec3(arr: PyReadonlyArray1<'_, Real>) -> PyResult<Vec3> {
    let shape = arr.shape();
    if shape[0] != 3 {
        return Err(PyValueError::new_err(format!(
            "expected (3,) array, got shape ({})",
            shape[0]
        )));
    }
    Ok(Vec3::new(
        *arr.get(0).unwrap(),
        *arr.get(1).unwrap(),
        *arr.get(2).unwrap(),
    ))
}

/// Extract correspondences from an (N, 4) numpy array: columns [x1, y1, x2, y2].
pub fn numpy_to_correspondences(
    arr: PyReadonlyArray2<'_, Real>,
) -> PyResult<Vec<vision_mvg::Correspondence2D>> {
    let shape = arr.shape();
    if shape[1] != 4 {
        return Err(PyValueError::new_err(format!(
            "expected (N, 4) array, got shape ({}, {})",
            shape[0], shape[1]
        )));
    }
    let n = shape[0];
    let mut corrs = Vec::with_capacity(n);
    for i in 0..n {
        let pt1 = Pt2::new(*arr.get([i, 0]).unwrap(), *arr.get([i, 1]).unwrap());
        let pt2 = Pt2::new(*arr.get([i, 2]).unwrap(), *arr.get([i, 3]).unwrap());
        corrs.push(vision_mvg::Correspondence2D::new(pt1, pt2));
    }
    Ok(corrs)
}

/// Parse `RansacOptions` from a Python object (expects dataclass with named fields).
pub fn ransac_opts_from_py(obj: &Bound<'_, PyAny>) -> PyResult<vision_calibration_core::RansacOptions> {
    Ok(vision_calibration_core::RansacOptions {
        max_iters: obj.getattr("max_iters")?.extract()?,
        thresh: obj.getattr("thresh")?.extract()?,
        min_inliers: obj.getattr("min_inliers")?.extract()?,
        confidence: obj.getattr("confidence")?.extract()?,
        seed: obj.getattr("seed")?.extract()?,
        refit_on_inliers: obj.getattr("refit_on_inliers")?.extract()?,
    })
}
