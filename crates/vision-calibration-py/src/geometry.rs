//! PyO3 wrappers for `vision-geometry` low-level solvers.

use numpy::{PyArray1, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;
use vision_calibration_core::Real;

use crate::convert::*;

/// 5-point essential matrix solver (Nister). Input: calibrated (N=5, 2) arrays.
#[pyfunction]
fn essential_5point<'py>(
    py: Python<'py>,
    pts1: PyReadonlyArray2<'py, Real>,
    pts2: PyReadonlyArray2<'py, Real>,
) -> PyResult<Vec<Bound<'py, PyArray2<Real>>>> {
    let p1 = numpy_to_pt2_list(pts1)?;
    let p2 = numpy_to_pt2_list(pts2)?;
    let results =
        vision_geometry::epipolar::essential_5point(&p1, &p2).map_err(crate::runtime_err)?;
    Ok(results.iter().map(|m| mat3_to_numpy(py, m)).collect())
}

/// 7-point fundamental matrix solver. Input: pixel (N=7, 2) arrays.
#[pyfunction]
fn fundamental_7point<'py>(
    py: Python<'py>,
    pts1: PyReadonlyArray2<'py, Real>,
    pts2: PyReadonlyArray2<'py, Real>,
) -> PyResult<Vec<Bound<'py, PyArray2<Real>>>> {
    let p1 = numpy_to_pt2_list(pts1)?;
    let p2 = numpy_to_pt2_list(pts2)?;
    let results =
        vision_geometry::epipolar::fundamental_7point(&p1, &p2).map_err(crate::runtime_err)?;
    Ok(results.iter().map(|m| mat3_to_numpy(py, m)).collect())
}

/// Normalized 8-point fundamental matrix solver. Input: pixel (N>=8, 2) arrays.
#[pyfunction]
fn fundamental_8point<'py>(
    py: Python<'py>,
    pts1: PyReadonlyArray2<'py, Real>,
    pts2: PyReadonlyArray2<'py, Real>,
) -> PyResult<Bound<'py, PyArray2<Real>>> {
    let p1 = numpy_to_pt2_list(pts1)?;
    let p2 = numpy_to_pt2_list(pts2)?;
    let f = vision_geometry::epipolar::fundamental_8point(&p1, &p2).map_err(crate::runtime_err)?;
    Ok(mat3_to_numpy(py, &f))
}

/// RANSAC 8-point fundamental matrix. Returns (F, inlier_indices).
#[pyfunction]
fn fundamental_8point_ransac<'py>(
    py: Python<'py>,
    pts1: PyReadonlyArray2<'py, Real>,
    pts2: PyReadonlyArray2<'py, Real>,
    opts: &Bound<'py, PyAny>,
) -> PyResult<(Bound<'py, PyArray2<Real>>, Vec<usize>)> {
    let p1 = numpy_to_pt2_list(pts1)?;
    let p2 = numpy_to_pt2_list(pts2)?;
    let ransac_opts = ransac_opts_from_py(opts)?;
    let (f, inliers) = vision_geometry::epipolar::fundamental_8point_ransac(&p1, &p2, &ransac_opts)
        .map_err(crate::runtime_err)?;
    Ok((mat3_to_numpy(py, &f), inliers))
}

/// Decompose essential matrix into 4 candidate (R, t) pairs.
#[pyfunction]
#[allow(clippy::type_complexity)]
fn decompose_essential<'py>(
    py: Python<'py>,
    e: PyReadonlyArray2<'py, Real>,
) -> PyResult<Vec<(Bound<'py, PyArray2<Real>>, Bound<'py, PyArray1<Real>>)>> {
    let e_mat = numpy_to_mat3(e)?;
    let decomps =
        vision_geometry::epipolar::decompose_essential(&e_mat).map_err(crate::runtime_err)?;
    Ok(decomps
        .iter()
        .map(|(r, t)| (mat3_to_numpy(py, r), vec3_to_numpy(py, t)))
        .collect())
}

/// Normalized DLT homography estimation. Input: (N>=4, 2) arrays.
#[pyfunction]
fn dlt_homography<'py>(
    py: Python<'py>,
    src: PyReadonlyArray2<'py, Real>,
    dst: PyReadonlyArray2<'py, Real>,
) -> PyResult<Bound<'py, PyArray2<Real>>> {
    let s = numpy_to_pt2_list(src)?;
    let d = numpy_to_pt2_list(dst)?;
    let h = vision_geometry::homography::dlt_homography(&s, &d).map_err(crate::runtime_err)?;
    Ok(mat3_to_numpy(py, &h))
}

/// RANSAC DLT homography. Returns (H, inlier_indices).
#[pyfunction]
fn dlt_homography_ransac<'py>(
    py: Python<'py>,
    src: PyReadonlyArray2<'py, Real>,
    dst: PyReadonlyArray2<'py, Real>,
    opts: &Bound<'py, PyAny>,
) -> PyResult<(Bound<'py, PyArray2<Real>>, Vec<usize>)> {
    let s = numpy_to_pt2_list(src)?;
    let d = numpy_to_pt2_list(dst)?;
    let ransac_opts = ransac_opts_from_py(opts)?;
    let (h, inliers) = vision_geometry::homography::dlt_homography_ransac(&s, &d, &ransac_opts)
        .map_err(crate::runtime_err)?;
    Ok((mat3_to_numpy(py, &h), inliers))
}

/// DLT camera matrix estimation. Input: (N>=6, 3) world, (N, 2) image.
#[pyfunction]
fn dlt_camera_matrix<'py>(
    py: Python<'py>,
    world: PyReadonlyArray2<'py, Real>,
    image: PyReadonlyArray2<'py, Real>,
) -> PyResult<Bound<'py, PyArray2<Real>>> {
    let w = numpy_to_pt3_list(world)?;
    let i = numpy_to_pt2_list(image)?;
    let p =
        vision_geometry::camera_matrix::dlt_camera_matrix(&w, &i).map_err(crate::runtime_err)?;
    Ok(mat34_to_numpy(py, &p))
}

/// Decompose a 3x4 camera matrix into K, R, t.
#[pyfunction]
fn decompose_camera_matrix<'py>(
    py: Python<'py>,
    p: PyReadonlyArray2<'py, Real>,
) -> PyResult<Py<PyAny>> {
    let p_mat = numpy_to_mat34(p)?;
    let decomp = vision_geometry::camera_matrix::decompose_camera_matrix(&p_mat)
        .map_err(crate::runtime_err)?;
    let dict = pyo3::types::PyDict::new(py);
    dict.set_item("k", mat3_to_numpy(py, &decomp.k))?;
    dict.set_item("r", mat3_to_numpy(py, &decomp.r))?;
    dict.set_item("t", vec3_to_numpy(py, &decomp.t))?;
    Ok(dict.into())
}

/// Linear DLT triangulation from multiple views.
#[pyfunction]
fn triangulate_point_linear<'py>(
    py: Python<'py>,
    cameras: Vec<PyReadonlyArray2<'py, Real>>,
    points: Vec<(Real, Real)>,
) -> PyResult<(Real, Real, Real)> {
    let cams: Vec<_> = cameras
        .into_iter()
        .map(|c| numpy_to_mat34(c))
        .collect::<PyResult<_>>()?;
    let pts: Vec<_> = points
        .into_iter()
        .map(|(x, y)| vision_calibration_core::Pt2::new(x, y))
        .collect();
    let pt = vision_geometry::triangulation::triangulate_point_linear(&cams, &pts)
        .map_err(crate::runtime_err)?;
    let _ = py;
    Ok((pt.x, pt.y, pt.z))
}

/// Register geometry functions on the given module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(essential_5point, m)?)?;
    m.add_function(wrap_pyfunction!(fundamental_7point, m)?)?;
    m.add_function(wrap_pyfunction!(fundamental_8point, m)?)?;
    m.add_function(wrap_pyfunction!(fundamental_8point_ransac, m)?)?;
    m.add_function(wrap_pyfunction!(decompose_essential, m)?)?;
    m.add_function(wrap_pyfunction!(dlt_homography, m)?)?;
    m.add_function(wrap_pyfunction!(dlt_homography_ransac, m)?)?;
    m.add_function(wrap_pyfunction!(dlt_camera_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(decompose_camera_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(triangulate_point_linear, m)?)?;
    Ok(())
}
