//! PyO3 wrappers for `vision-mvg` multi-view geometry functions.

use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use vision_calibration_core::Real;

use crate::convert::*;

/// Recover relative pose from calibrated correspondences.
/// Input: (N>=5, 4) array with columns [x1, y1, x2, y2].
#[pyfunction]
fn recover_relative_pose<'py>(
    py: Python<'py>,
    corrs: PyReadonlyArray2<'py, Real>,
) -> PyResult<Py<PyAny>> {
    let c = numpy_to_correspondences(corrs)?;
    let result = vision_mvg::recover_relative_pose(&c).map_err(crate::runtime_err)?;
    let dict = pyo3::types::PyDict::new(py);
    dict.set_item("r", mat3_to_numpy(py, &result.r))?;
    dict.set_item("t", vec3_to_numpy(py, &result.t))?;
    dict.set_item("essential", mat3_to_numpy(py, &result.essential))?;
    let points: Vec<Py<PyAny>> = result
        .points
        .iter()
        .map(|tp| {
            let d = pyo3::types::PyDict::new(py);
            d.set_item("point", (tp.point.x, tp.point.y, tp.point.z))
                .unwrap();
            d.set_item("reprojection_error", tp.reprojection_error)
                .unwrap();
            d.set_item("parallax_deg", tp.parallax_deg).unwrap();
            d.set_item("in_front", tp.in_front).unwrap();
            d.into()
        })
        .collect();
    dict.set_item("points", points)?;
    Ok(dict.into())
}

/// Robust relative pose recovery using RANSAC.
/// Input: (N, 4) correspondences array, RansacOptions dataclass.
#[pyfunction]
fn recover_relative_pose_robust<'py>(
    py: Python<'py>,
    corrs: PyReadonlyArray2<'py, Real>,
    opts: &Bound<'py, PyAny>,
) -> PyResult<Py<PyAny>> {
    let c = numpy_to_correspondences(corrs)?;
    let ransac_opts = ransac_opts_from_py(opts)?;
    let result =
        vision_mvg::recover_relative_pose_robust(&c, &ransac_opts).map_err(crate::runtime_err)?;
    let dict = pyo3::types::PyDict::new(py);
    dict.set_item("r", mat3_to_numpy(py, &result.r))?;
    dict.set_item("t", vec3_to_numpy(py, &result.t))?;
    dict.set_item("essential", mat3_to_numpy(py, &result.essential))?;
    dict.set_item("inliers", &result.inliers)?;
    dict.set_item("inlier_rms", result.inlier_rms)?;
    Ok(dict.into())
}

/// RANSAC essential matrix estimation.
#[pyfunction]
fn estimate_essential<'py>(
    py: Python<'py>,
    corrs: PyReadonlyArray2<'py, Real>,
    opts: &Bound<'py, PyAny>,
) -> PyResult<Py<PyAny>> {
    let c = numpy_to_correspondences(corrs)?;
    let ransac_opts = ransac_opts_from_py(opts)?;
    let est = vision_mvg::estimate_essential(&c, &ransac_opts).map_err(crate::runtime_err)?;
    let dict = pyo3::types::PyDict::new(py);
    dict.set_item("essential", mat3_to_numpy(py, &est.essential))?;
    dict.set_item("inliers", &est.inliers)?;
    dict.set_item("inlier_rms", est.inlier_rms)?;
    Ok(dict.into())
}

/// RANSAC homography estimation.
#[pyfunction]
fn estimate_homography<'py>(
    py: Python<'py>,
    corrs: PyReadonlyArray2<'py, Real>,
    opts: &Bound<'py, PyAny>,
) -> PyResult<Py<PyAny>> {
    let c = numpy_to_correspondences(corrs)?;
    let ransac_opts = ransac_opts_from_py(opts)?;
    let est = vision_mvg::estimate_homography(&c, &ransac_opts).map_err(crate::runtime_err)?;
    let dict = pyo3::types::PyDict::new(py);
    dict.set_item("homography", mat3_to_numpy(py, &est.homography))?;
    dict.set_item("inliers", &est.inliers)?;
    dict.set_item("inlier_rms", est.inlier_rms)?;
    Ok(dict.into())
}

/// Decompose homography into candidate (R, t, normal) tuples.
#[pyfunction]
fn decompose_homography<'py>(
    py: Python<'py>,
    h: PyReadonlyArray2<'py, Real>,
) -> PyResult<Vec<Py<PyAny>>> {
    let h_mat = numpy_to_mat3(h)?;
    let decomps = vision_mvg::decompose_homography(&h_mat).map_err(crate::runtime_err)?;
    Ok(decomps
        .iter()
        .map(|d| {
            let dict = pyo3::types::PyDict::new(py);
            dict.set_item("r", mat3_to_numpy(py, &d.r)).unwrap();
            dict.set_item("t", vec3_to_numpy(py, &d.t)).unwrap();
            dict.set_item("normal", vec3_to_numpy(py, &d.normal))
                .unwrap();
            dict.into()
        })
        .collect())
}

/// Apply homography to a 2D point. Returns (x', y').
#[pyfunction]
fn homography_transfer(h: PyReadonlyArray2<'_, Real>, pt: (Real, Real)) -> PyResult<(Real, Real)> {
    let h_mat = numpy_to_mat3(h)?;
    let p = vision_calibration_core::Pt2::new(pt.0, pt.1);
    let result = vision_mvg::homography_transfer(&h_mat, &p);
    Ok((result.x, result.y))
}

/// Triangulate points from two calibrated views.
/// Returns list of dicts with keys: point, reprojection_error, parallax_deg, in_front.
#[pyfunction]
fn triangulate_two_view<'py>(
    py: Python<'py>,
    r: PyReadonlyArray2<'py, Real>,
    t: PyReadonlyArray1<'py, Real>,
    pts1: PyReadonlyArray2<'py, Real>,
    pts2: PyReadonlyArray2<'py, Real>,
) -> PyResult<Vec<Py<PyAny>>> {
    let r_mat = numpy_to_mat3(r)?;
    let t_vec = numpy_to_vec3(t)?;
    let p1 = numpy_to_pt2_list(pts1)?;
    let p2 = numpy_to_pt2_list(pts2)?;
    let tps = vision_mvg::triangulation::triangulate_two_view(&r_mat, &t_vec, &p1, &p2)
        .map_err(crate::runtime_err)?;
    Ok(tps
        .iter()
        .map(|tp| {
            let d = pyo3::types::PyDict::new(py);
            d.set_item("point", (tp.point.x, tp.point.y, tp.point.z))
                .unwrap();
            d.set_item("reprojection_error", tp.reprojection_error)
                .unwrap();
            d.set_item("parallax_deg", tp.parallax_deg).unwrap();
            d.set_item("in_front", tp.in_front).unwrap();
            d.into()
        })
        .collect())
}

/// Analyze a two-view scene for degeneracies.
#[pyfunction]
fn analyze_scene<'py>(
    py: Python<'py>,
    corrs: PyReadonlyArray2<'py, Real>,
    e: PyReadonlyArray2<'py, Real>,
    r: PyReadonlyArray2<'py, Real>,
    t: PyReadonlyArray1<'py, Real>,
) -> PyResult<Py<PyAny>> {
    let c = numpy_to_correspondences(corrs)?;
    let e_mat = numpy_to_mat3(e)?;
    let r_mat = numpy_to_mat3(r)?;
    let t_vec = numpy_to_vec3(t)?;
    let diag = vision_mvg::degeneracy::analyze_scene(&c, &e_mat, &r_mat, &t_vec);
    let dict = pyo3::types::PyDict::new(py);
    dict.set_item("median_parallax_deg", diag.median_parallax_deg)?;
    dict.set_item("is_pure_rotation", diag.is_pure_rotation)?;
    dict.set_item("is_planar", diag.is_planar)?;
    dict.set_item("baseline_ratio", diag.baseline_ratio)?;
    Ok(dict.into())
}

/// Sampson distance between a fundamental/essential matrix and a correspondence.
#[pyfunction]
fn sampson_distance(
    f: PyReadonlyArray2<'_, Real>,
    pt1: (Real, Real),
    pt2: (Real, Real),
) -> PyResult<Real> {
    let f_mat = numpy_to_mat3(f)?;
    let c = vision_mvg::Correspondence2D::new(
        vision_calibration_core::Pt2::new(pt1.0, pt1.1),
        vision_calibration_core::Pt2::new(pt2.0, pt2.1),
    );
    Ok(vision_mvg::residuals::sampson_distance(&f_mat, &c))
}

/// Symmetric transfer error for a homography and a correspondence.
#[pyfunction]
fn symmetric_transfer_error(
    h: PyReadonlyArray2<'_, Real>,
    pt1: (Real, Real),
    pt2: (Real, Real),
) -> PyResult<Real> {
    let h_mat = numpy_to_mat3(h)?;
    let c = vision_mvg::Correspondence2D::new(
        vision_calibration_core::Pt2::new(pt1.0, pt1.1),
        vision_calibration_core::Pt2::new(pt2.0, pt2.1),
    );
    Ok(vision_mvg::residuals::symmetric_transfer_error(&h_mat, &c))
}

/// Register MVG functions on the given module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(recover_relative_pose, m)?)?;
    m.add_function(wrap_pyfunction!(recover_relative_pose_robust, m)?)?;
    m.add_function(wrap_pyfunction!(estimate_essential, m)?)?;
    m.add_function(wrap_pyfunction!(estimate_homography, m)?)?;
    m.add_function(wrap_pyfunction!(decompose_homography, m)?)?;
    m.add_function(wrap_pyfunction!(homography_transfer, m)?)?;
    m.add_function(wrap_pyfunction!(triangulate_two_view, m)?)?;
    m.add_function(wrap_pyfunction!(analyze_scene, m)?)?;
    m.add_function(wrap_pyfunction!(sampson_distance, m)?)?;
    m.add_function(wrap_pyfunction!(symmetric_transfer_error, m)?)?;
    Ok(())
}
