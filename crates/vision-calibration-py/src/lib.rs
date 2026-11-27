//! PyO3 extension module (`_vision_calibration`) exposing the
//! `vision-calibration` Rust API to Python.

use pyo3::exceptions::{PyRuntimeError, PyTypeError};
use pyo3::prelude::*;
use pythonize::{depythonize, pythonize};
use serde::de::DeserializeOwned;

mod camera;
mod control;
mod dataset;
mod detect;
mod device_seed;
mod errors;
mod mvg;
mod project;
mod session;
mod validation;
use validation::{reject_non_finite, value_err};

pub(crate) fn runtime_err(message: impl std::fmt::Display) -> PyErr {
    PyRuntimeError::new_err(message.to_string())
}

pub(crate) fn parse_payload<T>(payload: &Bound<'_, PyAny>, name: &str) -> PyResult<T>
where
    T: DeserializeOwned,
{
    depythonize(payload)
        .map_err(|err| PyTypeError::new_err(format!("invalid {name} payload: {err}")))
}

/// The cameras of a calibration export payload, by its `kind`.
pub(crate) fn parse_cameras(
    payload: &Bound<'_, PyAny>,
    name: &str,
) -> PyResult<Box<dyn vision_calibration::common::CalibratedCameras>> {
    let value: serde_json::Value = parse_payload(payload, name)?;
    vision_calibration::common::calibrated_cameras(&value.to_string())
        .map_err(|err| PyTypeError::new_err(format!("invalid {name} payload: {err}")))
}

pub(crate) fn parse_optional_payload<T>(
    payload: Option<&Bound<'_, PyAny>>,
    name: &str,
) -> PyResult<Option<T>>
where
    T: DeserializeOwned,
{
    match payload {
        None => Ok(None),
        Some(value) if value.is_none() => Ok(None),
        Some(value) => parse_payload::<T>(value, name).map(Some),
    }
}

/// A rig hand-eye export as the upstream calibration of a rig laserline
/// input.
///
/// Parameters
/// ----------
/// export:
///     Rig hand-eye calibration export (serde-compatible with
///     `RigHandeyeExport`).
/// rig_se3_target:
///     One rig-from-target pose per view of the laserline dataset (list of
///     serde-compatible `Iso3`), through the export's hand-eye chain.
///
/// Returns
/// -------
/// dict
///     Serde payload of `RigUpstreamCalibration`, with the export's
///     covariance in its terms.
///
/// Raises
/// ------
/// ValueError
///     If the export's camera and extrinsic counts differ, its covariance
///     lists part of a pose, or a payload is malformed or not finite.
#[pyfunction]
fn rig_upstream_calibration(
    py: Python<'_>,
    export: &Bound<'_, PyAny>,
    rig_se3_target: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    use vision_calibration::core::Iso3;
    use vision_calibration::rig_handeye::RigHandeyeExport;

    reject_non_finite("rig_se3_target", rig_se3_target)?;
    let export: RigHandeyeExport = parse_payload(export, "export")?;
    let poses: Vec<Iso3> = parse_payload(rig_se3_target, "rig_se3_target")?;
    let upstream = export
        .to_upstream_calibration(poses)
        .map_err(|err| value_err(err.to_string()))?;
    let output = pythonize(py, &upstream)
        .map_err(|err| runtime_err(format!("failed to convert the upstream to Python: {err}")))?;
    Ok(output.unbind())
}

/// Projection uncertainty of one camera of a calibration export.
///
/// Parameters
/// ----------
/// export:
///     A calibration export (any problem type; its `kind` field selects it).
/// camera:
///     Camera index, 0 for single-camera results.
/// image_size:
///     ``[width, height]`` of the camera's images, pixels.
///
/// Returns
/// -------
/// dict
///     Serde payload of `ProjectionUncertainty`.
///
/// Raises
/// ------
/// ValueError
///     If the export has no covariance of the camera's parameters (none, or
///     the final solve held them fixed), `camera` is out of range, or the
///     image is smaller than 2×2 pixels.
/// TypeError
///     If `export` is not a calibration export.
#[pyfunction]
fn projection_uncertainty(
    py: Python<'_>,
    export: &Bound<'_, PyAny>,
    camera: usize,
    image_size: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    let image_size: [u32; 2] = parse_payload(image_size, "image_size")?;
    let export = parse_cameras(export, "export")?;
    let map = py
        .detach(|| export.projection_uncertainty(camera, image_size))
        .map_err(analysis_err)?;
    let output = pythonize(py, &map)
        .map_err(|err| runtime_err(format!("failed to convert result to Python: {err}")))?;
    Ok(output.unbind())
}

/// Quality report of a calibration export: a pass / warn / fail verdict and
/// the checks behind it.
///
/// Parameters
/// ----------
/// export:
///     A calibration export (any problem type; its `kind` field selects it).
/// thresholds:
///     Serde payload of `QualityThresholds`; `None` for the defaults.
/// image_sizes:
///     ``[[width, height], ...]`` per camera, pixels; `None` when only the
///     export's image manifest gives the sizes.
///
/// Returns
/// -------
/// dict
///     Serde payload of `QualityReport`.
///
/// Raises
/// ------
/// ValueError
///     If a threshold is out of range, `image_sizes` does not give one size
///     per camera, or a camera's size is ambiguous.
/// TypeError
///     If `export` is not a calibration export or a payload has the wrong
///     shape.
#[pyfunction(signature = (export, thresholds=None, image_sizes=None))]
fn assess(
    py: Python<'_>,
    export: &Bound<'_, PyAny>,
    thresholds: Option<&Bound<'_, PyAny>>,
    image_sizes: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    use vision_calibration::analysis::QualityThresholds;

    let thresholds: QualityThresholds =
        parse_optional_payload(thresholds, "thresholds")?.unwrap_or_default();
    let image_sizes: Vec<[u32; 2]> =
        parse_optional_payload(image_sizes, "image_sizes")?.unwrap_or_default();
    let export = parse_cameras(export, "export")?;
    let report = py
        .detach(|| vision_calibration::analysis::assess(export.as_ref(), &image_sizes, &thresholds))
        .map_err(|err| match &err {
            vision_calibration::Error::InvalidInput { .. } => value_err(err.to_string()),
            _ => runtime_err(err.to_string()),
        })?;
    let output = pythonize(py, &report)
        .map_err(|err| runtime_err(format!("failed to convert result to Python: {err}")))?;
    Ok(output.unbind())
}

/// A calibration export as one self-contained HTML page: the verdict and its
/// checks, each camera's parameters with σ, and the residual and
/// projection-σ figures.
///
/// Parameters
/// ----------
/// export:
///     A calibration export (any problem type; its `kind` field selects it).
/// image_sizes:
///     ``[[width, height], ...]`` per camera, pixels, or `None`.
/// title:
///     The page's heading; `None` for "Calibration report".
/// camera_names:
///     Each camera's name, in camera order.
/// thresholds:
///     Serde payload of `QualityThresholds`; `None` for the defaults.
///
/// Raises
/// ------
/// ValueError
///     If a threshold is out of range or `image_sizes` does not give one size
///     per camera.
/// TypeError
///     If `export` is not a calibration export or a payload has the wrong
///     shape.
#[pyfunction(signature = (export, image_sizes=None, title=None, camera_names=None, thresholds=None))]
fn html_report(
    py: Python<'_>,
    export: &Bound<'_, PyAny>,
    image_sizes: Option<&Bound<'_, PyAny>>,
    title: Option<String>,
    camera_names: Option<Vec<String>>,
    thresholds: Option<&Bound<'_, PyAny>>,
) -> PyResult<String> {
    use vision_calibration::analysis::ReportOptions;

    let mut options = ReportOptions::default();
    options.title = title.unwrap_or_default();
    options.camera_names = camera_names.unwrap_or_default();
    options.thresholds = parse_optional_payload(thresholds, "thresholds")?.unwrap_or_default();
    let image_sizes: Vec<[u32; 2]> =
        parse_optional_payload(image_sizes, "image_sizes")?.unwrap_or_default();
    let export = parse_cameras(export, "export")?;
    py.detach(|| vision_calibration::analysis::html_report(export.as_ref(), &image_sizes, &options))
        .map_err(|err| match &err {
            vision_calibration::Error::InvalidInput { .. } => value_err(err.to_string()),
            _ => runtime_err(err.to_string()),
        })
}

/// A pipeline error as a Python exception: an input the caller can fix (a
/// precondition, too little data, no covariance) is a `ValueError`; a
/// numerical failure a `RuntimeError`.
fn analysis_err(err: vision_calibration::Error) -> PyErr {
    match &err {
        vision_calibration::Error::InvalidInput { .. }
        | vision_calibration::Error::InsufficientData { .. }
        | vision_calibration::Error::CovarianceUnavailable { .. } => value_err(err.to_string()),
        _ => runtime_err(err.to_string()),
    }
}

/// The planar-intrinsics export a payload holds.
fn planar_export(
    export: &Bound<'_, PyAny>,
) -> PyResult<vision_calibration::planar_intrinsics::PlanarIntrinsicsExport> {
    let value: serde_json::Value = parse_payload(export, "export")?;
    let kind = value.get("kind").and_then(serde_json::Value::as_str);
    if kind != Some("planar_intrinsics") {
        return Err(PyTypeError::new_err(format!(
            "next-view planning needs a planar intrinsics export, not {}",
            kind.unwrap_or("a payload without a kind")
        )));
    }
    serde_json::from_value(value)
        .map_err(|err| PyTypeError::new_err(format!("invalid export payload: {err}")))
}

/// Parse candidate poses, target points and the image size.
fn next_view_inputs(
    candidates: &Bound<'_, PyAny>,
    target_points: &Bound<'_, PyAny>,
    image_size: &Bound<'_, PyAny>,
) -> PyResult<(
    Vec<vision_calibration::core::Iso3>,
    Vec<vision_calibration::core::Pt3>,
    [u32; 2],
)> {
    reject_non_finite("candidates", candidates)?;
    let candidates = parse_payload(candidates, "candidates")?;
    let points = parse_target_points(target_points)?;
    let image_size = parse_payload(image_size, "image_size")?;
    Ok((candidates, points, image_size))
}

/// A target's points, ``[[x, y, z], ...]``.
fn parse_target_points(
    target_points: &Bound<'_, PyAny>,
) -> PyResult<Vec<vision_calibration::core::Pt3>> {
    reject_non_finite("target_points", target_points)?;
    let points: Vec<[f64; 3]> = parse_payload(target_points, "target_points")?;
    Ok(points
        .into_iter()
        .map(vision_calibration::core::Pt3::from)
        .collect())
}

/// The covariance a planar calibration is predicted to have with candidate
/// views added, without a re-solve.
///
/// Parameters
/// ----------
/// export:
///     A planar intrinsics export.
/// candidates:
///     ``cam_se3_target`` poses, serde `Iso3` payloads.
/// target_points:
///     The target's points in its own frame, metres, ``[[x, y, z], ...]``.
/// image_size:
///     ``[width, height]`` of the camera's images, pixels.
///
/// Returns
/// -------
/// dict
///     Serde payload of `PredictedUncertainty`.
///
/// Raises
/// ------
/// ValueError
///     If the export has no covariance, `candidates` or `target_points` is
///     empty, or the image is smaller than 2×2 pixels.
/// TypeError
///     If `export` is not a planar intrinsics export or a payload has the
///     wrong shape.
#[pyfunction]
fn predict_with_views(
    py: Python<'_>,
    export: &Bound<'_, PyAny>,
    candidates: &Bound<'_, PyAny>,
    target_points: &Bound<'_, PyAny>,
    image_size: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    let export = planar_export(export)?;
    let (candidates, points, image_size) = next_view_inputs(candidates, target_points, image_size)?;
    let predicted = py
        .detach(|| {
            vision_calibration::analysis::predict_with_views(
                &export,
                &candidates,
                &points,
                image_size,
            )
        })
        .map_err(analysis_err)?;
    let output = pythonize(py, &predicted)
        .map_err(|err| runtime_err(format!("failed to convert result to Python: {err}")))?;
    Ok(output.unbind())
}

/// Candidate views of a planar calibration, each added alone, ranked by how
/// much they reduce the largest projection σ.
///
/// Parameters and errors as `predict_with_views`.
///
/// Returns
/// -------
/// list[dict]
///     Serde payloads of `RankedCandidate`, best first.
#[pyfunction]
fn rank_candidates(
    py: Python<'_>,
    export: &Bound<'_, PyAny>,
    candidates: &Bound<'_, PyAny>,
    target_points: &Bound<'_, PyAny>,
    image_size: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    let export = planar_export(export)?;
    let (candidates, points, image_size) = next_view_inputs(candidates, target_points, image_size)?;
    let ranked = py
        .detach(|| {
            vision_calibration::analysis::rank_candidates(&export, &candidates, &points, image_size)
        })
        .map_err(analysis_err)?;
    let output = pythonize(py, &ranked)
        .map_err(|err| runtime_err(format!("failed to convert result to Python: {err}")))?;
    Ok(output.unbind())
}

/// Parse a rig hand-eye export payload.
fn rig_handeye_export(
    export: &Bound<'_, PyAny>,
) -> PyResult<vision_calibration::rig_handeye::RigHandeyeExport> {
    let value: serde_json::Value = parse_payload(export, "export")?;
    let kind = value.get("kind").and_then(serde_json::Value::as_str);
    if kind != Some("rig_handeye") {
        return Err(PyTypeError::new_err(format!(
            "robot pose planning needs a rig hand-eye export, not {}",
            kind.unwrap_or("a payload without a kind")
        )));
    }
    serde_json::from_value(value)
        .map_err(|err| PyTypeError::new_err(format!("invalid export payload: {err}")))
}

/// Candidate robot poses, target points and one image size per camera.
type RobotPoseInputs = (
    Vec<vision_calibration::core::Iso3>,
    Vec<vision_calibration::core::Pt3>,
    Vec<[u32; 2]>,
);

/// Parse candidate robot poses, target points and one image size per camera.
fn robot_pose_inputs(
    candidates: &Bound<'_, PyAny>,
    target_points: &Bound<'_, PyAny>,
    image_sizes: &Bound<'_, PyAny>,
) -> PyResult<RobotPoseInputs> {
    reject_non_finite("candidates", candidates)?;
    let candidates = parse_payload(candidates, "candidates")?;
    let points = parse_target_points(target_points)?;
    let image_sizes = parse_payload(image_sizes, "image_sizes")?;
    Ok((candidates, points, image_sizes))
}

/// The covariance a rig hand-eye calibration is predicted to have with views
/// at candidate robot poses added, without a re-solve.
///
/// Parameters
/// ----------
/// export:
///     A rig hand-eye export.
/// candidates:
///     ``base_se3_gripper`` robot poses, serde `Iso3` payloads.
/// target_points:
///     The target's points in its own frame, metres, ``[[x, y, z], ...]``.
/// image_sizes:
///     ``[width, height]`` of each camera's images, pixels.
///
/// Returns
/// -------
/// dict
///     Serde payload of `PredictedRigUncertainty`.
///
/// Raises
/// ------
/// ValueError
///     If the export has no covariance, `candidates` or `target_points` is
///     empty, `image_sizes` does not give one size per camera, or the export
///     cannot be planned from.
/// TypeError
///     If `export` is not a rig hand-eye export or a payload has the wrong
///     shape.
#[pyfunction]
fn predict_with_robot_poses(
    py: Python<'_>,
    export: &Bound<'_, PyAny>,
    candidates: &Bound<'_, PyAny>,
    target_points: &Bound<'_, PyAny>,
    image_sizes: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    let export = rig_handeye_export(export)?;
    let (candidates, points, sizes) = robot_pose_inputs(candidates, target_points, image_sizes)?;
    let predicted = py
        .detach(|| {
            vision_calibration::analysis::predict_with_robot_poses(
                &export,
                &candidates,
                &points,
                &sizes,
            )
        })
        .map_err(analysis_err)?;
    let output = pythonize(py, &predicted)
        .map_err(|err| runtime_err(format!("failed to convert result to Python: {err}")))?;
    Ok(output.unbind())
}

/// Candidate robot poses of a rig hand-eye calibration, each added alone,
/// ranked by how much they reduce the log-volume of the covariance.
///
/// Parameters and errors as `predict_with_robot_poses`.
///
/// Returns
/// -------
/// list[dict]
///     Serde payloads of `RankedRobotPose`, best first.
#[pyfunction]
fn rank_robot_poses(
    py: Python<'_>,
    export: &Bound<'_, PyAny>,
    candidates: &Bound<'_, PyAny>,
    target_points: &Bound<'_, PyAny>,
    image_sizes: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    let export = rig_handeye_export(export)?;
    let (candidates, points, sizes) = robot_pose_inputs(candidates, target_points, image_sizes)?;
    let ranked = py
        .detach(|| {
            vision_calibration::analysis::rank_robot_poses(&export, &candidates, &points, &sizes)
        })
        .map_err(analysis_err)?;
    let output = pythonize(py, &ranked)
        .map_err(|err| runtime_err(format!("failed to convert result to Python: {err}")))?;
    Ok(output.unbind())
}

/// A deterministic set of candidate target poses over the image.
///
/// Parameters
/// ----------
/// camera:
///     Serde payload of the camera's `CameraParams`.
/// target_points:
///     The target's points in its own frame, metres, ``[[x, y, z], ...]``.
/// image_size:
///     ``[width, height]`` of the camera's images, pixels.
/// grid:
///     Serde payload of `CandidateGrid`.
///
/// Returns
/// -------
/// list[dict]
///     ``cam_se3_target`` poses, serde `Iso3` payloads.
///
/// Raises
/// ------
/// ValueError
///     If a grid setting is out of range, `target_points` is empty, or the
///     image is smaller than 2×2 pixels.
#[pyfunction]
fn candidate_poses(
    py: Python<'_>,
    camera: &Bound<'_, PyAny>,
    target_points: &Bound<'_, PyAny>,
    image_size: &Bound<'_, PyAny>,
    grid: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    reject_non_finite("camera", camera)?;
    reject_non_finite("grid", grid)?;
    let camera: vision_calibration::core::CameraParams = parse_payload(camera, "camera")?;
    let grid: vision_calibration::analysis::CandidateGrid = parse_payload(grid, "grid")?;
    let points = parse_target_points(target_points)?;
    let image_size = parse_payload(image_size, "image_size")?;
    let poses = py
        .detach(|| {
            vision_calibration::analysis::candidate_poses(&camera, &points, image_size, &grid)
        })
        .map_err(analysis_err)?;
    let output = pythonize(py, &poses)
        .map_err(|err| runtime_err(format!("failed to convert result to Python: {err}")))?;
    Ok(output.unbind())
}

/// Candidate camera configurations of a planar calibration, scored on views
/// each fit did not see (K-fold over views) and ranked by held-out
/// reprojection error.
///
/// Parameters
/// ----------
/// input:
///     Serde payload of `PlanarIntrinsicsInput`.
/// alternatives:
///     Serde payloads of `CameraConfig`, the candidates after the config's
///     own camera.
/// config:
///     Serde payload of `PlanarIntrinsicsConfig` (the estimator; its
///     `camera` is candidate 0); `None` for the default.
/// folds:
///     Number of folds, at least 2.
/// seed:
///     Seed of the shuffle that assigns views to folds.
/// progress:
///     Called with each progress report of the fits; raising stops the run.
///
/// Returns
/// -------
/// dict
///     Serde payload of `ModelComparison`.
///
/// Raises
/// ------
/// ValueError
///     If `folds` is below 2, there are too few views for the folds, a
///     candidate is invalid or repeated, `config.optimize.fix_poses` is not
///     empty, or the input fails the planar checks.
/// TypeError
///     If a payload has the wrong shape.
#[pyfunction(signature = (input, alternatives, config=None, folds=5, seed=0, progress=None))]
fn compare_camera_models(
    py: Python<'_>,
    input: &Bound<'_, PyAny>,
    alternatives: &Bound<'_, PyAny>,
    config: Option<&Bound<'_, PyAny>>,
    folds: usize,
    seed: u64,
    progress: Option<Py<PyAny>>,
) -> PyResult<Py<PyAny>> {
    use vision_calibration::analysis::CrossValidationOptions;
    use vision_calibration::common::config::CameraConfig;
    use vision_calibration::planar_intrinsics::{PlanarIntrinsicsConfig, PlanarIntrinsicsInput};

    let input: PlanarIntrinsicsInput = session::parse_checked(input, "input")?;
    let alternatives: Vec<CameraConfig> = session::parse_checked(alternatives, "alternatives")?;
    let config: PlanarIntrinsicsConfig =
        session::parse_optional_checked(config, "config")?.unwrap_or_default();
    let opts = CrossValidationOptions::new(folds, seed);
    let control = control::PyRunControl::new(progress);
    let run = control.control();
    let result = py.detach(|| {
        vision_calibration::analysis::compare_camera_models(
            &input,
            &config,
            &alternatives,
            &opts,
            run,
        )
    });
    let comparison = control.finish(result, |err| match err {
        // Normally replaced by the exception the progress callable raised or
        // the Ctrl-C that cancelled the token.
        vision_calibration::Error::Cancelled => {
            pyo3::exceptions::PyKeyboardInterrupt::new_err("cancelled")
        }
        // Only the input checks raise a core error.
        err @ vision_calibration::Error::Core(_) => value_err(format!("invalid input: {err}")),
        err => errors::calib_err("model comparison failed", err),
    })?;
    session::to_py(py, &comparison)
}

/// The plane of sharp focus and the depth of field, fitted to the per-point
/// blur of a calibrated planar dataset with the camera and the poses held.
///
/// Parameters
/// ----------
/// input:
///     Serde payload of `PlanarIntrinsicsInput` (views with blur).
/// camera:
///     Serde payload of `CameraParams`, the calibrated camera.
/// cam_se3_target:
///     Serde payloads of the calibrated `cam_se3_target` poses, one per view.
/// blur_limit_px:
///     The largest acceptable blur σ for the depth of field, px; `None` for
///     none.
/// view_jackknife:
///     Whether to add the delete-one-view jackknife covariance.
///
/// Returns
/// -------
/// dict
///     Serde payload of `FocusDiagnostic`.
///
/// Raises
/// ------
/// ValueError
///     If the pose count differs from the view count, the camera or a pose
///     is not finite, the projection is not a pinhole, the sensor faces
///     away, some blur points state a variance of their width and others
///     do not, or the blur limit is not finite and positive.
/// TypeError
///     If a payload has the wrong shape.
#[pyfunction(signature = (input, camera, cam_se3_target, blur_limit_px=None, view_jackknife=false))]
fn fit_focus(
    py: Python<'_>,
    input: &Bound<'_, PyAny>,
    camera: &Bound<'_, PyAny>,
    cam_se3_target: &Bound<'_, PyAny>,
    blur_limit_px: Option<f64>,
    view_jackknife: bool,
) -> PyResult<Py<PyAny>> {
    use vision_calibration::analysis::FocusOptions;
    use vision_calibration::core::{CameraParams, Iso3};
    use vision_calibration::planar_intrinsics::PlanarIntrinsicsInput;

    if let Some(limit) = blur_limit_px
        && !(limit.is_finite() && limit > 0.0)
    {
        return Err(value_err(format!(
            "blur_limit_px must be finite and positive, got {limit}"
        )));
    }
    let input: PlanarIntrinsicsInput = session::parse_checked(input, "input")?;
    let camera: CameraParams = session::parse_checked(camera, "camera")?;
    let poses: Vec<Iso3> = session::parse_checked(cam_se3_target, "cam_se3_target")?;
    let opts = FocusOptions::new(blur_limit_px, view_jackknife);
    let diagnostic = py
        .detach(|| vision_calibration::analysis::fit_focus(&input, &camera, &poses, &opts))
        .map_err(|err| value_err(format!("invalid input: {err}")))?;
    session::to_py(py, &diagnostic)
}

/// `values` (row-major, `rows` × `columns`) as a 2-D numpy array.
fn plane<'py>(
    py: Python<'py>,
    values: Vec<f64>,
    rows: usize,
    columns: usize,
) -> PyResult<Bound<'py, numpy::PyArray2<f64>>> {
    use numpy::PyArrayMethods;
    numpy::PyArray1::from_vec(py, values).reshape([rows, columns])
}

/// The grid fields of a map payload.
fn grid_payload<'py>(
    py: Python<'py>,
    grid: &vision_calibration::analysis::ImageGrid,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
    let out = pyo3::types::PyDict::new(py);
    out.set_item("image_size", grid.image_size().to_vec())?;
    out.set_item("stride", grid.stride())?;
    out.set_item("columns", grid.columns())?;
    out.set_item("rows", grid.rows())?;
    Ok(out)
}

/// The image grid of `columns` columns over `image_size`.
fn image_grid(
    image_size: &Bound<'_, PyAny>,
    columns: u32,
) -> PyResult<vision_calibration::analysis::ImageGrid> {
    let image_size: [u32; 2] = parse_payload(image_size, "image_size")?;
    vision_calibration::analysis::ImageGrid::with_columns(image_size, columns)
        .map_err(|err| value_err(err.to_string()))
}

/// The lens and sensor displacement of a camera over its image.
///
/// Parameters
/// ----------
/// camera:
///     Serde payload of `CameraParams`.
/// image_size:
///     ``[width, height]`` of the camera's images, pixels.
/// columns:
///     Grid columns; the stride is ``ceil(width / columns)``.
///
/// Returns
/// -------
/// dict
///     ``image_size``, ``stride``, ``columns``, ``rows``, ``valid_nodes`` and
///     the ``total``, ``distortion`` and ``sensor`` layers, each a dict of
///     ``dx`` and ``dy`` float64 arrays shaped ``(rows, columns)``, NaN where
///     a node has no value.
///
/// Raises
/// ------
/// ValueError
///     If the camera is not finite, its sensor homography is not invertible,
///     or the image or ``columns`` is empty.
#[pyfunction(signature = (camera, image_size, columns=256))]
fn distortion_map(
    py: Python<'_>,
    camera: &Bound<'_, PyAny>,
    image_size: &Bound<'_, PyAny>,
    columns: u32,
) -> PyResult<Py<PyAny>> {
    use vision_calibration::core::CameraParams;

    let camera: CameraParams = session::parse_checked(camera, "camera")?;
    let grid = image_grid(image_size, columns)?;
    let map = py
        .detach(|| vision_calibration::analysis::distortion_map(&camera, &grid))
        .map_err(|err| value_err(format!("invalid camera: {err}")))?;
    let (rows, cols) = (grid.rows(), grid.columns());
    let out = grid_payload(py, &grid)?;
    out.set_item("valid_nodes", map.valid_nodes)?;
    for (name, layer) in [
        ("total", map.total),
        ("distortion", map.distortion),
        ("sensor", map.sensor),
    ] {
        let planes = pyo3::types::PyDict::new(py);
        planes.set_item("dx", plane(py, layer.dx, rows, cols)?)?;
        planes.set_item("dy", plane(py, layer.dy, rows, cols)?)?;
        out.set_item(name, planes)?;
    }
    Ok(out.into_any().unbind())
}

/// Where two calibrations of one camera project the same ray, after the
/// rotation of the second camera's frame that brings them closest.
///
/// Parameters
/// ----------
/// a:
///     Serde payload of the reference `CameraParams`; the grid is its image.
/// b:
///     Serde payload of the compared `CameraParams`.
/// image_size:
///     ``[width, height]`` of the cameras' images, pixels.
/// columns:
///     Grid columns; the stride is ``ceil(width / columns)``.
/// compensation:
///     ``"rotation"`` (fit the rotation) or ``"none"``.
/// fit_points:
///     Pixels of ``a``'s image whose blocks the rotation is fitted on, as
///     ``[[x, y], ...]``; ``None`` fits every node with a value.
///
/// Returns
/// -------
/// dict
///     The grid fields, ``difference`` (a dict of ``dx`` and ``dy`` float64
///     arrays shaped ``(rows, columns)``, NaN where a node has no value),
///     ``fitted`` (a bool array of the same shape), ``camb_so3_cama``
///     (``[qx, qy, qz, qw]``), ``rotation_angle_rad``, ``valid_nodes``,
///     ``fit_nodes``, ``rms_fit_px``, ``max_fit_px``, ``rms_valid_px`` and
///     ``max_valid_px``.
///
/// Raises
/// ------
/// ValueError
///     If a camera is not finite or its sensor homography is not
///     invertible, ``compensation`` is unknown, the image or ``columns`` is
///     empty, or fewer than three nodes of the fit region have a value.
#[pyfunction(signature = (a, b, image_size, columns=64, compensation="rotation", fit_points=None))]
fn projection_difference(
    py: Python<'_>,
    a: &Bound<'_, PyAny>,
    b: &Bound<'_, PyAny>,
    image_size: &Bound<'_, PyAny>,
    columns: u32,
    compensation: &str,
    fit_points: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    use numpy::PyArrayMethods;
    use vision_calibration::analysis::{
        Compensation, FitRegion, ProjectionDifferenceOptions, projection_difference,
    };
    use vision_calibration::core::CameraParams;

    let compensation = match compensation {
        "rotation" => Compensation::Rotation,
        "none" => Compensation::None,
        other => {
            return Err(value_err(format!(
                "compensation must be \"rotation\" or \"none\", got {other:?}"
            )));
        }
    };
    let a: CameraParams = session::parse_checked(a, "a")?;
    let b: CameraParams = session::parse_checked(b, "b")?;
    let grid = image_grid(image_size, columns)?;
    let fit_region = match fit_points {
        None => FitRegion::WholeImage,
        Some(points) => FitRegion::Covered(parse_payload(points, "fit_points")?),
    };
    let options = ProjectionDifferenceOptions::new(compensation, fit_region);
    let diff = py
        .detach(|| projection_difference(&a, &b, &grid, &options))
        .map_err(|err| value_err(format!("invalid input: {err}")))?;
    let (rows, cols) = (grid.rows(), grid.columns());
    let out = grid_payload(py, &grid)?;
    let planes = pyo3::types::PyDict::new(py);
    planes.set_item("dx", plane(py, diff.difference.dx, rows, cols)?)?;
    planes.set_item("dy", plane(py, diff.difference.dy, rows, cols)?)?;
    out.set_item("difference", planes)?;
    out.set_item(
        "fitted",
        numpy::PyArray1::from_vec(py, diff.fitted).reshape([rows, cols])?,
    )?;
    let q = diff.camb_so3_cama.coords;
    out.set_item("camb_so3_cama", vec![q.x, q.y, q.z, q.w])?;
    out.set_item("rotation_angle_rad", diff.camb_so3_cama.angle())?;
    out.set_item("valid_nodes", diff.valid_nodes)?;
    out.set_item("fit_nodes", diff.fit_nodes)?;
    out.set_item("rms_fit_px", diff.rms_fit_px)?;
    out.set_item("max_fit_px", diff.max_fit_px)?;
    out.set_item("rms_valid_px", diff.rms_valid_px)?;
    out.set_item("max_valid_px", diff.max_valid_px)?;
    Ok(out.into_any().unbind())
}

/// Calibrations of one rig side by side against the first, the reference.
///
/// Parameters
/// ----------
/// runs:
///     ``[{"name": str, "export": dict, "camera_ids": [str, ...],
///     "image_sizes": [[width, height], ...], "config": dict | None}, ...]``:
///     each run's export (any problem type, all of one), its cameras' ids and
///     image sizes in camera order, and optionally the serde payload of its
///     `CalibrationConfig`.
/// options:
///     Serde payload of `CompareRunsOptions`; `None` for the defaults.
///
/// Returns
/// -------
/// dict
///     Serde payload of `RunComparison`.
///
/// Raises
/// ------
/// ValueError
///     If fewer than two runs are given, the runs are of different
///     workflows, the ids or sizes are not one per camera, a run shares no
///     camera id with the reference, a shared camera's image sizes differ, or
///     the relations are not one per run after the first.
/// TypeError
///     If an export or a payload has the wrong shape.
#[pyfunction(signature = (runs, options=None))]
fn compare_runs(
    py: Python<'_>,
    runs: &Bound<'_, pyo3::types::PyList>,
    options: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    use pyo3::types::PyDictMethods;
    use vision_calibration::analysis::{CompareRunsOptions, ComparedRun};
    use vision_calibration::dataset_runner::CalibrationConfig;

    struct Parsed {
        name: String,
        export: Box<dyn vision_calibration::common::CalibratedCameras>,
        camera_ids: Vec<String>,
        image_sizes: Vec<[u32; 2]>,
        config: Option<CalibrationConfig>,
    }
    let options: CompareRunsOptions =
        parse_optional_payload(options, "options")?.unwrap_or_default();
    let mut parsed = Vec::with_capacity(runs.len());
    for (i, run) in runs.iter().enumerate() {
        let run = run
            .cast::<pyo3::types::PyDict>()
            .map_err(|_| PyTypeError::new_err(format!("runs[{i}] is not a dict")))?;
        let item = |key: &str| -> PyResult<Bound<'_, PyAny>> {
            run.get_item(key)?
                .ok_or_else(|| PyTypeError::new_err(format!("runs[{i}] has no {key:?}")))
        };
        let config = match run.get_item("config")? {
            Some(c) if !c.is_none() => Some(session::parse_checked(&c, "config")?),
            _ => None,
        };
        parsed.push(Parsed {
            name: item("name")?.extract()?,
            export: parse_cameras(&item("export")?, "export")?,
            camera_ids: parse_payload(&item("camera_ids")?, "camera_ids")?,
            image_sizes: parse_payload(&item("image_sizes")?, "image_sizes")?,
            config,
        });
    }
    let comparison = py
        .detach(|| {
            let compared: Vec<ComparedRun<'_>> = parsed
                .iter()
                .map(|p| {
                    let run = ComparedRun::new(
                        p.name.clone(),
                        p.export.as_ref(),
                        p.camera_ids.clone(),
                        p.image_sizes.clone(),
                    );
                    match &p.config {
                        Some(c) => run.with_config(c.clone()),
                        None => run,
                    }
                })
                .collect();
            vision_calibration::analysis::compare_runs(&compared, &options)
        })
        .map_err(analysis_err)?;
    let output = pythonize(py, &comparison)
        .map_err(|err| runtime_err(format!("failed to convert result to Python: {err}")))?;
    Ok(output.unbind())
}

/// The systematic part of one camera's reprojection residuals over its
/// image.
///
/// Parameters
/// ----------
/// export:
///     A calibration export (any problem type; its `kind` field selects it).
/// image_size:
///     ``[width, height]`` of the camera's images, pixels.
/// camera:
///     Camera index, 0 for single-camera results.
/// columns:
///     Grid columns; the stride is ``ceil(width / columns)``.
/// options:
///     Serde payload of `ResidualMapOptions`; `None` for the defaults.
///
/// Returns
/// -------
/// dict
///     The grid fields, ``camera``, ``bandwidth_px``, ``bandwidth_selection``
///     (serde payload), ``used``, ``unprojected``, ``outliers`` and
///     ``views``, and the planes ``mean_dx``, ``mean_dy``, ``rms``,
///     ``std_error``, ``significance``, ``effective_samples`` and
///     ``effective_views`` as float64 arrays shaped ``(rows, columns)``.
///
/// Raises
/// ------
/// ValueError
///     If an option is out of range, the export has no record of the camera
///     or of a selected view, or the image or ``columns`` is empty.
/// TypeError
///     If `export` is not a calibration export or a payload has the wrong
///     shape.
#[pyfunction(signature = (export, image_size, camera=0, columns=64, options=None))]
fn residual_map(
    py: Python<'_>,
    export: &Bound<'_, PyAny>,
    image_size: &Bound<'_, PyAny>,
    camera: usize,
    columns: u32,
    options: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    use vision_calibration::analysis::ResidualMapOptions;

    let options: ResidualMapOptions =
        session::parse_optional_checked(options, "options")?.unwrap_or_default();
    let grid = image_grid(image_size, columns)?;
    let export = parse_cameras(export, "export")?;
    let map = py
        .detach(|| {
            vision_calibration::analysis::residual_map(
                &export.diagnostics().per_feature_residuals.target,
                camera,
                &grid,
                &options,
            )
        })
        .map_err(analysis_err)?;
    let (rows, cols) = (grid.rows(), grid.columns());
    let out = grid_payload(py, &grid)?;
    out.set_item("camera", map.camera)?;
    out.set_item("bandwidth_px", map.bandwidth_px)?;
    out.set_item(
        "bandwidth_selection",
        session::to_py(py, &map.bandwidth_selection)?,
    )?;
    out.set_item("used", map.used)?;
    out.set_item("unprojected", map.unprojected)?;
    out.set_item("outliers", map.outliers)?;
    out.set_item("views", map.views)?;
    for (name, values) in [
        ("mean_dx", map.mean_dx),
        ("mean_dy", map.mean_dy),
        ("rms", map.rms),
        ("std_error", map.std_error),
        ("significance", map.significance),
        ("effective_samples", map.effective_samples),
        ("effective_views", map.effective_views),
    ] {
        out.set_item(name, plane(py, values, rows, cols)?)?;
    }
    Ok(out.into_any().unbind())
}

/// Return the Rust library version embedded in the extension module.
#[pyfunction]
fn library_version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[pymodule]
fn _vision_calibration(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(rig_upstream_calibration, m)?)?;
    m.add_function(wrap_pyfunction!(projection_uncertainty, m)?)?;
    m.add_function(wrap_pyfunction!(assess, m)?)?;
    m.add_function(wrap_pyfunction!(html_report, m)?)?;
    m.add_function(wrap_pyfunction!(predict_with_views, m)?)?;
    m.add_function(wrap_pyfunction!(rank_candidates, m)?)?;
    m.add_function(wrap_pyfunction!(predict_with_robot_poses, m)?)?;
    m.add_function(wrap_pyfunction!(rank_robot_poses, m)?)?;
    m.add_function(wrap_pyfunction!(candidate_poses, m)?)?;
    m.add_function(wrap_pyfunction!(compare_camera_models, m)?)?;
    m.add_function(wrap_pyfunction!(fit_focus, m)?)?;
    m.add_function(wrap_pyfunction!(distortion_map, m)?)?;
    m.add_function(wrap_pyfunction!(residual_map, m)?)?;
    m.add_function(wrap_pyfunction!(projection_difference, m)?)?;
    m.add_function(wrap_pyfunction!(compare_runs, m)?)?;
    m.add_function(wrap_pyfunction!(library_version, m)?)?;
    m.add("RunError", m.py().get_type::<errors::RunError>())?;
    m.add("AskUserError", m.py().get_type::<errors::AskUserError>())?;
    m.add("SniffError", m.py().get_type::<errors::SniffError>())?;
    m.add(
        "SampleFetchError",
        m.py().get_type::<errors::SampleFetchError>(),
    )?;
    m.add_class::<camera::PyCameraModel>()?;
    session::register(m)?;
    dataset::register(m)?;
    detect::register(m)?;
    device_seed::register(m)?;
    mvg::register(m)?;
    project::register(m)?;
    Ok(())
}
