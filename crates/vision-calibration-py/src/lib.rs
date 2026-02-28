use pyo3::exceptions::{PyRuntimeError, PyTypeError};
use pyo3::prelude::*;
use pythonize::{depythonize, pythonize};
use serde::Serialize;
use serde::de::DeserializeOwned;
use vision_calibration::session::{CalibrationSession, ProblemType};

fn runtime_err(message: impl std::fmt::Display) -> PyErr {
    PyRuntimeError::new_err(message.to_string())
}

fn parse_payload<T>(payload: &Bound<'_, PyAny>, name: &str) -> PyResult<T>
where
    T: DeserializeOwned,
{
    depythonize(payload)
        .map_err(|err| PyTypeError::new_err(format!("invalid {name} payload: {err}")))
}

fn parse_optional_payload<T>(payload: Option<&Bound<'_, PyAny>>, name: &str) -> PyResult<Option<T>>
where
    T: DeserializeOwned,
{
    match payload {
        None => Ok(None),
        Some(value) if value.is_none() => Ok(None),
        Some(value) => parse_payload::<T>(value, name).map(Some),
    }
}

fn run_problem<P, F>(
    py: Python<'_>,
    input: &Bound<'_, PyAny>,
    config: Option<&Bound<'_, PyAny>>,
    run_fn: F,
) -> PyResult<Py<PyAny>>
where
    P: ProblemType,
    P::Input: DeserializeOwned,
    P::Config: DeserializeOwned,
    P::Export: Serialize,
    F: FnOnce(&mut CalibrationSession<P>) -> anyhow::Result<()>,
{
    let input: P::Input = parse_payload(input, "input")?;
    let config: Option<P::Config> = parse_optional_payload(config, "config")?;

    let mut session = CalibrationSession::<P>::new();
    session
        .set_input(input)
        .map_err(|err| runtime_err(format!("failed to set input: {err}")))?;

    if let Some(cfg) = config {
        session
            .set_config(cfg)
            .map_err(|err| runtime_err(format!("failed to set config: {err}")))?;
    }

    run_fn(&mut session).map_err(|err| runtime_err(format!("calibration failed: {err}")))?;

    let export = session
        .export()
        .map_err(|err| runtime_err(format!("failed to export results: {err}")))?;

    let output = pythonize(py, &export)
        .map_err(|err| runtime_err(format!("failed to convert output to Python object: {err}")))?;

    Ok(output.unbind())
}

/// Run planar intrinsics calibration.
///
/// Parameters
/// ----------
/// input:
///     Planar dataset payload (serde-compatible with `PlanarDataset`).
/// config:
///     Optional planar configuration payload (serde-compatible with `PlanarConfig`).
///
/// Returns
/// -------
/// dict
///     Planar intrinsics export payload.
#[pyfunction(signature = (input, config=None))]
fn run_planar_intrinsics(
    py: Python<'_>,
    input: &Bound<'_, PyAny>,
    config: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    run_problem::<vision_calibration::planar_intrinsics::PlanarIntrinsicsProblem, _>(
        py,
        input,
        config,
        vision_calibration::planar_intrinsics::run_calibration,
    )
}

/// Run single-camera hand-eye calibration.
///
/// Parameters
/// ----------
/// input:
///     Single-camera hand-eye input payload (serde-compatible with `SingleCamHandeyeInput`).
/// config:
///     Optional single-camera hand-eye config payload.
///
/// Returns
/// -------
/// dict
///     Single-camera hand-eye export payload.
#[pyfunction(signature = (input, config=None))]
fn run_single_cam_handeye(
    py: Python<'_>,
    input: &Bound<'_, PyAny>,
    config: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    run_problem::<vision_calibration::single_cam_handeye::SingleCamHandeyeProblem, _>(
        py,
        input,
        config,
        vision_calibration::single_cam_handeye::run_calibration,
    )
}

/// Run multi-camera rig extrinsics calibration.
///
/// Parameters
/// ----------
/// input:
///     Rig extrinsics input payload (serde-compatible with `RigExtrinsicsInput`).
/// config:
///     Optional rig extrinsics config payload.
///
/// Returns
/// -------
/// dict
///     Rig extrinsics export payload.
#[pyfunction(signature = (input, config=None))]
fn run_rig_extrinsics(
    py: Python<'_>,
    input: &Bound<'_, PyAny>,
    config: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    run_problem::<vision_calibration::rig_extrinsics::RigExtrinsicsProblem, _>(
        py,
        input,
        config,
        vision_calibration::rig_extrinsics::run_calibration,
    )
}

/// Run multi-camera rig hand-eye calibration.
///
/// Parameters
/// ----------
/// input:
///     Rig hand-eye input payload (serde-compatible with `RigHandeyeInput`).
/// config:
///     Optional rig hand-eye config payload.
///
/// Returns
/// -------
/// dict
///     Rig hand-eye export payload.
#[pyfunction(signature = (input, config=None))]
fn run_rig_handeye(
    py: Python<'_>,
    input: &Bound<'_, PyAny>,
    config: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    run_problem::<vision_calibration::rig_handeye::RigHandeyeProblem, _>(
        py,
        input,
        config,
        vision_calibration::rig_handeye::run_calibration,
    )
}

/// Run laserline device calibration.
///
/// Parameters
/// ----------
/// input:
///     Laserline input payload (serde-compatible with `LaserlineDeviceInput`).
/// config:
///     Optional laserline device config payload.
///
/// Returns
/// -------
/// dict
///     Laserline device export payload.
#[pyfunction(signature = (input, config=None))]
fn run_laserline_device(
    py: Python<'_>,
    input: &Bound<'_, PyAny>,
    config: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    run_problem::<vision_calibration::laserline_device::LaserlineDeviceProblem, _>(
        py,
        input,
        config,
        |session| vision_calibration::laserline_device::run_calibration(session, None),
    )
}

/// Return the Rust library version embedded in the extension module.
#[pyfunction]
fn library_version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[pymodule]
fn _vision_calibration(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(run_planar_intrinsics, m)?)?;
    m.add_function(wrap_pyfunction!(run_single_cam_handeye, m)?)?;
    m.add_function(wrap_pyfunction!(run_rig_extrinsics, m)?)?;
    m.add_function(wrap_pyfunction!(run_rig_handeye, m)?)?;
    m.add_function(wrap_pyfunction!(run_laserline_device, m)?)?;
    m.add_function(wrap_pyfunction!(library_version, m)?)?;
    Ok(())
}
