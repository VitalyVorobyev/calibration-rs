use pyo3::exceptions::{PyRuntimeError, PyTypeError};
use pyo3::prelude::*;
use pythonize::{depythonize, pythonize};
use serde::Serialize;
use serde::de::DeserializeOwned;
use vision_calibration::session::{CalibrationSession, ProblemType};

mod validation;
use validation::{reject_non_finite, value_err};

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
    F: FnOnce(&mut CalibrationSession<P>) -> Result<(), vision_calibration::Error>,
{
    reject_non_finite("input", input)?;
    if let Some(cfg) = config {
        reject_non_finite("config", cfg)?;
    }

    let input: P::Input = parse_payload(input, "input")?;
    let config: Option<P::Config> = parse_optional_payload(config, "config")?;

    let mut session = CalibrationSession::<P>::new();
    session
        .set_input(input)
        .map_err(|err| value_err(format!("invalid input: {err}")))?;

    if let Some(cfg) = config {
        session
            .set_config(cfg)
            .map_err(|err| value_err(format!("invalid config: {err}")))?;
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
///     Optional planar configuration payload (serde-compatible with `PlanarIntrinsicsConfig`).
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

/// Run planar Scheimpflug intrinsics calibration.
///
/// Parameters
/// ----------
/// input:
///     Planar dataset payload (serde-compatible with `PlanarDataset`).
/// config:
///     Optional Scheimpflug intrinsics config payload.
///
/// Returns
/// -------
/// dict
///     Scheimpflug intrinsics export payload.
#[pyfunction(signature = (input, config=None))]
fn run_scheimpflug_intrinsics(
    py: Python<'_>,
    input: &Bound<'_, PyAny>,
    config: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    run_problem::<vision_calibration::scheimpflug_intrinsics::ScheimpflugIntrinsicsProblem, _>(
        py,
        input,
        config,
        |session| vision_calibration::scheimpflug_intrinsics::run_calibration(session, None),
    )
}

/// Run rig laserline device calibration.
///
/// Parameters
/// ----------
/// input:
///     Rig laserline input payload (serde-compatible with `RigLaserlineDeviceInput`).
/// config:
///     Optional config payload (serde-compatible with `RigLaserlineDeviceConfig`).
///
/// Returns
/// -------
/// dict
///     Rig laserline device export payload.
#[pyfunction(signature = (input, config=None))]
fn run_rig_laserline_device(
    py: Python<'_>,
    input: &Bound<'_, PyAny>,
    config: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    run_problem::<vision_calibration::rig_laserline_device::RigLaserlineDeviceProblem, _>(
        py,
        input,
        config,
        vision_calibration::rig_laserline_device::run_calibration,
    )
}

/// Map a laser pixel in a rig camera to a 3D point in the robot gripper frame.
///
/// Parameters
/// ----------
/// cam_idx:
///     Index of the camera that observed the pixel.
/// pixel:
///     Observed pixel as ``[u, v]`` (2-element list or tuple).
/// rig_cal:
///     Rig hand-eye calibration export (serde-compatible with
///     `RigHandeyeExport`). Must be a Scheimpflug rig (i.e. its `sensors` field
///     populated); pinhole rigs are not yet supported by this helper.
/// laser_planes_rig:
///     Per-camera laser planes expressed in rig frame (list of serde-compatible
///     `LaserPlane` objects).
/// base_se3_gripper:
///     Robot gripper pose at observation time (serde-compatible with `Iso3`).
///     Required for `EyeToHand` mode; ignored for `EyeInHand`.
///
/// Returns
/// -------
/// list[float]
///     3D point ``[x, y, z]`` in gripper frame.
///
/// Raises
/// ------
/// ValueError
///     If `cam_idx` is out of range, any payload contains non-finite values,
///     a required argument is missing, or the rig calibration is pinhole.
/// RuntimeError
///     If undistortion fails or the ray is parallel to the laser plane.
#[pyfunction(signature = (cam_idx, pixel, rig_cal, laser_planes_rig, base_se3_gripper=None))]
fn pixel_to_gripper_point(
    py: Python<'_>,
    cam_idx: usize,
    pixel: &Bound<'_, PyAny>,
    rig_cal: &Bound<'_, PyAny>,
    laser_planes_rig: &Bound<'_, PyAny>,
    base_se3_gripper: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    use vision_calibration::core::{Iso3, Pt2};
    use vision_calibration::optim::LaserPlane;
    use vision_calibration::rig_handeye::RigHandeyeExport;

    reject_non_finite("pixel", pixel)?;
    reject_non_finite("rig_cal", rig_cal)?;
    reject_non_finite("laser_planes_rig", laser_planes_rig)?;
    if let Some(p) = base_se3_gripper {
        reject_non_finite("base_se3_gripper", p)?;
    }

    let pixel_arr: [f64; 2] = parse_payload(pixel, "pixel")?;
    let px = Pt2::new(pixel_arr[0], pixel_arr[1]);

    let rig_cal: RigHandeyeExport = parse_payload(rig_cal, "rig_cal")?;
    let planes: Vec<LaserPlane> = parse_payload(laser_planes_rig, "laser_planes_rig")?;
    let pose: Option<Iso3> = match base_se3_gripper {
        None => None,
        Some(p) => Some(parse_payload(p, "base_se3_gripper")?),
    };

    let pt = vision_calibration::pixel_to_gripper_point(cam_idx, px, &rig_cal, &planes, pose)
        .map_err(|err| {
            // Match on the typed enum: any input-validation variant → ValueError;
            // numerical / propagated errors → RuntimeError.
            match &err {
                vision_calibration::Error::InvalidInput { .. }
                | vision_calibration::Error::InsufficientData { .. }
                | vision_calibration::Error::NotAvailable { .. } => value_err(err.to_string()),
                _ => runtime_err(err.to_string()),
            }
        })?;

    let arr = vec![pt.x, pt.y, pt.z];
    let output = pythonize(py, &arr)
        .map_err(|err| runtime_err(format!("failed to convert point to Python: {err}")))?;
    Ok(output.unbind())
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
    m.add_function(wrap_pyfunction!(run_scheimpflug_intrinsics, m)?)?;
    m.add_function(wrap_pyfunction!(run_rig_laserline_device, m)?)?;
    m.add_function(wrap_pyfunction!(pixel_to_gripper_point, m)?)?;
    m.add_function(wrap_pyfunction!(library_version, m)?)?;
    Ok(())
}
