//! Device seeds: the device quantities a calibration starts from, from a
//! device spec or a calibration export.

use pyo3::prelude::*;
use pythonize::pythonize;
use vision_calibration::device_seed::{DeviceSeed, DeviceSeedError, DeviceSpec};

use crate::validation::value_err;
use crate::{parse_cameras, parse_payload, runtime_err};

fn seed_err(err: DeviceSeedError) -> PyErr {
    value_err(err.to_string())
}

/// A seed payload, parsed and validated.
fn parse_seed(seed: &Bound<'_, PyAny>) -> PyResult<DeviceSeed> {
    let seed: DeviceSeed = parse_payload(seed, "seed")?;
    seed.validate().map_err(seed_err)?;
    Ok(seed)
}

fn seed_payload(py: Python<'_>, seed: &DeviceSeed) -> PyResult<Py<PyAny>> {
    let output = pythonize(py, seed)
        .map_err(|err| runtime_err(format!("failed to convert the seed to Python: {err}")))?;
    Ok(output.unbind())
}

/// The device seed of a calibration export: its cameras whole, the mounts of
/// a rig, its laser planes and its hand-eye.
///
/// Raises
/// ------
/// ValueError
///     If the ids or sizes are not one per camera, an id repeats, or a camera
///     has a homography sensor.
/// TypeError
///     If `export` is not a calibration export.
#[pyfunction]
fn device_seed_from_calibration(
    py: Python<'_>,
    export: &Bound<'_, PyAny>,
    camera_ids: Vec<String>,
    image_sizes: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    let image_sizes: Vec<[u32; 2]> = parse_payload(image_sizes, "image_sizes")?;
    let export = parse_cameras(export, "export")?;
    let seed = DeviceSeed::from_calibration(export.as_ref(), &camera_ids, &image_sizes)
        .map_err(seed_err)?;
    seed_payload(py, &seed)
}

/// The datasheet seed of a device spec payload.
///
/// Raises
/// ------
/// ValueError
///     If the spec does not validate or mounts only some cameras.
/// TypeError
///     If `spec` is not a device spec payload.
#[pyfunction]
fn device_seed_from_spec(py: Python<'_>, spec: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let spec: DeviceSpec = parse_payload(spec, "spec")?;
    let seed = DeviceSeed::from_spec(&spec).map_err(seed_err)?;
    seed_payload(py, &seed)
}

/// A seed payload, validated and written back.
#[pyfunction]
fn device_seed_validate(py: Python<'_>, seed: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let seed = parse_seed(seed)?;
    seed_payload(py, &seed)
}

/// The seed with every calibrated lens reduced to its intrinsics and tilt.
#[pyfunction]
fn device_seed_nominal(py: Python<'_>, seed: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let seed = parse_seed(seed)?.nominal();
    seed_payload(py, &seed)
}

/// The seed of the cameras `camera_ids` only, in that order.
#[pyfunction]
fn device_seed_select(
    py: Python<'_>,
    seed: &Bound<'_, PyAny>,
    camera_ids: Vec<String>,
) -> PyResult<Py<PyAny>> {
    let ids: Vec<&str> = camera_ids.iter().map(String::as_str).collect();
    let seed = parse_seed(seed)?.select(&ids).map_err(seed_err)?;
    seed_payload(py, &seed)
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(device_seed_from_calibration, m)?)?;
    m.add_function(wrap_pyfunction!(device_seed_from_spec, m)?)?;
    m.add_function(wrap_pyfunction!(device_seed_validate, m)?)?;
    m.add_function(wrap_pyfunction!(device_seed_nominal, m)?)?;
    m.add_function(wrap_pyfunction!(device_seed_select, m)?)?;
    Ok(())
}
