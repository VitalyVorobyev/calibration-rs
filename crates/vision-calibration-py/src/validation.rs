use numpy::{PyReadonlyArray2, PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{
    PyBool, PyByteArray, PyBytes, PyDict, PyFloat, PyInt, PyList, PySequence, PyString, PyTuple,
};

pub(crate) fn value_err(message: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(message.to_string())
}

/// Row count of an `(N, cols)` array, or a `ValueError` naming `name`.
pub(crate) fn array_rows(
    array: &PyReadonlyArray2<'_, f64>,
    cols: usize,
    name: &str,
) -> PyResult<usize> {
    let shape = array.shape();
    if shape[1] != cols {
        return Err(value_err(format!(
            "{name} must have shape (N, {cols}), got ({}, {})",
            shape[0], shape[1]
        )));
    }
    Ok(shape[0])
}

/// Recursively walk a Python payload and reject non-finite float leaves.
///
/// `depythonize` happily accepts NaN / ±Inf and passes them into the Rust
/// algorithms, where they corrupt least-squares solves without a clear error.
/// Rejecting them at the Python boundary gives users an actionable message
/// pointing at their input.
pub(crate) fn reject_non_finite(name: &str, value: &Bound<'_, PyAny>) -> PyResult<()> {
    walk(name, value, 0)
}

const MAX_DEPTH: u32 = 32;

fn walk(name: &str, value: &Bound<'_, PyAny>, depth: u32) -> PyResult<()> {
    if depth > MAX_DEPTH {
        return Ok(());
    }
    if value.is_none()
        || value.is_instance_of::<PyString>()
        || value.is_instance_of::<PyBool>()
        || value.is_instance_of::<PyInt>()
        || value.is_instance_of::<PyBytes>()
        || value.is_instance_of::<PyByteArray>()
    {
        return Ok(());
    }
    if let Ok(f) = value.cast::<PyFloat>() {
        let v: f64 = f.extract()?;
        if !v.is_finite() {
            return Err(value_err(format!(
                "{name}: non-finite float ({v}); inputs must contain only finite values"
            )));
        }
        return Ok(());
    }
    if let Ok(arr) = value.extract::<PyReadonlyArrayDyn<'_, f64>>() {
        if let Some(v) = arr.as_array().iter().copied().find(|v| !v.is_finite()) {
            return Err(value_err(format!(
                "{name}: non-finite float ({v}); inputs must contain only finite values"
            )));
        }
        return Ok(());
    }
    if let Ok(arr) = value.extract::<PyReadonlyArrayDyn<'_, f32>>() {
        if let Some(v) = arr.as_array().iter().copied().find(|v| !v.is_finite()) {
            return Err(value_err(format!(
                "{name}: non-finite float ({v}); inputs must contain only finite values"
            )));
        }
        return Ok(());
    }
    if let Ok(dict) = value.cast::<PyDict>() {
        for (k, v) in dict.iter() {
            let key_repr: String = k
                .str()
                .map(|s| s.to_string())
                .unwrap_or_else(|_| "?".into());
            walk(&format!("{name}.{key_repr}"), &v, depth + 1)?;
        }
        return Ok(());
    }
    if let Ok(list) = value.cast::<PyList>() {
        for (i, item) in list.iter().enumerate() {
            walk(&format!("{name}[{i}]"), &item, depth + 1)?;
        }
        return Ok(());
    }
    if let Ok(tup) = value.cast::<PyTuple>() {
        for (i, item) in tup.iter().enumerate() {
            walk(&format!("{name}[{i}]"), &item, depth + 1)?;
        }
        return Ok(());
    }
    if let Ok(seq) = value.cast::<PySequence>() {
        let len = seq.len()?;
        for i in 0..len {
            let item = seq.get_item(i)?;
            walk(&format!("{name}[{i}]"), &item, depth + 1)?;
        }
        return Ok(());
    }
    // Catch numpy float scalars (np.float64/32/16) and other Python numeric scalars.
    if let Ok(v) = value.extract::<f64>()
        && !v.is_finite()
    {
        return Err(value_err(format!(
            "{name}: non-finite float ({v}); inputs must contain only finite values"
        )));
    }
    Ok(())
}
