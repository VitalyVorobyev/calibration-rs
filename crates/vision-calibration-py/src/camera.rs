//! `CameraModel`: a calibrated camera built from its `CameraParams`, with
//! vectorized projection over numpy arrays.

use numpy::{PyArray2, PyArrayMethods, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::prelude::*;
use pythonize::pythonize;
use vision_calibration::core::{CameraModel, CameraParams, Pt2, Vec3};

use crate::validation::{reject_non_finite, value_err};
use crate::{parse_payload, runtime_err};

/// Arrays at least this long are processed with the GIL released; shorter
/// ones are not worth the hand-off.
const DETACH_MIN_LEN: usize = 1024;

/// A camera model built from `CameraParams`: projection, distortion, sensor
/// (identity or Scheimpflug tilt) and intrinsics.
#[pyclass(
    module = "vision_calibration._vision_calibration",
    name = "CameraModel",
    frozen
)]
pub(crate) struct PyCameraModel {
    model: CameraModel,
    params: CameraParams,
}

#[pymethods]
impl PyCameraModel {
    /// Build the model from a `CameraParams` payload.
    #[new]
    fn new(params: &Bound<'_, PyAny>) -> PyResult<Self> {
        reject_non_finite("params", params)?;
        let params: CameraParams = parse_payload(params, "params")?;
        let model = params
            .build()
            .map_err(|err| value_err(format!("cannot build the camera model: {err}")))?;
        Ok(Self { model, params })
    }

    /// The `CameraParams` payload the model was built from.
    fn params(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        Ok(pythonize(py, &self.params)
            .map_err(|err| runtime_err(format!("failed to convert params: {err}")))?
            .unbind())
    }

    /// Project `(N, 3)` camera-frame points to `(N, 2)` pixels; NaN rows for
    /// points outside the projection's domain.
    fn project<'py>(
        &self,
        py: Python<'py>,
        points: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let n = rows(&points, 3, "points_3d")?;
        let input = points.as_slice()?;
        let mut out = vec![0.0; n * 2];
        run(py, n, || project_points(&self.model, input, &mut out));
        to_array(py, out, n, 2)
    }

    /// Back-project `(N, 2)` pixels to camera rays: `(origins, directions)`,
    /// each `(N, 3)`.
    #[allow(clippy::type_complexity)] // a (origins, directions) pair of arrays
    fn unproject<'py>(
        &self,
        py: Python<'py>,
        pixels: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<f64>>)> {
        let n = rows(&pixels, 2, "pixels")?;
        let input = pixels.as_slice()?;
        let mut origins = vec![0.0; n * 3];
        let mut directions = vec![0.0; n * 3];
        run(py, n, || {
            unproject_pixels(&self.model, input, &mut origins, &mut directions)
        });
        Ok((
            to_array(py, origins, n, 3)?,
            to_array(py, directions, n, 3)?,
        ))
    }

    /// Map `(N, 2)` pixels to where they land in an undistorted pinhole image
    /// with the same intrinsics; NaN rows for pixels whose ray misses the
    /// image plane.
    fn undistort<'py>(
        &self,
        py: Python<'py>,
        pixels: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let n = rows(&pixels, 2, "pixels")?;
        let input = pixels.as_slice()?;
        let mut out = vec![0.0; n * 2];
        run(py, n, || undistort_pixels(&self.model, input, &mut out));
        to_array(py, out, n, 2)
    }
}

/// Row count of an `(N, cols)` array, or a `ValueError` naming `name`.
fn rows(array: &PyReadonlyArray2<'_, f64>, cols: usize, name: &str) -> PyResult<usize> {
    let shape = array.shape();
    if shape[1] != cols {
        return Err(value_err(format!(
            "{name} must have shape (N, {cols}), got ({}, {})",
            shape[0], shape[1]
        )));
    }
    Ok(shape[0])
}

/// Run `work` over `n` items, releasing the GIL when `n` is large.
fn run<F: FnOnce() + Send>(py: Python<'_>, n: usize, work: F) {
    if n >= DETACH_MIN_LEN {
        py.detach(work);
    } else {
        work();
    }
}

fn to_array<'py>(
    py: Python<'py>,
    data: Vec<f64>,
    n: usize,
    cols: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    numpy::PyArray1::from_vec(py, data).reshape([n, cols])
}

/// Project row-major `(N, 3)` camera-frame points into row-major `(N, 2)`
/// pixels. A point outside the projection's domain yields `[NaN, NaN]`.
pub(crate) fn project_points(model: &CameraModel, points: &[f64], out: &mut [f64]) {
    for (p, o) in points
        .as_chunks::<3>()
        .0
        .iter()
        .zip(out.as_chunks_mut::<2>().0)
    {
        let px = model.project_point_c(&Vec3::new(p[0], p[1], p[2]));
        *o = px.map_or([f64::NAN; 2], |px| [px.x, px.y]);
    }
}

/// Back-project row-major `(N, 2)` pixels into row-major `(N, 3)` ray
/// origins and directions, in the camera frame.
pub(crate) fn unproject_pixels(
    model: &CameraModel,
    pixels: &[f64],
    origins: &mut [f64],
    directions: &mut [f64],
) {
    for ((p, o), d) in pixels
        .as_chunks::<2>()
        .0
        .iter()
        .zip(origins.as_chunks_mut::<3>().0)
        .zip(directions.as_chunks_mut::<3>().0)
    {
        let ray = model.backproject_pixel(&Pt2::new(p[0], p[1]));
        o.copy_from_slice(ray.origin.as_slice());
        d.copy_from_slice(ray.direction.as_slice());
    }
}

/// Map row-major `(N, 2)` pixels to the undistorted pinhole image with the
/// model's intrinsics: the pixel's ray, intersected with the plane `z = 1`,
/// through `K`. A ray that misses that plane yields `[NaN, NaN]`.
pub(crate) fn undistort_pixels(model: &CameraModel, pixels: &[f64], out: &mut [f64]) {
    let k = &model.k;
    for (p, o) in pixels
        .as_chunks::<2>()
        .0
        .iter()
        .zip(out.as_chunks_mut::<2>().0)
    {
        let ray = model.backproject_pixel(&Pt2::new(p[0], p[1]));
        *o = ray
            .point_at_unit_depth()
            .map_or([f64::NAN; 2], |n| [k.fx * n.x + k.cx, k.fy * n.y + k.cy]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params(json: serde_json::Value) -> CameraModel {
        serde_json::from_value::<CameraParams>(json)
            .unwrap()
            .build()
            .unwrap()
    }

    fn brown_conrady() -> CameraModel {
        params(serde_json::json!({
            "projection": {"type": "pinhole"},
            "distortion": {"type": "brown_conrady5",
                           "k1": -0.21, "k2": 0.08, "k3": 0.0, "p1": 0.001, "p2": -0.0005,
                           "iters": 8},
            "sensor": {"type": "identity"},
            "intrinsics": {"fx": 800.0, "fy": 780.0, "cx": 640.0, "cy": 360.0},
        }))
    }

    fn scheimpflug() -> CameraModel {
        params(serde_json::json!({
            "projection": {"type": "pinhole"},
            "distortion": {"type": "none"},
            "sensor": {"type": "scheimpflug", "tilt_x": 0.05, "tilt_y": -0.03},
            "intrinsics": {"fx": 1200.0, "fy": 1190.0, "cx": 512.0, "cy": 384.0},
        }))
    }

    #[test]
    fn project_matches_the_core_model_and_marks_points_behind() {
        let model = brown_conrady();
        let points = [0.1, -0.05, 1.0, 0.0, 0.0, -1.0];
        let mut out = [0.0; 4];
        project_points(&model, &points, &mut out);
        let expected = model.project_point_c(&Vec3::new(0.1, -0.05, 1.0)).unwrap();
        assert_eq!([out[0], out[1]], [expected.x, expected.y]);
        assert!(out[2].is_nan() && out[3].is_nan());
    }

    #[test]
    fn unproject_inverts_project() {
        for model in [brown_conrady(), scheimpflug()] {
            let points = [0.1, -0.05, 1.0, -0.2, 0.15, 2.0];
            let mut px = [0.0; 4];
            project_points(&model, &points, &mut px);
            let mut origins = [0.0; 6];
            let mut dirs = [0.0; 6];
            unproject_pixels(&model, &px, &mut origins, &mut dirs);
            for i in 0..2 {
                let p = &points[3 * i..3 * i + 3];
                let d = &dirs[3 * i..3 * i + 3];
                // The direction is parallel to the point: the cross product vanishes.
                let cross = [
                    p[1] * d[2] - p[2] * d[1],
                    p[2] * d[0] - p[0] * d[2],
                    p[0] * d[1] - p[1] * d[0],
                ];
                assert!(cross.iter().all(|c| c.abs() < 1e-9), "{cross:?}");
                assert_eq!(&origins[3 * i..3 * i + 3], &[0.0; 3]);
            }
        }
    }

    #[test]
    fn undistort_lands_on_the_pinhole_projection() {
        for model in [brown_conrady(), scheimpflug()] {
            let points = [0.1, -0.05, 1.0, -0.2, 0.15, 2.0];
            let mut px = [0.0; 4];
            project_points(&model, &points, &mut px);
            let mut out = [0.0; 4];
            undistort_pixels(&model, &px, &mut out);
            let k = &model.k;
            for i in 0..2 {
                let p = &points[3 * i..3 * i + 3];
                let ideal = [k.fx * p[0] / p[2] + k.cx, k.fy * p[1] / p[2] + k.cy];
                assert!((out[2 * i] - ideal[0]).abs() < 1e-6, "{out:?} vs {ideal:?}");
                assert!(
                    (out[2 * i + 1] - ideal[1]).abs() < 1e-6,
                    "{out:?} vs {ideal:?}"
                );
            }
        }
    }
}
