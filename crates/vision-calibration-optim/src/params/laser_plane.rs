//! Laser plane parameters for optimization.
//!
//! A laser plane is parameterized as (n̂, d) where n̂ is a unit normal and d
//! the signed distance from the camera origin ([`LaserPlane`]). In
//! optimization, the normal is optimized on the S2 manifold (unit sphere),
//! while the distance is a separate Euclidean scalar.

use crate::Error;
use nalgebra::{DVector, DVectorView, Vector3};
use vision_calibration_core::LaserPlane;

/// Convert the unit normal to a 3D parameter vector [nx, ny, nz].
pub(crate) fn normal_to_dvec(plane: &LaserPlane) -> DVector<f64> {
    let n = plane.normal();
    DVector::from_vec(vec![n.x, n.y, n.z])
}

/// Convert the distance to a 1D parameter vector \[d\].
pub(crate) fn distance_to_dvec(plane: &LaserPlane) -> DVector<f64> {
    DVector::from_vec(vec![plane.distance()])
}

/// Parse from split normal + distance vectors; the normal is renormalized
/// (the optimizer's iterate is only unit up to its retraction).
///
/// # Errors
///
/// [`Error::InvalidInput`] if `normal` is not length 3, `distance` is not
/// length 1, or a value is not finite or the normal is zero.
pub(crate) fn plane_from_split_dvec(
    normal: DVectorView<f64>,
    distance: DVectorView<f64>,
) -> Result<LaserPlane, Error> {
    if normal.len() != 3 {
        return Err(Error::invalid_input(format!(
            "LaserPlane normal requires 3D vector, got {}",
            normal.len()
        )));
    }
    if distance.len() != 1 {
        return Err(Error::invalid_input(format!(
            "LaserPlane distance requires 1D vector, got {}",
            distance.len()
        )));
    }
    let n = Vector3::new(normal[0], normal[1], normal[2]);
    let length = n.norm();
    if !(length.is_finite() && length > 0.0) {
        return Err(Error::invalid_input(
            "LaserPlane normal must be finite and nonzero",
        ));
    }
    Ok(LaserPlane::new(n / length, distance[0])?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plane_split_roundtrip() {
        let plane = LaserPlane::new((Vector3::new(1.0, 2.0, 3.0)).normalize(), -0.4).unwrap();

        let normal = normal_to_dvec(&plane);
        let distance = distance_to_dvec(&plane);
        let plane2 = plane_from_split_dvec(normal.as_view(), distance.as_view()).unwrap();

        assert!((plane.normal().x - plane2.normal().x).abs() < 1e-10);
        assert!((plane.normal().y - plane2.normal().y).abs() < 1e-10);
        assert!((plane.normal().z - plane2.normal().z).abs() < 1e-10);
        assert!((plane.distance() - plane2.distance()).abs() < 1e-10);
    }

    #[test]
    fn split_rejects_degenerate_vectors() {
        let d = DVector::from_vec(vec![0.1]);
        assert!(plane_from_split_dvec(DVector::zeros(3).as_view(), d.as_view()).is_err());
        assert!(plane_from_split_dvec(DVector::zeros(2).as_view(), d.as_view()).is_err());
    }
}
