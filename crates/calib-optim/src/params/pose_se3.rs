//! SE(3) parameter conversions for tiny-solver.

use anyhow::{ensure, Result};
use calib_core::Iso3;
use nalgebra::{DVector, DVectorView, Quaternion, UnitQuaternion, Vector3};

/// Convert an `Iso3` into a 7D SE(3) parameter vector `[qx, qy, qz, qw, tx, ty, tz]`.
pub fn iso3_to_se3_dvec(pose: &Iso3) -> DVector<f64> {
    let q = pose.rotation.clone().into_inner();
    let t = pose.translation.vector;
    nalgebra::dvector![
        q.coords[0],
        q.coords[1],
        q.coords[2],
        q.coords[3],
        t.x,
        t.y,
        t.z
    ]
}

/// Convert a 7D SE(3) vector `[qx, qy, qz, qw, tx, ty, tz]` into an `Iso3`.
pub fn se3_dvec_to_iso3(v: DVectorView<'_, f64>) -> Result<Iso3> {
    ensure!(
        v.len() == 7,
        "expected se3 vector of length 7, got {}",
        v.len()
    );
    let quat = Quaternion::new(v[3], v[0], v[1], v[2]);
    let rot = UnitQuaternion::from_quaternion(quat);
    let trans = Vector3::new(v[4], v[5], v[6]);
    Ok(Iso3::from_parts(trans.into(), rot))
}
