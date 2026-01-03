//! Backend-independent reprojection residual models.

use crate::math::projection::project_pinhole;
use nalgebra::{DVector, DVectorView, Quaternion, RealField, SVector, UnitQuaternion, Vector3};

/// Compute a 2D reprojection residual for pinhole intrinsics and SE3 pose.
///
/// The residual is scaled by `sqrt(w)` and ordered `[u_residual, v_residual]`.
pub fn reproj_residual_pinhole4_se3(
    intr: &DVector<f64>,
    pose: &DVector<f64>,
    pw: [f64; 3],
    uv: [f64; 2],
    w: f64,
) -> SVector<f64, 2> {
    reproj_residual_pinhole4_se3_generic(intr.as_view(), pose.as_view(), pw, uv, w)
}

/// Generic reprojection residual evaluator for backend adapters.
pub(crate) fn reproj_residual_pinhole4_se3_generic<T: RealField>(
    intr: DVectorView<'_, T>,
    pose: DVectorView<'_, T>,
    pw: [f64; 3],
    uv: [f64; 2],
    w: f64,
) -> SVector<T, 2> {
    debug_assert!(intr.len() >= 4, "intrinsics must have 4 params");
    debug_assert!(pose.len() == 7, "pose must have 7 params");

    let fx = intr[0].clone();
    let fy = intr[1].clone();
    let cx = intr[2].clone();
    let cy = intr[3].clone();

    let qx = pose[0].clone();
    let qy = pose[1].clone();
    let qz = pose[2].clone();
    let qw = pose[3].clone();
    let tx = pose[4].clone();
    let ty = pose[5].clone();
    let tz = pose[6].clone();

    let quat = Quaternion::new(qw, qx, qy, qz);
    let rot = UnitQuaternion::from_quaternion(quat);
    let t = Vector3::new(tx, ty, tz);

    let pw_t = Vector3::new(
        T::from_f64(pw[0]).unwrap(),
        T::from_f64(pw[1]).unwrap(),
        T::from_f64(pw[2]).unwrap(),
    );
    let pc = rot.transform_vector(&pw_t) + t;

    let proj = project_pinhole(fx, fy, cx, cy, pc);
    let sqrt_w = T::from_f64(w.sqrt()).unwrap();
    let u_meas = T::from_f64(uv[0]).unwrap();
    let v_meas = T::from_f64(uv[1]).unwrap();
    let ru = (u_meas - proj.x.clone()) * sqrt_w.clone();
    let rv = (v_meas - proj.y.clone()) * sqrt_w;
    SVector::<T, 2>::new(ru, rv)
}
