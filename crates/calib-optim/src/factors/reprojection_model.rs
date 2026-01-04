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

/// Apply Brown-Conrady distortion to normalized coordinates (generic for autodiff).
///
/// This implements the Brown-Conrady distortion model with radial (k1, k2, k3)
/// and tangential (p1, p2) coefficients. The function is generic over `RealField`
/// to support automatic differentiation.
fn distort_brown_conrady_generic<T: RealField>(
    x: T,
    y: T,
    k1: T,
    k2: T,
    k3: T,
    p1: T,
    p2: T,
) -> (T, T) {
    let r2 = x.clone() * x.clone() + y.clone() * y.clone();
    let r4 = r2.clone() * r2.clone();
    let r6 = r4.clone() * r2.clone();

    let radial = T::one() + k1 * r2.clone() + k2 * r4 + k3 * r6;

    let two = T::one() + T::one();
    let x2 = x.clone() * x.clone();
    let y2 = y.clone() * y.clone();
    let xy = x.clone() * y.clone();

    let x_tan =
        two.clone() * p1.clone() * xy.clone() + p2.clone() * (r2.clone() + two.clone() * x2);
    let y_tan = p1 * (r2 + two.clone() * y2) + two * p2 * xy;

    (x.clone() * radial.clone() + x_tan, y * radial + y_tan)
}

/// Compute a 2D reprojection residual with pinhole intrinsics, distortion, and SE3 pose.
///
/// The residual is scaled by `sqrt(w)` and ordered `[u_residual, v_residual]`.
///
/// # Parameters
/// - `intr`: Intrinsics vector `[fx, fy, cx, cy]`
/// - `dist`: Distortion vector `[k1, k2, k3, p1, p2]`
/// - `pose`: SE3 pose `[qx, qy, qz, qw, tx, ty, tz]`
/// - `pw`: 3D point in world coordinates
/// - `uv`: 2D measured pixel coordinates
/// - `w`: Weight for this observation
pub(crate) fn reproj_residual_pinhole4_dist5_se3_generic<T: RealField>(
    intr: DVectorView<'_, T>,
    dist: DVectorView<'_, T>,
    pose: DVectorView<'_, T>,
    pw: [f64; 3],
    uv: [f64; 2],
    w: f64,
) -> SVector<T, 2> {
    debug_assert!(intr.len() >= 4, "intrinsics must have 4 params");
    debug_assert!(dist.len() >= 5, "distortion must have 5 params");
    debug_assert!(pose.len() == 7, "pose must have 7 params");

    let fx = intr[0].clone();
    let fy = intr[1].clone();
    let cx = intr[2].clone();
    let cy = intr[3].clone();

    let k1 = dist[0].clone();
    let k2 = dist[1].clone();
    let k3 = dist[2].clone();
    let p1 = dist[3].clone();
    let p2 = dist[4].clone();

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

    // Transform to camera frame
    let pw_t = Vector3::new(
        T::from_f64(pw[0]).unwrap(),
        T::from_f64(pw[1]).unwrap(),
        T::from_f64(pw[2]).unwrap(),
    );
    let pc = rot.transform_vector(&pw_t) + t;

    // Project to normalized coordinates
    let eps = T::from_f64(1e-12).unwrap();
    let z_safe = if pc.z.clone() > eps.clone() {
        pc.z.clone()
    } else {
        eps
    };
    let x_norm = pc.x.clone() / z_safe.clone();
    let y_norm = pc.y.clone() / z_safe;

    // Apply distortion
    let (x_dist, y_dist) = distort_brown_conrady_generic(x_norm, y_norm, k1, k2, k3, p1, p2);

    // Apply intrinsics
    let u_proj = fx * x_dist + cx;
    let v_proj = fy * y_dist + cy;

    // Compute weighted residual
    let sqrt_w = T::from_f64(w.sqrt()).unwrap();
    let u_meas = T::from_f64(uv[0]).unwrap();
    let v_meas = T::from_f64(uv[1]).unwrap();

    let ru = (u_meas - u_proj) * sqrt_w.clone();
    let rv = (v_meas - v_proj) * sqrt_w;

    SVector::<T, 2>::new(ru, rv)
}

/// Compute tilt projection matrix for Scheimpflug sensor (generic for autodiff).
///
/// This implements the OpenCV-compatible tilted sensor model using rotations
/// around X and Y axes followed by z-normalization projection.
fn tilt_projection_matrix_generic<T: RealField>(tau_x: T, tau_y: T) -> nalgebra::Matrix3<T> {
    let s_tx = tau_x.clone().sin();
    let c_tx = tau_x.cos();
    let s_ty = tau_y.clone().sin();
    let c_ty = tau_y.cos();

    let zero = T::zero();
    let one = T::one();

    let rot_x = nalgebra::Matrix3::new(
        one.clone(),
        zero.clone(),
        zero.clone(),
        zero.clone(),
        c_tx.clone(),
        s_tx.clone(),
        zero.clone(),
        -s_tx.clone(),
        c_tx,
    );
    let rot_y = nalgebra::Matrix3::new(
        c_ty.clone(),
        zero.clone(),
        -s_ty.clone(),
        zero.clone(),
        one.clone(),
        zero.clone(),
        s_ty.clone(),
        zero.clone(),
        c_ty,
    );
    let rot_xy = rot_y * rot_x;

    let r22 = rot_xy[(2, 2)].clone();
    let r02 = rot_xy[(0, 2)].clone();
    let r12 = rot_xy[(1, 2)].clone();

    nalgebra::Matrix3::new(
        r22.clone(),
        zero.clone(),
        -r02,
        zero.clone(),
        r22.clone(),
        -r12,
        zero.clone(),
        zero.clone(),
        one,
    ) * rot_xy
}

/// Apply Scheimpflug sensor homography to normalized coordinates (generic for autodiff).
fn apply_scheimpflug_generic<T: RealField>(x_norm: T, y_norm: T, tau_x: T, tau_y: T) -> (T, T) {
    let h = tilt_projection_matrix_generic(tau_x, tau_y);
    let p = nalgebra::Vector3::new(x_norm, y_norm, T::one());
    let p_tilted = h * p;
    let eps = T::from_f64(1e-12).unwrap();
    let z_safe = if p_tilted.z.clone() > eps.clone() {
        p_tilted.z.clone()
    } else {
        eps
    };
    (
        p_tilted.x.clone() / z_safe.clone(),
        p_tilted.y.clone() / z_safe,
    )
}

/// Compute reprojection residual with Scheimpflug sensor, distortion, intrinsics, and SE3 pose.
///
/// # Parameters
/// - `intr`: Intrinsics vector `[fx, fy, cx, cy]`
/// - `dist`: Distortion vector `[k1, k2, k3, p1, p2]`
/// - `sensor`: Scheimpflug parameters `[tilt_x, tilt_y]`
/// - `pose`: SE3 pose `[qx, qy, qz, qw, tx, ty, tz]`
/// - `pw`: 3D point in world coordinates
/// - `uv`: 2D measured pixel coordinates
/// - `w`: Weight for this observation
pub(crate) fn reproj_residual_pinhole4_dist5_scheimpflug2_se3_generic<T: RealField>(
    intr: DVectorView<'_, T>,
    dist: DVectorView<'_, T>,
    sensor: DVectorView<'_, T>,
    pose: DVectorView<'_, T>,
    pw: [f64; 3],
    uv: [f64; 2],
    w: f64,
) -> SVector<T, 2> {
    debug_assert!(intr.len() >= 4, "intrinsics must have 4 params");
    debug_assert!(dist.len() >= 5, "distortion must have 5 params");
    debug_assert!(sensor.len() >= 2, "sensor must have 2 params");
    debug_assert!(pose.len() == 7, "pose must have 7 params");

    let fx = intr[0].clone();
    let fy = intr[1].clone();
    let cx = intr[2].clone();
    let cy = intr[3].clone();

    let k1 = dist[0].clone();
    let k2 = dist[1].clone();
    let k3 = dist[2].clone();
    let p1 = dist[3].clone();
    let p2 = dist[4].clone();

    let tau_x = sensor[0].clone();
    let tau_y = sensor[1].clone();

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

    // Transform to camera frame
    let pw_t = Vector3::new(
        T::from_f64(pw[0]).unwrap(),
        T::from_f64(pw[1]).unwrap(),
        T::from_f64(pw[2]).unwrap(),
    );
    let pc = rot.transform_vector(&pw_t) + t;

    // Project to normalized coordinates
    let eps = T::from_f64(1e-12).unwrap();
    let z_safe = if pc.z.clone() > eps.clone() {
        pc.z.clone()
    } else {
        eps
    };
    let x_norm = pc.x.clone() / z_safe.clone();
    let y_norm = pc.y.clone() / z_safe;

    // Apply distortion
    let (x_dist, y_dist) = distort_brown_conrady_generic(x_norm, y_norm, k1, k2, k3, p1, p2);

    // Apply Scheimpflug sensor transformation
    let (x_sensor, y_sensor) = apply_scheimpflug_generic(x_dist, y_dist, tau_x, tau_y);

    // Apply intrinsics
    let u_proj = fx * x_sensor + cx;
    let v_proj = fy * y_sensor + cy;

    // Compute weighted residual
    let sqrt_w = T::from_f64(w.sqrt()).unwrap();
    let u_meas = T::from_f64(uv[0]).unwrap();
    let v_meas = T::from_f64(uv[1]).unwrap();

    let ru = (u_meas - u_proj) * sqrt_w.clone();
    let rv = (v_meas - v_proj) * sqrt_w;

    SVector::<T, 2>::new(ru, rv)
}

/// Reprojection residual with two composed SE3 transforms (for rig extrinsics).
///
/// Transform chain: P_camera = extr^-1 * pose * P_world
/// where extr is camera-to-rig and pose is rig-to-target.
///
/// # Parameters
/// - `intr`: [fx, fy, cx, cy] intrinsics
/// - `dist`: [k1, k2, k3, p1, p2] Brown-Conrady distortion
/// - `extr`: [qx, qy, qz, qw, tx, ty, tz] camera-to-rig SE3
/// - `pose`: [qx, qy, qz, qw, tx, ty, tz] rig-to-target SE3
/// - `pw`: 3D point in target/world frame
/// - `uv`: observed pixel coordinates
/// - `w`: weight for the residual
pub(crate) fn reproj_residual_pinhole4_dist5_two_se3_generic<T: RealField>(
    intr: DVectorView<'_, T>,
    dist: DVectorView<'_, T>,
    extr: DVectorView<'_, T>,
    pose: DVectorView<'_, T>,
    pw: [f64; 3],
    uv: [f64; 2],
    w: f64,
) -> SVector<T, 2> {
    use nalgebra::{Quaternion, UnitQuaternion, Vector3};

    // Extract SE3 from extr: [qx, qy, qz, qw, tx, ty, tz]
    let extr_q = UnitQuaternion::from_quaternion(Quaternion::new(
        extr[3].clone(),
        extr[0].clone(),
        extr[1].clone(),
        extr[2].clone(),
    ));
    let extr_t = Vector3::new(extr[4].clone(), extr[5].clone(), extr[6].clone());

    // Extract SE3 from pose
    let pose_q = UnitQuaternion::from_quaternion(Quaternion::new(
        pose[3].clone(),
        pose[0].clone(),
        pose[1].clone(),
        pose[2].clone(),
    ));
    let pose_t = Vector3::new(pose[4].clone(), pose[5].clone(), pose[6].clone());

    // Point in target frame
    let pw_t = Vector3::new(
        T::from_f64(pw[0]).unwrap(),
        T::from_f64(pw[1]).unwrap(),
        T::from_f64(pw[2]).unwrap(),
    );

    // Transform: target -> rig -> camera
    let p_rig = pose_q.transform_vector(&pw_t) + pose_t;
    let p_camera = extr_q.inverse_transform_vector(&(p_rig - extr_t));

    // Project to normalized coordinates
    let eps = T::from_f64(1e-12).unwrap();
    let z_safe = if p_camera.z.clone() > eps.clone() {
        p_camera.z.clone()
    } else {
        eps
    };
    let x_norm = p_camera.x.clone() / z_safe.clone();
    let y_norm = p_camera.y.clone() / z_safe;

    // Apply Brown-Conrady distortion
    let k1 = dist[0].clone();
    let k2 = dist[1].clone();
    let k3 = dist[2].clone();
    let p1 = dist[3].clone();
    let p2 = dist[4].clone();
    let (xd, yd) = distort_brown_conrady_generic(x_norm, y_norm, k1, k2, k3, p1, p2);

    // Apply intrinsics
    let fx = intr[0].clone();
    let fy = intr[1].clone();
    let cx = intr[2].clone();
    let cy = intr[3].clone();

    let u_pred = fx * xd + cx;
    let v_pred = fy * yd + cy;

    // Weighted residual
    let sqrt_w = T::from_f64(w.sqrt()).unwrap();
    let u_meas = T::from_f64(uv[0]).unwrap();
    let v_meas = T::from_f64(uv[1]).unwrap();

    SVector::<T, 2>::new(
        (u_meas - u_pred) * sqrt_w.clone(),
        (v_meas - v_pred) * sqrt_w,
    )
}

/// Reprojection residual for hand-eye calibration with robot pose as measurement.
///
/// Eye-in-hand: P_camera = extr^-1 * handeye^-1 * robot^-1 * target * P_world
/// Eye-to-hand: P_camera = extr^-1 * handeye * robot * target * P_world
///
/// # Parameters
/// - `intr`: [fx, fy, cx, cy] intrinsics
/// - `dist`: [k1, k2, k3, p1, p2] Brown-Conrady distortion
/// - `extr`: [qx, qy, qz, qw, tx, ty, tz] camera-to-rig SE3
/// - `handeye`: [qx, qy, qz, qw, tx, ty, tz] hand-eye SE3 (rig-to-gripper or rig-to-base)
/// - `target`: [qx, qy, qz, qw, tx, ty, tz] target pose SE3
/// - `robot_se3`: [qx, qy, qz, qw, tx, ty, tz] known robot pose (base-to-gripper)
/// - `mode`: `HandEyeMode` specifying transform chain
/// - `pw`: 3D point in target frame
/// - `uv`: observed pixel coordinates
/// - `w`: weight for the residual
pub(crate) fn reproj_residual_pinhole4_dist5_handeye_generic<T: RealField>(
    intr: DVectorView<'_, T>,
    dist: DVectorView<'_, T>,
    extr: DVectorView<'_, T>,
    handeye: DVectorView<'_, T>,
    target: DVectorView<'_, T>,
    robot_se3: &[f64; 7],
    mode: crate::ir::HandEyeMode,
    pw: [f64; 3],
    uv: [f64; 2],
    w: f64,
) -> SVector<T, 2> {
    use nalgebra::{Quaternion, UnitQuaternion, Vector3};

    // Extract all SE3 parameters
    let extr_q = UnitQuaternion::from_quaternion(Quaternion::new(
        extr[3].clone(),
        extr[0].clone(),
        extr[1].clone(),
        extr[2].clone(),
    ));
    let extr_t = Vector3::new(extr[4].clone(), extr[5].clone(), extr[6].clone());

    let handeye_q = UnitQuaternion::from_quaternion(Quaternion::new(
        handeye[3].clone(),
        handeye[0].clone(),
        handeye[1].clone(),
        handeye[2].clone(),
    ));
    let handeye_t = Vector3::new(handeye[4].clone(), handeye[5].clone(), handeye[6].clone());

    let target_q = UnitQuaternion::from_quaternion(Quaternion::new(
        target[3].clone(),
        target[0].clone(),
        target[1].clone(),
        target[2].clone(),
    ));
    let target_t = Vector3::new(target[4].clone(), target[5].clone(), target[6].clone());

    // Convert robot measurement to generic
    let robot_q: UnitQuaternion<T> = UnitQuaternion::from_quaternion(Quaternion::new(
        T::from_f64(robot_se3[3]).unwrap(),
        T::from_f64(robot_se3[0]).unwrap(),
        T::from_f64(robot_se3[1]).unwrap(),
        T::from_f64(robot_se3[2]).unwrap(),
    ));
    let robot_t = Vector3::new(
        T::from_f64(robot_se3[4]).unwrap(),
        T::from_f64(robot_se3[5]).unwrap(),
        T::from_f64(robot_se3[6]).unwrap(),
    );

    let pw_t = Vector3::new(
        T::from_f64(pw[0]).unwrap(),
        T::from_f64(pw[1]).unwrap(),
        T::from_f64(pw[2]).unwrap(),
    );

    // Compose transforms based on mode
    let p_camera = match mode {
        crate::ir::HandEyeMode::EyeInHand => {
            // target -> robot_base -> gripper -> rig -> camera
            let p_base = target_q.transform_vector(&pw_t) + target_t.clone();
            let p_gripper = robot_q.inverse_transform_vector(&(p_base - robot_t.clone()));
            let p_rig = handeye_q.inverse_transform_vector(&(p_gripper - handeye_t.clone()));
            extr_q.inverse_transform_vector(&(p_rig - extr_t.clone()))
        }
        crate::ir::HandEyeMode::EyeToHand => {
            // target -> gripper -> robot_base -> rig -> camera
            let p_gripper = target_q.transform_vector(&pw_t) + target_t.clone();
            let p_base = robot_q.transform_vector(&p_gripper) + robot_t.clone();
            let p_rig = handeye_q.transform_vector(&p_base) + handeye_t.clone();
            extr_q.inverse_transform_vector(&(p_rig - extr_t.clone()))
        }
    };

    // Project (same as two_se3)
    let eps = T::from_f64(1e-12).unwrap();
    let z_safe = if p_camera.z.clone() > eps.clone() {
        p_camera.z.clone()
    } else {
        eps
    };
    let x_norm = p_camera.x.clone() / z_safe.clone();
    let y_norm = p_camera.y.clone() / z_safe;

    // Apply Brown-Conrady distortion
    let k1 = dist[0].clone();
    let k2 = dist[1].clone();
    let k3 = dist[2].clone();
    let p1 = dist[3].clone();
    let p2 = dist[4].clone();
    let (xd, yd) = distort_brown_conrady_generic(x_norm, y_norm, k1, k2, k3, p1, p2);

    // Apply intrinsics
    let fx = intr[0].clone();
    let fy = intr[1].clone();
    let cx = intr[2].clone();
    let cy = intr[3].clone();

    let u_pred = fx * xd + cx;
    let v_pred = fy * yd + cy;

    // Weighted residual
    let sqrt_w = T::from_f64(w.sqrt()).unwrap();
    let u_meas = T::from_f64(uv[0]).unwrap();
    let v_meas = T::from_f64(uv[1]).unwrap();

    SVector::<T, 2>::new(
        (u_meas - u_pred) * sqrt_w.clone(),
        (v_meas - v_pred) * sqrt_w,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn distortion_changes_projection() {
        // Test that distortion affects output
        // Use a point far from center to amplify distortion effect
        let intr = DVector::from_row_slice(&[800.0, 800.0, 640.0, 360.0]);
        let dist_zero = DVector::from_row_slice(&[0.0, 0.0, 0.0, 0.0, 0.0]);
        let dist_barrel = DVector::from_row_slice(&[-0.3, 0.1, 0.0, 0.0, 0.0]);
        let pose = DVector::from_row_slice(&[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0]);

        // Point far from center in camera space creates larger distortion
        let pw = [0.5, 0.5, 1.0];
        let uv = [1000.0, 700.0];

        let r1: SVector<f64, 2> = reproj_residual_pinhole4_dist5_se3_generic(
            intr.as_view(),
            dist_zero.as_view(),
            pose.as_view(),
            pw,
            uv,
            1.0,
        );

        let r2: SVector<f64, 2> = reproj_residual_pinhole4_dist5_se3_generic(
            intr.as_view(),
            dist_barrel.as_view(),
            pose.as_view(),
            pw,
            uv,
            1.0,
        );

        // Residuals should differ significantly due to distortion
        let diff = (r1[0] - r2[0]).abs();
        assert!(
            diff > 1.0,
            "Expected residuals to differ by >1.0, got diff={diff}"
        );
    }

    #[test]
    fn zero_distortion_matches_no_distortion() {
        // Zero distortion should give same result as pinhole-only
        let intr = DVector::from_row_slice(&[800.0, 780.0, 640.0, 360.0]);
        let dist_zero = DVector::from_row_slice(&[0.0, 0.0, 0.0, 0.0, 0.0]);
        let pose = DVector::from_row_slice(&[0.0, 0.0, 0.0, 1.0, 0.2, 0.1, 0.8]);

        let pw = [0.05, -0.03, 0.8];
        let uv = [650.0, 355.0];
        let w = 2.0;

        let r_nodist: SVector<f64, 2> =
            reproj_residual_pinhole4_se3_generic(intr.as_view(), pose.as_view(), pw, uv, w);

        let r_zerodist: SVector<f64, 2> = reproj_residual_pinhole4_dist5_se3_generic(
            intr.as_view(),
            dist_zero.as_view(),
            pose.as_view(),
            pw,
            uv,
            w,
        );

        let diff_u = (r_nodist[0] - r_zerodist[0]).abs();
        let diff_v = (r_nodist[1] - r_zerodist[1]).abs();
        // Tolerances are looser due to floating point precision in generic operations
        assert!(diff_u < 1e-6, "u residuals should match, diff={diff_u}");
        assert!(diff_v < 1e-6, "v residuals should match, diff={diff_v}");
    }
}
