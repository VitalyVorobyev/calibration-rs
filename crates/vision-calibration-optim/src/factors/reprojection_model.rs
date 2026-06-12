//! Backend-independent reprojection residual models.

use crate::factors::camera_kernels::{DistortionKernel, ProjectionKernel, SensorKernel};
use crate::ir::ReprojChain;
use crate::math::projection::project_pinhole;
use nalgebra::{
    DVector, DVectorView, Matrix3, Quaternion, RealField, SVector, UnitQuaternion, Vector3,
};

/// Observation data for reprojection residuals.
///
/// Groups together the 3D world point, 2D pixel observation, and weight.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ObservationData {
    /// 3D point in world/target frame.
    pub pw: [f64; 3],
    /// 2D pixel observation.
    pub uv: [f64; 2],
    /// Observation weight.
    pub w: f64,
}

/// Robot pose data for hand-eye calibration.
///
/// Groups together the known robot pose (base-to-gripper) and calibration mode.
#[derive(Debug, Clone, Copy)]
pub(crate) struct RobotPoseData {
    /// Known robot pose as SE3: [qx, qy, qz, qw, tx, ty, tz].
    pub robot_se3: [f64; 7],
    /// Hand-eye calibration mode.
    pub mode: crate::ir::HandEyeMode,
}

/// Observation + robot pose bundle for hand-eye factors with robot deltas.
#[derive(Debug, Clone, Copy)]
pub(crate) struct HandEyeRobotDeltaData {
    pub robot: RobotPoseData,
    pub obs: ObservationData,
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
pub(crate) fn distort_brown_conrady_generic<T: RealField>(
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

fn skew_matrix<T: RealField>(w: &Vector3<T>) -> Matrix3<T> {
    Matrix3::new(
        T::zero(),
        -w.z.clone(),
        w.y.clone(),
        w.z.clone(),
        T::zero(),
        -w.x.clone(),
        -w.y.clone(),
        w.x.clone(),
        T::zero(),
    )
}

pub(crate) fn se3_exp<T: RealField>(xi: DVectorView<'_, T>) -> (UnitQuaternion<T>, Vector3<T>) {
    debug_assert!(xi.len() == 6, "se3 tangent must have 6 params");
    let w = Vector3::new(xi[0].clone(), xi[1].clone(), xi[2].clone());
    let v = Vector3::new(xi[3].clone(), xi[4].clone(), xi[5].clone());

    let theta = w.norm();
    let eps = T::from_f64(1e-9).unwrap();

    let w_hat = skew_matrix(&w);
    let w_hat2 = w_hat.clone() * w_hat.clone();

    let (b, c) = if theta.clone() <= eps {
        let half = T::from_f64(0.5).unwrap();
        let sixth = T::from_f64(1.0 / 6.0).unwrap();
        (half, sixth)
    } else {
        let theta2 = theta.clone() * theta.clone();
        let theta3 = theta2.clone() * theta.clone();
        let sin_theta = theta.clone().sin();
        let cos_theta = theta.clone().cos();
        let b = (T::one() - cos_theta) / theta2;
        let c = (theta - sin_theta) / theta3;
        (b, c)
    };

    let v_mat = Matrix3::identity() + w_hat * b + w_hat2 * c;
    let t = v_mat * v;
    let rot = UnitQuaternion::from_scaled_axis(w);
    (rot, t)
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
pub(crate) fn tilt_projection_matrix_generic<T: RealField>(
    tau_x: T,
    tau_y: T,
) -> nalgebra::Matrix3<T> {
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
pub(crate) fn apply_scheimpflug_generic<T: RealField>(
    x_norm: T,
    y_norm: T,
    tau_x: T,
    tau_y: T,
) -> (T, T) {
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

/// Apply inverse Scheimpflug sensor homography to sensor coordinates (generic for autodiff).
///
/// Maps sensor-plane normalized coordinates back to the normalized camera plane.
pub(crate) fn apply_scheimpflug_inverse_generic<T: RealField>(
    x_sensor: T,
    y_sensor: T,
    tau_x: T,
    tau_y: T,
) -> (T, T) {
    let h = tilt_projection_matrix_generic(tau_x, tau_y);
    let h_inv = match h.try_inverse() {
        Some(inv) => inv,
        None => return (x_sensor, y_sensor),
    };
    let p = h_inv * Vector3::new(x_sensor, y_sensor, T::one());
    let eps = T::from_f64(1e-12).unwrap();
    let z_safe = if p.z.clone().abs() > eps.clone() {
        p.z.clone()
    } else {
        eps
    };
    (p.x.clone() / z_safe.clone(), p.y.clone() / z_safe)
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
/// Transform chain (using `T_dst_src` notation):
/// - `pose` is `T_R_T` (target -> rig)
/// - `extr` is `T_R_C` (camera -> rig)
/// - `p_rig = pose * p_target`
/// - `p_cam = extr^-1 * p_rig`
///
/// # Parameters
/// - `intr`: [fx, fy, cx, cy] intrinsics
/// - `dist`: [k1, k2, k3, p1, p2] Brown-Conrady distortion
/// - `extr`: [qx, qy, qz, qw, tx, ty, tz] camera-to-rig SE3
/// - `pose`: [qx, qy, qz, qw, tx, ty, tz] target-to-rig SE3
/// - `pw`: 3D point in target/world frame
/// - `uv`: observed pixel coordinates
/// - `w`: weight for the residual
pub(crate) fn reproj_residual_pinhole4_dist5_two_se3_generic<T: RealField>(
    intr: DVectorView<'_, T>,
    dist: DVectorView<'_, T>,
    extr: DVectorView<'_, T>,
    pose: DVectorView<'_, T>,
    obs: &ObservationData,
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
        T::from_f64(obs.pw[0]).unwrap(),
        T::from_f64(obs.pw[1]).unwrap(),
        T::from_f64(obs.pw[2]).unwrap(),
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
    let sqrt_w = T::from_f64(obs.w.sqrt()).unwrap();
    let u_meas = T::from_f64(obs.uv[0]).unwrap();
    let v_meas = T::from_f64(obs.uv[1]).unwrap();

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
/// - `handeye`: [qx, qy, qz, qw, tx, ty, tz] hand-eye SE3
///   (gripper-from-rig for eye-in-hand, rig-from-base for eye-to-hand)
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
    robot_data: &RobotPoseData,
    obs: &ObservationData,
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
        T::from_f64(robot_data.robot_se3[3]).unwrap(),
        T::from_f64(robot_data.robot_se3[0]).unwrap(),
        T::from_f64(robot_data.robot_se3[1]).unwrap(),
        T::from_f64(robot_data.robot_se3[2]).unwrap(),
    ));
    let robot_t = Vector3::new(
        T::from_f64(robot_data.robot_se3[4]).unwrap(),
        T::from_f64(robot_data.robot_se3[5]).unwrap(),
        T::from_f64(robot_data.robot_se3[6]).unwrap(),
    );

    let pw_t = Vector3::new(
        T::from_f64(obs.pw[0]).unwrap(),
        T::from_f64(obs.pw[1]).unwrap(),
        T::from_f64(obs.pw[2]).unwrap(),
    );

    // Compose transforms based on mode
    let p_camera = match robot_data.mode {
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
    let sqrt_w = T::from_f64(obs.w.sqrt()).unwrap();
    let u_meas = T::from_f64(obs.uv[0]).unwrap();
    let v_meas = T::from_f64(obs.uv[1]).unwrap();

    SVector::<T, 2>::new(
        (u_meas - u_pred) * sqrt_w.clone(),
        (v_meas - v_pred) * sqrt_w,
    )
}

/// Reprojection residual with two composed SE3 transforms and a Scheimpflug sensor.
///
/// Equivalent to [`reproj_residual_pinhole4_dist5_two_se3_generic`] with a Scheimpflug
/// homography inserted between distortion and intrinsics application.
///
/// # Parameters
/// - `intr`: [fx, fy, cx, cy] intrinsics
/// - `dist`: [k1, k2, k3, p1, p2] Brown-Conrady distortion
/// - `sensor`: [tilt_x, tilt_y] Scheimpflug tilt angles
/// - `extr`: [qx, qy, qz, qw, tx, ty, tz] camera-to-rig SE3
/// - `pose`: [qx, qy, qz, qw, tx, ty, tz] target-to-rig SE3
pub(crate) fn reproj_residual_pinhole4_dist5_scheimpflug2_two_se3_generic<T: RealField>(
    intr: DVectorView<'_, T>,
    dist: DVectorView<'_, T>,
    sensor: DVectorView<'_, T>,
    extr: DVectorView<'_, T>,
    pose: DVectorView<'_, T>,
    obs: &ObservationData,
) -> SVector<T, 2> {
    debug_assert!(sensor.len() >= 2, "sensor must have 2 params");

    let extr_q = UnitQuaternion::from_quaternion(Quaternion::new(
        extr[3].clone(),
        extr[0].clone(),
        extr[1].clone(),
        extr[2].clone(),
    ));
    let extr_t = Vector3::new(extr[4].clone(), extr[5].clone(), extr[6].clone());

    let pose_q = UnitQuaternion::from_quaternion(Quaternion::new(
        pose[3].clone(),
        pose[0].clone(),
        pose[1].clone(),
        pose[2].clone(),
    ));
    let pose_t = Vector3::new(pose[4].clone(), pose[5].clone(), pose[6].clone());

    let pw_t = Vector3::new(
        T::from_f64(obs.pw[0]).unwrap(),
        T::from_f64(obs.pw[1]).unwrap(),
        T::from_f64(obs.pw[2]).unwrap(),
    );

    let p_rig = pose_q.transform_vector(&pw_t) + pose_t;
    let p_camera = extr_q.inverse_transform_vector(&(p_rig - extr_t));

    let eps = T::from_f64(1e-12).unwrap();
    let z_safe = if p_camera.z.clone() > eps.clone() {
        p_camera.z.clone()
    } else {
        eps
    };
    let x_norm = p_camera.x.clone() / z_safe.clone();
    let y_norm = p_camera.y.clone() / z_safe;

    let k1 = dist[0].clone();
    let k2 = dist[1].clone();
    let k3 = dist[2].clone();
    let p1 = dist[3].clone();
    let p2 = dist[4].clone();
    let (xd, yd) = distort_brown_conrady_generic(x_norm, y_norm, k1, k2, k3, p1, p2);

    let tau_x = sensor[0].clone();
    let tau_y = sensor[1].clone();
    let (xs, ys) = apply_scheimpflug_generic(xd, yd, tau_x, tau_y);

    let fx = intr[0].clone();
    let fy = intr[1].clone();
    let cx = intr[2].clone();
    let cy = intr[3].clone();

    let u_pred = fx * xs + cx;
    let v_pred = fy * ys + cy;

    let sqrt_w = T::from_f64(obs.w.sqrt()).unwrap();
    let u_meas = T::from_f64(obs.uv[0]).unwrap();
    let v_meas = T::from_f64(obs.uv[1]).unwrap();

    SVector::<T, 2>::new(
        (u_meas - u_pred) * sqrt_w.clone(),
        (v_meas - v_pred) * sqrt_w,
    )
}

/// Reprojection residual for hand-eye calibration with a Scheimpflug sensor.
///
/// Equivalent to [`reproj_residual_pinhole4_dist5_handeye_generic`] with a Scheimpflug
/// homography inserted between distortion and intrinsics application.
#[allow(clippy::too_many_arguments)]
pub(crate) fn reproj_residual_pinhole4_dist5_scheimpflug2_handeye_generic<T: RealField>(
    intr: DVectorView<'_, T>,
    dist: DVectorView<'_, T>,
    sensor: DVectorView<'_, T>,
    extr: DVectorView<'_, T>,
    handeye: DVectorView<'_, T>,
    target: DVectorView<'_, T>,
    robot_data: &RobotPoseData,
    obs: &ObservationData,
) -> SVector<T, 2> {
    debug_assert!(sensor.len() >= 2, "sensor must have 2 params");

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

    let robot_q: UnitQuaternion<T> = UnitQuaternion::from_quaternion(Quaternion::new(
        T::from_f64(robot_data.robot_se3[3]).unwrap(),
        T::from_f64(robot_data.robot_se3[0]).unwrap(),
        T::from_f64(robot_data.robot_se3[1]).unwrap(),
        T::from_f64(robot_data.robot_se3[2]).unwrap(),
    ));
    let robot_t = Vector3::new(
        T::from_f64(robot_data.robot_se3[4]).unwrap(),
        T::from_f64(robot_data.robot_se3[5]).unwrap(),
        T::from_f64(robot_data.robot_se3[6]).unwrap(),
    );

    let pw_t = Vector3::new(
        T::from_f64(obs.pw[0]).unwrap(),
        T::from_f64(obs.pw[1]).unwrap(),
        T::from_f64(obs.pw[2]).unwrap(),
    );

    let p_camera = match robot_data.mode {
        crate::ir::HandEyeMode::EyeInHand => {
            let p_base = target_q.transform_vector(&pw_t) + target_t.clone();
            let p_gripper = robot_q.inverse_transform_vector(&(p_base - robot_t.clone()));
            let p_rig = handeye_q.inverse_transform_vector(&(p_gripper - handeye_t.clone()));
            extr_q.inverse_transform_vector(&(p_rig - extr_t.clone()))
        }
        crate::ir::HandEyeMode::EyeToHand => {
            let p_gripper = target_q.transform_vector(&pw_t) + target_t.clone();
            let p_base = robot_q.transform_vector(&p_gripper) + robot_t.clone();
            let p_rig = handeye_q.transform_vector(&p_base) + handeye_t.clone();
            extr_q.inverse_transform_vector(&(p_rig - extr_t.clone()))
        }
    };

    let eps = T::from_f64(1e-12).unwrap();
    let z_safe = if p_camera.z.clone() > eps.clone() {
        p_camera.z.clone()
    } else {
        eps
    };
    let x_norm = p_camera.x.clone() / z_safe.clone();
    let y_norm = p_camera.y.clone() / z_safe;

    let k1 = dist[0].clone();
    let k2 = dist[1].clone();
    let k3 = dist[2].clone();
    let p1 = dist[3].clone();
    let p2 = dist[4].clone();
    let (xd, yd) = distort_brown_conrady_generic(x_norm, y_norm, k1, k2, k3, p1, p2);

    let tau_x = sensor[0].clone();
    let tau_y = sensor[1].clone();
    let (xs, ys) = apply_scheimpflug_generic(xd, yd, tau_x, tau_y);

    let fx = intr[0].clone();
    let fy = intr[1].clone();
    let cx = intr[2].clone();
    let cy = intr[3].clone();

    let u_pred = fx * xs + cx;
    let v_pred = fy * ys + cy;

    let sqrt_w = T::from_f64(obs.w.sqrt()).unwrap();
    let u_meas = T::from_f64(obs.uv[0]).unwrap();
    let v_meas = T::from_f64(obs.uv[1]).unwrap();

    SVector::<T, 2>::new(
        (u_meas - u_pred) * sqrt_w.clone(),
        (v_meas - v_pred) * sqrt_w,
    )
}

/// Reprojection residual for hand-eye calibration with per-view robot pose
/// corrections and a Scheimpflug sensor.
///
/// Equivalent to [`reproj_residual_pinhole4_dist5_handeye_robot_delta_generic`] with a
/// Scheimpflug homography inserted between distortion and intrinsics application.
#[allow(clippy::too_many_arguments)]
pub(crate) fn reproj_residual_pinhole4_dist5_scheimpflug2_handeye_robot_delta_generic<
    T: RealField,
>(
    intr: DVectorView<'_, T>,
    dist: DVectorView<'_, T>,
    sensor: DVectorView<'_, T>,
    extr: DVectorView<'_, T>,
    handeye: DVectorView<'_, T>,
    target: DVectorView<'_, T>,
    robot_delta: DVectorView<'_, T>,
    data: &HandEyeRobotDeltaData,
) -> SVector<T, 2> {
    debug_assert!(sensor.len() >= 2, "sensor must have 2 params");

    let robot_data = &data.robot;
    let obs = &data.obs;

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

    let robot_q: UnitQuaternion<T> = UnitQuaternion::from_quaternion(Quaternion::new(
        T::from_f64(robot_data.robot_se3[3]).unwrap(),
        T::from_f64(robot_data.robot_se3[0]).unwrap(),
        T::from_f64(robot_data.robot_se3[1]).unwrap(),
        T::from_f64(robot_data.robot_se3[2]).unwrap(),
    ));
    let robot_t = Vector3::new(
        T::from_f64(robot_data.robot_se3[4]).unwrap(),
        T::from_f64(robot_data.robot_se3[5]).unwrap(),
        T::from_f64(robot_data.robot_se3[6]).unwrap(),
    );

    let (delta_q, delta_t) = se3_exp(robot_delta);
    let robot_q = delta_q.clone() * robot_q;
    let robot_t = delta_q.transform_vector(&robot_t) + delta_t;

    let pw_t = Vector3::new(
        T::from_f64(obs.pw[0]).unwrap(),
        T::from_f64(obs.pw[1]).unwrap(),
        T::from_f64(obs.pw[2]).unwrap(),
    );

    let p_camera = match robot_data.mode {
        crate::ir::HandEyeMode::EyeInHand => {
            let p_base = target_q.transform_vector(&pw_t) + target_t.clone();
            let p_gripper = robot_q.inverse_transform_vector(&(p_base - robot_t.clone()));
            let p_rig = handeye_q.inverse_transform_vector(&(p_gripper - handeye_t.clone()));
            extr_q.inverse_transform_vector(&(p_rig - extr_t.clone()))
        }
        crate::ir::HandEyeMode::EyeToHand => {
            let p_gripper = target_q.transform_vector(&pw_t) + target_t.clone();
            let p_base = robot_q.transform_vector(&p_gripper) + robot_t.clone();
            let p_rig = handeye_q.transform_vector(&p_base) + handeye_t.clone();
            extr_q.inverse_transform_vector(&(p_rig - extr_t.clone()))
        }
    };

    let eps = T::from_f64(1e-12).unwrap();
    let z_safe = if p_camera.z.clone() > eps.clone() {
        p_camera.z.clone()
    } else {
        eps
    };
    let x_norm = p_camera.x.clone() / z_safe.clone();
    let y_norm = p_camera.y.clone() / z_safe;

    let k1 = dist[0].clone();
    let k2 = dist[1].clone();
    let k3 = dist[2].clone();
    let p1 = dist[3].clone();
    let p2 = dist[4].clone();
    let (xd, yd) = distort_brown_conrady_generic(x_norm, y_norm, k1, k2, k3, p1, p2);

    let tau_x = sensor[0].clone();
    let tau_y = sensor[1].clone();
    let (xs, ys) = apply_scheimpflug_generic(xd, yd, tau_x, tau_y);

    let fx = intr[0].clone();
    let fy = intr[1].clone();
    let cx = intr[2].clone();
    let cy = intr[3].clone();

    let u_pred = fx * xs + cx;
    let v_pred = fy * ys + cy;

    let sqrt_w = T::from_f64(obs.w.sqrt()).unwrap();
    let u_meas = T::from_f64(obs.uv[0]).unwrap();
    let v_meas = T::from_f64(obs.uv[1]).unwrap();

    SVector::<T, 2>::new(
        (u_meas - u_pred) * sqrt_w.clone(),
        (v_meas - v_pred) * sqrt_w,
    )
}

/// Reprojection residual for hand-eye calibration with per-view robot pose corrections.
///
/// The correction is applied as `exp(delta) * T_B_E` (left-multiply).
pub(crate) fn reproj_residual_pinhole4_dist5_handeye_robot_delta_generic<T: RealField>(
    intr: DVectorView<'_, T>,
    dist: DVectorView<'_, T>,
    extr: DVectorView<'_, T>,
    handeye: DVectorView<'_, T>,
    target: DVectorView<'_, T>,
    robot_delta: DVectorView<'_, T>,
    data: &HandEyeRobotDeltaData,
) -> SVector<T, 2> {
    use nalgebra::{Quaternion, UnitQuaternion, Vector3};

    let robot_data = &data.robot;
    let obs = &data.obs;

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
        T::from_f64(robot_data.robot_se3[3]).unwrap(),
        T::from_f64(robot_data.robot_se3[0]).unwrap(),
        T::from_f64(robot_data.robot_se3[1]).unwrap(),
        T::from_f64(robot_data.robot_se3[2]).unwrap(),
    ));
    let robot_t = Vector3::new(
        T::from_f64(robot_data.robot_se3[4]).unwrap(),
        T::from_f64(robot_data.robot_se3[5]).unwrap(),
        T::from_f64(robot_data.robot_se3[6]).unwrap(),
    );

    let (delta_q, delta_t) = se3_exp(robot_delta);
    let robot_q = delta_q.clone() * robot_q;
    let robot_t = delta_q.transform_vector(&robot_t) + delta_t;

    let pw_t = Vector3::new(
        T::from_f64(obs.pw[0]).unwrap(),
        T::from_f64(obs.pw[1]).unwrap(),
        T::from_f64(obs.pw[2]).unwrap(),
    );

    // Compose transforms based on mode
    let p_camera = match robot_data.mode {
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
    let sqrt_w = T::from_f64(obs.w.sqrt()).unwrap();
    let u_meas = T::from_f64(obs.uv[0]).unwrap();
    let v_meas = T::from_f64(obs.uv[1]).unwrap();

    SVector::<T, 2>::new(
        (u_meas - u_pred) * sqrt_w.clone(),
        (v_meas - v_pred) * sqrt_w,
    )
}

/// Extract `(rotation, translation)` from an SE3 parameter block
/// `[qx, qy, qz, qw, tx, ty, tz]`.
pub(crate) fn se3_from_block<T: RealField>(v: &DVector<T>) -> (UnitQuaternion<T>, Vector3<T>) {
    debug_assert!(v.len() == 7, "SE3 block must have 7 params");
    let q = UnitQuaternion::from_quaternion(Quaternion::new(
        v[3].clone(),
        v[0].clone(),
        v[1].clone(),
        v[2].clone(),
    ));
    let t = Vector3::new(v[4].clone(), v[5].clone(), v[6].clone());
    (q, t)
}

/// Lift a measured SE3 `[qx, qy, qz, qw, tx, ty, tz]` into the scalar type.
pub(crate) fn se3_from_f64_array<T: RealField>(a: &[f64; 7]) -> (UnitQuaternion<T>, Vector3<T>) {
    let q = UnitQuaternion::from_quaternion(Quaternion::new(
        T::from_f64(a[3]).unwrap(),
        T::from_f64(a[0]).unwrap(),
        T::from_f64(a[1]).unwrap(),
        T::from_f64(a[2]).unwrap(),
    ));
    let t = Vector3::new(
        T::from_f64(a[4]).unwrap(),
        T::from_f64(a[5]).unwrap(),
        T::from_f64(a[6]).unwrap(),
    );
    (q, t)
}

/// Transform a target-frame point into the camera frame through a
/// [`ReprojChain`].
///
/// `blocks` are the chain's parameter blocks in IR order (camera blocks
/// already stripped). The point is pushed through the chain step by step,
/// matching the per-chain operation order of the enumerated kernels exactly.
pub(crate) fn reproj_chain_transform<T: RealField>(
    chain: &ReprojChain,
    blocks: &[DVector<T>],
    pw: [f64; 3],
) -> Vector3<T> {
    let pw_t = Vector3::new(
        T::from_f64(pw[0]).unwrap(),
        T::from_f64(pw[1]).unwrap(),
        T::from_f64(pw[2]).unwrap(),
    );
    match chain {
        ReprojChain::SinglePose => {
            debug_assert!(blocks.len() == 1, "SinglePose chain expects 1 block");
            let (rot, t) = se3_from_block(&blocks[0]);
            rot.transform_vector(&pw_t) + t
        }
        ReprojChain::TwoSe3 => {
            debug_assert!(blocks.len() == 2, "TwoSe3 chain expects 2 blocks");
            let (extr_q, extr_t) = se3_from_block(&blocks[0]);
            let (pose_q, pose_t) = se3_from_block(&blocks[1]);
            let p_rig = pose_q.transform_vector(&pw_t) + pose_t;
            extr_q.inverse_transform_vector(&(p_rig - extr_t))
        }
        ReprojChain::HandEye {
            base_se3_gripper,
            mode,
        } => {
            debug_assert!(blocks.len() == 3, "HandEye chain expects 3 blocks");
            let (robot_q, robot_t) = se3_from_f64_array::<T>(base_se3_gripper);
            handeye_chain_transform(
                &blocks[0], &blocks[1], &blocks[2], robot_q, robot_t, *mode, &pw_t,
            )
        }
        ReprojChain::HandEyeRobotDelta {
            base_se3_gripper,
            mode,
        } => {
            debug_assert!(
                blocks.len() == 4,
                "HandEyeRobotDelta chain expects 4 blocks"
            );
            let (robot_q, robot_t) = se3_from_f64_array::<T>(base_se3_gripper);
            let (delta_q, delta_t) = se3_exp(blocks[3].as_view());
            let robot_q = delta_q.clone() * robot_q;
            let robot_t = delta_q.transform_vector(&robot_t) + delta_t;
            handeye_chain_transform(
                &blocks[0], &blocks[1], &blocks[2], robot_q, robot_t, *mode, &pw_t,
            )
        }
    }
}

/// Shared hand-eye chain: target -> (robot, hand-eye per mode) -> rig -> camera.
fn handeye_chain_transform<T: RealField>(
    extr: &DVector<T>,
    handeye: &DVector<T>,
    target: &DVector<T>,
    robot_q: UnitQuaternion<T>,
    robot_t: Vector3<T>,
    mode: crate::ir::HandEyeMode,
    pw_t: &Vector3<T>,
) -> Vector3<T> {
    let (extr_q, extr_t) = se3_from_block(extr);
    let (handeye_q, handeye_t) = se3_from_block(handeye);
    let (target_q, target_t) = se3_from_block(target);
    match mode {
        crate::ir::HandEyeMode::EyeInHand => {
            // target -> robot_base -> gripper -> rig -> camera
            let p_base = target_q.transform_vector(pw_t) + target_t.clone();
            let p_gripper = robot_q.inverse_transform_vector(&(p_base - robot_t.clone()));
            let p_rig = handeye_q.inverse_transform_vector(&(p_gripper - handeye_t.clone()));
            extr_q.inverse_transform_vector(&(p_rig - extr_t.clone()))
        }
        crate::ir::HandEyeMode::EyeToHand => {
            // target -> gripper -> robot_base -> rig -> camera
            let p_gripper = target_q.transform_vector(pw_t) + target_t.clone();
            let p_base = robot_q.transform_vector(&p_gripper) + robot_t.clone();
            let p_rig = handeye_q.transform_vector(&p_base) + handeye_t.clone();
            extr_q.inverse_transform_vector(&(p_rig - extr_t.clone()))
        }
    }
}

/// Reprojection residual generic over the camera-model kernels and pose chain.
///
/// `params` is the full IR-ordered block list: `[intrinsics, distortion?,
/// sensor?, <chain blocks>]`, with the optional blocks present exactly when
/// the corresponding kernel dimension is non-zero.
pub(crate) fn reproj_residual_model_generic<P, D, S, T>(
    chain: &ReprojChain,
    params: &[DVector<T>],
    pw: [f64; 3],
    uv: [f64; 2],
    w: f64,
) -> SVector<T, 2>
where
    P: ProjectionKernel,
    D: DistortionKernel,
    S: SensorKernel,
    T: RealField,
{
    let mut idx = 1;
    let dist = (D::DIM > 0).then(|| {
        let v = params[idx].as_view();
        idx += 1;
        v
    });
    let sensor = (S::DIM > 0).then(|| {
        let v = params[idx].as_view();
        idx += 1;
        v
    });

    let p_camera = reproj_chain_transform(chain, &params[idx..], pw);
    let (x_norm, y_norm) = P::normalize(&p_camera);
    let (x_dist, y_dist) = D::distort(dist, x_norm, y_norm);
    let (x_sensor, y_sensor) = S::to_sensor(sensor, x_dist, y_dist);

    let intr = &params[0];
    debug_assert!(intr.len() >= 4, "intrinsics must have 4 params");
    let fx = intr[0].clone();
    let fy = intr[1].clone();
    let cx = intr[2].clone();
    let cy = intr[3].clone();
    let u_proj = fx * x_sensor + cx;
    let v_proj = fy * y_sensor + cy;

    let sqrt_w = T::from_f64(w.sqrt()).unwrap();
    let u_meas = T::from_f64(uv[0]).unwrap();
    let v_meas = T::from_f64(uv[1]).unwrap();
    SVector::<T, 2>::new(
        (u_meas - u_proj) * sqrt_w.clone(),
        (v_meas - v_proj) * sqrt_w,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::DVector;

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
    fn scheimpflug_two_se3_zero_tilt_matches_pinhole() {
        let intr = DVector::from_row_slice(&[820.0, 810.0, 640.5, 360.5]);
        let dist = DVector::from_row_slice(&[-0.12, 0.08, 0.0, 0.001, -0.002]);
        let sensor_zero = DVector::from_row_slice(&[0.0, 0.0]);
        let extr = DVector::from_row_slice(&[0.02, 0.03, -0.01, 0.999_049_9, 0.12, -0.05, 0.03]);
        let pose = DVector::from_row_slice(&[0.05, -0.02, 0.04, 0.997_549_9, 0.4, 0.2, 0.9]);
        let obs = ObservationData {
            pw: [0.11, -0.07, 0.0],
            uv: [680.0, 340.0],
            w: 1.5,
        };

        let r_pin: SVector<f64, 2> = reproj_residual_pinhole4_dist5_two_se3_generic(
            intr.as_view(),
            dist.as_view(),
            extr.as_view(),
            pose.as_view(),
            &obs,
        );
        let r_sch: SVector<f64, 2> = reproj_residual_pinhole4_dist5_scheimpflug2_two_se3_generic(
            intr.as_view(),
            dist.as_view(),
            sensor_zero.as_view(),
            extr.as_view(),
            pose.as_view(),
            &obs,
        );

        assert!((r_pin[0] - r_sch[0]).abs() < 1e-10);
        assert!((r_pin[1] - r_sch[1]).abs() < 1e-10);
    }

    #[test]
    fn scheimpflug_handeye_zero_tilt_matches_pinhole() {
        let intr = DVector::from_row_slice(&[700.0, 698.0, 320.5, 240.5]);
        let dist = DVector::from_row_slice(&[-0.05, 0.02, 0.0, -0.001, 0.0005]);
        let sensor_zero = DVector::from_row_slice(&[0.0, 0.0]);
        let extr = DVector::from_row_slice(&[0.01, -0.02, 0.015, 0.999_6, 0.05, 0.01, -0.02]);
        let handeye = DVector::from_row_slice(&[-0.03, 0.02, 0.01, 0.999_3, 0.08, -0.04, 0.12]);
        let target = DVector::from_row_slice(&[0.04, 0.05, -0.02, 0.997_7, 0.3, 0.4, 0.7]);
        let robot = RobotPoseData {
            robot_se3: [0.02, 0.01, 0.03, 0.999_3, 0.5, -0.2, 0.8],
            mode: crate::ir::HandEyeMode::EyeInHand,
        };
        let obs = ObservationData {
            pw: [0.09, 0.04, 0.0],
            uv: [305.0, 265.0],
            w: 1.0,
        };

        let r_pin: SVector<f64, 2> = reproj_residual_pinhole4_dist5_handeye_generic(
            intr.as_view(),
            dist.as_view(),
            extr.as_view(),
            handeye.as_view(),
            target.as_view(),
            &robot,
            &obs,
        );
        let r_sch: SVector<f64, 2> = reproj_residual_pinhole4_dist5_scheimpflug2_handeye_generic(
            intr.as_view(),
            dist.as_view(),
            sensor_zero.as_view(),
            extr.as_view(),
            handeye.as_view(),
            target.as_view(),
            &robot,
            &obs,
        );

        assert!((r_pin[0] - r_sch[0]).abs() < 1e-10);
        assert!((r_pin[1] - r_sch[1]).abs() < 1e-10);
    }

    #[test]
    fn scheimpflug_handeye_delta_zero_tilt_matches_pinhole() {
        let intr = DVector::from_row_slice(&[700.0, 698.0, 320.5, 240.5]);
        let dist = DVector::from_row_slice(&[-0.05, 0.02, 0.0, -0.001, 0.0005]);
        let sensor_zero = DVector::from_row_slice(&[0.0, 0.0]);
        let extr = DVector::from_row_slice(&[0.01, -0.02, 0.015, 0.999_6, 0.05, 0.01, -0.02]);
        let handeye = DVector::from_row_slice(&[-0.03, 0.02, 0.01, 0.999_3, 0.08, -0.04, 0.12]);
        let target = DVector::from_row_slice(&[0.04, 0.05, -0.02, 0.997_7, 0.3, 0.4, 0.7]);
        let delta = DVector::from_row_slice(&[0.001, -0.002, 0.003, 0.0005, -0.001, 0.002]);
        let data = HandEyeRobotDeltaData {
            robot: RobotPoseData {
                robot_se3: [0.02, 0.01, 0.03, 0.999_3, 0.5, -0.2, 0.8],
                mode: crate::ir::HandEyeMode::EyeInHand,
            },
            obs: ObservationData {
                pw: [0.09, 0.04, 0.0],
                uv: [305.0, 265.0],
                w: 1.0,
            },
        };

        let r_pin: SVector<f64, 2> = reproj_residual_pinhole4_dist5_handeye_robot_delta_generic(
            intr.as_view(),
            dist.as_view(),
            extr.as_view(),
            handeye.as_view(),
            target.as_view(),
            delta.as_view(),
            &data,
        );
        let r_sch: SVector<f64, 2> =
            reproj_residual_pinhole4_dist5_scheimpflug2_handeye_robot_delta_generic(
                intr.as_view(),
                dist.as_view(),
                sensor_zero.as_view(),
                extr.as_view(),
                handeye.as_view(),
                target.as_view(),
                delta.as_view(),
                &data,
            );

        assert!((r_pin[0] - r_sch[0]).abs() < 1e-10);
        assert!((r_pin[1] - r_sch[1]).abs() < 1e-10);
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

    // ─── Old-vs-new kernel equivalence (descriptor-based generic kernel) ────
    //
    // The clamp-based legacy kernels must agree bit-for-bit with
    // `reproj_residual_model_generic`; the lone `project_pinhole` z-guard
    // divergence (Pinhole4) is bounded far below test tolerances.

    use crate::factors::camera_kernels::{
        BrownConrady5Kernel, IdentitySensorKernel, NoDistortionKernel, PinholeKernel,
        Scheimpflug2Kernel,
    };
    use crate::ir::HandEyeMode;

    #[allow(clippy::type_complexity)]
    fn eq_fixture() -> (
        DVector<f64>,
        DVector<f64>,
        DVector<f64>,
        Vec<DVector<f64>>,
        ObservationData,
    ) {
        let intr = DVector::from_row_slice(&[812.3, 798.7, 645.2, 357.9]);
        let dist = DVector::from_row_slice(&[-0.11, 0.07, 0.012, 0.0015, -0.0023]);
        let sensor = DVector::from_row_slice(&[0.021, -0.013]);
        let poses = vec![
            DVector::from_row_slice(&[0.021, 0.034, -0.012, 0.999_03, 0.12, -0.05, 0.83]),
            DVector::from_row_slice(&[0.051, -0.022, 0.041, 0.997_55, 0.41, 0.21, 0.92]),
            DVector::from_row_slice(&[-0.031, 0.018, 0.009, 0.999_24, 0.08, -0.04, 1.12]),
        ];
        let obs = ObservationData {
            pw: [0.113, -0.072, 0.004],
            uv: [684.2, 341.7],
            w: 1.7,
        };
        (intr, dist, sensor, poses, obs)
    }

    #[test]
    fn model_generic_matches_legacy_single_pose() {
        let (intr, dist, sensor, poses, obs) = eq_fixture();
        let pose = &poses[0];

        // Pinhole4Dist5 — bit-identical.
        let old: SVector<f64, 2> = reproj_residual_pinhole4_dist5_se3_generic(
            intr.as_view(),
            dist.as_view(),
            pose.as_view(),
            obs.pw,
            obs.uv,
            obs.w,
        );
        let new = reproj_residual_model_generic::<
            PinholeKernel,
            BrownConrady5Kernel,
            IdentitySensorKernel,
            f64,
        >(
            &ReprojChain::SinglePose,
            &[intr.clone(), dist.clone(), pose.clone()],
            obs.pw,
            obs.uv,
            obs.w,
        );
        assert_eq!(old, new, "Pinhole4Dist5/SinglePose must be bit-identical");

        // Pinhole4Dist5Scheimpflug2 — bit-identical.
        let old: SVector<f64, 2> = reproj_residual_pinhole4_dist5_scheimpflug2_se3_generic(
            intr.as_view(),
            dist.as_view(),
            sensor.as_view(),
            pose.as_view(),
            obs.pw,
            obs.uv,
            obs.w,
        );
        let new = reproj_residual_model_generic::<
            PinholeKernel,
            BrownConrady5Kernel,
            Scheimpflug2Kernel,
            f64,
        >(
            &ReprojChain::SinglePose,
            &[intr.clone(), dist.clone(), sensor.clone(), pose.clone()],
            obs.pw,
            obs.uv,
            obs.w,
        );
        assert_eq!(
            old, new,
            "Pinhole4Dist5Scheimpflug2/SinglePose must be bit-identical"
        );

        // Pinhole4 (no distortion) — the legacy kernel normalizes via
        // `project_pinhole` (z + 1e-9) while the unified kernel clamps; the
        // drift is bounded by ~1e-9 relative in normalized coordinates.
        let old: SVector<f64, 2> = reproj_residual_pinhole4_se3_generic(
            intr.as_view(),
            pose.as_view(),
            obs.pw,
            obs.uv,
            obs.w,
        );
        let new = reproj_residual_model_generic::<
            PinholeKernel,
            NoDistortionKernel,
            IdentitySensorKernel,
            f64,
        >(
            &ReprojChain::SinglePose,
            &[intr.clone(), pose.clone()],
            obs.pw,
            obs.uv,
            obs.w,
        );
        assert!(
            (old[0] - new[0]).abs() < 1e-5 && (old[1] - new[1]).abs() < 1e-5,
            "Pinhole4/SinglePose z-guard drift exceeds bound: old={old:?} new={new:?}"
        );
    }

    #[test]
    fn model_generic_matches_legacy_two_se3() {
        let (intr, dist, sensor, poses, obs) = eq_fixture();
        let extr = &poses[0];
        let pose = &poses[1];

        let old: SVector<f64, 2> = reproj_residual_pinhole4_dist5_two_se3_generic(
            intr.as_view(),
            dist.as_view(),
            extr.as_view(),
            pose.as_view(),
            &obs,
        );
        let new = reproj_residual_model_generic::<
            PinholeKernel,
            BrownConrady5Kernel,
            IdentitySensorKernel,
            f64,
        >(
            &ReprojChain::TwoSe3,
            &[intr.clone(), dist.clone(), extr.clone(), pose.clone()],
            obs.pw,
            obs.uv,
            obs.w,
        );
        assert_eq!(old, new, "Pinhole4Dist5/TwoSe3 must be bit-identical");

        let old: SVector<f64, 2> = reproj_residual_pinhole4_dist5_scheimpflug2_two_se3_generic(
            intr.as_view(),
            dist.as_view(),
            sensor.as_view(),
            extr.as_view(),
            pose.as_view(),
            &obs,
        );
        let new = reproj_residual_model_generic::<
            PinholeKernel,
            BrownConrady5Kernel,
            Scheimpflug2Kernel,
            f64,
        >(
            &ReprojChain::TwoSe3,
            &[
                intr.clone(),
                dist.clone(),
                sensor.clone(),
                extr.clone(),
                pose.clone(),
            ],
            obs.pw,
            obs.uv,
            obs.w,
        );
        assert_eq!(
            old, new,
            "Pinhole4Dist5Scheimpflug2/TwoSe3 must be bit-identical"
        );
    }

    #[test]
    fn model_generic_matches_legacy_handeye_all_modes() {
        let (intr, dist, sensor, poses, obs) = eq_fixture();
        let extr = &poses[0];
        let handeye = &poses[1];
        let target = &poses[2];
        let robot_se3 = [0.024, 0.011, 0.032, 0.999_15, 0.51, -0.22, 0.78];
        let delta = DVector::from_row_slice(&[0.0012, -0.0021, 0.0033, 0.0006, -0.0011, 0.0024]);

        for mode in [HandEyeMode::EyeInHand, HandEyeMode::EyeToHand] {
            let robot = RobotPoseData { robot_se3, mode };

            // HandEye, pinhole sensor.
            let old: SVector<f64, 2> = reproj_residual_pinhole4_dist5_handeye_generic(
                intr.as_view(),
                dist.as_view(),
                extr.as_view(),
                handeye.as_view(),
                target.as_view(),
                &robot,
                &obs,
            );
            let chain = ReprojChain::HandEye {
                base_se3_gripper: robot_se3,
                mode,
            };
            let new = reproj_residual_model_generic::<
                PinholeKernel,
                BrownConrady5Kernel,
                IdentitySensorKernel,
                f64,
            >(
                &chain,
                &[
                    intr.clone(),
                    dist.clone(),
                    extr.clone(),
                    handeye.clone(),
                    target.clone(),
                ],
                obs.pw,
                obs.uv,
                obs.w,
            );
            assert_eq!(old, new, "HandEye {mode:?} must be bit-identical");

            // HandEye, Scheimpflug sensor.
            let old: SVector<f64, 2> = reproj_residual_pinhole4_dist5_scheimpflug2_handeye_generic(
                intr.as_view(),
                dist.as_view(),
                sensor.as_view(),
                extr.as_view(),
                handeye.as_view(),
                target.as_view(),
                &robot,
                &obs,
            );
            let new = reproj_residual_model_generic::<
                PinholeKernel,
                BrownConrady5Kernel,
                Scheimpflug2Kernel,
                f64,
            >(
                &chain,
                &[
                    intr.clone(),
                    dist.clone(),
                    sensor.clone(),
                    extr.clone(),
                    handeye.clone(),
                    target.clone(),
                ],
                obs.pw,
                obs.uv,
                obs.w,
            );
            assert_eq!(
                old, new,
                "Scheimpflug2 HandEye {mode:?} must be bit-identical"
            );

            // HandEyeRobotDelta, both sensors.
            let data = HandEyeRobotDeltaData { robot, obs };
            let chain_delta = ReprojChain::HandEyeRobotDelta {
                base_se3_gripper: robot_se3,
                mode,
            };
            let old: SVector<f64, 2> = reproj_residual_pinhole4_dist5_handeye_robot_delta_generic(
                intr.as_view(),
                dist.as_view(),
                extr.as_view(),
                handeye.as_view(),
                target.as_view(),
                delta.as_view(),
                &data,
            );
            let new = reproj_residual_model_generic::<
                PinholeKernel,
                BrownConrady5Kernel,
                IdentitySensorKernel,
                f64,
            >(
                &chain_delta,
                &[
                    intr.clone(),
                    dist.clone(),
                    extr.clone(),
                    handeye.clone(),
                    target.clone(),
                    delta.clone(),
                ],
                obs.pw,
                obs.uv,
                obs.w,
            );
            assert_eq!(old, new, "HandEyeRobotDelta {mode:?} must be bit-identical");

            let old: SVector<f64, 2> =
                reproj_residual_pinhole4_dist5_scheimpflug2_handeye_robot_delta_generic(
                    intr.as_view(),
                    dist.as_view(),
                    sensor.as_view(),
                    extr.as_view(),
                    handeye.as_view(),
                    target.as_view(),
                    delta.as_view(),
                    &data,
                );
            let new = reproj_residual_model_generic::<
                PinholeKernel,
                BrownConrady5Kernel,
                Scheimpflug2Kernel,
                f64,
            >(
                &chain_delta,
                &[
                    intr.clone(),
                    dist.clone(),
                    sensor.clone(),
                    extr.clone(),
                    handeye.clone(),
                    target.clone(),
                    delta.clone(),
                ],
                obs.pw,
                obs.uv,
                obs.w,
            );
            assert_eq!(
                old, new,
                "Scheimpflug2 HandEyeRobotDelta {mode:?} must be bit-identical"
            );
        }
    }
}
