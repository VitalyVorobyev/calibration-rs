//! Laserline plane residual factors.
//!
//! These residuals constrain laser line pixels to lie on a laser plane.
//! Two approaches are provided:
//! 1. Point-to-plane distance: ray-target intersection then distance to laser plane
//! 2. Line-distance in normalized plane: projects laser-target line to normalized plane
//!
//! Both come in two flavours:
//! - Single-pose (`*_residual_generic`): the residual takes a direct
//!   `cam_se3_target` SE3 parameter block.
//! - Rig + hand-eye (`*_rig_handeye_residual_generic`): the residual composes
//!   `cam_se3_target` from (`cam_to_rig`, `handeye`, `target_ref`, robot pose)
//!   according to [`HandEyeMode`], so that rig extrinsics, hand-eye transform,
//!   and target reference pose all move under a single cost with the laser
//!   geometry constraint.

use crate::factors::reprojection_model::{
    RobotPoseData, apply_scheimpflug_inverse_generic, se3_exp,
};
use crate::ir::HandEyeMode;
use nalgebra::{DVectorView, Quaternion, RealField, SVector, UnitQuaternion, Vector2, Vector3};

/// Undistort a pixel to normalized coordinates using a fixed-point iteration.
///
/// Returns `(x_u, y_u)` in normalized camera coordinates.
fn undistort_pixel_to_normalized<T: RealField>(
    intr: DVectorView<'_, T>,
    dist: DVectorView<'_, T>,
    sensor: DVectorView<'_, T>,
    laser_pixel: [f64; 2],
) -> (T, T) {
    debug_assert!(intr.len() >= 4, "intrinsics must have 4 params");
    debug_assert!(dist.len() >= 5, "distortion must have 5 params");
    debug_assert!(sensor.len() >= 2, "sensor must have 2 params");
    // Extract intrinsics
    let fx = intr[0].clone();
    let fy = intr[1].clone();
    let cx = intr[2].clone();
    let cy = intr[3].clone();

    let tau_x = sensor[0].clone();
    let tau_y = sensor[1].clone();

    // Extract distortion
    let k1 = dist[0].clone();
    let k2 = dist[1].clone();
    let k3 = dist[2].clone();
    let p1 = dist[3].clone();
    let p2 = dist[4].clone();

    let u_px = T::from_f64(laser_pixel[0]).unwrap();
    let v_px = T::from_f64(laser_pixel[1]).unwrap();

    // Sensor-plane normalized coordinates
    let x_s = (u_px - cx) / fx;
    let y_s = (v_px - cy) / fy;

    // Map back to distorted normalized plane using inverse Scheimpflug transform
    let (x_d, y_d) = apply_scheimpflug_inverse_generic(x_s, y_s, tau_x, tau_y);

    // Fixed-point iteration to invert distortion
    let mut x_u = x_d.clone();
    let mut y_u = y_d.clone();
    for _ in 0..5 {
        let r2 = x_u.clone() * x_u.clone() + y_u.clone() * y_u.clone();
        let r4 = r2.clone() * r2.clone();
        let r6 = r4.clone() * r2.clone();

        let radial = T::one() + k1.clone() * r2.clone() + k2.clone() * r4.clone() + k3.clone() * r6;
        let xy = x_u.clone() * y_u.clone();
        let two = T::from_f64(2.0).unwrap();
        let dx_t = two.clone() * p1.clone() * xy.clone()
            + p2.clone() * (r2.clone() + two.clone() * x_u.clone() * x_u.clone());
        let dy_t = p1.clone() * (r2.clone() + two.clone() * y_u.clone() * y_u.clone())
            + two.clone() * p2.clone() * xy;

        x_u = (x_d.clone() - dx_t) / radial.clone();
        y_u = (y_d.clone() - dy_t) / radial;
    }

    (x_u, y_u)
}

/// Generic residual for laser plane constraint (point-to-plane distance).
///
/// # Parameters
///
/// - `intr`: [fx, fy, cx, cy] (4D) - Pinhole intrinsics
/// - `dist`: [k1, k2, k3, p1, p2] (5D) - Brown-Conrady distortion
/// - `pose`: [qx, qy, qz, qw, tx, ty, tz] (7D) - SE(3) camera-to-target transform
/// - `plane_normal`: [nx, ny, nz] (3D) - Laser plane unit normal (S2)
/// - `plane_distance`: [d] (1D) - Laser plane signed distance
/// - `laser_pixel`: [u, v] - Observed laser line pixel
/// - `w`: Weight
///
/// # Algorithm
///
/// 1. Undistort pixel to normalized coordinates
/// 2. Back-project to ray in camera frame
/// 3. Transform ray to target frame and intersect with Z=0 plane
/// 4. Transform intersection point back to camera frame
/// 5. Compute signed distance from point to laser plane
///
/// # Returns
///
/// 1D residual scaled by sqrt(w): [distance * sqrt(w)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn laser_plane_pixel_residual_generic<T: RealField>(
    intr: DVectorView<'_, T>,
    dist: DVectorView<'_, T>,
    sensor: DVectorView<'_, T>,
    pose: DVectorView<'_, T>,
    plane_normal: DVectorView<'_, T>,
    plane_distance: DVectorView<'_, T>,
    laser_pixel: [f64; 2],
    w: f64,
) -> SVector<T, 1> {
    debug_assert!(intr.len() >= 4, "intrinsics must have 4 params");
    debug_assert!(dist.len() >= 5, "distortion must have 5 params");
    debug_assert!(sensor.len() >= 2, "sensor must have 2 params");
    debug_assert!(pose.len() == 7, "pose must have 7 params (SE3)");
    debug_assert!(plane_normal.len() == 3, "plane normal must have 3 params");
    debug_assert!(
        plane_distance.len() == 1,
        "plane distance must have 1 param"
    );

    // Extract pose (T_C_T: camera-to-target)
    let pose_qx = pose[0].clone();
    let pose_qy = pose[1].clone();
    let pose_qz = pose[2].clone();
    let pose_qw = pose[3].clone();
    let pose_tx = pose[4].clone();
    let pose_ty = pose[5].clone();
    let pose_tz = pose[6].clone();

    let pose_quat = Quaternion::new(pose_qw, pose_qx, pose_qy, pose_qz);
    let pose_rot = UnitQuaternion::from_quaternion(pose_quat);
    let pose_t = Vector3::new(pose_tx, pose_ty, pose_tz);

    // 1. Undistort pixel to normalized coordinates
    let (x_u, y_u) = undistort_pixel_to_normalized(intr, dist, sensor, laser_pixel);

    // 2. Back-project to ray in camera frame (ray on z=1 plane)
    let ray_dir_camera = Vector3::new(x_u, y_u, T::one()).normalize();

    // 3. Transform ray to target frame
    // T_T_C = (T_C_T)^-1
    // ray_dir_target = R_T_C * ray_dir_camera = R_C_T^T * ray_dir_camera
    let ray_dir_target = pose_rot.inverse_transform_vector(&ray_dir_camera);

    // Ray origin in target frame: -R_C_T^T * t_C_T
    let ray_origin_target = pose_rot.inverse_transform_vector(&(-pose_t.clone()));

    // 4. Intersect ray with target plane (Z=0)
    // Ray: p(t) = ray_origin_target + t * ray_dir_target
    // Plane: Z = 0
    // Solve: ray_origin_target.z + t * ray_dir_target.z = 0
    let eps = T::from_f64(1e-9).unwrap();
    if ray_dir_target.z.clone().abs() < eps {
        let large_residual = T::from_f64(1e6).unwrap();
        return SVector::<T, 1>::new(large_residual);
    }
    let t_intersect = -ray_origin_target.z.clone() / ray_dir_target.z.clone();
    if t_intersect < -eps {
        let large_residual = T::from_f64(1e6).unwrap();
        return SVector::<T, 1>::new(large_residual);
    }
    let pt_target = ray_origin_target + ray_dir_target * t_intersect;

    // 5. Transform intersection point back to camera frame
    // p_camera = R_C_T * p_target + t_C_T
    let pt_camera = pose_rot.transform_vector(&pt_target) + pose_t;

    // 6. Read plane parameters (unit normal + distance)
    let n_c = Vector3::new(
        plane_normal[0].clone(),
        plane_normal[1].clone(),
        plane_normal[2].clone(),
    );
    let d_c = plane_distance[0].clone();

    // 7. Compute signed distance from pt_camera to laser plane
    // Plane equation: n · p + d = 0
    let dist_to_plane = n_c.dot(&pt_camera) + d_c;

    // 8. Scale by sqrt(weight)
    let sqrt_w = T::from_f64(w.sqrt()).unwrap();
    let residual = dist_to_plane * sqrt_w;

    SVector::<T, 1>::new(residual)
}

/// Helper: Compute point on line of intersection of two planes.
///
/// Given two planes n1·p + d1 = 0 and n2·p + d2 = 0, and their intersection
/// direction v = n1 × n2, find a point p0 on the line of intersection.
///
/// Uses the method of choosing the coordinate with the largest direction component
/// and solving a 2x2 system for numerical stability.
fn compute_line_origin<T: RealField>(
    n1: &Vector3<T>,
    d1: &T,
    n2: &Vector3<T>,
    d2: &T,
    v: &Vector3<T>,
) -> Vector3<T> {
    // Find the coordinate with largest absolute value in v for numerical stability
    let abs_vx = v.x.clone().abs();
    let abs_vy = v.y.clone().abs();
    let abs_vz = v.z.clone().abs();

    if abs_vz >= abs_vx.clone() && abs_vz >= abs_vy.clone() {
        // z has largest component, solve for x and y
        // n1.x * x + n1.y * y = -d1 - n1.z * 0
        // n2.x * x + n2.y * y = -d2 - n2.z * 0
        let det = n1.x.clone() * n2.y.clone() - n1.y.clone() * n2.x.clone();
        let eps = T::from_f64(1e-12).unwrap();
        if det.clone().abs() < eps {
            return Vector3::new(T::zero(), T::zero(), T::zero());
        }
        let x = ((-d1.clone()) * n2.y.clone() - (-d2.clone()) * n1.y.clone()) / det.clone();
        let y = (n1.x.clone() * (-d2.clone()) - n2.x.clone() * (-d1.clone())) / det;
        Vector3::new(x, y, T::zero())
    } else if abs_vy >= abs_vx {
        // y has largest component, solve for x and z
        let det = n1.x.clone() * n2.z.clone() - n1.z.clone() * n2.x.clone();
        let eps = T::from_f64(1e-12).unwrap();
        if det.clone().abs() < eps {
            return Vector3::new(T::zero(), T::zero(), T::zero());
        }
        let x = ((-d1.clone()) * n2.z.clone() - (-d2.clone()) * n1.z.clone()) / det.clone();
        let z = (n1.x.clone() * (-d2.clone()) - n2.x.clone() * (-d1.clone())) / det;
        Vector3::new(x, T::zero(), z)
    } else {
        // x has largest component, solve for y and z
        let det = n1.y.clone() * n2.z.clone() - n1.z.clone() * n2.y.clone();
        let eps = T::from_f64(1e-12).unwrap();
        if det.clone().abs() < eps {
            return Vector3::new(T::zero(), T::zero(), T::zero());
        }
        let y = ((-d1.clone()) * n2.z.clone() - (-d2.clone()) * n1.z.clone()) / det.clone();
        let z = (n1.y.clone() * (-d2.clone()) - n2.y.clone() * (-d1.clone())) / det;
        Vector3::new(T::zero(), y, z)
    }
}

/// Helper: Project 3D line onto z=1 normalized plane.
///
/// Given a 3D line parameterized as L(s) = p0 + s*v, computes the projection
/// of this line onto the z=1 plane in camera coordinates.
///
/// Returns (origin_2d, direction_2d) where both are in normalized camera coordinates.
fn project_line_to_normalized_plane<T: RealField>(
    p0: &Vector3<T>,
    v: &Vector3<T>,
) -> (Vector2<T>, Vector2<T>) {
    // Project origin point onto z=1: [x/z, y/z]
    let p0_norm = Vector2::new(p0.x.clone() / p0.z.clone(), p0.y.clone() / p0.z.clone());

    // Project direction: derivative of [x(s)/z(s), y(s)/z(s)] at s=0
    // x(s) = p0.x + s*v.x, z(s) = p0.z + s*v.z
    // x_norm(s) = x(s)/z(s)
    // dx_norm/ds|_{s=0} = (v.x * z - x * v.z) / z^2 at s=0
    //                    = (v.x * p0.z - p0.x * v.z) / p0.z^2
    let z2_inv = T::one() / (p0.z.clone() * p0.z.clone());
    let v_norm_x = (v.x.clone() * p0.z.clone() - p0.x.clone() * v.z.clone()) * z2_inv.clone();
    let v_norm_y = (v.y.clone() * p0.z.clone() - p0.y.clone() * v.z.clone()) * z2_inv;

    // Normalize direction vector
    let v_norm_len =
        (v_norm_x.clone() * v_norm_x.clone() + v_norm_y.clone() * v_norm_y.clone()).sqrt();
    let eps = T::from_f64(1e-12).unwrap();
    if v_norm_len.clone() < eps {
        return (p0_norm, Vector2::new(T::one(), T::zero()));
    }
    let v_norm = Vector2::new(v_norm_x / v_norm_len.clone(), v_norm_y / v_norm_len);

    (p0_norm, v_norm)
}

/// Generic residual for laser plane constraint (line-distance in normalized plane).
///
/// # Parameters
///
/// - `intr`: [fx, fy, cx, cy] (4D) - Pinhole intrinsics
/// - `dist`: [k1, k2, k3, p1, p2] (5D) - Brown-Conrady distortion
/// - `pose`: [qx, qy, qz, qw, tx, ty, tz] (7D) - SE(3) camera-to-target transform
/// - `plane_normal`: [nx, ny, nz] (3D) - Laser plane unit normal (S2)
/// - `plane_distance`: [d] (1D) - Laser plane signed distance
/// - `laser_pixel`: [u, v] - Observed laser line pixel
/// - `w`: Weight
///
/// # Algorithm
///
/// 1. Compute 3D intersection line of laser plane and target plane (in camera frame)
/// 2. Project this line onto normalized camera plane (z=1)
/// 3. Undistort laser pixel to normalized coordinates
/// 4. Compute perpendicular distance from normalized pixel to projected line
/// 5. Scale by sqrt(fx*fy) to get pixel-comparable residual
///
/// # Returns
///
/// 1D residual scaled by sqrt(w): [distance_pixels * sqrt(w)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn laser_line_dist_normalized_generic<T: RealField>(
    intr: DVectorView<'_, T>,
    dist: DVectorView<'_, T>,
    sensor: DVectorView<'_, T>,
    pose: DVectorView<'_, T>,
    plane_normal: DVectorView<'_, T>,
    plane_distance: DVectorView<'_, T>,
    laser_pixel: [f64; 2],
    w: f64,
) -> SVector<T, 1> {
    debug_assert!(intr.len() >= 4, "intrinsics must have 4 params");
    debug_assert!(dist.len() >= 5, "distortion must have 5 params");
    debug_assert!(sensor.len() >= 2, "sensor must have 2 params");
    debug_assert!(pose.len() == 7, "pose must have 7 params (SE3)");
    debug_assert!(plane_normal.len() == 3, "plane normal must have 3 params");
    debug_assert!(
        plane_distance.len() == 1,
        "plane distance must have 1 param"
    );

    // Extract intrinsics
    let fx = intr[0].clone();
    let fy = intr[1].clone();

    // Extract pose (T_C_T: camera-to-target)
    let pose_qx = pose[0].clone();
    let pose_qy = pose[1].clone();
    let pose_qz = pose[2].clone();
    let pose_qw = pose[3].clone();
    let pose_tx = pose[4].clone();
    let pose_ty = pose[5].clone();
    let pose_tz = pose[6].clone();

    let pose_quat = Quaternion::new(pose_qw, pose_qx, pose_qy, pose_qz);
    let pose_rot = UnitQuaternion::from_quaternion(pose_quat);
    let pose_t = Vector3::new(pose_tx, pose_ty, pose_tz);

    // Extract laser plane in camera frame
    let n_c = Vector3::new(
        plane_normal[0].clone(),
        plane_normal[1].clone(),
        plane_normal[2].clone(),
    );
    let d_c = plane_distance[0].clone();

    // 1. Compute target plane normal in camera frame
    // Target plane is Z=0 in target frame, so normal in target frame is [0, 0, 1]
    // Transform to camera frame: n_target_c = R_C_T * [0, 0, 1] = third column of R
    let rot_matrix = pose_rot.to_rotation_matrix();
    let n_target_c = rot_matrix.matrix().column(2).into_owned();

    // Target plane distance in camera frame: d = -n · t
    let d_target_c = -n_target_c.dot(&pose_t);

    // 2. Compute intersection line direction (cross product of normals)
    let v_3d = n_c.cross(&n_target_c);
    let v_norm_3d = v_3d.norm();

    // Check for parallel planes (degenerate case)
    let epsilon = T::from_f64(1e-9).unwrap();
    if v_norm_3d < epsilon {
        // Planes are parallel - return large residual
        let large_residual = T::from_f64(1e6).unwrap();
        return SVector::<T, 1>::new(large_residual);
    }

    let v_3d_unit = v_3d / v_norm_3d;

    // 3. Find point on intersection line
    let p0_3d = compute_line_origin(&n_c, &d_c, &n_target_c, &d_target_c, &v_3d_unit);

    // Check if point has valid z coordinate for projection
    let p0_z_abs = p0_3d.z.clone().abs();
    if p0_z_abs < epsilon {
        // Point at or behind camera - return large residual
        let large_residual = T::from_f64(1e6).unwrap();
        return SVector::<T, 1>::new(large_residual);
    }

    // 4. Project 3D line onto normalized plane (z=1)
    let (p0_norm, v_norm) = project_line_to_normalized_plane(&p0_3d, &v_3d_unit);

    // 5. Undistort laser pixel to normalized coordinates
    let (x_u, y_u) = undistort_pixel_to_normalized(intr, dist, sensor, laser_pixel);

    // 6. Compute perpendicular distance from pixel to line (2D geometry)
    // Line: p0_norm + t * v_norm
    // Point: [x_u, y_u]
    // Distance = |v_norm × (pixel - p0_norm)| (2D cross product)
    let to_pixel_x = x_u - p0_norm.x.clone();
    let to_pixel_y = y_u - p0_norm.y.clone();

    // 2D cross product: det([v_norm, to_pixel]) = v_norm.x * to_pixel.y - v_norm.y * to_pixel.x
    let cross = v_norm.x.clone() * to_pixel_y - v_norm.y.clone() * to_pixel_x;
    let dist_normalized = cross.abs();

    // 7. Scale by geometric mean focal length to get pixel-comparable residual
    let f_geom = (fx * fy).sqrt();
    let dist_pixels = dist_normalized * f_geom;

    // 8. Apply weight
    let sqrt_w = T::from_f64(w.sqrt()).unwrap();
    let residual = dist_pixels * sqrt_w;

    SVector::<T, 1>::new(residual)
}

/// Build `cam_se3_target` from the hand-eye chain for a rig-level laser factor.
///
/// Inputs are the SE3 param blocks and the per-view robot pose data; output is
/// the composed `(rotation, translation)` that maps points in the target frame
/// into the camera frame, identical to the pose the single-camera
/// `laser_*_residual_generic` functions consume directly.
///
/// The chain matches `reproj_residual_pinhole4_dist5_scheimpflug2_handeye_*`
/// so a rig-level target-corner reprojection and a rig-level laser-line
/// residual share the exact same chain semantics.
///
/// - `cam_to_rig`: 7D SE3 `T_rig_from_cam`.
/// - `handeye`:   7D SE3 — `gripper_se3_rig` (EyeInHand) or `rig_se3_base`
///   (EyeToHand).
/// - `target_ref`: 7D SE3 — `base_se3_target` (EyeInHand) or
///   `gripper_se3_target` (EyeToHand).
/// - `robot_se3`: known `base_se3_gripper` for this view.
#[allow(clippy::too_many_arguments)]
fn compose_cam_se3_target_generic<T: RealField>(
    cam_to_rig: DVectorView<'_, T>,
    handeye: DVectorView<'_, T>,
    target_ref: DVectorView<'_, T>,
    robot_se3: [f64; 7],
    mode: HandEyeMode,
) -> (UnitQuaternion<T>, Vector3<T>) {
    compose_cam_se3_target_with_delta_generic(
        cam_to_rig, handeye, target_ref, robot_se3, None, mode,
    )
}

#[allow(clippy::too_many_arguments)]
fn compose_cam_se3_target_with_delta_generic<T: RealField>(
    cam_to_rig: DVectorView<'_, T>,
    handeye: DVectorView<'_, T>,
    target_ref: DVectorView<'_, T>,
    robot_se3: [f64; 7],
    robot_delta: Option<DVectorView<'_, T>>,
    mode: HandEyeMode,
) -> (UnitQuaternion<T>, Vector3<T>) {
    debug_assert!(cam_to_rig.len() == 7, "cam_to_rig must be 7 SE3");
    debug_assert!(handeye.len() == 7, "handeye must be 7 SE3");
    debug_assert!(target_ref.len() == 7, "target_ref must be 7 SE3");

    let extr_q = UnitQuaternion::from_quaternion(Quaternion::new(
        cam_to_rig[3].clone(),
        cam_to_rig[0].clone(),
        cam_to_rig[1].clone(),
        cam_to_rig[2].clone(),
    ));
    let extr_t = Vector3::new(
        cam_to_rig[4].clone(),
        cam_to_rig[5].clone(),
        cam_to_rig[6].clone(),
    );
    let handeye_q = UnitQuaternion::from_quaternion(Quaternion::new(
        handeye[3].clone(),
        handeye[0].clone(),
        handeye[1].clone(),
        handeye[2].clone(),
    ));
    let handeye_t = Vector3::new(handeye[4].clone(), handeye[5].clone(), handeye[6].clone());
    let target_q = UnitQuaternion::from_quaternion(Quaternion::new(
        target_ref[3].clone(),
        target_ref[0].clone(),
        target_ref[1].clone(),
        target_ref[2].clone(),
    ));
    let target_t = Vector3::new(
        target_ref[4].clone(),
        target_ref[5].clone(),
        target_ref[6].clone(),
    );
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
    let (robot_q, robot_t) = if let Some(delta) = robot_delta {
        let (delta_q, delta_t) = se3_exp(delta);
        (
            delta_q.clone() * robot_q,
            delta_q.transform_vector(&robot_t) + delta_t,
        )
    } else {
        (robot_q, robot_t)
    };

    // Compute (R_net, t_net) such that p_cam = R_net * p_target + t_net, by
    // pushing the origin of the target frame through each stage of the chain.
    match mode {
        HandEyeMode::EyeInHand => {
            // p_cam = extr^-1 ( handeye^-1 ( robot^-1 ( target * p_t ) ) )
            //
            // Composing R: R_net = extr^-1 * handeye^-1 * robot^-1 * target
            let r_net = extr_q.inverse() * handeye_q.inverse() * robot_q.inverse() * target_q;

            // Translation: apply chain to origin (p_t = 0):
            // p_base   = target_t
            // p_grip   = robot_q^-1 * (p_base - robot_t)
            // p_rig    = handeye_q^-1 * (p_grip - handeye_t)
            // p_cam    = extr_q^-1 * (p_rig - extr_t)
            let p_base = target_t;
            let p_grip = robot_q.inverse_transform_vector(&(p_base - robot_t));
            let p_rig = handeye_q.inverse_transform_vector(&(p_grip - handeye_t));
            let t_net = extr_q.inverse_transform_vector(&(p_rig - extr_t));
            (r_net, t_net)
        }
        HandEyeMode::EyeToHand => {
            // p_cam = extr^-1 ( handeye * ( robot * ( target * p_t ) ) )
            //
            // Composing R: R_net = extr^-1 * handeye * robot * target
            let r_net = extr_q.inverse() * handeye_q.clone() * robot_q.clone() * target_q;

            // Translation: origin propagation:
            // p_grip   = target_t
            // p_base   = robot_q * p_grip + robot_t
            // p_rig    = handeye_q * p_base + handeye_t
            // p_cam    = extr_q^-1 * (p_rig - extr_t)
            let p_grip = target_t;
            let p_base = robot_q.transform_vector(&p_grip) + robot_t;
            let p_rig = handeye_q.transform_vector(&p_base) + handeye_t;
            let t_net = extr_q.inverse_transform_vector(&(p_rig - extr_t));
            (r_net, t_net)
        }
    }
}

/// Pack `(rotation, translation)` into a heap-allocated 7D SE3 DVector suitable
/// for the `laser_*_residual_generic` functions.
fn se3_from_rot_trans<T: RealField>(
    rot: &UnitQuaternion<T>,
    t: &Vector3<T>,
) -> nalgebra::DVector<T> {
    let q = rot.quaternion();
    // Layout matches params::pose_se3::iso3_to_se3_dvec: [qx, qy, qz, qw, tx, ty, tz]
    nalgebra::DVector::from_vec(vec![
        q.i.clone(),
        q.j.clone(),
        q.k.clone(),
        q.w.clone(),
        t.x.clone(),
        t.y.clone(),
        t.z.clone(),
    ])
}

/// Rig + hand-eye version of [`laser_plane_pixel_residual_generic`].
///
/// Composes `cam_se3_target` from (`cam_to_rig`, `handeye`, `target_ref`,
/// `robot_data.robot_se3`) via the mode-dependent chain, then delegates to the
/// single-camera point-to-plane residual. Residual is 1D (meters, signed
/// distance from laser-target ray intersection to the laser plane).
#[allow(clippy::too_many_arguments)]
pub(crate) fn laser_plane_pixel_rig_handeye_residual_generic<T: RealField>(
    intr: DVectorView<'_, T>,
    dist: DVectorView<'_, T>,
    sensor: DVectorView<'_, T>,
    cam_to_rig: DVectorView<'_, T>,
    handeye: DVectorView<'_, T>,
    target_ref: DVectorView<'_, T>,
    plane_normal: DVectorView<'_, T>,
    plane_distance: DVectorView<'_, T>,
    robot_data: RobotPoseData,
    laser_pixel: [f64; 2],
    w: f64,
) -> SVector<T, 1> {
    let (rot, t) = compose_cam_se3_target_generic(
        cam_to_rig,
        handeye,
        target_ref,
        robot_data.robot_se3,
        robot_data.mode,
    );
    let pose = se3_from_rot_trans(&rot, &t);
    laser_plane_pixel_residual_generic(
        intr,
        dist,
        sensor,
        pose.as_view(),
        plane_normal,
        plane_distance,
        laser_pixel,
        w,
    )
}

/// Rig + hand-eye + robot-delta version of [`laser_plane_pixel_residual_generic`].
///
/// `robot_delta` is a 6D se(3) tangent correction applied as
/// `exp(delta) * T_B_G`, matching the Scheimpflug hand-eye reprojection
/// robot-delta factor.
#[allow(clippy::too_many_arguments)]
pub(crate) fn laser_plane_pixel_rig_handeye_robot_delta_residual_generic<T: RealField>(
    intr: DVectorView<'_, T>,
    dist: DVectorView<'_, T>,
    sensor: DVectorView<'_, T>,
    cam_to_rig: DVectorView<'_, T>,
    handeye: DVectorView<'_, T>,
    target_ref: DVectorView<'_, T>,
    plane_normal: DVectorView<'_, T>,
    plane_distance: DVectorView<'_, T>,
    robot_delta: DVectorView<'_, T>,
    robot_data: RobotPoseData,
    laser_pixel: [f64; 2],
    w: f64,
) -> SVector<T, 1> {
    let (rot, t) = compose_cam_se3_target_with_delta_generic(
        cam_to_rig,
        handeye,
        target_ref,
        robot_data.robot_se3,
        Some(robot_delta),
        robot_data.mode,
    );
    let pose = se3_from_rot_trans(&rot, &t);
    laser_plane_pixel_residual_generic(
        intr,
        dist,
        sensor,
        pose.as_view(),
        plane_normal,
        plane_distance,
        laser_pixel,
        w,
    )
}

/// Rig + hand-eye version of [`laser_line_dist_normalized_generic`].
///
/// Composes `cam_se3_target` from (`cam_to_rig`, `handeye`, `target_ref`,
/// `robot_data.robot_se3`) via the mode-dependent chain, then delegates to the
/// single-camera line-distance-in-normalized-plane residual. Residual is 1D
/// (pixels, perpendicular distance from the undistorted laser pixel to the
/// projected laser-target intersection line, scaled by sqrt(fx*fy)).
#[allow(clippy::too_many_arguments)]
pub(crate) fn laser_line_dist_normalized_rig_handeye_residual_generic<T: RealField>(
    intr: DVectorView<'_, T>,
    dist: DVectorView<'_, T>,
    sensor: DVectorView<'_, T>,
    cam_to_rig: DVectorView<'_, T>,
    handeye: DVectorView<'_, T>,
    target_ref: DVectorView<'_, T>,
    plane_normal: DVectorView<'_, T>,
    plane_distance: DVectorView<'_, T>,
    robot_data: RobotPoseData,
    laser_pixel: [f64; 2],
    w: f64,
) -> SVector<T, 1> {
    let (rot, t) = compose_cam_se3_target_generic(
        cam_to_rig,
        handeye,
        target_ref,
        robot_data.robot_se3,
        robot_data.mode,
    );
    let pose = se3_from_rot_trans(&rot, &t);
    laser_line_dist_normalized_generic(
        intr,
        dist,
        sensor,
        pose.as_view(),
        plane_normal,
        plane_distance,
        laser_pixel,
        w,
    )
}

/// Rig + hand-eye + robot-delta version of [`laser_line_dist_normalized_generic`].
///
/// `robot_delta` is a 6D se(3) tangent correction applied as
/// `exp(delta) * T_B_G`, matching the Scheimpflug hand-eye reprojection
/// robot-delta factor.
#[allow(clippy::too_many_arguments)]
pub(crate) fn laser_line_dist_normalized_rig_handeye_robot_delta_residual_generic<T: RealField>(
    intr: DVectorView<'_, T>,
    dist: DVectorView<'_, T>,
    sensor: DVectorView<'_, T>,
    cam_to_rig: DVectorView<'_, T>,
    handeye: DVectorView<'_, T>,
    target_ref: DVectorView<'_, T>,
    plane_normal: DVectorView<'_, T>,
    plane_distance: DVectorView<'_, T>,
    robot_delta: DVectorView<'_, T>,
    robot_data: RobotPoseData,
    laser_pixel: [f64; 2],
    w: f64,
) -> SVector<T, 1> {
    let (rot, t) = compose_cam_se3_target_with_delta_generic(
        cam_to_rig,
        handeye,
        target_ref,
        robot_data.robot_se3,
        Some(robot_delta),
        robot_data.mode,
    );
    let pose = se3_from_rot_trans(&rot, &t);
    laser_line_dist_normalized_generic(
        intr,
        dist,
        sensor,
        pose.as_view(),
        plane_normal,
        plane_distance,
        laser_pixel,
        w,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::factors::reprojection_model::apply_scheimpflug_generic;
    use nalgebra::DVector;

    #[test]
    fn laser_residual_smoke_test() {
        // Simple smoke test: plane parallel to XY at z=0.5, pixel at center
        let intr = DVector::from_vec(vec![800.0, 800.0, 640.0, 360.0]);
        let dist = DVector::from_vec(vec![0.0, 0.0, 0.0, 0.0, 0.0]); // No distortion
        let sensor = DVector::from_vec(vec![0.0, 0.0]); // Identity sensor

        // Identity pose (camera frame = target frame)
        let pose = DVector::from_vec(vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]);

        // Plane: z = 0.5, normal = [0, 0, 1], distance = -0.5
        let plane_normal = DVector::from_vec(vec![0.0, 0.0, 1.0]);
        let plane_distance = DVector::from_vec(vec![-0.5]);

        // Pixel at center (will project to z=0 in target frame since pose is identity)
        let pixel = [640.0, 360.0];
        let w = 1.0;

        let residual = laser_plane_pixel_residual_generic(
            intr.as_view(),
            dist.as_view(),
            sensor.as_view(),
            pose.as_view(),
            plane_normal.as_view(),
            plane_distance.as_view(),
            pixel,
            w,
        );

        // Ray from center pixel goes through (0,0,0) in target frame
        // Distance from (0,0,0) to plane z=0.5 is -0.5
        assert!(
            (residual[0] + 0.5_f64).abs() < 0.1,
            "residual: {}",
            residual[0]
        );
    }

    #[test]
    fn laser_line_dist_normalized_smoke_test() {
        // Smoke test: laser plane intersecting with target plane
        let intr = DVector::from_vec(vec![800.0, 800.0, 640.0, 360.0]);
        let dist = DVector::from_vec(vec![0.0, 0.0, 0.0, 0.0, 0.0]); // No distortion
        let sensor = DVector::from_vec(vec![0.0, 0.0]); // Identity sensor

        // Non-identity pose: camera looking at target from distance
        // Camera at (0, 0, 0.5), looking down at target at (0, 0, 0)
        // Rotation: identity (no rotation), Translation: (0, 0, 0.5)
        let pose = DVector::from_vec(vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.5]);

        // Laser plane: tilted, passing through camera frame
        // Normal: [0.1, 0, 1] (tilted), distance: -0.4 (40cm in front)
        let normal = nalgebra::Vector3::new(0.1, 0.0, 1.0).normalize();
        let plane_normal = DVector::from_vec(vec![normal.x, normal.y, normal.z]);
        let plane_distance = DVector::from_vec(vec![-0.4]);

        // Pixel at center
        let pixel = [640.0, 360.0];
        let w = 1.0;

        let residual = laser_line_dist_normalized_generic(
            intr.as_view(),
            dist.as_view(),
            sensor.as_view(),
            pose.as_view(),
            plane_normal.as_view(),
            plane_distance.as_view(),
            pixel,
            w,
        );

        // Residual should be finite (not NaN or Inf)
        // The magnitude depends on geometry - we just verify it's computed correctly
        let res: f64 = residual[0];
        assert!(
            res.is_finite(),
            "residual should be finite (not NaN/Inf): {}",
            res
        );
        // Sanity check: should be less than image width (reasonable for pixel distance)
        assert!(
            res.abs() < 10000.0,
            "residual magnitude unreasonable: {}",
            res
        );
    }

    #[test]
    fn undistort_center_pixel_is_zero() {
        let intr = DVector::from_vec(vec![800.0, 800.0, 640.0, 360.0]);
        let dist = DVector::from_vec(vec![0.0, 0.0, 0.0, 0.0, 0.0]);
        let sensor = DVector::from_vec(vec![0.0, 0.0]);

        let (x_u, y_u): (f64, f64) = undistort_pixel_to_normalized(
            intr.as_view(),
            dist.as_view(),
            sensor.as_view(),
            [640.0, 360.0],
        );

        assert!(x_u.abs() < 1e-12_f64, "x_u should be 0 for principal point");
        assert!(y_u.abs() < 1e-12_f64, "y_u should be 0 for principal point");
    }

    #[test]
    fn undistort_inverts_scheimpflug_for_zero_distortion() {
        let intr = DVector::from_vec(vec![800.0, 820.0, 640.0, 360.0]);
        let dist = DVector::from_vec(vec![0.0, 0.0, 0.0, 0.0, 0.0]);
        let sensor = DVector::from_vec(vec![0.02, -0.01]);

        let x_norm = 0.1_f64;
        let y_norm = -0.05_f64;
        let (x_sensor, y_sensor) = apply_scheimpflug_generic(x_norm, y_norm, sensor[0], sensor[1]);

        let u = intr[0] * x_sensor + intr[2];
        let v = intr[1] * y_sensor + intr[3];

        let (x_u, y_u): (f64, f64) =
            undistort_pixel_to_normalized(intr.as_view(), dist.as_view(), sensor.as_view(), [u, v]);

        assert!((x_u - x_norm).abs() < 1e-8, "x_u mismatch: {}", x_u);
        assert!((y_u - y_norm).abs() < 1e-8, "y_u mismatch: {}", y_u);
    }

    #[test]
    fn test_line_projection_geometry() {
        // Test the helper function for line projection
        use nalgebra::Vector3;

        // Simple case: line parallel to XY plane at z=1
        let p0 = Vector3::new(1.0, 0.5, 1.0);
        let v = Vector3::new(1.0, 0.0, 0.0); // Direction along x-axis

        let (p0_norm, v_norm) = project_line_to_normalized_plane(&p0, &v);

        // Origin should project to (1.0/1.0, 0.5/1.0) = (1.0, 0.5)
        assert!((p0_norm.x - 1.0_f64).abs() < 1e-10);
        assert!((p0_norm.y - 0.5_f64).abs() < 1e-10);

        // Direction should remain along x (1, 0) since line is at constant z
        assert!((v_norm.x - 1.0_f64).abs() < 1e-10);
        assert!(v_norm.y.abs() < 1e-10_f64);
    }

    #[test]
    fn test_compute_line_origin_simple() {
        // Test plane intersection: XY plane (z=0) and YZ plane (x=0)
        // Should intersect along y-axis (x=0, z=0)
        use nalgebra::Vector3;

        let n1 = Vector3::new(0.0, 0.0, 1.0); // XY plane normal
        let d1 = 0.0;
        let n2 = Vector3::new(1.0, 0.0, 0.0); // YZ plane normal
        let d2 = 0.0;
        let v = n1.cross(&n2); // Direction: (0, 1, 0)

        let p0 = compute_line_origin(&n1, &d1, &n2, &d2, &v);

        // Point should satisfy both planes
        let plane1_dist: f64 = n1.dot(&p0) + d1;
        let plane2_dist: f64 = n2.dot(&p0) + d2;
        assert!(plane1_dist.abs() < 1e-10);
        assert!(plane2_dist.abs() < 1e-10);
        // And should have x=0, z=0
        assert!(p0.x.abs() < 1e-10);
        assert!(p0.z.abs() < 1e-10);
    }

    /// Compose ground-truth `cam_se3_target` via the hand-eye chain in each
    /// mode and check that `compose_cam_se3_target_generic` produces the
    /// same transform. This pins down chain semantics and guards against
    /// flipped inverses.
    ///
    /// Conventions (mirror `reproj_residual_pinhole4_dist5_scheimpflug2_handeye_*`):
    /// - `cam_to_rig` param ≡ T_rig_from_cam (forward maps cam→rig).
    /// - EyeInHand: `handeye` = T_gripper_from_rig; `target_ref` = T_base_from_target.
    /// - EyeToHand: `handeye` = T_rig_from_base;    `target_ref` = T_gripper_from_target.
    /// - Robot data: `robot_se3` = T_base_from_gripper (as published by the robot).
    #[test]
    fn compose_cam_se3_target_matches_explicit_chain() {
        use nalgebra::{Isometry3, Translation3, UnitQuaternion as UQ, Vector3 as V3};

        let rot_xyz = |rx: f64, ry: f64, rz: f64| -> UQ<f64> {
            UQ::from_axis_angle(&Vector3::x_axis(), rx)
                * UQ::from_axis_angle(&Vector3::y_axis(), ry)
                * UQ::from_axis_angle(&Vector3::z_axis(), rz)
        };

        // Arbitrary non-identity transforms, distinct enough to catch
        // convention mix-ups.
        let t_rig_from_cam: Isometry3<f64> = Isometry3::from_parts(
            Translation3::new(0.10, -0.05, 0.20),
            rot_xyz(0.1, -0.2, 0.3),
        );
        let handeye_iso: Isometry3<f64> = Isometry3::from_parts(
            Translation3::new(-0.02, 0.03, 0.15),
            rot_xyz(-0.05, 0.04, 0.08),
        );
        let target_ref_iso: Isometry3<f64> =
            Isometry3::from_parts(Translation3::new(0.30, 0.20, 0.40), rot_xyz(0.3, 0.0, -0.1));
        let t_base_from_gripper: Isometry3<f64> =
            Isometry3::from_parts(Translation3::new(0.50, 0.10, 0.20), rot_xyz(0.2, 0.2, 0.2));

        // Helper to pack an Isometry3 into [qx,qy,qz,qw,tx,ty,tz].
        let pack = |iso: &Isometry3<f64>| -> DVector<f64> {
            let q = iso.rotation.into_inner();
            DVector::from_vec(vec![
                q.i,
                q.j,
                q.k,
                q.w,
                iso.translation.vector.x,
                iso.translation.vector.y,
                iso.translation.vector.z,
            ])
        };
        let pack_arr = |iso: &Isometry3<f64>| -> [f64; 7] {
            let q = iso.rotation.into_inner();
            [
                q.i,
                q.j,
                q.k,
                q.w,
                iso.translation.vector.x,
                iso.translation.vector.y,
                iso.translation.vector.z,
            ]
        };

        let cam_to_rig = pack(&t_rig_from_cam);
        let handeye = pack(&handeye_iso);
        let target_ref = pack(&target_ref_iso);
        let robot_arr = pack_arr(&t_base_from_gripper);

        // --- EyeInHand --- T_C_T = T_C_R * T_R_G * T_G_B * T_B_T
        //                       = t_rig_from_cam^-1 * handeye^-1 * robot^-1 * target_ref
        let t_ct_eih: Isometry3<f64> = t_rig_from_cam.inverse()
            * handeye_iso.inverse()
            * t_base_from_gripper.inverse()
            * target_ref_iso;
        let (r, t) = compose_cam_se3_target_generic::<f64>(
            cam_to_rig.as_view(),
            handeye.as_view(),
            target_ref.as_view(),
            robot_arr,
            HandEyeMode::EyeInHand,
        );
        for pw in [[0.01, 0.02, 0.0], [-0.03, 0.04, 0.1], [0.0, 0.0, 0.0]] {
            let p = V3::new(pw[0], pw[1], pw[2]);
            let expected = t_ct_eih * nalgebra::Point3::from(p);
            let actual = r.transform_vector(&p) + t;
            assert!(
                (expected.coords - actual).norm() < 1e-12,
                "EyeInHand chain mismatch: expected={:?} actual={:?}",
                expected.coords,
                actual
            );
        }

        // --- EyeToHand --- T_C_T = T_C_R * T_R_B * T_B_G * T_G_T
        //                       = t_rig_from_cam^-1 * handeye * robot * target_ref
        let t_ct_eth: Isometry3<f64> =
            t_rig_from_cam.inverse() * handeye_iso * t_base_from_gripper * target_ref_iso;
        let (r2, t2) = compose_cam_se3_target_generic::<f64>(
            cam_to_rig.as_view(),
            handeye.as_view(),
            target_ref.as_view(),
            robot_arr,
            HandEyeMode::EyeToHand,
        );
        for pw in [[0.01, 0.02, 0.0], [-0.03, 0.04, 0.1], [0.0, 0.0, 0.0]] {
            let p = V3::new(pw[0], pw[1], pw[2]);
            let expected = t_ct_eth * nalgebra::Point3::from(p);
            let actual = r2.transform_vector(&p) + t2;
            assert!(
                (expected.coords - actual).norm() < 1e-12,
                "EyeToHand chain mismatch: expected={:?} actual={:?}",
                expected.coords,
                actual
            );
        }
    }

    /// Ground-truth laser-plane pixel residual via the rig/hand-eye composition
    /// must equal the single-camera residual when we pass the same composed
    /// `cam_se3_target` directly — in both modes, with both residual types.
    #[test]
    fn rig_handeye_residuals_match_single_camera_via_composition() {
        use nalgebra::{Isometry3, Translation3, UnitQuaternion as UQ};

        let intr = DVector::from_vec(vec![900.0, 900.0, 640.0, 360.0]);
        let dist = DVector::from_vec(vec![0.05, -0.1, 0.0, 0.0, 0.0]);
        let sensor = DVector::from_vec(vec![0.02, -0.01]);
        let plane_normal = {
            let n = nalgebra::Vector3::new(0.1, 0.2, 1.0).normalize();
            DVector::from_vec(vec![n.x, n.y, n.z])
        };
        let plane_distance = DVector::from_vec(vec![-0.35]);
        let laser_pixel = [712.0, 388.0];
        let w = 2.0;

        let t_rig_from_cam: Isometry3<f64> = Isometry3::from_parts(
            Translation3::new(0.05, -0.03, 0.10),
            UQ::from_axis_angle(&Vector3::y_axis(), 0.25_f64),
        );
        let handeye_iso: Isometry3<f64> = Isometry3::from_parts(
            Translation3::new(-0.02, 0.01, 0.08),
            UQ::from_axis_angle(&Vector3::x_axis(), -0.1_f64),
        );
        let target_ref_iso: Isometry3<f64> = Isometry3::from_parts(
            Translation3::new(0.22, 0.15, 0.42),
            UQ::from_axis_angle(&Vector3::z_axis(), 0.35_f64),
        );
        let t_base_from_gripper: Isometry3<f64> = Isometry3::from_parts(
            Translation3::new(0.50, 0.10, 0.30),
            UQ::from_axis_angle(&Vector3::y_axis(), 0.2_f64),
        );

        let pack = |iso: &Isometry3<f64>| -> DVector<f64> {
            let q = iso.rotation.into_inner();
            DVector::from_vec(vec![
                q.i,
                q.j,
                q.k,
                q.w,
                iso.translation.vector.x,
                iso.translation.vector.y,
                iso.translation.vector.z,
            ])
        };
        let pack_arr = |iso: &Isometry3<f64>| -> [f64; 7] {
            let q = iso.rotation.into_inner();
            [
                q.i,
                q.j,
                q.k,
                q.w,
                iso.translation.vector.x,
                iso.translation.vector.y,
                iso.translation.vector.z,
            ]
        };

        let cam_to_rig_dv = pack(&t_rig_from_cam);
        let handeye_dv = pack(&handeye_iso);
        let target_ref_dv = pack(&target_ref_iso);
        let robot_arr = pack_arr(&t_base_from_gripper);

        for mode in [HandEyeMode::EyeInHand, HandEyeMode::EyeToHand] {
            let cam_se3_target: Isometry3<f64> = match mode {
                HandEyeMode::EyeInHand => {
                    t_rig_from_cam.inverse()
                        * handeye_iso.inverse()
                        * t_base_from_gripper.inverse()
                        * target_ref_iso
                }
                HandEyeMode::EyeToHand => {
                    t_rig_from_cam.inverse() * handeye_iso * t_base_from_gripper * target_ref_iso
                }
            };
            let pose_dv = pack(&cam_se3_target);

            let robot_data = RobotPoseData {
                robot_se3: robot_arr,
                mode,
            };

            // Point-to-plane variant.
            let r_single = laser_plane_pixel_residual_generic::<f64>(
                intr.as_view(),
                dist.as_view(),
                sensor.as_view(),
                pose_dv.as_view(),
                plane_normal.as_view(),
                plane_distance.as_view(),
                laser_pixel,
                w,
            );
            let r_rig = laser_plane_pixel_rig_handeye_residual_generic::<f64>(
                intr.as_view(),
                dist.as_view(),
                sensor.as_view(),
                cam_to_rig_dv.as_view(),
                handeye_dv.as_view(),
                target_ref_dv.as_view(),
                plane_normal.as_view(),
                plane_distance.as_view(),
                robot_data,
                laser_pixel,
                w,
            );
            assert!(
                (r_single[0] - r_rig[0]).abs() < 1e-10,
                "PointToPlane mismatch mode={:?}: single={} rig={}",
                mode,
                r_single[0],
                r_rig[0]
            );

            // Line-distance in normalized plane variant.
            let r_single_ln = laser_line_dist_normalized_generic::<f64>(
                intr.as_view(),
                dist.as_view(),
                sensor.as_view(),
                pose_dv.as_view(),
                plane_normal.as_view(),
                plane_distance.as_view(),
                laser_pixel,
                w,
            );
            let r_rig_ln = laser_line_dist_normalized_rig_handeye_residual_generic::<f64>(
                intr.as_view(),
                dist.as_view(),
                sensor.as_view(),
                cam_to_rig_dv.as_view(),
                handeye_dv.as_view(),
                target_ref_dv.as_view(),
                plane_normal.as_view(),
                plane_distance.as_view(),
                robot_data,
                laser_pixel,
                w,
            );
            assert!(
                (r_single_ln[0] - r_rig_ln[0]).abs() < 1e-10,
                "LineDistNormalized mismatch mode={:?}: single={} rig={}",
                mode,
                r_single_ln[0],
                r_rig_ln[0]
            );
        }
    }

    #[test]
    fn rig_handeye_robot_delta_laser_residual_matches_corrected_robot_pose() {
        use nalgebra::{Isometry3, Translation3, UnitQuaternion as UQ};

        let intr = DVector::from_vec(vec![900.0, 880.0, 640.0, 360.0]);
        let dist = DVector::from_vec(vec![0.03, -0.02, 0.0, 0.001, -0.001]);
        let sensor = DVector::from_vec(vec![0.015, -0.008]);
        let plane_normal = {
            let n = nalgebra::Vector3::new(0.08, -0.12, 1.0).normalize();
            DVector::from_vec(vec![n.x, n.y, n.z])
        };
        let plane_distance = DVector::from_vec(vec![-0.32]);
        let laser_pixel = [705.0, 402.0];
        let w = 3.0;

        let t_rig_from_cam: Isometry3<f64> = Isometry3::from_parts(
            Translation3::new(0.05, -0.03, 0.10),
            UQ::from_axis_angle(&Vector3::y_axis(), 0.25_f64),
        );
        let rig_se3_base: Isometry3<f64> = Isometry3::from_parts(
            Translation3::new(-0.02, 0.01, 0.08),
            UQ::from_axis_angle(&Vector3::x_axis(), -0.1_f64),
        );
        let gripper_se3_target: Isometry3<f64> = Isometry3::from_parts(
            Translation3::new(0.22, 0.15, 0.42),
            UQ::from_axis_angle(&Vector3::z_axis(), 0.35_f64),
        );
        let base_se3_gripper: Isometry3<f64> = Isometry3::from_parts(
            Translation3::new(0.50, 0.10, 0.30),
            UQ::from_axis_angle(&Vector3::y_axis(), 0.2_f64),
        );
        let delta = DVector::from_vec(vec![0.003, -0.002, 0.004, 0.0005, -0.0004, 0.0003]);

        let pack = |iso: &Isometry3<f64>| -> DVector<f64> {
            let q = iso.rotation.into_inner();
            DVector::from_vec(vec![
                q.i,
                q.j,
                q.k,
                q.w,
                iso.translation.vector.x,
                iso.translation.vector.y,
                iso.translation.vector.z,
            ])
        };
        let pack_arr = |iso: &Isometry3<f64>| -> [f64; 7] {
            let q = iso.rotation.into_inner();
            [
                q.i,
                q.j,
                q.k,
                q.w,
                iso.translation.vector.x,
                iso.translation.vector.y,
                iso.translation.vector.z,
            ]
        };

        let (delta_q, delta_t) = se3_exp(delta.as_view());
        let corrected_robot = Isometry3::from_parts(
            Translation3::from(
                delta_q.transform_vector(&base_se3_gripper.translation.vector) + delta_t,
            ),
            delta_q * base_se3_gripper.rotation,
        );

        let cam_to_rig_dv = pack(&t_rig_from_cam);
        let handeye_dv = pack(&rig_se3_base);
        let target_ref_dv = pack(&gripper_se3_target);
        let robot_arr = pack_arr(&base_se3_gripper);
        let corrected_robot_arr = pack_arr(&corrected_robot);
        let robot_data = RobotPoseData {
            robot_se3: robot_arr,
            mode: HandEyeMode::EyeToHand,
        };
        let corrected_robot_data = RobotPoseData {
            robot_se3: corrected_robot_arr,
            mode: HandEyeMode::EyeToHand,
        };

        let p2p_delta = laser_plane_pixel_rig_handeye_robot_delta_residual_generic::<f64>(
            intr.as_view(),
            dist.as_view(),
            sensor.as_view(),
            cam_to_rig_dv.as_view(),
            handeye_dv.as_view(),
            target_ref_dv.as_view(),
            plane_normal.as_view(),
            plane_distance.as_view(),
            delta.as_view(),
            robot_data,
            laser_pixel,
            w,
        );
        let p2p_corrected = laser_plane_pixel_rig_handeye_residual_generic::<f64>(
            intr.as_view(),
            dist.as_view(),
            sensor.as_view(),
            cam_to_rig_dv.as_view(),
            handeye_dv.as_view(),
            target_ref_dv.as_view(),
            plane_normal.as_view(),
            plane_distance.as_view(),
            corrected_robot_data,
            laser_pixel,
            w,
        );
        assert!((p2p_delta[0] - p2p_corrected[0]).abs() < 1e-10);

        let line_delta = laser_line_dist_normalized_rig_handeye_robot_delta_residual_generic::<f64>(
            intr.as_view(),
            dist.as_view(),
            sensor.as_view(),
            cam_to_rig_dv.as_view(),
            handeye_dv.as_view(),
            target_ref_dv.as_view(),
            plane_normal.as_view(),
            plane_distance.as_view(),
            delta.as_view(),
            robot_data,
            laser_pixel,
            w,
        );
        let line_corrected = laser_line_dist_normalized_rig_handeye_residual_generic::<f64>(
            intr.as_view(),
            dist.as_view(),
            sensor.as_view(),
            cam_to_rig_dv.as_view(),
            handeye_dv.as_view(),
            target_ref_dv.as_view(),
            plane_normal.as_view(),
            plane_distance.as_view(),
            corrected_robot_data,
            laser_pixel,
            w,
        );
        assert!((line_delta[0] - line_corrected[0]).abs() < 1e-10);
    }
}
