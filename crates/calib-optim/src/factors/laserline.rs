//! Laserline plane residual factors.
//!
//! These residuals constrain laser line pixels to lie on a laser plane.
//! Two approaches are provided:
//! 1. Point-to-plane distance: ray-target intersection then distance to laser plane
//! 2. Line-distance in normalized plane: projects laser-target line to normalized plane

use crate::factors::reprojection_model::apply_scheimpflug_inverse_generic;
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
}
