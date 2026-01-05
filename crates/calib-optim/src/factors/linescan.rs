//! Linescan laser plane residual factors.
//!
//! These residuals constrain laser line pixels to lie on a laser plane
//! by computing the point-to-plane distance for ray-target intersections.

use nalgebra::{DVectorView, Quaternion, RealField, SVector, UnitQuaternion, Vector3};

/// Generic residual for laser plane constraint (point-to-plane distance).
///
/// # Parameters
///
/// - `intr`: [fx, fy, cx, cy] (4D) - Pinhole intrinsics
/// - `dist`: [k1, k2, k3, p1, p2] (5D) - Brown-Conrady distortion
/// - `pose`: [qx, qy, qz, qw, tx, ty, tz] (7D) - SE(3) camera-to-target transform
/// - `plane`: [nx, ny, nz, d] (4D) - Laser plane (normal + distance, normalized in function)
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
pub(crate) fn laser_plane_pixel_residual_generic<T: RealField>(
    intr: DVectorView<'_, T>,
    dist: DVectorView<'_, T>,
    pose: DVectorView<'_, T>,
    plane: DVectorView<'_, T>,
    laser_pixel: [f64; 2],
    w: f64,
) -> SVector<T, 1> {
    debug_assert!(intr.len() >= 4, "intrinsics must have 4 params");
    debug_assert!(dist.len() >= 5, "distortion must have 5 params");
    debug_assert!(pose.len() == 7, "pose must have 7 params (SE3)");
    debug_assert!(plane.len() == 4, "plane must have 4 params");

    // Extract intrinsics
    let fx = intr[0].clone();
    let fy = intr[1].clone();
    let cx = intr[2].clone();
    let cy = intr[3].clone();

    // Extract distortion
    let k1 = dist[0].clone();
    let k2 = dist[1].clone();
    let k3 = dist[2].clone();
    let p1 = dist[3].clone();
    let p2 = dist[4].clone();

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
    let u_px = T::from_f64(laser_pixel[0]).unwrap();
    let v_px = T::from_f64(laser_pixel[1]).unwrap();

    // Convert to normalized coordinates (apply K^-1)
    let x_n = (u_px - cx.clone()) / fx.clone();
    let y_n = (v_px - cy.clone()) / fy.clone();

    // Undistort (iterative approximation: use distorted coordinates as initial guess)
    // For autodiff compatibility, we use a simplified single-iteration undistortion
    let r2 = x_n.clone() * x_n.clone() + y_n.clone() * y_n.clone();
    let r4 = r2.clone() * r2.clone();
    let r6 = r4.clone() * r2.clone();

    let radial =
        T::one() + k1.clone() * r2.clone() + k2.clone() * r4.clone() + k3.clone() * r6.clone();
    let xy = x_n.clone() * y_n.clone();
    let two = T::from_f64(2.0).unwrap();
    let dx_t = two.clone() * p1.clone() * xy.clone()
        + p2.clone() * (r2.clone() + two.clone() * x_n.clone() * x_n.clone());
    let dy_t = p1.clone() * (r2.clone() + two.clone() * y_n.clone() * y_n.clone())
        + two.clone() * p2.clone() * xy;

    // Approximate undistortion (inverse): x_u ≈ x_n / radial - delta_tangential
    let x_u = (x_n - dx_t.clone()) / radial.clone();
    let y_u = (y_n - dy_t.clone()) / radial;

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
    let t_intersect = -ray_origin_target.z.clone() / ray_dir_target.z.clone();
    let pt_target = ray_origin_target + ray_dir_target * t_intersect;

    // 5. Transform intersection point back to camera frame
    // p_camera = R_C_T * p_target + t_C_T
    let pt_camera = pose_rot.transform_vector(&pt_target) + pose_t;

    // 6. Normalize plane parameters (enforce manifold constraint)
    let n_x = plane[0].clone();
    let n_y = plane[1].clone();
    let n_z = plane[2].clone();
    let d_raw = plane[3].clone();

    let n_norm =
        (n_x.clone() * n_x.clone() + n_y.clone() * n_y.clone() + n_z.clone() * n_z.clone()).sqrt();
    let n_x_unit = n_x / n_norm.clone();
    let n_y_unit = n_y / n_norm.clone();
    let n_z_unit = n_z / n_norm.clone();
    let d_unit = d_raw / n_norm;

    // 7. Compute signed distance from pt_camera to laser plane
    // Plane equation: n · p + d = 0
    let dist_to_plane = n_x_unit * pt_camera.x.clone()
        + n_y_unit * pt_camera.y.clone()
        + n_z_unit * pt_camera.z.clone()
        + d_unit;

    // 8. Scale by sqrt(weight)
    let sqrt_w = T::from_f64(w.sqrt()).unwrap();
    let residual = dist_to_plane * sqrt_w;

    SVector::<T, 1>::new(residual)
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::DVector;

    #[test]
    fn laser_residual_smoke_test() {
        // Simple smoke test: plane parallel to XY at z=0.5, pixel at center
        let intr = DVector::from_vec(vec![800.0, 800.0, 640.0, 360.0]);
        let dist = DVector::from_vec(vec![0.0, 0.0, 0.0, 0.0, 0.0]); // No distortion

        // Identity pose (camera frame = target frame)
        let pose = DVector::from_vec(vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]);

        // Plane: z = 0.5, normal = [0, 0, 1], distance = -0.5
        let plane = DVector::from_vec(vec![0.0, 0.0, 1.0, -0.5]);

        // Pixel at center (will project to z=0 in target frame since pose is identity)
        let pixel = [640.0, 360.0];
        let w = 1.0;

        let residual = laser_plane_pixel_residual_generic(
            intr.as_view(),
            dist.as_view(),
            pose.as_view(),
            plane.as_view(),
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
}
