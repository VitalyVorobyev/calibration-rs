//! Laserline plane residual factors.
//!
//! These residuals constrain laser line pixels to lie on a laser plane.
//! Two approaches are provided:
//! 1. Point-to-plane distance: ray-target intersection then distance to laser plane
//! 2. Line-distance in normalized plane: projects laser-target line to normalized plane
//!
//! Both are generic over the camera-model kernels (distortion + sensor) and a
//! [`LaserChain`]:
//! - [`LaserChain::SinglePose`]: the residual takes a direct `cam_se3_target`
//!   SE3 parameter block.
//! - [`LaserChain::Rig`]: the residual composes `cam_se3_target =
//!   cam_se3_rig · rig_se3_target`, as the rig reprojection chain does.
//! - [`LaserChain::RigHandeye`] / [`LaserChain::RigHandeyeRobotDelta`]: the
//!   residual composes `cam_se3_target` from (`rig_se3_cam`, `handeye`,
//!   `target_ref`, robot pose) according to [`HandeyeMode`], so that rig
//!   extrinsics, hand-eye transform, and target reference pose all move under
//!   a single cost with the laser geometry constraint.

use crate::factors::camera_blocks::CameraBlocks;
use crate::factors::reprojection_model::{se3_exp, se3_from_block};
use crate::ir::{CameraModelSpec, HandeyeMode, LaserChain};
use nalgebra::{DVector, DVectorView, Quaternion, RealField, SVector, UnitQuaternion, Vector3};

/// Generic residual for laser plane constraint (point-to-plane distance).
///
/// # Parameters
///
/// - `camera`: the camera model and its parameter blocks
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
/// Model-generic core of the point-to-plane laser residual.
pub(crate) fn laser_point_to_plane_core<T: RealField>(
    camera: &CameraBlocks<'_, T>,
    pose: DVectorView<'_, T>,
    plane_normal: DVectorView<'_, T>,
    plane_distance: DVectorView<'_, T>,
    laser_pixel: [f64; 2],
    w: f64,
) -> SVector<T, 1> {
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

    // 1-2. Back-project the pixel to its camera ray.
    let ray = camera.pixel_ray(laser_pixel);
    let ray_dir_camera = ray.direction.normalize();

    // 3. Transform ray to target frame
    // T_T_C = (T_C_T)^-1
    // ray_dir_target = R_T_C * ray_dir_camera = R_C_T^T * ray_dir_camera
    let ray_dir_target = pose_rot.inverse_transform_vector(&ray_dir_camera);

    // Ray origin in target frame: R_C_T^T * (o_C - t_C_T)
    let ray_origin_target = pose_rot.inverse_transform_vector(&(ray.origin - pose_t.clone()));

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

/// Model-generic core of the line-distance laser residual.
///
/// # Parameters
///
/// - `camera`: the camera model and its parameter blocks
/// - `pose`: [qx, qy, qz, qw, tx, ty, tz] (7D) - SE(3) camera-to-target transform
/// - `plane_normal`: [nx, ny, nz] (3D) - Laser plane unit normal (S2)
/// - `plane_distance`: [d] (1D) - Laser plane signed distance
/// - `laser_pixel`: [u, v] - Observed laser line pixel
/// - `w`: Weight
///
/// # Returns
///
/// 1D residual scaled by sqrt(w): [distance_pixels * sqrt(w)], from
/// [`laser_line_distance`]; a large constant where that is undefined.
pub(crate) fn laser_line_dist_core<T: RealField>(
    camera: &CameraBlocks<'_, T>,
    pose: DVectorView<'_, T>,
    plane_normal: DVectorView<'_, T>,
    plane_distance: DVectorView<'_, T>,
    laser_pixel: [f64; 2],
    w: f64,
) -> SVector<T, 1> {
    let residual = laser_line_distance(camera, pose, plane_normal, plane_distance, laser_pixel)
        .map_or_else(
            || T::from_f64(1e6).unwrap(),
            |d| d * T::from_f64(w.sqrt()).unwrap(),
        );
    SVector::<T, 1>::new(residual)
}

/// Distance from a laser pixel to the laser line, in pixels.
///
/// 1. Intersect the laser plane with the target plane (camera frame).
/// 2. Project that 3D line onto the normalized camera plane (z = 1), where it
///    stays straight.
/// 3. Undistort the pixel to the same plane.
/// 4. Take the perpendicular distance, scaled by `sqrt(fx·fy)` to pixels.
///
/// `None` when the planes are parallel or the line passes through the
/// camera centre. Parameters as in [`laser_line_dist_core`].
///
/// The camera's projection must image 3D lines as straight lines
/// ([`ProjectionKind::preserves_lines`](vision_calibration_core::ProjectionKind::preserves_lines));
/// `ProblemIR::validate` rejects a line-distance factor on any other.
pub(crate) fn laser_line_distance<T: RealField>(
    camera: &CameraBlocks<'_, T>,
    pose: DVectorView<'_, T>,
    plane_normal: DVectorView<'_, T>,
    plane_distance: DVectorView<'_, T>,
    laser_pixel: [f64; 2],
) -> Option<T> {
    debug_assert!(pose.len() == 7, "pose must have 7 params (SE3)");
    debug_assert!(plane_normal.len() == 3, "plane normal must have 3 params");
    debug_assert!(
        plane_distance.len() == 1,
        "plane distance must have 1 param"
    );

    let fx = camera.intrinsics()[0].clone();
    let fy = camera.intrinsics()[1].clone();

    // Pose (T_C_T: camera-to-target)
    let pose_quat = Quaternion::new(
        pose[3].clone(),
        pose[0].clone(),
        pose[1].clone(),
        pose[2].clone(),
    );
    let pose_rot = UnitQuaternion::from_quaternion(pose_quat);
    let pose_t = Vector3::new(pose[4].clone(), pose[5].clone(), pose[6].clone());

    // Laser plane in camera frame
    let n_c = Vector3::new(
        plane_normal[0].clone(),
        plane_normal[1].clone(),
        plane_normal[2].clone(),
    );
    let d_c = plane_distance[0].clone();

    // Target plane (z = 0 in the target frame) in the camera frame: its
    // normal is the third column of R_C_T, its distance -n · t.
    let rot_matrix = pose_rot.to_rotation_matrix();
    let n_target_c = rot_matrix.matrix().column(2).into_owned();
    let d_target_c = -n_target_c.dot(&pose_t);

    // Projective 2D line of the two planes' intersection on z = 1:
    // elimination of Z from n_c·p + d_c = 0 and n_target_c·p + d_target_c = 0
    // gives (n_c * d_target_c - n_target_c * d_c) · [x_u, y_u, 1] = 0.
    // Smooth everywhere with no axis-selection branch or abs() cusp.
    let line = n_c * d_target_c - n_target_c * d_c;
    let line_norm2 = line.x.clone() * line.x.clone() + line.y.clone() * line.y.clone();
    let epsilon = T::from_f64(1e-9).unwrap();
    if line_norm2 < epsilon.clone() * epsilon {
        return None;
    }
    let line_norm = line_norm2.sqrt();
    let [x_u, y_u]: [T; 2] = camera.pixel_to_normalized(laser_pixel).coords.into();
    let dist_norm = (line.x.clone() * x_u + line.y.clone() * y_u + line.z.clone()) / line_norm;
    Some(dist_norm * (fx * fy).sqrt())
}

/// Build `cam_se3_target` from the hand-eye chain for a rig-level laser factor.
///
/// Inputs are the SE3 param blocks, the per-view robot pose, and an optional
/// 6D se(3) robot-pose correction (applied as `T_B_G * exp(delta)`); output
/// is the composed `(rotation, translation)` that maps points in the target
/// frame into the camera frame, identical to the pose the single-camera laser
/// cores consume directly.
///
/// The chain matches the hand-eye reprojection chains so a rig-level
/// target-corner reprojection and a rig-level laser residual share the exact
/// same chain semantics.
///
/// - `rig_se3_cam`: 7D SE3 `T_rig_from_cam`.
/// - `handeye`:   7D SE3 — `gripper_se3_rig` (EyeInHand) or `rig_se3_base`
///   (EyeToHand).
/// - `target_ref`: 7D SE3 — `base_se3_target` (EyeInHand) or
///   `gripper_se3_target` (EyeToHand).
/// - `robot_se3`: known `base_se3_gripper` for this view.
fn compose_cam_se3_target_with_delta_generic<T: RealField>(
    rig_se3_cam: DVectorView<'_, T>,
    handeye: DVectorView<'_, T>,
    target_ref: DVectorView<'_, T>,
    robot_se3: [f64; 7],
    robot_delta: Option<DVectorView<'_, T>>,
    mode: HandeyeMode,
) -> (UnitQuaternion<T>, Vector3<T>) {
    debug_assert!(rig_se3_cam.len() == 7, "rig_se3_cam must be 7 SE3");
    debug_assert!(handeye.len() == 7, "handeye must be 7 SE3");
    debug_assert!(target_ref.len() == 7, "target_ref must be 7 SE3");

    let extr_q = UnitQuaternion::from_quaternion(Quaternion::new(
        rig_se3_cam[3].clone(),
        rig_se3_cam[0].clone(),
        rig_se3_cam[1].clone(),
        rig_se3_cam[2].clone(),
    ));
    let extr_t = Vector3::new(
        rig_se3_cam[4].clone(),
        rig_se3_cam[5].clone(),
        rig_se3_cam[6].clone(),
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
        // Gripper-frame correction: T_B_G · exp(δ).
        let (delta_q, delta_t) = se3_exp(delta);
        (
            robot_q.clone() * delta_q,
            robot_q.transform_vector(&delta_t) + robot_t,
        )
    } else {
        (robot_q, robot_t)
    };

    // Compute (R_net, t_net) such that p_cam = R_net * p_target + t_net, by
    // pushing the origin of the target frame through each stage of the chain.
    match mode {
        HandeyeMode::EyeInHand => {
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
        HandeyeMode::EyeToHand => {
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

/// Resolve a [`LaserChain`]'s pose and plane blocks against the IR-ordered
/// chain block slice, composing the rig/hand-eye pose when needed.
///
/// Returns `(pose, plane_normal_index, plane_distance_index)` where `pose` is
/// an owned 7D SE3 vector and the indices are into `blocks`.
fn resolve_laser_chain_pose<T: RealField>(
    chain: &LaserChain,
    blocks: &[DVector<T>],
) -> (DVector<T>, usize, usize) {
    match chain {
        LaserChain::SinglePose => {
            debug_assert!(blocks.len() == 3, "SinglePose laser chain expects 3 blocks");
            (blocks[0].clone(), 1, 2)
        }
        LaserChain::Rig => {
            debug_assert!(blocks.len() == 4, "Rig laser chain expects 4 blocks");
            let (cam_q, cam_t) = se3_from_block(&blocks[0]);
            let (rig_q, rig_t) = se3_from_block(&blocks[1]);
            // cam_se3_target = cam_se3_rig · rig_se3_target.
            let rot = cam_q.clone() * rig_q;
            let t = cam_q.transform_vector(&rig_t) + cam_t;
            (se3_from_rot_trans(&rot, &t), 2, 3)
        }
        LaserChain::RigHandeye {
            base_se3_gripper,
            mode,
        } => {
            debug_assert!(blocks.len() == 5, "RigHandeye laser chain expects 5 blocks");
            let (rot, t) = compose_cam_se3_target_with_delta_generic(
                blocks[0].as_view(),
                blocks[1].as_view(),
                blocks[2].as_view(),
                *base_se3_gripper,
                None,
                *mode,
            );
            (se3_from_rot_trans(&rot, &t), 3, 4)
        }
        LaserChain::RigHandeyeRobotDelta {
            base_se3_gripper,
            mode,
        } => {
            debug_assert!(
                blocks.len() == 6,
                "RigHandeyeRobotDelta laser chain expects 6 blocks"
            );
            let (rot, t) = compose_cam_se3_target_with_delta_generic(
                blocks[0].as_view(),
                blocks[1].as_view(),
                blocks[2].as_view(),
                *base_se3_gripper,
                Some(blocks[5].as_view()),
                *mode,
            );
            (se3_from_rot_trans(&rot, &t), 3, 4)
        }
    }
}

/// Point-to-plane laser residual of a camera model and chain.
///
/// `params` is the full IR-ordered block list: the camera blocks of `model`
/// followed by the chain blocks.
pub(crate) fn laser_point_to_plane_residual<T: RealField>(
    model: CameraModelSpec,
    chain: &LaserChain,
    params: &[DVector<T>],
    laser_pixel: [f64; 2],
    w: f64,
) -> SVector<T, 1> {
    let (camera, blocks) = CameraBlocks::split(model, params);
    let (pose, n_idx, d_idx) = resolve_laser_chain_pose(chain, blocks);
    laser_point_to_plane_core(
        &camera,
        pose.as_view(),
        blocks[n_idx].as_view(),
        blocks[d_idx].as_view(),
        laser_pixel,
        w,
    )
}

/// Line-distance laser residual of a camera model and chain.
///
/// `params` is the full IR-ordered block list: the camera blocks of `model`
/// followed by the chain blocks.
pub(crate) fn laser_line_distance_residual<T: RealField>(
    model: CameraModelSpec,
    chain: &LaserChain,
    params: &[DVector<T>],
    laser_pixel: [f64; 2],
    w: f64,
) -> SVector<T, 1> {
    let (camera, blocks) = CameraBlocks::split(model, params);
    let (pose, n_idx, d_idx) = resolve_laser_chain_pose(chain, blocks);
    laser_line_dist_core(
        &camera,
        pose.as_view(),
        blocks[n_idx].as_view(),
        blocks[d_idx].as_view(),
        laser_pixel,
        w,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{DistortionKind, SensorKind};
    use nalgebra::Point2;

    const SPEC: CameraModelSpec =
        CameraModelSpec::pinhole(DistortionKind::BrownConrady5, SensorKind::Scheimpflug);
    use nalgebra::{DVector, Isometry3, Translation3, UnitQuaternion as UQ};

    fn pack(iso: &Isometry3<f64>) -> DVector<f64> {
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
    }

    fn pack_arr(iso: &Isometry3<f64>) -> [f64; 7] {
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
    }

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
        let residual = laser_point_to_plane_core(
            &CameraBlocks::new(
                SPEC,
                intr.as_slice(),
                &[],
                dist.as_slice(),
                sensor.as_slice(),
            ),
            pose.as_view(),
            plane_normal.as_view(),
            plane_distance.as_view(),
            [640.0, 360.0],
            1.0,
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
    fn laser_line_dist_smoke_test() {
        let intr = DVector::from_vec(vec![800.0, 800.0, 640.0, 360.0]);
        let dist = DVector::from_vec(vec![0.0, 0.0, 0.0, 0.0, 0.0]);
        let sensor = DVector::from_vec(vec![0.0, 0.0]);
        let pose = DVector::from_vec(vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.5]);
        let normal = nalgebra::Vector3::new(0.1, 0.0, 1.0).normalize();
        let plane_normal = DVector::from_vec(vec![normal.x, normal.y, normal.z]);
        let plane_distance = DVector::from_vec(vec![-0.4]);

        let residual = laser_line_dist_core(
            &CameraBlocks::new(
                SPEC,
                intr.as_slice(),
                &[],
                dist.as_slice(),
                sensor.as_slice(),
            ),
            pose.as_view(),
            plane_normal.as_view(),
            plane_distance.as_view(),
            [640.0, 360.0],
            1.0,
        );

        let res: f64 = residual[0];
        assert!(res.is_finite(), "residual should be finite: {res}");
        assert!(
            res.abs() < 10000.0,
            "residual magnitude unreasonable: {res}"
        );
    }

    #[test]
    fn undistort_center_pixel_is_zero() {
        let intr = DVector::from_vec(vec![800.0, 800.0, 640.0, 360.0]);
        let dist = DVector::from_vec(vec![0.0, 0.0, 0.0, 0.0, 0.0]);
        let sensor = DVector::from_vec(vec![0.0, 0.0]);

        let n = CameraBlocks::new(
            SPEC,
            intr.as_slice(),
            &[],
            dist.as_slice(),
            sensor.as_slice(),
        )
        .pixel_to_normalized([640.0, 360.0]);
        let (x_u, y_u): (f64, f64) = (n.x, n.y);

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
        let p_sensor =
            SensorKind::Scheimpflug.to_sensor(sensor.as_slice(), &Point2::new(x_norm, y_norm));
        let (x_sensor, y_sensor) = (p_sensor.x, p_sensor.y);

        let u = intr[0] * x_sensor + intr[2];
        let v = intr[1] * y_sensor + intr[3];

        let n = CameraBlocks::new(
            SPEC,
            intr.as_slice(),
            &[],
            dist.as_slice(),
            sensor.as_slice(),
        )
        .pixel_to_normalized([u, v]);
        let (x_u, y_u): (f64, f64) = (n.x, n.y);

        assert!((x_u - x_norm).abs() < 1e-8, "x_u mismatch: {}", x_u);
        assert!((y_u - y_norm).abs() < 1e-8, "y_u mismatch: {}", y_u);
    }

    /// Compose ground-truth `cam_se3_target` via the hand-eye chain in each
    /// mode and check that `compose_cam_se3_target_with_delta_generic`
    /// produces the same transform. This pins down chain semantics and guards
    /// against flipped inverses.
    ///
    /// Conventions (mirror the hand-eye reprojection chains):
    /// - `rig_se3_cam` param ≡ T_rig_from_cam (forward maps cam→rig).
    /// - EyeInHand: `handeye` = T_gripper_from_rig; `target_ref` = T_base_from_target.
    /// - EyeToHand: `handeye` = T_rig_from_base;    `target_ref` = T_gripper_from_target.
    /// - Robot data: `robot_se3` = T_base_from_gripper (as published by the robot).
    #[test]
    fn compose_cam_se3_target_matches_explicit_chain() {
        use nalgebra::Vector3 as V3;

        let rot_xyz = |rx: f64, ry: f64, rz: f64| -> UQ<f64> {
            UQ::from_axis_angle(&Vector3::x_axis(), rx)
                * UQ::from_axis_angle(&Vector3::y_axis(), ry)
                * UQ::from_axis_angle(&Vector3::z_axis(), rz)
        };

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

        let rig_se3_cam = pack(&t_rig_from_cam);
        let handeye = pack(&handeye_iso);
        let target_ref = pack(&target_ref_iso);
        let robot_arr = pack_arr(&t_base_from_gripper);

        // --- EyeInHand --- T_C_T = T_C_R * T_R_G * T_G_B * T_B_T
        let t_ct_eih: Isometry3<f64> = t_rig_from_cam.inverse()
            * handeye_iso.inverse()
            * t_base_from_gripper.inverse()
            * target_ref_iso;
        let (r, t) = compose_cam_se3_target_with_delta_generic::<f64>(
            rig_se3_cam.as_view(),
            handeye.as_view(),
            target_ref.as_view(),
            robot_arr,
            None,
            HandeyeMode::EyeInHand,
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
        let t_ct_eth: Isometry3<f64> =
            t_rig_from_cam.inverse() * handeye_iso * t_base_from_gripper * target_ref_iso;
        let (r2, t2) = compose_cam_se3_target_with_delta_generic::<f64>(
            rig_se3_cam.as_view(),
            handeye.as_view(),
            target_ref.as_view(),
            robot_arr,
            None,
            HandeyeMode::EyeToHand,
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

    /// The rig/hand-eye chain residual must equal the single-pose residual
    /// evaluated at the explicitly composed `cam_se3_target` — in both modes,
    /// for both residual types.
    #[test]
    fn rig_handeye_residuals_match_single_camera_via_composition() {
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

        let robot_arr = pack_arr(&t_base_from_gripper);

        for mode in [HandeyeMode::EyeInHand, HandeyeMode::EyeToHand] {
            let cam_se3_target: Isometry3<f64> = match mode {
                HandeyeMode::EyeInHand => {
                    t_rig_from_cam.inverse()
                        * handeye_iso.inverse()
                        * t_base_from_gripper.inverse()
                        * target_ref_iso
                }
                HandeyeMode::EyeToHand => {
                    t_rig_from_cam.inverse() * handeye_iso * t_base_from_gripper * target_ref_iso
                }
            };
            let pose_dv = pack(&cam_se3_target);
            let chain = LaserChain::RigHandeye {
                base_se3_gripper: robot_arr,
                mode,
            };
            let chain_params = [
                intr.clone(),
                dist.clone(),
                sensor.clone(),
                pack(&t_rig_from_cam),
                pack(&handeye_iso),
                pack(&target_ref_iso),
                plane_normal.clone(),
                plane_distance.clone(),
            ];

            // Point-to-plane variant.
            let r_single = laser_point_to_plane_core(
                &CameraBlocks::new(
                    SPEC,
                    intr.as_slice(),
                    &[],
                    dist.as_slice(),
                    sensor.as_slice(),
                ),
                pose_dv.as_view(),
                plane_normal.as_view(),
                plane_distance.as_view(),
                laser_pixel,
                w,
            );
            let r_rig = laser_point_to_plane_residual(SPEC, &chain, &chain_params, laser_pixel, w);
            assert!(
                (r_single[0] - r_rig[0]).abs() < 1e-10,
                "PointToPlane mismatch mode={:?}: single={} rig={}",
                mode,
                r_single[0],
                r_rig[0]
            );

            // Line-distance variant.
            let r_single_ln = laser_line_dist_core(
                &CameraBlocks::new(
                    SPEC,
                    intr.as_slice(),
                    &[],
                    dist.as_slice(),
                    sensor.as_slice(),
                ),
                pose_dv.as_view(),
                plane_normal.as_view(),
                plane_distance.as_view(),
                laser_pixel,
                w,
            );
            let r_rig_ln =
                laser_line_distance_residual(SPEC, &chain, &chain_params, laser_pixel, w);
            assert!(
                (r_single_ln[0] - r_rig_ln[0]).abs() < 1e-10,
                "LineDistNormalized mismatch mode={:?}: single={} rig={}",
                mode,
                r_single_ln[0],
                r_rig_ln[0]
            );
        }
    }

    /// The rig chain equals the single-pose chain at the composed pose
    /// `cam_se3_rig · rig_se3_target`, for both laser metrics.
    #[test]
    fn rig_laser_residuals_match_single_pose_via_composition() {
        let intr = DVector::from_vec(vec![900.0, 880.0, 640.0, 360.0]);
        let dist = DVector::from_vec(vec![0.05, -0.1, 0.001, -0.002, 0.01]);
        let sensor = DVector::from_vec(vec![0.02, -0.01]);
        let n = Vector3::new(0.1, 0.2, 1.0).normalize();
        let plane_normal = DVector::from_vec(vec![n.x, n.y, n.z]);
        let plane_distance = DVector::from_vec(vec![-0.35]);
        let laser_pixel = [712.0, 388.0];
        let w = 2.0;
        let cam_se3_rig = Isometry3::from_parts(
            Translation3::new(0.05, -0.03, 0.10),
            UQ::from_axis_angle(&Vector3::y_axis(), 0.25_f64),
        );
        let rig_se3_target = Isometry3::from_parts(
            Translation3::new(0.02, 0.05, 0.62),
            UQ::from_axis_angle(&Vector3::z_axis(), 0.35_f64)
                * UQ::from_axis_angle(&Vector3::x_axis(), -0.2_f64),
        );
        let single = [
            intr.clone(),
            dist.clone(),
            sensor.clone(),
            pack(&(cam_se3_rig * rig_se3_target)),
            plane_normal.clone(),
            plane_distance.clone(),
        ];
        let rig = [
            intr,
            dist,
            sensor,
            pack(&cam_se3_rig),
            pack(&rig_se3_target),
            plane_normal,
            plane_distance,
        ];
        let p2p_single =
            laser_point_to_plane_residual(SPEC, &LaserChain::SinglePose, &single, laser_pixel, w);
        let p2p_rig = laser_point_to_plane_residual(SPEC, &LaserChain::Rig, &rig, laser_pixel, w);
        assert!(p2p_single[0].abs() > 1e-6, "a non-trivial residual");
        assert!((p2p_single[0] - p2p_rig[0]).abs() < 1e-12);
        let line_single =
            laser_line_distance_residual(SPEC, &LaserChain::SinglePose, &single, laser_pixel, w);
        let line_rig = laser_line_distance_residual(SPEC, &LaserChain::Rig, &rig, laser_pixel, w);
        assert!((line_single[0] - line_rig[0]).abs() < 1e-9 * line_single[0].abs().max(1.0));
    }

    /// The robot-delta chain must equal the no-delta chain evaluated at the
    /// explicitly corrected robot pose (`T_B_G * exp(delta)`).
    #[test]
    fn rig_handeye_robot_delta_laser_residual_matches_corrected_robot_pose() {
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

        let (delta_q, delta_t) = se3_exp(delta.as_view());
        let corrected_robot =
            base_se3_gripper * Isometry3::from_parts(Translation3::from(delta_t), delta_q);

        let mode = HandeyeMode::EyeToHand;
        let chain_delta = LaserChain::RigHandeyeRobotDelta {
            base_se3_gripper: pack_arr(&base_se3_gripper),
            mode,
        };
        let chain_corrected = LaserChain::RigHandeye {
            base_se3_gripper: pack_arr(&corrected_robot),
            mode,
        };
        let base_params = [
            intr.clone(),
            dist.clone(),
            sensor.clone(),
            pack(&t_rig_from_cam),
            pack(&rig_se3_base),
            pack(&gripper_se3_target),
            plane_normal.clone(),
            plane_distance.clone(),
        ];
        let mut delta_params = base_params.to_vec();
        delta_params.push(delta.clone());

        let p2p_delta =
            laser_point_to_plane_residual(SPEC, &chain_delta, &delta_params, laser_pixel, w);
        let p2p_corrected =
            laser_point_to_plane_residual(SPEC, &chain_corrected, &base_params, laser_pixel, w);
        assert!((p2p_delta[0] - p2p_corrected[0]).abs() < 1e-10);

        let line_delta =
            laser_line_distance_residual(SPEC, &chain_delta, &delta_params, laser_pixel, w);
        let line_corrected =
            laser_line_distance_residual(SPEC, &chain_corrected, &base_params, laser_pixel, w);
        assert!((line_delta[0] - line_corrected[0]).abs() < 1e-10);
    }

    #[test]
    fn laser_line_distance_handles_tilted_target_past_45_deg_with_vertical_plane() {
        // Vertical laser plane (n = [1, 0, 0], d = 0) and a target tilted 60 deg about X
        // make the intersection line mostly vertical along the camera Z/Y plane where the old
        // Z=0 intercept solver picked p0.z = 0 and rejected valid visible pixels.
        let intr = DVector::from_vec(vec![800.0, 800.0, 640.0, 360.0]);
        let dist = DVector::from_vec(vec![0.0, 0.0, 0.0, 0.0, 0.0]);
        let sensor = DVector::from_vec(vec![0.0, 0.0]);
        let cam = CameraBlocks::new(
            SPEC,
            intr.as_slice(),
            &[],
            dist.as_slice(),
            sensor.as_slice(),
        );
        let pose_iso = Isometry3::from_parts(
            Translation3::new(0.0, 0.0, 0.6),
            UQ::from_axis_angle(&Vector3::x_axis(), 60.0_f64.to_radians()),
        );
        let pose = pack(&pose_iso);
        let plane_normal = DVector::from_vec(vec![1.0, 0.0, 0.0]);
        let plane_distance = DVector::from_vec(vec![0.0]);

        // Intersection line lies in x = 0 in camera frame -> projects to u = cx = 640.0.
        let on_line = laser_line_distance(
            &cam,
            pose.as_view(),
            plane_normal.as_view(),
            plane_distance.as_view(),
            [640.0, 420.0],
        )
        .expect("intersection line is visible and well-defined");
        assert!(on_line.abs() < 1e-12, "got {on_line}");

        let off_line = laser_line_distance(
            &cam,
            pose.as_view(),
            plane_normal.as_view(),
            plane_distance.as_view(),
            [643.5, 420.0],
        )
        .expect("intersection line is visible");
        assert!((off_line.abs() - 3.5).abs() < 1e-9, "got {off_line}");
    }

    #[test]
    fn laser_line_distance_is_smooth_across_zero_for_dual_numbers() {
        use num_dual::Dual64;
        // Smoothness across zero is required for exact first derivatives and second_order
        // curvature (previous .abs() had a non-differentiable kink at distance == 0).
        let intr: Vec<Dual64> = [800.0, 800.0, 640.0, 360.0]
            .into_iter()
            .map(Dual64::from)
            .collect();
        let dist = DVector::from_element(5, Dual64::from(0.0));
        let sensor = DVector::from_element(2, Dual64::from(0.0));
        let cam = CameraBlocks::new(
            SPEC,
            intr.as_slice(),
            &[],
            dist.as_slice(),
            sensor.as_slice(),
        );
        let pose_iso = Isometry3::from_parts(
            Translation3::new(0.0, 0.0, 0.6),
            UQ::from_axis_angle(&Vector3::x_axis(), 0.4),
        );
        let pose_f = pack(&pose_iso);
        let pose_d = DVector::from_iterator(7, pose_f.iter().copied().map(Dual64::from));
        let pn = DVector::from_vec(vec![
            Dual64::from(1.0),
            Dual64::from(0.0),
            Dual64::from(0.0),
        ]);
        // Perturb plane_d about 0 so the exact pixel [640.0, 360.0] has residual == 0.
        let pd = DVector::from_vec(vec![Dual64::new(0.0, 1.0)]);

        let res_plus = laser_line_dist_core(
            &cam,
            pose_d.as_view(),
            pn.as_view(),
            pd.as_view(),
            [640.0, 360.0],
            1.0,
        );
        let pd_neg = DVector::from_vec(vec![Dual64::new(-1e-7, 1.0)]);
        let res_minus = laser_line_dist_core(
            &cam,
            pose_d.as_view(),
            pn.as_view(),
            pd_neg.as_view(),
            [640.0, 360.0],
            1.0,
        );
        assert!(res_plus[0].re.abs() < 1e-12);
        assert!(res_plus[0].eps.is_finite() && res_plus[0].eps.abs() > 1e2);
        assert!(
            (res_plus[0].eps - res_minus[0].eps).abs() < 1e-3,
            "derivative jumped across 0: {} vs {}",
            res_plus[0].eps,
            res_minus[0].eps
        );
    }
}
