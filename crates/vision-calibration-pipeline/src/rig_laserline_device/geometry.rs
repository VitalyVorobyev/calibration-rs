//! Geometry helpers for rig-laserline devices.
//!
//! Maps observed laser pixels into 3D points using an upstream rig hand-eye
//! calibration together with per-camera laser planes.

use vision_calibration_core::{Iso3, Pt2, Pt3, Vec3};
use vision_calibration_optim::{HandEyeMode, LaserPlane};

use crate::Error;
use crate::rig_handeye::RigHandeyeExport;

/// Map a laser pixel in a specific camera to a 3D point in the robot gripper frame.
///
/// Given:
/// - `cam_idx`: which camera of the rig captured the pixel.
/// - `pixel`: observed pixel on the laser line.
/// - `rig_cal`: upstream rig hand-eye calibration.
/// - `laser_planes_rig`: laser planes (one per camera) expressed in rig frame.
/// - `base_se3_gripper`: the robot gripper pose at the time the pixel was
///   observed. Required for `EyeToHand` (where the rig is fixed in base and
///   the gripper frame depends on the robot pose); ignored for `EyeInHand`.
///
/// Returns the 3D point in gripper (robot flange) frame:
///
/// 1. Back-project `pixel` to a camera-frame ray through the camera's model
///    (intrinsics, sensor and distortion inverted).
/// 2. Transform the ray into rig frame via `cam_se3_rig[cam_idx].inverse()`.
/// 3. Intersect the ray with `laser_planes_rig[cam_idx]` (in rig frame).
/// 4. Map the rig-frame point into the gripper frame using the hand-eye
///    transform plus, for `EyeToHand`, the provided robot pose.
///
/// # Errors
///
/// Returns [`Error`] if `cam_idx` is out of range, if the camera model cannot
/// be built, if the ray never intersects the plane, or if `base_se3_gripper`
/// is missing in `EyeToHand` mode.
pub fn pixel_to_gripper_point(
    cam_idx: usize,
    pixel: Pt2,
    rig_cal: &RigHandeyeExport,
    laser_planes_rig: &[LaserPlane],
    base_se3_gripper: Option<Iso3>,
) -> Result<Pt3, Error> {
    let n_cams = rig_cal.cameras.len();
    if cam_idx >= n_cams {
        return Err(Error::InvalidInput {
            reason: format!("cam_idx {cam_idx} out of range (num_cameras = {n_cams})"),
        });
    }
    if laser_planes_rig.len() != n_cams {
        return Err(Error::InvalidInput {
            reason: format!(
                "laser_planes_rig has {} entries, expected {n_cams}",
                laser_planes_rig.len()
            ),
        });
    }
    if cam_idx >= rig_cal.cam_se3_rig.len() {
        return Err(Error::InvalidInput {
            reason: "rig calibration missing per-cam data".to_string(),
        });
    }

    // Back-project the pixel through the camera's full model.
    let camera = rig_cal.cameras[cam_idx]
        .build()
        .map_err(|e| Error::invalid_input(format!("camera {cam_idx}: {e}")))?;
    let ray = camera.backproject_pixel(&pixel);

    // Transform ray origin/direction from camera to rig.
    let cam_to_rig = rig_cal.cam_se3_rig[cam_idx].inverse();
    let origin_rig: Vec3 = cam_to_rig.transform_point(&Pt3::from(ray.origin)).coords;
    let dir_rig = cam_to_rig.rotation.transform_vector(&ray.direction);

    // Intersect with laser plane (in rig frame): n · (o + t d) + d_plane = 0.
    let plane = &laser_planes_rig[cam_idx];
    let n = plane.normal.into_inner();
    let denom = n.dot(&dir_rig);
    if denom.abs() < 1e-12 {
        return Err(Error::Numerical(
            "ray is parallel to laser plane; no intersection".to_string(),
        ));
    }
    let t = -(n.dot(&origin_rig) + plane.distance) / denom;
    let p_rig: Vec3 = origin_rig + t * dir_rig;

    // Map rig-frame point into the gripper frame. The chain depends on mode:
    // - EyeInHand: rig is mounted on the gripper, so p_G = T_G_R * p_R where
    //   T_G_R = gripper_se3_rig (the fixed hand-eye transform).
    // - EyeToHand: rig is fixed in base, so p_G depends on the robot pose:
    //   p_G = T_G_B * T_B_R * p_R where T_B_R = rig_se3_base.inverse() and
    //   T_G_B = base_se3_gripper.inverse().
    let p_rig_pt = Pt3::from(p_rig);
    let p_gripper = match rig_cal.handeye_mode {
        HandEyeMode::EyeInHand => rig_cal
            .gripper_se3_rig
            .ok_or_else(|| Error::InvalidInput {
                reason: "EyeInHand export missing gripper_se3_rig".to_string(),
            })?
            .transform_point(&p_rig_pt),
        HandEyeMode::EyeToHand => {
            let rig_se3_base = rig_cal.rig_se3_base.ok_or_else(|| Error::InvalidInput {
                reason: "EyeToHand export missing rig_se3_base".to_string(),
            })?;
            let base_se3_gripper = base_se3_gripper.ok_or_else(|| Error::InvalidInput {
                reason: "EyeToHand mode requires `base_se3_gripper` to map into the \
                         gripper frame; call pixel_to_rig_point for a pose-free result"
                    .to_string(),
            })?;
            // p_base = rig_se3_base.inverse() * p_rig = T_B_R * p_rig
            let p_base = rig_se3_base.inverse().transform_point(&p_rig_pt);
            // p_gripper = base_se3_gripper.inverse() * p_base = T_G_B * p_base
            base_se3_gripper.inverse().transform_point(&p_base)
        }
    };
    Ok(p_gripper)
}
