//! Multi-camera rig extrinsics initialization.
//!
//! Estimates per-camera rig transforms and per-view rig-to-target poses from
//! per-camera target observations.

use crate::Error;
use nalgebra::{Quaternion, Translation3, UnitQuaternion};
use vision_calibration_core::{Iso3, Real, Vec3};

/// Result of multi-camera extrinsics initialization:
/// - `cam_to_rig[cam] = T_R_C`: transform from camera frame to rig frame.
/// - `rig_from_target[view] = T_R_T`: transform from target frame to rig frame.
#[derive(Debug, Clone)]
pub struct ExtrinsicPoses {
    /// Per-camera transform `T_R_C` (camera frame -> rig frame).
    pub cam_to_rig: Vec<Iso3>,
    /// Per-view transform `T_R_T` (target frame -> rig frame).
    pub rig_from_target: Vec<Iso3>,
}

/// Linear initialisation of a camera rig from per-camera target poses.
#[derive(Debug, Clone, Copy)]
pub struct MultiCamExtrinsicsInit;

/// Simple SE(3) averaging:
/// - translations are averaged arithmetically
/// - rotations are averaged in quaternion space (with hemisphere correction)
///
/// Use this only for initialization; it does not preserve full rotation
/// statistics and should be refined downstream.
fn average_isometries(poses: &[Iso3]) -> Result<Iso3, Error> {
    if poses.is_empty() {
        return Err(Error::InsufficientData { need: 1, got: 0 });
    }

    // 1) Average translation
    let mut t_sum = Vec3::zeros();
    for iso in poses {
        t_sum += iso.translation.vector;
    }
    let t_avg = t_sum / (poses.len() as Real);
    let t_avg = Translation3::from(t_avg);

    // 2) Average rotation via quaternions
    let q0 = poses[0].rotation; // reference for hemisphere
    let mut acc = nalgebra::Vector4::<Real>::zeros();

    for iso in poses {
        let q = iso.rotation;
        let coords = q.coords;
        // enforce same hemisphere to avoid cancellation
        let sign = if q0.coords.dot(&coords) < 0.0 {
            -1.0
        } else {
            1.0
        };
        acc += coords * sign;
    }

    if acc.norm_squared() == 0.0 {
        // fallback: identity rotation
        return Ok(Iso3::from_parts(t_avg, UnitQuaternion::identity()));
    }

    let acc = acc / (poses.len() as Real);
    // `UnitQuaternion` stores its quaternion coordinates as (i, j, k, w),
    // which matches the layout of `Quaternion::from_vector`.
    let q = Quaternion::from_vector(acc).normalize();
    let r_avg = UnitQuaternion::from_quaternion(q);

    Ok(Iso3::from_parts(t_avg, r_avg))
}

/// Estimate rig extrinsics from per-view camera-to-target poses.
///
/// `cam_se3_target[view][cam] = Some(T_C_T)` where `T_C_T` maps points from the
/// target frame into the camera frame (i.e., target pose in the camera frame).
///
/// `ref_cam_idx` defines the rig frame by enforcing
/// `cam_to_rig[ref_cam_idx] = Identity` (rig frame = reference camera frame).
///
/// A camera need not share a view with the reference camera: the cameras are
/// chained through their co-visibility graph (two cameras are linked when
/// they see the target in the same view). Each camera is reached from the
/// reference along a path with the fewest links, and among those through
/// the neighbour it shares the most views with (then the lowest index), so
/// a camera that shares views with the reference is always estimated from
/// them directly. Each link's relative pose is the average over the views
/// its two cameras share.
///
/// Returns `ExtrinsicPoses { cam_to_rig, rig_from_target }`.
///
/// # Errors
///
/// [`Error::InvalidInput`] when a camera shares no view with the reference
/// camera, directly or through other cameras (the message names the
/// cameras that cannot be reached), or a view has no valid camera poses;
/// [`Error::InsufficientData`] for no views or no cameras.
pub fn estimate_extrinsics_from_cam_target_poses(
    cam_se3_target: &[Vec<Option<Iso3>>],
    ref_cam_idx: usize,
) -> Result<ExtrinsicPoses, Error> {
    let num_views = cam_se3_target.len();
    if num_views == 0 {
        return Err(Error::InsufficientData { need: 1, got: 0 });
    }

    let num_cameras = cam_se3_target[0].len();
    if num_cameras == 0 {
        return Err(Error::InsufficientData { need: 1, got: 0 });
    }
    if ref_cam_idx >= num_cameras {
        return Err(Error::invalid_input(format!(
            "invalid ref_cam_idx {} for {} cameras",
            ref_cam_idx, num_cameras
        )));
    }

    for (v_idx, view) in cam_se3_target.iter().enumerate() {
        if view.len() != num_cameras {
            return Err(Error::invalid_input(format!(
                "view {} has camera count {}, expected {}",
                v_idx,
                view.len(),
                num_cameras
            )));
        }
    }

    // 1) Estimate cam_to_rig (camera -> rig), with rig = reference camera frame.
    //
    // Two cameras i, j seeing the target in one view give their relative
    // pose T_Ci_Cj = T_Ci_T * (T_Cj_T)^(-1). The cameras are reached from the
    // reference breadth-first through this co-visibility graph, so every
    // camera takes a path with the fewest links; among the cameras of the
    // previous layer, a camera's parent is the one it shares the most views
    // with (ties: the lowest index). Then T_R_Cj = T_R_Ci * T_Ci_Cj.
    let shared = |i: usize, j: usize| {
        cam_se3_target
            .iter()
            .filter(|view| view[i].is_some() && view[j].is_some())
            .count()
    };
    let mut cam_to_rig: Vec<Option<Iso3>> = vec![None; num_cameras];
    cam_to_rig[ref_cam_idx] = Some(Iso3::identity());
    let mut layer = vec![ref_cam_idx];
    while !layer.is_empty() {
        let mut next = Vec::new();
        for cam_idx in 0..num_cameras {
            if cam_to_rig[cam_idx].is_some() {
                continue;
            }
            // The parent in this layer sharing the most views (lowest index on ties).
            let Some((parent, _)) = layer
                .iter()
                .map(|&p| (p, shared(p, cam_idx)))
                .filter(|&(_, n)| n > 0)
                .min_by_key(|&(p, n)| (std::cmp::Reverse(n), p))
            else {
                continue;
            };
            let candidates: Vec<Iso3> = cam_se3_target
                .iter()
                .filter_map(|view| match (&view[parent], &view[cam_idx]) {
                    // T_Cparent_Ccam = T_Cparent_T * (T_Ccam_T)^(-1)
                    (Some(ct_parent), Some(ct_cam)) => Some(ct_parent * ct_cam.inverse()),
                    _ => None,
                })
                .collect();
            let parent_se3_cam = average_isometries(&candidates)?;
            let rig_se3_parent =
                cam_to_rig[parent].expect("a camera of the current layer is placed");
            cam_to_rig[cam_idx] = Some(rig_se3_parent * parent_se3_cam);
            next.push(cam_idx);
        }
        layer = next;
    }
    let unreached: Vec<usize> = (0..num_cameras)
        .filter(|&c| cam_to_rig[c].is_none())
        .collect();
    if !unreached.is_empty() {
        return Err(Error::invalid_input(format!(
            "camera(s) {unreached:?} share no view with reference camera {ref_cam_idx}, \
             directly or through other cameras: a camera's rig pose needs a view in \
             which it and an already placed camera both see the target"
        )));
    }
    let cam_to_rig: Vec<Iso3> = cam_to_rig.into_iter().flatten().collect();

    // 2) Estimate rig_from_target (target -> rig) for each view by averaging over cameras.
    //
    // From T_Ci_T and T_R_Ci:
    //   T_R_T = T_R_Ci * T_Ci_T
    let mut rig_from_target: Vec<Iso3> = Vec::with_capacity(num_views);

    for (v_idx, view) in cam_se3_target.iter().enumerate() {
        let mut candidates: Vec<Iso3> = Vec::new();

        for (cam_idx, opt_ct) in view.iter().enumerate() {
            if let Some(ct) = opt_ct {
                let tr = cam_to_rig[cam_idx] * ct;
                candidates.push(tr);
            }
        }

        if candidates.is_empty() {
            return Err(Error::invalid_input(format!(
                "view {} has no valid camera poses",
                v_idx
            )));
        }

        let avg = average_isometries(&candidates)?;
        rig_from_target.push(avg);
    }

    Ok(ExtrinsicPoses {
        cam_to_rig,
        rig_from_target,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{Isometry3, Rotation3, Translation3};

    fn make_iso(angles: (Real, Real, Real), t: (Real, Real, Real)) -> Iso3 {
        let rot = Rotation3::from_euler_angles(angles.0, angles.1, angles.2);
        let tr = Translation3::new(t.0, t.1, t.2);
        Isometry3::from_parts(tr, rot.into())
    }

    #[test]
    fn extrinsics_from_cam_target_poses_two_cameras() {
        let num_cams = 2;
        let num_views = 4;

        // --- Ground-truth extrinsics ---

        // camera -> rig (T_R_C)
        let cam0_to_rig_gt = Iso3::identity();
        let cam1_to_rig_gt = make_iso((0.1, -0.05, 0.2), (0.2, -0.1, 0.0));

        // target -> rig per view (T_R_T)
        let rig_from_target_gt = [
            make_iso((0.2, 0.1, 0.0), (0.0, 0.0, 1.0)),
            make_iso((-0.1, 0.0, 0.15), (0.1, -0.05, 1.2)),
            make_iso((0.05, -0.2, 0.1), (-0.2, 0.05, 1.1)),
            make_iso((0.0, 0.1, -0.1), (0.05, 0.1, 0.9)),
        ];

        // --- Build cam<-target poses: T_C_T = (T_R_C)^(-1) * T_R_T ---

        let mut cam_se3_target: Vec<Vec<Option<Iso3>>> = vec![vec![None; num_cams]; num_views];

        let cam_to_rig_gt = [cam0_to_rig_gt, cam1_to_rig_gt];
        for (v_idx, rig_from_target) in rig_from_target_gt.iter().enumerate() {
            for cam_idx in 0..num_cams {
                let cam_se3_target_pose = cam_to_rig_gt[cam_idx].inverse() * rig_from_target;
                cam_se3_target[v_idx][cam_idx] = Some(cam_se3_target_pose);
            }
        }

        // --- Run extrinsics estimation ---

        let est = estimate_extrinsics_from_cam_target_poses(&cam_se3_target, 0).unwrap();

        assert_eq!(est.cam_to_rig.len(), num_cams);
        assert_eq!(est.rig_from_target.len(), num_views);

        // Helper: compare two Iso3 with angle + translation norms
        fn pose_error(a: &Iso3, b: &Iso3) -> (Real, Real) {
            let dt = (a.translation.vector - b.translation.vector).norm();

            let r_a = a.rotation.to_rotation_matrix();
            let r_b = b.rotation.to_rotation_matrix();
            let r_diff = r_a.transpose() * r_b;
            let trace = r_diff.matrix().trace();
            let cos_theta = ((trace - 1.0) * 0.5).clamp(-1.0, 1.0);
            let angle = cos_theta.acos();

            (dt, angle)
        }

        // camera 0 should be identity (rig frame)
        let (dt0, ang0) = pose_error(&est.cam_to_rig[0], &cam0_to_rig_gt);
        assert!(dt0 < 1e-10, "cam0 translation error {}", dt0);
        assert!(ang0 < 1e-10, "cam0 rotation error {}", ang0);

        // camera 1 extrinsics
        let (dt1, ang1) = pose_error(&est.cam_to_rig[1], &cam1_to_rig_gt);
        assert!(dt1 < 1e-10, "cam1 translation error {}", dt1);
        assert!(ang1 < 1e-10, "cam1 rotation error {}", ang1);

        // target->rig per view
        for (v, rig_from_target) in rig_from_target_gt.iter().enumerate().take(num_views) {
            let (dt, ang) = pose_error(&est.rig_from_target[v], rig_from_target);
            assert!(dt < 1e-10, "view {} translation error {}", v, dt);
            assert!(ang < 1e-10, "view {} rotation error {}", v, ang);
        }
    }

    /// Four cameras on a ring where each sees the target only in the views it
    /// shares with its two neighbours: camera 2 never shares a view with the
    /// reference and is reached through camera 1 or 3.
    fn ring(views_per_pair: usize) -> (Vec<Iso3>, Vec<Iso3>, Vec<Vec<Option<Iso3>>>) {
        let cam_to_rig_gt: Vec<Iso3> = (0..4)
            .map(|c| {
                let a = c as Real * std::f64::consts::FRAC_PI_2;
                make_iso(
                    (0.0, a, 0.02 * c as Real),
                    (0.3 * a.sin(), 0.01 * c as Real, 0.3 * (1.0 - a.cos())),
                )
            })
            .collect();
        let mut rig_from_target_gt = Vec::new();
        let mut cam_se3_target = Vec::new();
        for pair in 0..4 {
            let (a, b) = (pair, (pair + 1) % 4);
            for k in 0..views_per_pair {
                let t = make_iso(
                    (0.05 * k as Real, 0.1 * pair as Real, -0.03 * k as Real),
                    (
                        0.02 * k as Real,
                        -0.01 * pair as Real,
                        1.0 + 0.05 * k as Real,
                    ),
                );
                let mut view = vec![None; 4];
                for c in [a, b] {
                    view[c] = Some(cam_to_rig_gt[c].inverse() * t);
                }
                rig_from_target_gt.push(t);
                cam_se3_target.push(view);
            }
        }
        (cam_to_rig_gt, rig_from_target_gt, cam_se3_target)
    }

    #[test]
    fn a_ring_where_only_neighbours_overlap_is_chained_through_the_graph() {
        let (cam_to_rig_gt, rig_from_target_gt, cam_se3_target) = ring(3);
        // Camera 2 shares no view with camera 0.
        assert!(
            cam_se3_target
                .iter()
                .all(|v| !(v[0].is_some() && v[2].is_some()))
        );
        let est = estimate_extrinsics_from_cam_target_poses(&cam_se3_target, 0).unwrap();
        for (c, gt) in cam_to_rig_gt.iter().enumerate() {
            let rel = gt.inverse() * est.cam_to_rig[c];
            assert!(rel.translation.vector.norm() < 1e-10, "camera {c}: {rel:?}");
            assert!(rel.rotation.angle() < 1e-10, "camera {c}: {rel:?}");
        }
        for (v, gt) in rig_from_target_gt.iter().enumerate() {
            let rel = gt.inverse() * est.rig_from_target[v];
            assert!(rel.translation.vector.norm() < 1e-10, "view {v}");
            assert!(rel.rotation.angle() < 1e-10, "view {v}");
        }
        // Any camera can be the reference.
        let from_2 = estimate_extrinsics_from_cam_target_poses(&cam_se3_target, 2).unwrap();
        let rel = cam_to_rig_gt[2].inverse() * cam_to_rig_gt[0];
        let got = from_2.cam_to_rig[0];
        assert!((got.translation.vector - rel.translation.vector).norm() < 1e-10);
    }

    #[test]
    fn a_camera_no_path_reaches_is_named() {
        let (_, _, mut cam_se3_target) = ring(2);
        // Cut camera 3 off: drop its observations from every view, then cut
        // camera 2's link to camera 1 so cameras 2 and 3 form their own island.
        for view in &mut cam_se3_target {
            if view[1].is_some() && view[2].is_some() {
                view[2] = None;
            }
            if view[3].is_some() && view[0].is_some() {
                view[3] = None;
            }
        }
        let err = estimate_extrinsics_from_cam_target_poses(&cam_se3_target, 0).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("[2, 3]") && msg.contains("reference camera 0"),
            "{msg}"
        );
    }

    #[test]
    fn extrinsics_allow_missing_ref_camera_in_view() {
        let num_cams = 2;
        let num_views = 3;

        let cam0_to_rig_gt = Iso3::identity();
        let cam1_to_rig_gt = make_iso((0.0, 0.0, 0.2), (0.2, 0.0, 0.0));
        let rig_from_target_gt = [
            make_iso((0.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
            make_iso((0.1, 0.0, 0.0), (0.1, 0.0, 1.1)),
            make_iso((0.0, 0.1, 0.0), (0.0, 0.1, 0.9)),
        ];

        let cam_to_rig_gt = [cam0_to_rig_gt, cam1_to_rig_gt];
        let mut cam_se3_target: Vec<Vec<Option<Iso3>>> = vec![vec![None; num_cams]; num_views];
        for (v_idx, rig_from_target) in rig_from_target_gt.iter().enumerate() {
            for cam_idx in 0..num_cams {
                cam_se3_target[v_idx][cam_idx] =
                    Some(cam_to_rig_gt[cam_idx].inverse() * rig_from_target);
            }
        }

        // Hide reference camera (0) in the middle view. Rig pose should still be estimable.
        cam_se3_target[1][0] = None;

        let est = estimate_extrinsics_from_cam_target_poses(&cam_se3_target, 0).unwrap();
        assert_eq!(est.cam_to_rig.len(), num_cams);
        assert_eq!(est.rig_from_target.len(), num_views);

        // View 1 rig pose still should be close.
        let dt = (est.rig_from_target[1].translation.vector
            - rig_from_target_gt[1].translation.vector)
            .norm();
        assert!(dt < 1e-10, "translation error {}", dt);

        let r_est = est.rig_from_target[1].rotation.to_rotation_matrix();
        let r_gt = rig_from_target_gt[1].rotation.to_rotation_matrix();
        let r_diff = r_est.transpose() * r_gt;
        let trace = r_diff.matrix().trace();
        let cos_theta = ((trace - 1.0) * 0.5).clamp(-1.0, 1.0);
        let angle = cos_theta.acos();
        assert!(angle < 1e-10, "rotation error {}", angle);
    }
}
