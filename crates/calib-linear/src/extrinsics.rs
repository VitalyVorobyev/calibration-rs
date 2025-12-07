// crates/calib-linear/src/extrinsics.rs

use calib_core::{Iso3, Real};
use nalgebra::{Quaternion, Translation3, UnitQuaternion, Vector3};

/// Result of multi-camera extrinsics initialization:
/// - cam_to_rig[cam]: transform from camera frame to rig frame
/// - rig_to_target[view]: transform from rig frame to target frame
#[derive(Debug, Clone)]
pub struct ExtrinsicPoses {
    pub cam_to_rig: Vec<Iso3>,
    pub rig_to_target: Vec<Iso3>,
}

/// Linear initialisation of a camera rig from per-camera target poses.
#[derive(Debug, Clone, Copy)]
pub struct MultiCamExtrinsicsInit;

/// Simple SE(3) averaging:
/// - translations are averaged arithmetically
/// - rotations are averaged in quaternion space (with hemisphere correction)
pub fn average_isometries(poses: &[Iso3]) -> Iso3 {
    assert!(!poses.is_empty(), "average_isometries: empty input");

    // 1) Average translation
    let mut t_sum = Vector3::<Real>::zeros();
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
        return Iso3::from_parts(t_avg, UnitQuaternion::identity());
    }

    let acc = acc / (poses.len() as Real);
    // `UnitQuaternion` stores its quaternion coordinates as (i, j, k, w),
    // which matches the layout of `Quaternion::from_vector`.
    let q = Quaternion::from_vector(acc).normalize();
    let r_avg = UnitQuaternion::from_quaternion(q);

    Iso3::from_parts(t_avg, r_avg)
}

/// cam_se3_target[view][cam] = Some(T_CT) where T_CT: camera -> target pose
///
/// ref_cam_idx: index of the camera whose frame will define the rig frame.
///              For it we enforce cam_to_rig[ref_cam_idx] = Identity.
///
/// Returns ExtrinsicPoses { cam_to_rig, rig_to_target }.
///
/// Panics if there is not enough overlap (no views where both cameras
/// see the target, or view with no cameras).
pub fn estimate_extrinsics_from_cam_target_poses(
    cam_se3_target: &[Vec<Option<Iso3>>],
    ref_cam_idx: usize,
) -> ExtrinsicPoses {
    MultiCamExtrinsicsInit::from_cam_target_poses(cam_se3_target, ref_cam_idx)
}

impl MultiCamExtrinsicsInit {
    /// Estimate rig and camera poses from per-camera target observations.
    pub fn from_cam_target_poses(
        cam_se3_target: &[Vec<Option<Iso3>>],
        ref_cam_idx: usize,
    ) -> ExtrinsicPoses {
    let num_views = cam_se3_target.len();
    assert!(num_views > 0, "need at least one view");

    let num_cameras = cam_se3_target[0].len();
    assert!(ref_cam_idx < num_cameras, "invalid ref_cam_idx");

    for (v_idx, view) in cam_se3_target.iter().enumerate() {
        assert_eq!(
            view.len(),
            num_cameras,
            "view {} has different camera count",
            v_idx
        );
    }

    // 1) Estimate cam_to_rig (camera -> rig), with rig = ref camera frame
    let mut cam_to_rig: Vec<Iso3> = Vec::with_capacity(num_cameras);

    for cam_idx in 0..num_cameras {
        if cam_idx == ref_cam_idx {
            cam_to_rig.push(Iso3::identity());
            continue;
        }

        let mut candidates: Vec<Iso3> = Vec::new();

        for view in cam_se3_target {
            if let (Some(ct_cam), Some(ct_ref)) = (&view[cam_idx], &view[ref_cam_idx]) {
                // X = C_i->T * (C_ref->T)^(-1)
                let x = ct_cam * ct_ref.inverse();
                candidates.push(x);
            }
        }

        assert!(
            !candidates.is_empty(),
            "no overlapping views between camera {} and reference {}",
            cam_idx,
            ref_cam_idx
        );

        let avg = average_isometries(&candidates);
        cam_to_rig.push(avg);
    }

    // 2) Estimate rig_to_target for each view by averaging over cameras
    let mut rig_to_target: Vec<Iso3> = Vec::with_capacity(num_views);

    for (v_idx, view) in cam_se3_target.iter().enumerate() {
        let mut candidates: Vec<Iso3> = Vec::new();

        for (cam_idx, opt_ct) in view.iter().enumerate() {
            if let Some(ct) = opt_ct {
                // rig->target = (cam->rig)^(-1) * (cam->target)
                let rt = cam_to_rig[cam_idx].inverse() * ct;
                candidates.push(rt);
            }
        }

        assert!(
            !candidates.is_empty(),
            "view {} has no valid camera poses",
            v_idx
        );

        let avg = average_isometries(&candidates);
        rig_to_target.push(avg);
    }

            ExtrinsicPoses {
                cam_to_rig,
                rig_to_target,
            }
        }
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

        // camera -> rig
        let cam0_to_rig_gt = Iso3::identity();
        let cam1_to_rig_gt = make_iso((0.1, -0.05, 0.2), (0.2, -0.1, 0.0));

        // rig -> target per view
        let rig_to_target_gt: Vec<Iso3> = vec![
            make_iso((0.2, 0.1, 0.0), (0.0, 0.0, 1.0)),
            make_iso((-0.1, 0.0, 0.15), (0.1, -0.05, 1.2)),
            make_iso((0.05, -0.2, 0.1), (-0.2, 0.05, 1.1)),
            make_iso((0.0, 0.1, -0.1), (0.05, 0.1, 0.9)),
        ];

        // --- Build cam->target poses: C->T = (C->R) * (R->T) ---

        let mut cam_se3_target: Vec<Vec<Option<Iso3>>> = vec![vec![None; num_cams]; num_views];

        for (v_idx, rt) in rig_to_target_gt.iter().enumerate() {
            let ct0 = cam0_to_rig_gt * rt;
            let ct1 = cam1_to_rig_gt * rt;
            cam_se3_target[v_idx][0] = Some(ct0);
            cam_se3_target[v_idx][1] = Some(ct1);
        }

        // --- Run extrinsics estimation ---

        let est = estimate_extrinsics_from_cam_target_poses(&cam_se3_target, 0);

        assert_eq!(est.cam_to_rig.len(), num_cams);
        assert_eq!(est.rig_to_target.len(), num_views);

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

        // rig->target per view
        for (v, item) in rig_to_target_gt.iter().enumerate().take(num_views) {
            let (dt, ang) = pose_error(&est.rig_to_target[v], item);
            assert!(dt < 1e-10, "view {} translation error {}", v, dt);
            assert!(ang < 1e-10, "view {} rotation error {}", v, ang);
        }
    }
}
