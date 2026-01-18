//! Integration test for hand-eye calibration.
//!
//! This test validates:
//! 1. Eye-in-hand calibration compiles and runs
//! 2. Convergence to ground truth for synthetic data
//! 3. Per-camera intrinsics and distortion optimization
//! 4. Hand-eye transform estimation

use calib_core::{
    BrownConrady5, Camera, CameraFixMask, CorrespondenceView, FxFyCxCySkew, IdentitySensor,
    Pinhole, Pt3, Real, Vec2,
};
use calib_optim::backend::{
    solve_with_backend, BackendKind, BackendSolveOptions, LinearSolverKind,
};
use calib_optim::ir::HandEyeMode;
use calib_optim::params::pose_se3::se3_dvec_to_iso3;
use calib_optim::problems::handeye::*;
use nalgebra::{Isometry3, Matrix3, Rotation3, Translation3, UnitQuaternion, Vector3};

#[test]
fn eye_in_hand_calibration_converges() {
    // Ground truth camera parameters (homogeneous rig with 2 cameras)
    let intrinsics_gt = FxFyCxCySkew {
        fx: 800.0,
        fy: 780.0,
        cx: 640.0,
        cy: 360.0,
        skew: 0.0,
    };

    let distortion_gt = BrownConrady5 {
        k1: -0.25,
        k2: 0.08,
        k3: 0.0,
        p1: 0.001,
        p2: -0.001,
        iters: 5,
    };

    let camera_gt = Camera::new(Pinhole, distortion_gt, IdentitySensor, intrinsics_gt);

    // Ground truth extrinsics: 2-camera stereo rig
    // Camera 0 at rig origin
    let cam0_to_rig_gt = Isometry3::identity();

    // Camera 1 is 0.12m to the right
    let cam1_to_rig_gt = Isometry3::from_parts(
        Translation3::new(0.12, 0.0, 0.0),
        Rotation3::identity().into(),
    );

    let cam_to_rig_gt = [cam0_to_rig_gt, cam1_to_rig_gt];

    // Ground truth hand-eye transform (rig-to-gripper for eye-in-hand)
    let handeye_gt = Isometry3::from_parts(
        Translation3::new(0.05, 0.03, 0.15),
        Rotation3::from_euler_angles(0.1, -0.05, 0.15).into(),
    );

    // For eye-in-hand: camera moves with robot, target is fixed in base/world frame
    // We want the target to be in front of the camera at ~1m distance

    // Ground truth robot poses (base-to-gripper) - 4 different robot configurations
    // Position the gripper so that camera (through handeye) can see the target
    let robot_poses_gt = [
        Isometry3::from_parts(
            Translation3::new(0.0, 0.0, -1.0), // Move gripper back so camera sees forward
            Rotation3::from_euler_angles(0.0, 0.0, 0.0).into(),
        ),
        Isometry3::from_parts(
            Translation3::new(-0.05, 0.05, -1.05),
            Rotation3::from_euler_angles(0.0, 0.0, 0.1).into(),
        ),
        Isometry3::from_parts(
            Translation3::new(0.05, -0.05, -0.95),
            Rotation3::from_euler_angles(0.0, 0.0, -0.1).into(),
        ),
        Isometry3::from_parts(
            Translation3::new(-0.03, 0.03, -1.02),
            Rotation3::from_euler_angles(0.0, 0.0, 0.05).into(),
        ),
    ];

    // Ground truth calibration target pose (fixed in base/world frame at origin)
    let target_pose_gt = Isometry3::identity();

    // Generate synthetic observations
    let mut views = Vec::new();
    for robot_pose in &robot_poses_gt {
        let mut cameras_obs = Vec::new();

        // Use only camera 0 for simplified test
        let cam_to_rig = &cam_to_rig_gt[0];
        let mut points_3d = Vec::new();
        let mut points_2d = Vec::new();

        // Planar calibration target at z=0 in target frame
        for y in -2..=2 {
            for x in -3..=3 {
                let pw = Pt3::new(x as Real * 0.05, y as Real * 0.05, 0.0);

                // Eye-in-hand transform chain:
                // P_camera = extr^-1 * handeye^-1 * robot^-1 * target * P_world
                let p_base = target_pose_gt.transform_point(&pw);
                let p_gripper = robot_pose.inverse_transform_point(&p_base);
                let p_rig = handeye_gt.inverse_transform_point(&p_gripper);
                let p_cam = cam_to_rig.inverse_transform_point(&p_rig);

                if let Some(pixel) = camera_gt.project_point(&p_cam) {
                    points_3d.push(pw);
                    points_2d.push(pixel);
                }
            }
        }

        cameras_obs.push(Some(CorrespondenceView::new(points_3d, points_2d).unwrap()));

        views.push(RigViewObservations {
            cameras: cameras_obs,
            robot_pose: *robot_pose,
        });
    }

    let dataset = HandEyeDataset::new(views, 1, HandEyeMode::EyeInHand).unwrap();

    // Initial values (perturbed from ground truth)
    // Single camera initialization
    let intrinsics_init = vec![FxFyCxCySkew {
        fx: 810.0, // Perturbed
        fy: 790.0,
        cx: 645.0,
        cy: 365.0,
        skew: 0.0,
    }];

    let distortion_init = vec![BrownConrady5 {
        k1: -0.22, // Perturbed
        k2: 0.06,
        k3: 0.0,
        p1: 0.0,
        p2: 0.0,
        iters: 8,
    }];

    // Perturb camera extrinsics
    let cam0_to_rig_init = cam0_to_rig_gt; // Keep camera 0 at origin

    // Perturb hand-eye transform
    let handeye_init = Isometry3::from_parts(
        Translation3::new(0.048, 0.032, 0.148),
        Rotation3::from_euler_angles(0.098, -0.048, 0.148).into(),
    );

    let target_poses_init = vec![Isometry3::identity()];

    let initial = HandEyeInit {
        intrinsics: intrinsics_init,
        distortion: distortion_init,
        cam_to_rig: vec![cam0_to_rig_init],
        handeye: handeye_init,
        target_poses: target_poses_init,
    };

    // Solve options - fix camera 0 extrinsics for gauge freedom
    let opts = HandEyeSolveOptions {
        fix_extrinsics: vec![true], // Fix cam0 (gauge freedom)
        ..Default::default()
    };

    let backend_opts = BackendSolveOptions {
        max_iters: 50,
        verbosity: 0,
        linear_solver: Some(LinearSolverKind::SparseCholesky),
        min_abs_decrease: Some(1e-10),
        min_rel_decrease: Some(1e-10),
        min_error: Some(1e-12),
    };

    // Optimize
    let result = optimize_handeye(dataset, initial, opts, backend_opts).unwrap();

    // Verify per-camera intrinsics and distortion convergence
    for (cam_idx, camera) in result.cameras.iter().enumerate() {
        let intr_final = camera.k;
        let fx_error = (intr_final.fx - intrinsics_gt.fx).abs();
        let fy_error = (intr_final.fy - intrinsics_gt.fy).abs();
        let cx_error = (intr_final.cx - intrinsics_gt.cx).abs();
        let cy_error = (intr_final.cy - intrinsics_gt.cy).abs();

        println!(
            "Camera {} intrinsics errors: fx={:.2e}, fy={:.2e}, cx={:.2e}, cy={:.2e}",
            cam_idx, fx_error, fy_error, cx_error, cy_error
        );

        assert!(
            fx_error < 0.5,
            "camera {} fx error too large: {}",
            cam_idx,
            fx_error
        );
        assert!(
            fy_error < 0.5,
            "camera {} fy error too large: {}",
            cam_idx,
            fy_error
        );
        assert!(
            cx_error < 0.5,
            "camera {} cx error too large: {}",
            cam_idx,
            cx_error
        );
        assert!(
            cy_error < 0.5,
            "camera {} cy error too large: {}",
            cam_idx,
            cy_error
        );

        // Verify distortion convergence
        let dist_final = camera.dist;
        let k1_error = (dist_final.k1 - distortion_gt.k1).abs();
        let k2_error = (dist_final.k2 - distortion_gt.k2).abs();
        let p1_error = (dist_final.p1 - distortion_gt.p1).abs();
        let p2_error = (dist_final.p2 - distortion_gt.p2).abs();

        println!(
            "Camera {} distortion errors: k1={:.2e}, k2={:.2e}, p1={:.2e}, p2={:.2e}",
            cam_idx, k1_error, k2_error, p1_error, p2_error
        );

        assert!(
            k1_error < 0.01,
            "camera {} k1 error too large: {}",
            cam_idx,
            k1_error
        );
        assert!(
            k2_error < 0.01,
            "camera {} k2 error too large: {}",
            cam_idx,
            k2_error
        );
        assert!(
            p1_error < 0.001,
            "camera {} p1 error too large: {}",
            cam_idx,
            p1_error
        );
        assert!(
            p2_error < 0.001,
            "camera {} p2 error too large: {}",
            cam_idx,
            p2_error
        );
    }

    // Verify hand-eye transform convergence
    let handeye_final = &result.handeye;
    let handeye_dt = (handeye_final.translation.vector - handeye_gt.translation.vector).norm();
    let handeye_r_final = handeye_final.rotation.to_rotation_matrix();
    let handeye_r_gt = handeye_gt.rotation.to_rotation_matrix();
    let handeye_r_diff = handeye_r_final.transpose() * handeye_r_gt;
    let handeye_trace = handeye_r_diff.matrix().trace();
    let handeye_cos_theta = ((handeye_trace - 1.0) * 0.5).clamp(-1.0, 1.0);
    let handeye_ang = handeye_cos_theta.acos();

    println!(
        "Hand-eye transform errors: translation={:.2e}, rotation={:.2e}",
        handeye_dt, handeye_ang
    );

    assert!(
        handeye_dt < 1e-3,
        "hand-eye translation error too large: {}",
        handeye_dt
    );
    assert!(
        handeye_ang < 1e-3,
        "hand-eye rotation error too large: {}",
        handeye_ang
    );

    println!("✓ Eye-in-hand calibration converged to ground truth");
    println!("  Final cost: {:.6e}", result.final_cost);
}

fn rotation_angle(a: &Isometry3<Real>, b: &Isometry3<Real>) -> Real {
    let r_final = a.rotation.to_rotation_matrix();
    let r_gt = b.rotation.to_rotation_matrix();
    let r_diff = r_final.transpose() * r_gt;
    let trace = r_diff.matrix().trace();
    let cos_theta = ((trace - 1.0) * 0.5).clamp(-1.0, 1.0);
    cos_theta.acos()
}

fn skew_matrix(w: &Vector3<Real>) -> Matrix3<Real> {
    Matrix3::new(0.0, -w.z, w.y, w.z, 0.0, -w.x, -w.y, w.x, 0.0)
}

fn se3_exp_isometry(xi: [Real; 6]) -> Isometry3<Real> {
    let w = Vector3::new(xi[0], xi[1], xi[2]);
    let v = Vector3::new(xi[3], xi[4], xi[5]);

    let theta = w.norm();
    let w_hat = skew_matrix(&w);
    let w_hat2 = w_hat * w_hat;

    let (b, c) = if theta <= 1e-9 {
        (0.5, 1.0 / 6.0)
    } else {
        let theta2 = theta * theta;
        let theta3 = theta2 * theta;
        let b = (1.0 - theta.cos()) / theta2;
        let c = (theta - theta.sin()) / theta3;
        (b, c)
    };

    let v_mat = Matrix3::identity() + w_hat * b + w_hat2 * c;
    let t = v_mat * v;
    Isometry3::from_parts(Translation3::from(t), UnitQuaternion::from_scaled_axis(w))
}

fn lcg(seed: &mut u64) -> Real {
    *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    ((*seed >> 32) as u32) as Real / (u32::MAX as Real)
}

fn mean_reproj_error_eye_in_hand(
    views: &[RigViewObservations],
    intrinsics: FxFyCxCySkew<Real>,
    distortion: BrownConrady5<Real>,
    cam_to_rig: &Isometry3<Real>,
    handeye: &Isometry3<Real>,
    target_pose: &Isometry3<Real>,
    robot_deltas: Option<&Vec<Isometry3<Real>>>,
) -> Real {
    let camera = Camera::new(Pinhole, distortion, IdentitySensor, intrinsics);
    let mut total_error = 0.0;
    let mut total_points = 0usize;

    for (view_idx, view) in views.iter().enumerate() {
        let robot_pose = if let Some(deltas) = robot_deltas {
            deltas[view_idx] * view.robot_pose
        } else {
            view.robot_pose
        };

        let obs = view.cameras[0].as_ref().unwrap();
        for (pw, uv) in obs.points_3d.iter().zip(&obs.points_2d) {
            let p_base = target_pose.transform_point(pw);
            let p_gripper = robot_pose.inverse_transform_point(&p_base);
            let p_rig = handeye.inverse_transform_point(&p_gripper);
            let p_cam = cam_to_rig.inverse_transform_point(&p_rig);
            if let Some(pred) = camera.project_point(&p_cam) {
                total_error += (pred - *uv).norm();
                total_points += 1;
            }
        }
    }

    total_error / total_points as Real
}

#[test]
fn eye_in_hand_robot_pose_refinement_improves_handeye() {
    let intrinsics_gt = FxFyCxCySkew {
        fx: 820.0,
        fy: 810.0,
        cx: 640.0,
        cy: 360.0,
        skew: 0.0,
    };
    let distortion_gt = BrownConrady5 {
        k1: 0.0,
        k2: 0.0,
        k3: 0.0,
        p1: 0.0,
        p2: 0.0,
        iters: 8,
    };
    let camera_gt = Camera::new(Pinhole, distortion_gt, IdentitySensor, intrinsics_gt);

    let cam_to_rig_gt = Isometry3::identity();
    let handeye_gt = Isometry3::from_parts(
        Translation3::new(0.04, -0.02, 0.12),
        Rotation3::from_euler_angles(0.05, -0.03, 0.08).into(),
    );
    let target_pose_gt = Isometry3::identity();

    let robot_poses_gt = vec![
        Isometry3::from_parts(
            Translation3::new(0.0, 0.0, -1.0),
            Rotation3::from_euler_angles(0.0, 0.0, 0.0).into(),
        ),
        Isometry3::from_parts(
            Translation3::new(-0.06, 0.03, -1.05),
            Rotation3::from_euler_angles(0.02, -0.04, 0.08).into(),
        ),
        Isometry3::from_parts(
            Translation3::new(0.05, -0.04, -0.95),
            Rotation3::from_euler_angles(-0.03, 0.06, -0.07).into(),
        ),
        Isometry3::from_parts(
            Translation3::new(0.08, 0.02, -1.02),
            Rotation3::from_euler_angles(0.05, 0.02, 0.04).into(),
        ),
        Isometry3::from_parts(
            Translation3::new(-0.04, -0.05, -0.98),
            Rotation3::from_euler_angles(-0.02, 0.05, 0.03).into(),
        ),
    ];

    let rot_amp = 1.2_f64.to_radians();
    let trans_amp = 0.003;
    let mut pose_seed = 23_u64;
    let mut robot_poses_meas = Vec::with_capacity(robot_poses_gt.len());
    for pose in &robot_poses_gt {
        let rot = Vector3::new(
            (lcg(&mut pose_seed) * 2.0 - 1.0) * rot_amp,
            (lcg(&mut pose_seed) * 2.0 - 1.0) * rot_amp,
            (lcg(&mut pose_seed) * 2.0 - 1.0) * rot_amp,
        );
        let trans = Vector3::new(
            (lcg(&mut pose_seed) * 2.0 - 1.0) * trans_amp,
            (lcg(&mut pose_seed) * 2.0 - 1.0) * trans_amp,
            (lcg(&mut pose_seed) * 2.0 - 1.0) * trans_amp,
        );
        let bias = se3_exp_isometry([rot.x, rot.y, rot.z, trans.x, trans.y, trans.z]);
        robot_poses_meas.push(bias * *pose);
    }

    let nx = 6;
    let ny = 5;
    let spacing = 0.04_f64;
    let mut board_points = Vec::new();
    for j in 0..ny {
        for i in 0..nx {
            board_points.push(Pt3::new(i as f64 * spacing, j as f64 * spacing, 0.0));
        }
    }

    let mut views = Vec::new();
    let mut pixel_seed = 7_u64;
    for (robot_pose_gt, robot_pose_meas) in robot_poses_gt.iter().zip(&robot_poses_meas) {
        let mut points_3d = Vec::new();
        let mut points_2d = Vec::new();

        for pw in &board_points {
            let p_base = target_pose_gt.transform_point(pw);
            let p_gripper = robot_pose_gt.inverse_transform_point(&p_base);
            let p_rig = handeye_gt.inverse_transform_point(&p_gripper);
            let p_cam = cam_to_rig_gt.inverse_transform_point(&p_rig);

            if let Some(pixel) = camera_gt.project_point(&p_cam) {
                let noise_u = (lcg(&mut pixel_seed) - 0.5) * 0.4;
                let noise_v = (lcg(&mut pixel_seed) - 0.5) * 0.4;
                points_3d.push(*pw);
                points_2d.push(Vec2::new(pixel.x + noise_u, pixel.y + noise_v));
            }
        }

        let obs = CorrespondenceView::new(points_3d, points_2d).unwrap();
        views.push(RigViewObservations {
            cameras: vec![Some(obs)],
            robot_pose: *robot_pose_meas,
        });
    }

    let dataset = HandEyeDataset::new(views.clone(), 1, HandEyeMode::EyeInHand).unwrap();
    let init = HandEyeInit {
        intrinsics: vec![intrinsics_gt],
        distortion: vec![distortion_gt],
        cam_to_rig: vec![cam_to_rig_gt],
        handeye: Isometry3::from_parts(
            Translation3::new(0.038, -0.021, 0.118),
            Rotation3::from_euler_angles(0.052, -0.028, 0.075).into(),
        ),
        target_poses: vec![target_pose_gt],
    };

    let base_opts = HandEyeSolveOptions {
        default_fix: CameraFixMask::all_fixed(),
        fix_extrinsics: vec![true],
        ..Default::default()
    };

    let backend_opts = BackendSolveOptions {
        max_iters: 60,
        ..Default::default()
    };

    let result_no_refine = optimize_handeye(
        dataset.clone(),
        init.clone(),
        base_opts.clone(),
        backend_opts.clone(),
    )
    .unwrap();

    let dt_no =
        (result_no_refine.handeye.translation.vector - handeye_gt.translation.vector).norm();
    let ang_no = rotation_angle(&result_no_refine.handeye, &handeye_gt);
    let reproj_no = mean_reproj_error_eye_in_hand(
        &views,
        intrinsics_gt,
        distortion_gt,
        &cam_to_rig_gt,
        &result_no_refine.handeye,
        &result_no_refine.target_poses[0],
        None,
    );

    let mut opts_refine = base_opts.clone();
    opts_refine.refine_robot_poses = true;
    opts_refine.robot_rot_sigma = rot_amp;
    opts_refine.robot_trans_sigma = trans_amp;

    let (ir, initial_map) = build_handeye_ir(&dataset, &init, &opts_refine).unwrap();
    let solution =
        solve_with_backend(BackendKind::TinySolver, &ir, &initial_map, &backend_opts).unwrap();

    let handeye_ref = se3_dvec_to_iso3(solution.params.get("handeye").unwrap().as_view()).unwrap();
    let target_ref = se3_dvec_to_iso3(solution.params.get("target").unwrap().as_view()).unwrap();

    let mut deltas = Vec::with_capacity(dataset.num_views());
    let mut max_rot: f64 = 0.0;
    let mut max_trans: f64 = 0.0;
    for i in 0..dataset.num_views() {
        let key = format!("robot_delta/{}", i);
        let delta_vec = solution.params.get(&key).unwrap();
        let xi = [
            delta_vec[0],
            delta_vec[1],
            delta_vec[2],
            delta_vec[3],
            delta_vec[4],
            delta_vec[5],
        ];
        max_rot = max_rot.max(Vector3::new(xi[0], xi[1], xi[2]).norm());
        max_trans = max_trans.max(Vector3::new(xi[3], xi[4], xi[5]).norm());
        deltas.push(se3_exp_isometry(xi));
    }

    let dt_ref = (handeye_ref.translation.vector - handeye_gt.translation.vector).norm();
    let ang_ref = rotation_angle(&handeye_ref, &handeye_gt);
    let reproj_ref = mean_reproj_error_eye_in_hand(
        &views,
        intrinsics_gt,
        distortion_gt,
        &cam_to_rig_gt,
        &handeye_ref,
        &target_ref,
        Some(&deltas),
    );

    let err_no = dt_no + ang_no;
    let err_ref = dt_ref + ang_ref;
    assert!(
        err_ref < err_no,
        "expected refined hand-eye error to drop: {:.4} -> {:.4}",
        err_no,
        err_ref
    );
    assert!(
        reproj_ref < reproj_no,
        "expected reprojection error to drop: {:.4} -> {:.4}",
        reproj_no,
        reproj_ref
    );
    assert!(
        max_rot < opts_refine.robot_rot_sigma * 5.0,
        "robot rotation deltas too large: {:.4} rad",
        max_rot
    );
    assert!(
        max_trans < opts_refine.robot_trans_sigma * 5.0,
        "robot translation deltas too large: {:.4} m",
        max_trans
    );
}
