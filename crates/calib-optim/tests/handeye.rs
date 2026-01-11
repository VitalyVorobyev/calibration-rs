//! Integration test for hand-eye calibration.
//!
//! This test validates:
//! 1. Eye-in-hand calibration compiles and runs
//! 2. Convergence to ground truth for synthetic data
//! 3. Per-camera intrinsics and distortion optimization
//! 4. Hand-eye transform estimation

use calib_core::{BrownConrady5, Camera, FxFyCxCySkew, IdentitySensor, Pinhole, Pt3, Real};
use calib_optim::backend::{BackendSolveOptions, LinearSolverKind};
use calib_optim::ir::HandEyeMode;
use calib_optim::problems::handeye::*;
use nalgebra::{Isometry3, Rotation3, Translation3};

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

    // Ground truth calibration target poses (fixed in base/world frame at origin)
    let target_poses_gt = vec![
        Isometry3::identity(),
        Isometry3::identity(),
        Isometry3::identity(),
        Isometry3::identity(),
    ];

    // Generate synthetic observations
    let mut views = Vec::new();
    for (robot_pose, target_pose) in robot_poses_gt.iter().zip(&target_poses_gt) {
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
                let p_target = target_pose.transform_point(&pw);
                let p_base = robot_pose.inverse_transform_point(&p_target);
                let p_rig = handeye_gt.inverse_transform_point(&p_base);
                let p_cam = cam_to_rig.inverse_transform_point(&p_rig);

                if let Some(pixel) = camera_gt.project_point(&p_cam) {
                    points_3d.push(pw);
                    points_2d.push(pixel);
                }
            }
        }

        cameras_obs.push(Some(
            CameraViewObservations::new(points_3d, points_2d).unwrap(),
        ));

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

    // Perturb target poses (all identity, so just add small perturbations)
    let target_poses_init = vec![
        Isometry3::identity(), // Fix first pose for gauge freedom
        Isometry3::from_parts(
            Translation3::new(0.008, -0.005, 0.002),
            Rotation3::from_euler_angles(0.01, -0.008, 0.005).into(),
        ),
        Isometry3::from_parts(
            Translation3::new(-0.005, 0.008, -0.003),
            Rotation3::from_euler_angles(-0.008, 0.01, -0.005).into(),
        ),
        Isometry3::from_parts(
            Translation3::new(0.005, -0.003, 0.005),
            Rotation3::from_euler_angles(0.005, -0.005, 0.008).into(),
        ),
    ];

    let initial = HandEyeInit {
        intrinsics: intrinsics_init,
        distortion: distortion_init,
        cam_to_rig: vec![cam0_to_rig_init],
        handeye: handeye_init,
        target_poses: target_poses_init,
    };

    // Solve options - fix camera 0 extrinsics and first target pose for gauge freedom
    let opts = HandEyeSolveOptions {
        fix_extrinsics: vec![true], // Fix cam0 (gauge freedom)
        fix_target_poses: vec![0],  // Fix first target pose (gauge freedom)
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
