//! Integration test for rig extrinsics optimization.
//!
//! This test validates:
//! 1. Multi-camera rig extrinsics optimization compiles and runs
//! 2. Convergence to ground truth for synthetic data
//! 3. Shared intrinsics and distortion across cameras
//! 4. Per-camera extrinsics parameters

use calib_core::{BrownConrady5, Camera, FxFyCxCySkew, IdentitySensor, Pinhole, Pt3, Real};
use calib_optim::backend::{BackendSolveOptions, LinearSolverKind};
use calib_optim::params::distortion::BrownConrady5Params;
use calib_optim::params::intrinsics::Intrinsics4;
use calib_optim::problems::rig_extrinsics::*;
use nalgebra::{Isometry3, Rotation3, Translation3};

#[test]
fn stereo_rig_extrinsics_converges() {
    // Ground truth camera parameters (shared across cameras)
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
    // Camera 0 is at rig origin (identity)
    let cam0_to_rig_gt = Isometry3::identity();

    // Camera 1 is 0.12m to the right with slight rotation
    let cam1_to_rig_gt = Isometry3::from_parts(
        Translation3::new(0.12, 0.0, 0.0),
        Rotation3::from_euler_angles(0.0, 0.0, 0.0).into(),
    );

    let cam_to_rig_gt = vec![cam0_to_rig_gt, cam1_to_rig_gt];

    // Ground truth rig poses (3 views)
    let rig_to_target_gt = vec![
        Isometry3::from_parts(
            Translation3::new(0.1, -0.05, 1.0),
            Rotation3::from_euler_angles(0.1, -0.05, 0.2).into(),
        ),
        Isometry3::from_parts(
            Translation3::new(-0.08, 0.03, 1.1),
            Rotation3::from_euler_angles(-0.08, 0.06, -0.15).into(),
        ),
        Isometry3::from_parts(
            Translation3::new(0.05, 0.08, 0.95),
            Rotation3::from_euler_angles(0.05, -0.08, 0.12).into(),
        ),
    ];

    // Generate synthetic observations
    let mut views = Vec::new();
    for rig_pose in &rig_to_target_gt {
        let mut cameras_obs = Vec::new();

        for cam_to_rig in &cam_to_rig_gt {
            let mut points_3d = Vec::new();
            let mut points_2d = Vec::new();

            // Planar calibration target at z=0 in target frame
            for y in -2..=2 {
                for x in -3..=3 {
                    let pw = Pt3::new(x as Real * 0.05, y as Real * 0.05, 0.0);

                    // Transform: target -> rig -> camera
                    let p_rig = rig_pose.transform_point(&pw);
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
        }

        views.push(RigViewObservations {
            cameras: cameras_obs,
        });
    }

    let dataset = RigExtrinsicsDataset::new(views, 2).unwrap();

    // Initial values (perturbed from ground truth)
    let intrinsics_init = Intrinsics4 {
        fx: 810.0, // Perturbed
        fy: 790.0, // Perturbed
        cx: 645.0, // Perturbed
        cy: 365.0, // Perturbed
    };

    let distortion_init = BrownConrady5Params {
        k1: -0.22, // Perturbed
        k2: 0.06,  // Perturbed
        k3: 0.0,
        p1: 0.0, // Perturbed
        p2: 0.0, // Perturbed
    };

    // Camera 0 extrinsics: use ground truth since it will be fixed (gauge freedom)
    let cam0_to_rig_init = cam0_to_rig_gt;

    // Perturb camera 1 extrinsics
    let cam1_to_rig_init = Isometry3::from_parts(
        Translation3::new(0.115, 0.003, -0.002),
        Rotation3::from_euler_angles(0.008, 0.005, -0.01).into(),
    );

    // Perturb rig poses
    let rig_to_target_init = rig_to_target_gt
        .iter()
        .map(|iso| {
            let t = iso.translation.vector + nalgebra::Vector3::new(0.008, -0.005, 0.01);
            let r = iso.rotation.to_rotation_matrix()
                * Rotation3::from_euler_angles(0.01, -0.008, 0.005);
            Isometry3::from_parts(Translation3::from(t), r.into())
        })
        .collect::<Vec<_>>();

    let initial = RigExtrinsicsInit {
        intrinsics: intrinsics_init,
        distortion: distortion_init,
        cam_to_rig: vec![cam0_to_rig_init, cam1_to_rig_init],
        rig_to_target: rig_to_target_init,
    };

    // Solve options - fix camera 0 extrinsics to remove gauge freedom
    let mut opts = RigExtrinsicsSolveOptions::default();
    opts.fix_extrinsics = vec![true, false]; // Fix cam0, optimize cam1
    opts.fix_rig_poses = vec![]; // Optimize all rig poses

    let backend_opts = BackendSolveOptions {
        max_iters: 50,
        verbosity: 0,
        linear_solver: Some(LinearSolverKind::SparseCholesky),
        min_abs_decrease: Some(1e-10),
        min_rel_decrease: Some(1e-10),
        min_error: Some(1e-12),
    };

    // Optimize
    let result = optimize_rig_extrinsics(dataset, initial, opts, backend_opts).unwrap();

    // Verify intrinsics convergence
    let intr_final = result.camera.k;
    let fx_error = (intr_final.fx - intrinsics_gt.fx).abs();
    let fy_error = (intr_final.fy - intrinsics_gt.fy).abs();
    let cx_error = (intr_final.cx - intrinsics_gt.cx).abs();
    let cy_error = (intr_final.cy - intrinsics_gt.cy).abs();

    println!(
        "Intrinsics errors: fx={:.2e}, fy={:.2e}, cx={:.2e}, cy={:.2e}",
        fx_error, fy_error, cx_error, cy_error
    );

    assert!(fx_error < 0.5, "fx error too large: {}", fx_error);
    assert!(fy_error < 0.5, "fy error too large: {}", fy_error);
    assert!(cx_error < 0.5, "cx error too large: {}", cx_error);
    assert!(cy_error < 0.5, "cy error too large: {}", cy_error);

    // Verify distortion convergence
    let dist_final = result.camera.dist;
    let k1_error = (dist_final.k1 - distortion_gt.k1).abs();
    let k2_error = (dist_final.k2 - distortion_gt.k2).abs();
    let p1_error = (dist_final.p1 - distortion_gt.p1).abs();
    let p2_error = (dist_final.p2 - distortion_gt.p2).abs();

    println!(
        "Distortion errors: k1={:.2e}, k2={:.2e}, p1={:.2e}, p2={:.2e}",
        k1_error, k2_error, p1_error, p2_error
    );

    assert!(k1_error < 0.01, "k1 error too large: {}", k1_error);
    assert!(k2_error < 0.01, "k2 error too large: {}", k2_error);
    assert!(p1_error < 0.001, "p1 error too large: {}", p1_error);
    assert!(p2_error < 0.001, "p2 error too large: {}", p2_error);

    // Verify camera 1 extrinsics (camera 0 is fixed)
    let cam1_extr_final = &result.cam_to_rig[1];
    let cam1_extr_gt = &cam1_to_rig_gt;

    let dt = (cam1_extr_final.translation.vector - cam1_extr_gt.translation.vector).norm();
    let r_final = cam1_extr_final.rotation.to_rotation_matrix();
    let r_gt = cam1_extr_gt.rotation.to_rotation_matrix();
    let r_diff = r_final.transpose() * r_gt;
    let trace = r_diff.matrix().trace();
    let cos_theta = ((trace - 1.0) * 0.5).clamp(-1.0, 1.0);
    let ang = cos_theta.acos();

    println!(
        "Camera 1 extrinsics errors: translation={:.2e}, rotation={:.2e}",
        dt, ang
    );

    assert!(
        dt < 1e-3,
        "camera 1 translation error too large: {}",
        dt
    );
    assert!(
        ang < 1e-3,
        "camera 1 rotation error too large: {}",
        ang
    );

    println!("✓ Stereo rig extrinsics converged to ground truth");
    println!("  Final cost: {:.6e}", result.final_cost);
}
