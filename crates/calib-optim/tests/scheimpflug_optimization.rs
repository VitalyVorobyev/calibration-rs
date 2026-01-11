//! Integration test demonstrating Scheimpflug sensor parameter optimization.
//!
//! This test validates that:
//! 1. The Scheimpflug factor compiles correctly in the IR
//! 2. The tiny-solver backend can optimize Scheimpflug parameters
//! 3. Optimization converges to ground truth values for synthetic data

use calib_core::{BrownConrady5, Camera, FxFyCxCySkew, Pinhole, Pt3, Real, ScheimpflugParams};
use calib_optim::backend::{BackendKind, BackendSolveOptions, LinearSolverKind};
use calib_optim::ir::{FactorKind, FixedMask, ManifoldKind, ProblemIR, ResidualBlock, RobustLoss};
use calib_optim::params::distortion::{pack_distortion, DISTORTION_DIM};
use calib_optim::params::intrinsics::{pack_intrinsics, INTRINSICS_DIM};
use calib_optim::params::pose_se3::iso3_to_se3_dvec;
use nalgebra::{DVector, Isometry3, Rotation3, Translation3};
use std::collections::HashMap;

#[test]
fn scheimpflug_optimization_synthetic() {
    // Ground truth camera parameters
    let intrinsics_gt = FxFyCxCySkew {
        fx: 800.0,
        fy: 780.0,
        cx: 640.0,
        cy: 360.0,
        skew: 0.0,
    };

    let distortion_gt = BrownConrady5 {
        k1: -0.3,
        k2: 0.1,
        k3: 0.0,
        p1: 0.001,
        p2: -0.001,
        iters: 5,
    };

    let sensor_params_gt = ScheimpflugParams {
        tilt_x: 0.02,
        tilt_y: -0.01,
    };
    let sensor_gt = sensor_params_gt.compile();

    let camera_gt = Camera::new(Pinhole, distortion_gt, sensor_gt, intrinsics_gt);

    // Ground truth poses
    let poses_gt = vec![
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
    for pose in &poses_gt {
        let mut points_3d = Vec::new();
        let mut points_2d = Vec::new();

        // Planar calibration target at z=0 in board frame
        for y in -2..=2 {
            for x in -3..=3 {
                let pw = Pt3::new(x as Real * 0.05, y as Real * 0.05, 0.0);
                let pc = pose.transform_point(&pw);
                if let Some(pixel) = camera_gt.project_point(&pc) {
                    points_3d.push(pw);
                    points_2d.push(pixel);
                }
            }
        }

        views.push((points_3d, points_2d));
    }

    // Initial values (perturbed from ground truth)
    let intrinsics_init = FxFyCxCySkew {
        fx: 810.0, // Perturbed
        fy: 790.0, // Perturbed
        cx: 645.0, // Perturbed
        cy: 365.0, // Perturbed
        skew: 0.0,
    };

    let distortion_init = BrownConrady5 {
        k1: -0.25, // Perturbed
        k2: 0.08,  // Perturbed
        k3: 0.0,
        p1: 0.0, // Perturbed
        p2: 0.0, // Perturbed
        iters: 8,
    };

    let sensor_init = [0.015, -0.008]; // Perturbed from [0.02, -0.01]

    let poses_init = poses_gt
        .iter()
        .map(|iso| {
            // Slightly perturb poses
            let t = iso.translation.vector + nalgebra::Vector3::new(0.005, -0.003, 0.01);
            let r = iso.rotation.to_rotation_matrix()
                * Rotation3::from_euler_angles(0.01, -0.008, 0.005);
            Isometry3::from_parts(Translation3::from(t), r.into())
        })
        .collect::<Vec<_>>();

    // Build optimization problem IR
    let mut ir = ProblemIR::new();
    let mut initial_map: HashMap<String, DVector<f64>> = HashMap::new();

    // Add parameter blocks
    let cam_id = ir.add_param_block(
        "cam",
        INTRINSICS_DIM,
        ManifoldKind::Euclidean,
        FixedMask::all_free(),
        None,
    );
    initial_map.insert(
        "cam".to_string(),
        pack_intrinsics(&intrinsics_init).unwrap(),
    );

    let dist_id = ir.add_param_block(
        "dist",
        DISTORTION_DIM,
        ManifoldKind::Euclidean,
        FixedMask::fix_indices(&[2]), // Fix k3
        None,
    );
    initial_map.insert("dist".to_string(), pack_distortion(&distortion_init));

    let sensor_id = ir.add_param_block(
        "sensor",
        2,
        ManifoldKind::Euclidean,
        FixedMask::all_free(),
        None,
    );
    initial_map.insert("sensor".to_string(), DVector::from_row_slice(&sensor_init));

    // Add poses and residuals
    for (view_idx, (points_3d, points_2d)) in views.iter().enumerate() {
        let pose_key = format!("pose/{}", view_idx);
        let pose_id =
            ir.add_param_block(&pose_key, 7, ManifoldKind::SE3, FixedMask::all_free(), None);
        initial_map.insert(pose_key, iso3_to_se3_dvec(&poses_init[view_idx]));

        for (pw, uv) in points_3d.iter().zip(points_2d.iter()) {
            let factor = FactorKind::ReprojPointPinhole4Dist5Scheimpflug2 {
                pw: [pw.x, pw.y, pw.z],
                uv: [uv.x, uv.y],
                w: 1.0,
            };
            let residual = ResidualBlock {
                params: vec![cam_id, dist_id, sensor_id, pose_id],
                loss: RobustLoss::None,
                factor,
                residual_dim: 2,
            };
            ir.add_residual_block(residual);
        }
    }

    // Validate IR
    ir.validate().expect("IR validation failed");

    // Solve
    let backend_opts = BackendSolveOptions {
        max_iters: 50,
        verbosity: 0,
        linear_solver: Some(LinearSolverKind::SparseCholesky),
        min_abs_decrease: Some(1e-10),
        min_rel_decrease: Some(1e-10),
        min_error: Some(1e-12),
    };

    let solution = calib_optim::backend::solve_with_backend(
        BackendKind::TinySolver,
        &ir,
        &initial_map,
        &backend_opts,
    )
    .expect("optimization failed");

    // Extract results
    let cam_final = solution.params.get("cam").unwrap();
    let dist_final = solution.params.get("dist").unwrap();
    let sensor_final = solution.params.get("sensor").unwrap();

    // Verify convergence to ground truth
    let fx_error = (cam_final[0] - intrinsics_gt.fx).abs();
    let fy_error = (cam_final[1] - intrinsics_gt.fy).abs();
    let cx_error = (cam_final[2] - intrinsics_gt.cx).abs();
    let cy_error = (cam_final[3] - intrinsics_gt.cy).abs();

    println!(
        "Intrinsics errors: fx={:.2e}, fy={:.2e}, cx={:.2e}, cy={:.2e}",
        fx_error, fy_error, cx_error, cy_error
    );

    assert!(fx_error < 0.5, "fx error too large: {}", fx_error);
    assert!(fy_error < 0.5, "fy error too large: {}", fy_error);
    assert!(cx_error < 0.5, "cx error too large: {}", cx_error);
    assert!(cy_error < 0.5, "cy error too large: {}", cy_error);

    let k1_error = (dist_final[0] - distortion_gt.k1).abs();
    let k2_error = (dist_final[1] - distortion_gt.k2).abs();
    let p1_error = (dist_final[3] - distortion_gt.p1).abs();
    let p2_error = (dist_final[4] - distortion_gt.p2).abs();

    println!(
        "Distortion errors: k1={:.2e}, k2={:.2e}, p1={:.2e}, p2={:.2e}",
        k1_error, k2_error, p1_error, p2_error
    );

    assert!(k1_error < 0.01, "k1 error too large: {}", k1_error);
    assert!(k2_error < 0.01, "k2 error too large: {}", k2_error);
    assert!(p1_error < 0.001, "p1 error too large: {}", p1_error);
    assert!(p2_error < 0.001, "p2 error too large: {}", p2_error);

    let tilt_x_error = (sensor_final[0] - sensor_params_gt.tilt_x).abs();
    let tilt_y_error = (sensor_final[1] - sensor_params_gt.tilt_y).abs();

    println!(
        "Scheimpflug errors: tilt_x={:.2e}, tilt_y={:.2e}",
        tilt_x_error, tilt_y_error
    );

    assert!(
        tilt_x_error < 0.001,
        "tilt_x error too large: {}",
        tilt_x_error
    );
    assert!(
        tilt_y_error < 0.001,
        "tilt_y error too large: {}",
        tilt_y_error
    );

    println!("✓ Scheimpflug optimization converged to ground truth");
    println!("  Final cost: {:.6e}", solution.final_cost);
}
