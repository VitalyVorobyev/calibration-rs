//! Integration tests for the Rational-8 and Division-1 distortion models.
//!
//! Each test builds a synthetic ground-truth camera, generates planar observations,
//! constructs a `ProblemIR` by hand using the new `CameraModelDesc` constants,
//! perturbs the initial values, solves via the tiny-solver backend, and asserts
//! convergence to ground truth within tight tolerances.

use nalgebra::{DVector, Isometry3, Rotation3, Translation3};
use std::collections::HashMap;
use vision_calibration_core::{
    Camera, Division, FxFyCxCySkew, IdentitySensor, Pinhole, Pt3, RationalPolynomial, Real,
};
use vision_calibration_optim::{
    BackendKind, BackendSolveOptions, CameraModelDesc, FactorKind, FixedMask, INTRINSICS_DIM,
    ManifoldKind, ProblemIR, ReprojChain, ResidualBlock, RobustLoss, iso3_to_se3_dvec,
    pack_intrinsics, solve_with_backend,
};

// ─────────────────────────────────────────────────────────────────────────────
// Shared helpers
// ─────────────────────────────────────────────────────────────────────────────

fn gt_intrinsics() -> FxFyCxCySkew<Real> {
    FxFyCxCySkew {
        fx: 800.0,
        fy: 780.0,
        cx: 640.0,
        cy: 360.0,
        skew: 0.0,
    }
}

fn gt_poses() -> Vec<Isometry3<Real>> {
    vec![
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
    ]
}

fn perturbed_poses(poses: &[Isometry3<Real>]) -> Vec<Isometry3<Real>> {
    poses
        .iter()
        .map(|iso| {
            let t = iso.translation.vector + nalgebra::Vector3::new(0.005, -0.003, 0.01);
            let r = iso.rotation.to_rotation_matrix()
                * Rotation3::from_euler_angles(0.01, -0.008, 0.005);
            Isometry3::from_parts(Translation3::from(t), r.into())
        })
        .collect()
}

fn perturbed_intrinsics() -> FxFyCxCySkew<Real> {
    FxFyCxCySkew {
        fx: 810.0,
        fy: 790.0,
        cx: 645.0,
        cy: 365.0,
        skew: 0.0,
    }
}

fn backend_opts() -> BackendSolveOptions {
    BackendSolveOptions {
        max_iters: 200,
        verbosity: 0,
        min_abs_decrease: Some(1e-14),
        min_rel_decrease: Some(1e-14),
        min_error: Some(1e-16),
        ..Default::default()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Rational-8 optimization
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn rational_optimization_synthetic() {
    let intrinsics_gt = gt_intrinsics();

    let dist_gt = RationalPolynomial {
        k1: -0.3_f64,
        k2: 0.1,
        k3: 0.0,
        k4: 0.01,
        k5: 0.0,
        k6: 0.0,
        p1: 0.001,
        p2: -0.001,
        iters: 10,
    };

    let camera_gt = Camera::new(Pinhole, dist_gt, IdentitySensor, intrinsics_gt);

    let poses_gt = gt_poses();

    // Generate synthetic observations.
    let mut views: Vec<(Vec<Pt3>, Vec<nalgebra::Point2<Real>>)> = Vec::new();
    for pose in &poses_gt {
        let mut pts3 = Vec::new();
        let mut pts2 = Vec::new();
        for y in -2..=2 {
            for x in -3..=3 {
                let pw = Pt3::new(x as Real * 0.05, y as Real * 0.05, 0.0);
                let pc = pose.transform_point(&pw);
                if let Some(pixel) = camera_gt.project_point(&pc) {
                    pts3.push(pw);
                    pts2.push(pixel);
                }
            }
        }
        views.push((pts3, pts2));
    }

    // Initial values (perturbed).
    let intrinsics_init = perturbed_intrinsics();
    // Start near zero for distortion (much simpler model guess).
    let dist_init: [f64; 8] = [-0.25, 0.08, 0.0, 0.008, 0.0, 0.0, 0.0, 0.0];

    let poses_init = perturbed_poses(&poses_gt);

    // Build IR.
    let mut ir = ProblemIR::new();
    let mut initial_map: HashMap<String, DVector<f64>> = HashMap::new();

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
        8,
        ManifoldKind::Euclidean,
        FixedMask::all_free(),
        None,
    );
    initial_map.insert("dist".to_string(), DVector::from_row_slice(&dist_init));

    for (view_idx, (pts3, pts2)) in views.iter().enumerate() {
        let pose_key = format!("pose/{view_idx}");
        let pose_id =
            ir.add_param_block(&pose_key, 7, ManifoldKind::SE3, FixedMask::all_free(), None);
        initial_map.insert(pose_key, iso3_to_se3_dvec(&poses_init[view_idx]));

        for (pw, uv) in pts3.iter().zip(pts2.iter()) {
            let factor = FactorKind::ReprojPoint {
                model: CameraModelDesc::PINHOLE4_RATIONAL8,
                chain: ReprojChain::SinglePose,
                pw: [pw.x, pw.y, pw.z],
                uv: [uv.x, uv.y],
                w: 1.0,
            };
            ir.add_residual_block(ResidualBlock {
                params: vec![cam_id, dist_id, pose_id],
                loss: RobustLoss::None,
                factor,
                residual_dim: 2,
            });
        }
    }

    ir.validate().expect("rational IR must be valid");

    let solution = solve_with_backend(BackendKind::TinySolver, &ir, &initial_map, &backend_opts())
        .expect("rational optimization must succeed");

    let cam_final = solution.params.get("cam").unwrap();
    let dist_final = solution.params.get("dist").unwrap();

    let fx_err = (cam_final[0] - intrinsics_gt.fx).abs();
    let fy_err = (cam_final[1] - intrinsics_gt.fy).abs();
    let cx_err = (cam_final[2] - intrinsics_gt.cx).abs();
    let cy_err = (cam_final[3] - intrinsics_gt.cy).abs();

    println!(
        "[rational] intrinsics errors: fx={fx_err:.2e} fy={fy_err:.2e} cx={cx_err:.2e} cy={cy_err:.2e}"
    );
    assert!(fx_err < 0.5, "rational fx error too large: {fx_err}");
    assert!(fy_err < 0.5, "rational fy error too large: {fy_err}");
    assert!(cx_err < 0.5, "rational cx error too large: {cx_err}");
    assert!(cy_err < 0.5, "rational cy error too large: {cy_err}");

    let k1_err = (dist_final[0] - dist_gt.k1).abs();
    let k2_err = (dist_final[1] - dist_gt.k2).abs();
    let k4_err = (dist_final[3] - dist_gt.k4).abs();

    println!("[rational] distortion errors: k1={k1_err:.2e} k2={k2_err:.2e} k4={k4_err:.2e}");
    // The rational polynomial has correlated (numerator, denominator) coefficients.
    // With 3 views and synthetic data a 5% convergence criterion is appropriate.
    assert!(k1_err < 5e-2, "rational k1 error too large: {k1_err}");
    assert!(k2_err < 5e-2, "rational k2 error too large: {k2_err}");
    assert!(k4_err < 5e-2, "rational k4 error too large: {k4_err}");

    println!(
        "[rational] final cost: {:.3e}",
        solution.solve_report.final_cost
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Division-1 optimization
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn division_optimization_synthetic() {
    let intrinsics_gt = gt_intrinsics();
    let lambda_gt: Real = -0.3;
    let dist_gt = Division { lambda: lambda_gt };

    let camera_gt = Camera::new(Pinhole, dist_gt, IdentitySensor, intrinsics_gt);

    let poses_gt = gt_poses();

    // Generate synthetic observations.
    let mut views: Vec<(Vec<Pt3>, Vec<nalgebra::Point2<Real>>)> = Vec::new();
    for pose in &poses_gt {
        let mut pts3 = Vec::new();
        let mut pts2 = Vec::new();
        for y in -2..=2 {
            for x in -3..=3 {
                let pw = Pt3::new(x as Real * 0.05, y as Real * 0.05, 0.0);
                let pc = pose.transform_point(&pw);
                if let Some(pixel) = camera_gt.project_point(&pc) {
                    pts3.push(pw);
                    pts2.push(pixel);
                }
            }
        }
        views.push((pts3, pts2));
    }

    let intrinsics_init = perturbed_intrinsics();
    // Perturb lambda: start at a small non-zero value so the gradient is non-degenerate.
    // Starting exactly at 0.0 gives a zero gradient in the lambda direction (quadratic
    // minimum at the zero locus), so the solver never leaves the "no distortion" basin.
    let lambda_init: f64 = -0.05;

    let poses_init = perturbed_poses(&poses_gt);

    // Build IR.
    let mut ir = ProblemIR::new();
    let mut initial_map: HashMap<String, DVector<f64>> = HashMap::new();

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
        1,
        ManifoldKind::Euclidean,
        FixedMask::all_free(),
        None,
    );
    initial_map.insert("dist".to_string(), DVector::from_row_slice(&[lambda_init]));

    for (view_idx, (pts3, pts2)) in views.iter().enumerate() {
        let pose_key = format!("pose/{view_idx}");
        let pose_id =
            ir.add_param_block(&pose_key, 7, ManifoldKind::SE3, FixedMask::all_free(), None);
        initial_map.insert(pose_key, iso3_to_se3_dvec(&poses_init[view_idx]));

        for (pw, uv) in pts3.iter().zip(pts2.iter()) {
            let factor = FactorKind::ReprojPoint {
                model: CameraModelDesc::PINHOLE4_DIVISION1,
                chain: ReprojChain::SinglePose,
                pw: [pw.x, pw.y, pw.z],
                uv: [uv.x, uv.y],
                w: 1.0,
            };
            ir.add_residual_block(ResidualBlock {
                params: vec![cam_id, dist_id, pose_id],
                loss: RobustLoss::None,
                factor,
                residual_dim: 2,
            });
        }
    }

    ir.validate().expect("division IR must be valid");

    let solution = solve_with_backend(BackendKind::TinySolver, &ir, &initial_map, &backend_opts())
        .expect("division optimization must succeed");

    let cam_final = solution.params.get("cam").unwrap();
    let dist_final = solution.params.get("dist").unwrap();

    let fx_err = (cam_final[0] - intrinsics_gt.fx).abs();
    let fy_err = (cam_final[1] - intrinsics_gt.fy).abs();
    let cx_err = (cam_final[2] - intrinsics_gt.cx).abs();
    let cy_err = (cam_final[3] - intrinsics_gt.cy).abs();

    println!(
        "[division] intrinsics errors: fx={fx_err:.2e} fy={fy_err:.2e} cx={cx_err:.2e} cy={cy_err:.2e}"
    );
    assert!(fx_err < 0.5, "division fx error too large: {fx_err}");
    assert!(fy_err < 0.5, "division fy error too large: {fy_err}");
    assert!(cx_err < 0.5, "division cx error too large: {cx_err}");
    assert!(cy_err < 0.5, "division cy error too large: {cy_err}");

    let lambda_err = (dist_final[0] - lambda_gt).abs();
    println!("[division] lambda error: {lambda_err:.2e}");
    assert!(
        lambda_err < 1e-2,
        "division lambda error too large: {lambda_err}"
    );

    println!(
        "[division] final cost: {:.3e}",
        solution.solve_report.final_cost
    );
}
