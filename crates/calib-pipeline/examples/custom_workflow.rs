//! Example demonstrating custom calibration workflow using helper functions.
//!
//! This example shows:
//! - Using granular helper functions instead of session API
//! - Inspecting intermediate results before committing to optimization
//! - Full control over the calibration pipeline
//! - Custom decision-making based on initialization quality
//!
//! Run with: cargo run --example custom_workflow

use calib_pipeline::helpers::{initialize_planar_intrinsics, optimize_planar_intrinsics_from_init};
use calib_pipeline::iterative_intrinsics::IterativeIntrinsicsOptions;
use calib_pipeline::distortion_fit::DistortionFitOptions;
use calib_pipeline::{BackendSolveOptions, PlanarIntrinsicsSolveOptions, PlanarViewData};
use calib_core::{BrownConrady5, Camera, FxFyCxCySkew, IdentitySensor, Pinhole, Pt3, Vec2};
use nalgebra::{UnitQuaternion, Vector3};

fn main() -> anyhow::Result<()> {
    println!("=== Custom Workflow Example ===\n");

    // Generate synthetic calibration data
    let (views, ground_truth_k, ground_truth_dist) = generate_synthetic_data();
    println!("Generated {} calibration views", views.len());
    println!(
        "Ground truth intrinsics: fx={:.1}, fy={:.1}, cx={:.1}, cy={:.1}",
        ground_truth_k.fx, ground_truth_k.fy, ground_truth_k.cx, ground_truth_k.cy
    );
    println!(
        "Ground truth distortion: k1={:.4}, k2={:.4}\n",
        ground_truth_dist.k1, ground_truth_dist.k2
    );

    // Step 1: Linear initialization
    println!("Step 1: Running linear initialization...");
    let init_opts = IterativeIntrinsicsOptions {
        iterations: 2,
        distortion_opts: DistortionFitOptions {
            fix_k3: true,          // Conservative: only estimate k1, k2
            fix_tangential: false, // Estimate p1, p2
            iters: 8,
        },
    };

    let init_result = initialize_planar_intrinsics(&views, &init_opts)?;
    println!("✓ Linear initialization complete");

    // Step 2: Inspect initialization results
    println!("\n--- Initialization Results ---");
    println!(
        "Initial intrinsics: fx={:.1}, fy={:.1}, cx={:.1}, cy={:.1}, skew={:.4}",
        init_result.intrinsics.fx,
        init_result.intrinsics.fy,
        init_result.intrinsics.cx,
        init_result.intrinsics.cy,
        init_result.intrinsics.skew
    );
    println!(
        "Initial distortion: k1={:.4}, k2={:.4}, k3={:.4}, p1={:.4}, p2={:.4}",
        init_result.distortion.k1,
        init_result.distortion.k2,
        init_result.distortion.k3,
        init_result.distortion.p1,
        init_result.distortion.p2
    );

    // Compute initialization errors
    let fx_error = 100.0 * (init_result.intrinsics.fx - ground_truth_k.fx).abs() / ground_truth_k.fx;
    let fy_error = 100.0 * (init_result.intrinsics.fy - ground_truth_k.fy).abs() / ground_truth_k.fy;
    println!(
        "Initial errors: fx={:.1}%, fy={:.1}%",
        fx_error, fy_error
    );

    // Step 3: Decide whether to proceed with optimization
    if init_result.intrinsics.fx < 100.0 || init_result.intrinsics.fx > 2000.0 {
        eprintln!("\n⚠ Warning: Suspiciously unusual focal length!");
        eprintln!("  This might indicate poor corner detection or incorrect calibration target");
        eprintln!("  Proceeding anyway for demo purposes...");
    }

    if fx_error > 50.0 || fy_error > 50.0 {
        println!("\n⚠ Note: Large initialization error (expected for linear methods)");
        println!("  Non-linear optimization should significantly improve this");
    }

    // Step 4: Non-linear optimization
    println!("\nStep 2: Running non-linear optimization...");

    let solve_opts = PlanarIntrinsicsSolveOptions {
        fix_poses: vec![0], // Fix first pose for gauge freedom
        ..Default::default()
    };

    let backend_opts = BackendSolveOptions {
        max_iters: 50,
        verbosity: 0, // Set to 1 for solver progress
        ..Default::default()
    };

    let optim_result = optimize_planar_intrinsics_from_init(
        &views,
        &init_result,
        &solve_opts,
        &backend_opts,
    )?;
    println!("✓ Optimization complete");

    // Step 5: Analyze final results
    println!("\n=== Final Results ===");
    println!("Final cost: {:.6}", optim_result.final_cost);
    println!("Mean reprojection error: {:.3} pixels", optim_result.mean_reproj_error);

    println!(
        "\nFinal intrinsics: fx={:.1}, fy={:.1}, cx={:.1}, cy={:.1}, skew={:.4}",
        optim_result.intrinsics.fx,
        optim_result.intrinsics.fy,
        optim_result.intrinsics.cx,
        optim_result.intrinsics.cy,
        optim_result.intrinsics.skew
    );
    println!(
        "Final distortion: k1={:.4}, k2={:.4}, k3={:.4}, p1={:.4}, p2={:.4}",
        optim_result.distortion.k1,
        optim_result.distortion.k2,
        optim_result.distortion.k3,
        optim_result.distortion.p1,
        optim_result.distortion.p2
    );

    // Compare with ground truth
    println!("\n--- Accuracy Assessment ---");
    let final_fx_error = 100.0 * (optim_result.intrinsics.fx - ground_truth_k.fx).abs() / ground_truth_k.fx;
    let final_fy_error = 100.0 * (optim_result.intrinsics.fy - ground_truth_k.fy).abs() / ground_truth_k.fy;
    let final_cx_error = 100.0 * (optim_result.intrinsics.cx - ground_truth_k.cx).abs() / ground_truth_k.cx;
    let final_cy_error = 100.0 * (optim_result.intrinsics.cy - ground_truth_k.cy).abs() / ground_truth_k.cy;

    println!(
        "Final errors: fx={:.2}%, fy={:.2}%, cx={:.2}%, cy={:.2}%",
        final_fx_error, final_fy_error, final_cx_error, final_cy_error
    );

    println!("\n--- Improvement from Initialization ---");
    println!(
        "fx: {:.1}% → {:.2}% ({}x better)",
        fx_error,
        final_fx_error,
        fx_error / final_fx_error
    );
    println!(
        "fy: {:.1}% → {:.2}% ({}x better)",
        fy_error,
        final_fy_error,
        fy_error / final_fy_error
    );

    println!("\n✓ Custom workflow completed successfully!");

    Ok(())
}

/// Generate synthetic calibration data with known ground truth.
fn generate_synthetic_data() -> (Vec<PlanarViewData>, FxFyCxCySkew<f64>, BrownConrady5<f64>) {
    // Ground truth camera parameters
    let k_gt = FxFyCxCySkew {
        fx: 800.0,
        fy: 780.0,
        cx: 640.0,
        cy: 360.0,
        skew: 0.0,
    };
    let dist_gt = BrownConrady5 {
        k1: -0.1,
        k2: 0.01,
        k3: 0.0,
        p1: 0.0,
        p2: 0.0,
        iters: 8,
    };
    let cam_gt = Camera::new(Pinhole, dist_gt, IdentitySensor, k_gt);

    // Generate checkerboard pattern (6x5 grid, 4cm spacing)
    let nx = 6;
    let ny = 5;
    let spacing = 0.04;
    let mut board_points = Vec::new();
    for j in 0..ny {
        for i in 0..nx {
            board_points.push(Pt3::new(i as f64 * spacing, j as f64 * spacing, 0.0));
        }
    }

    // Generate views from different poses
    let mut views = Vec::new();
    for view_idx in 0..6 {
        // Vary rotation and translation for each view
        let angle = 0.2 * (view_idx as f64) - 0.5; // -0.5 to +0.7 radians
        let axis = Vector3::new(0.0, 1.0, 0.0);
        let rotation = UnitQuaternion::from_scaled_axis(axis * angle);
        let translation = Vector3::new(0.0, 0.0, 0.4 + 0.15 * view_idx as f64);
        let pose = calib_core::Iso3::from_parts(translation.into(), rotation);

        // Project points through camera
        let mut points_2d = Vec::new();
        for pw in &board_points {
            let pc = pose.transform_point(pw);
            if let Some(proj) = cam_gt.project_point(&pc) {
                points_2d.push(Vec2::new(proj.x, proj.y));
            }
        }

        views.push(PlanarViewData {
            points_3d: board_points.clone(),
            points_2d,
            weights: None,
        });
    }

    (views, k_gt, dist_gt)
}
