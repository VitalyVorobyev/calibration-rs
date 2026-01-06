//! Basic example demonstrating the Session API for planar intrinsics calibration.
//!
//! This example shows:
//! - Creating a calibration session
//! - Setting observations
//! - Running initialization
//! - Checkpointing state to JSON
//! - Running optimization
//! - Exporting final results
//!
//! Run with: cargo run --example session_basic

use calib_core::{BrownConrady5, Camera, FxFyCxCySkew, IdentitySensor, Pinhole, Pt3, Vec2};
use calib_pipeline::session::problem_types::{
    PlanarIntrinsicsInitOptions, PlanarIntrinsicsObservations, PlanarIntrinsicsOptimOptions,
    PlanarIntrinsicsProblem,
};
use calib_pipeline::session::CalibrationSession;
use calib_pipeline::PlanarViewData;
use nalgebra::{UnitQuaternion, Vector3};

fn main() -> anyhow::Result<()> {
    println!("=== Session API Example ===\n");

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

    // Step 1: Create session
    let mut session = CalibrationSession::<PlanarIntrinsicsProblem>::new_with_description(
        "Basic planar intrinsics calibration".to_string(),
    );
    println!("✓ Created calibration session");

    // Step 2: Set observations
    let observations = PlanarIntrinsicsObservations { views };
    session.set_observations(observations);
    println!("✓ Set observations");

    // Step 3: Initialize with linear solver
    println!("\nRunning linear initialization...");
    session.initialize(PlanarIntrinsicsInitOptions::default())?;
    println!("✓ Initialization complete");

    // Step 4: Checkpoint state (optional)
    let checkpoint_json = session.to_json()?;
    println!(
        "✓ Session state saved to JSON ({} bytes)",
        checkpoint_json.len()
    );

    // Could save to file:
    // std::fs::write("checkpoint.json", checkpoint_json)?;

    // Step 5: Optimize with non-linear refinement
    println!("\nRunning non-linear optimization...");
    session.optimize(PlanarIntrinsicsOptimOptions::default())?;
    println!("✓ Optimization complete");

    // Step 6: Export final results
    let final_results = session.export()?;
    println!("\n=== Final Results ===");
    println!("Final cost: {:.6}", final_results.report.final_cost);

    // Extract and display intrinsics
    let calib_core::IntrinsicsConfig::FxFyCxCySkew {
        fx,
        fy,
        cx,
        cy,
        skew,
    } = &final_results.report.camera.intrinsics;
    println!(
        "Estimated intrinsics: fx={:.1}, fy={:.1}, cx={:.1}, cy={:.1}, skew={:.4}",
        fx, fy, cx, cy, skew
    );
    println!(
        "Errors: fx={:.2}%, fy={:.2}%, cx={:.2}%, cy={:.2}%",
        100.0 * (fx - ground_truth_k.fx).abs() / ground_truth_k.fx,
        100.0 * (fy - ground_truth_k.fy).abs() / ground_truth_k.fy,
        100.0 * (cx - ground_truth_k.cx).abs() / ground_truth_k.cx,
        100.0 * (cy - ground_truth_k.cy).abs() / ground_truth_k.cy
    );

    // Extract and display distortion
    if let calib_core::DistortionConfig::BrownConrady5 {
        k1,
        k2,
        k3,
        p1,
        p2,
        iters: _,
    } = &final_results.report.camera.distortion
    {
        println!(
            "Estimated distortion: k1={:.4}, k2={:.4}, k3={:.4}, p1={:.4}, p2={:.4}",
            k1, k2, k3, p1, p2
        );
    }

    println!("\n✓ Session completed successfully!");

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

    // Generate checkerboard pattern (5x4 grid, 5cm spacing)
    let nx = 5;
    let ny = 4;
    let spacing = 0.05;
    let mut board_points = Vec::new();
    for j in 0..ny {
        for i in 0..nx {
            board_points.push(Pt3::new(i as f64 * spacing, j as f64 * spacing, 0.0));
        }
    }

    // Generate views from different poses
    let mut views = Vec::new();
    for view_idx in 0..5 {
        // Vary rotation and translation for each view
        let angle = 0.15 * (view_idx as f64) - 0.3; // -0.3 to +0.3 radians
        let axis = Vector3::new(0.0, 1.0, 0.0);
        let rotation = UnitQuaternion::from_scaled_axis(axis * angle);
        let translation = Vector3::new(0.0, 0.0, 0.5 + 0.1 * view_idx as f64);
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
