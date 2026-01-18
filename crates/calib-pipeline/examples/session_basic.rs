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

use calib_core::{synthetic::planar, BrownConrady5, Camera, FxFyCxCySkew, IdentitySensor, Pinhole};
use calib_pipeline::session::problem_types::{
    PlanarIntrinsicsInitOptions, PlanarIntrinsicsObservations, PlanarIntrinsicsOptimOptions,
    PlanarIntrinsicsProblem,
};
use calib_pipeline::session::CalibrationSession;
use calib_pipeline::CorrespondenceView;
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
    let calib_core::IntrinsicsParams::FxFyCxCySkew { params } =
        &final_results.report.camera.intrinsics;
    println!(
        "Estimated intrinsics: fx={:.1}, fy={:.1}, cx={:.1}, cy={:.1}, skew={:.4}",
        params.fx, params.fy, params.cx, params.cy, params.skew
    );
    println!(
        "Errors: fx={:.2}%, fy={:.2}%, cx={:.2}%, cy={:.2}%",
        100.0 * (params.fx - ground_truth_k.fx).abs() / ground_truth_k.fx,
        100.0 * (params.fy - ground_truth_k.fy).abs() / ground_truth_k.fy,
        100.0 * (params.cx - ground_truth_k.cx).abs() / ground_truth_k.cx,
        100.0 * (params.cy - ground_truth_k.cy).abs() / ground_truth_k.cy
    );

    // Extract and display distortion
    if let calib_core::DistortionParams::BrownConrady5 { params } =
        &final_results.report.camera.distortion
    {
        println!(
            "Estimated distortion: k1={:.4}, k2={:.4}, k3={:.4}, p1={:.4}, p2={:.4}",
            params.k1, params.k2, params.k3, params.p1, params.p2
        );
    }

    println!("\n✓ Session completed successfully!");

    Ok(())
}

/// Generate synthetic calibration data with known ground truth.
fn generate_synthetic_data() -> (
    Vec<CorrespondenceView>,
    FxFyCxCySkew<f64>,
    BrownConrady5<f64>,
) {
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
    let board_points = planar::grid_points(5, 4, 0.05);

    // Generate views from different poses (yaw around +Y, increasing distance).
    let poses = planar::poses_yaw_y_z(5, -0.3, 0.15, 0.5, 0.1);
    let views =
        planar::project_views_all(&cam_gt, &board_points, &poses).expect("synthetic projection");

    (views, k_gt, dist_gt)
}
