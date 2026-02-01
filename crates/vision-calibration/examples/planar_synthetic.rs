//! Planar intrinsics calibration with synthetic data.
//!
//! This example demonstrates the basic calibration workflow using synthetic data:
//! 1. Generate a synthetic planar calibration dataset
//! 2. Run initialization (Zhang's method with iterative distortion)
//! 3. Run non-linear optimization (bundle adjustment)
//! 4. Export and inspect results
//!
//! Run with: `cargo run -p vision-calibration --example planar_synthetic`

use anyhow::Result;
use vision_calibration::planar_intrinsics::{run_calibration, step_init, step_optimize};
use vision_calibration::prelude::*;
use vision_calibration::synthetic::planar;

fn main() -> Result<()> {
    println!("=== Planar Intrinsics Calibration (Synthetic Data) ===\n");

    // Ground truth camera parameters
    let k_gt = FxFyCxCySkew {
        fx: 800.0,
        fy: 780.0,
        cx: 640.0,
        cy: 360.0,
        skew: 0.0,
    };
    let dist_gt = BrownConrady5 {
        k1: 0.05,
        k2: -0.02,
        k3: 0.0,
        p1: 0.001,
        p2: -0.001,
        iters: 8,
    };
    let cam_gt = vision_calibration::make_pinhole_camera(k_gt, dist_gt);

    println!("Ground truth:");
    println!(
        "  Intrinsics: fx={:.1}, fy={:.1}, cx={:.1}, cy={:.1}",
        k_gt.fx, k_gt.fy, k_gt.cx, k_gt.cy
    );
    println!(
        "  Distortion: k1={:.4}, k2={:.4}, p1={:.4}, p2={:.4}\n",
        dist_gt.k1, dist_gt.k2, dist_gt.p1, dist_gt.p2
    );

    // Generate synthetic calibration data
    let board_points = planar::grid_points(8, 6, 0.04); // 8x6 grid, 40mm squares
    let poses = planar::poses_yaw_y_z(
        6,    // 6 views
        -0.2, // start yaw
        0.08, // yaw step
        0.5,  // start Z
        0.05, // Z step
    );
    let views = planar::project_views_all(&cam_gt, &board_points, &poses)?;

    println!(
        "Generated {} views with {} points each\n",
        views.len(),
        board_points.len()
    );

    // Create dataset from views
    let dataset = PlanarDataset::new(views.into_iter().map(View::without_meta).collect())?;

    // Create calibration session
    let mut session = CalibrationSession::<PlanarIntrinsicsProblem>::new();
    session.set_input(dataset)?;

    // Option 1: Step-by-step calibration (recommended for inspection)
    println!("--- Step 1: Initialization ---");
    step_init(&mut session, None)?;

    let init_k = session.state.initial_intrinsics.as_ref().unwrap();
    let init_dist = session.state.initial_distortion.as_ref().unwrap();
    println!(
        "  Intrinsics: fx={:.1}, fy={:.1}, cx={:.1}, cy={:.1}",
        init_k.fx, init_k.fy, init_k.cx, init_k.cy
    );
    println!(
        "  Distortion: k1={:.4}, k2={:.4}, p1={:.4}, p2={:.4}",
        init_dist.k1, init_dist.k2, init_dist.p1, init_dist.p2
    );
    println!(
        "  Error vs GT: fx={:.1}%, fy={:.1}%",
        100.0 * (init_k.fx - k_gt.fx).abs() / k_gt.fx,
        100.0 * (init_k.fy - k_gt.fy).abs() / k_gt.fy
    );
    println!();

    println!("--- Step 2: Optimization ---");
    step_optimize(&mut session, None)?;

    let state = &session.state;
    println!("  Final cost: {:.2e}", state.final_cost.unwrap());
    println!(
        "  Mean reprojection error: {:.4} px",
        state.mean_reproj_error.unwrap()
    );
    println!();

    // Export results
    let export = session.export()?;
    let final_k = export.params.intrinsics();
    let final_dist = export.params.distortion();

    println!("--- Final Results ---");
    println!(
        "  Intrinsics: fx={:.1}, fy={:.1}, cx={:.1}, cy={:.1}",
        final_k.fx, final_k.fy, final_k.cx, final_k.cy
    );
    println!(
        "  Distortion: k1={:.4}, k2={:.4}, k3={:.4}, p1={:.4}, p2={:.4}",
        final_dist.k1, final_dist.k2, final_dist.k3, final_dist.p1, final_dist.p2
    );
    println!(
        "  Error vs GT: fx={:.2}%, fy={:.2}%",
        100.0 * (final_k.fx - k_gt.fx).abs() / k_gt.fx,
        100.0 * (final_k.fy - k_gt.fy).abs() / k_gt.fy
    );
    println!();

    // Option 2: Pipeline function (convenience)
    println!("--- Alternative: run_calibration() ---");
    let mut session2 = CalibrationSession::<PlanarIntrinsicsProblem>::new();

    // Recreate dataset
    let views2 = planar::project_views_all(&cam_gt, &board_points, &poses)?;
    let dataset2 = PlanarDataset::new(views2.into_iter().map(View::without_meta).collect())?;
    session2.set_input(dataset2)?;

    run_calibration(&mut session2)?;

    let export2 = session2.export()?;
    println!(
        "  Mean reprojection error: {:.4} px",
        export2.mean_reproj_error
    );

    Ok(())
}
