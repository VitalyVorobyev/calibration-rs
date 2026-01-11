//! Example comparing Session API vs. Imperative Function API.
//!
//! This example demonstrates both approaches for the same calibration task,
//! highlighting their differences and trade-offs.
//!
//! Run with: cargo run --example compare_apis

use calib_core::{BrownConrady5, Camera, FxFyCxCySkew, IdentitySensor, Pinhole, Pt3, Vec2};
use calib_pipeline::distortion_fit::DistortionFitOptions;
use calib_pipeline::helpers::{initialize_planar_intrinsics, optimize_planar_intrinsics_from_init};
use calib_pipeline::iterative_intrinsics::IterativeIntrinsicsOptions;
use calib_pipeline::session::problem_types::{
    PlanarIntrinsicsInitOptions, PlanarIntrinsicsObservations, PlanarIntrinsicsOptimOptions,
    PlanarIntrinsicsProblem,
};
use calib_pipeline::session::CalibrationSession;
use calib_pipeline::{BackendSolveOptions, PlanarIntrinsicsSolveOptions, PlanarViewData};
use nalgebra::{UnitQuaternion, Vector3};

fn main() -> anyhow::Result<()> {
    println!("=== Comparing Session API vs. Imperative Function API ===\n");

    // Generate synthetic data (same for both approaches)
    let views = generate_synthetic_data();
    println!("Generated {} calibration views\n", views.len());

    println!("┌─────────────────────────────────────────────────────────────┐");
    println!("│ Approach 1: SESSION API (Type-Safe, Structured)            │");
    println!("└─────────────────────────────────────────────────────────────┘\n");

    let start = std::time::Instant::now();
    let session_result = run_with_session_api(views.clone())?;
    let session_time = start.elapsed();

    println!("\n┌─────────────────────────────────────────────────────────────┐");
    println!("│ Approach 2: IMPERATIVE API (Flexible, Inspectable)         │");
    println!("└─────────────────────────────────────────────────────────────┘\n");

    let start = std::time::Instant::now();
    let imperative_result = run_with_imperative_api(views)?;
    let imperative_time = start.elapsed();

    // Compare results
    println!("\n┌─────────────────────────────────────────────────────────────┐");
    println!("│ COMPARISON                                                  │");
    println!("└─────────────────────────────────────────────────────────────┘\n");

    println!("Execution Time:");
    println!("  Session API:    {:?}", session_time);
    println!("  Imperative API: {:?}", imperative_time);

    println!("\nFinal Cost:");
    println!("  Session API:    {:.6}", session_result.final_cost);
    println!("  Imperative API: {:.6}", imperative_result.final_cost);

    println!("\nIntrinsics (fx, fy, cx, cy):");
    println!(
        "  Session API:    ({:.1}, {:.1}, {:.1}, {:.1})",
        session_result.fx, session_result.fy, session_result.cx, session_result.cy
    );
    println!(
        "  Imperative API: ({:.1}, {:.1}, {:.1}, {:.1})",
        imperative_result.fx, imperative_result.fy, imperative_result.cx, imperative_result.cy
    );

    println!("\n┌─────────────────────────────────────────────────────────────┐");
    println!("│ WHEN TO USE EACH API                                       │");
    println!("└─────────────────────────────────────────────────────────────┘\n");

    println!("Use SESSION API when:");
    println!("  ✓ You have a standard calibration workflow");
    println!("  ✓ You want automatic state management");
    println!("  ✓ You need checkpointing between stages");
    println!("  ✓ Type safety is important");
    println!("  ✓ You prefer declarative style");

    println!("\nUse IMPERATIVE API when:");
    println!("  ✓ You need to inspect intermediate results");
    println!("  ✓ You want custom decision logic");
    println!("  ✓ You're composing a complex workflow");
    println!("  ✓ You need maximum flexibility");
    println!("  ✓ You prefer explicit control flow");

    Ok(())
}

struct CalibrationResult {
    fx: f64,
    fy: f64,
    cx: f64,
    cy: f64,
    final_cost: f64,
}

fn run_with_session_api(views: Vec<PlanarViewData>) -> anyhow::Result<CalibrationResult> {
    println!("Creating session...");
    let mut session = CalibrationSession::<PlanarIntrinsicsProblem>::new();

    println!("Setting observations...");
    session.set_observations(PlanarIntrinsicsObservations { views });

    println!("Initializing...");
    session.initialize(PlanarIntrinsicsInitOptions::default())?;

    // Could checkpoint here
    let _checkpoint = session.to_json()?;
    println!("  (checkpoint created: {} bytes)", _checkpoint.len());

    println!("Optimizing...");
    session.optimize(PlanarIntrinsicsOptimOptions::default())?;

    println!("Exporting results...");
    let report = session.export()?;

    // Extract results
    let calib_core::IntrinsicsParams::FxFyCxCySkew { params } = &report.report.camera.intrinsics;

    println!("✓ Session API completed");

    Ok(CalibrationResult {
        fx: params.fx,
        fy: params.fy,
        cx: params.cx,
        cy: params.cy,
        final_cost: report.report.final_cost,
    })
}

fn run_with_imperative_api(views: Vec<PlanarViewData>) -> anyhow::Result<CalibrationResult> {
    println!("Configuring initialization options...");
    let init_opts = IterativeIntrinsicsOptions {
        iterations: 2,
        distortion_opts: DistortionFitOptions {
            fix_k3: true,
            fix_tangential: false,
            iters: 8,
        },
    };

    println!("Running linear initialization...");
    let init_result = initialize_planar_intrinsics(&views, &init_opts)?;

    // Inspect initialization (this is the key advantage)
    println!(
        "  Initial fx={:.1}, fy={:.1}",
        init_result.intrinsics.fx, init_result.intrinsics.fy
    );

    // Custom decision logic could go here
    if init_result.intrinsics.fx < 100.0 {
        println!("  ⚠ Warning: Low focal length detected!");
    }

    println!("Configuring optimization options...");
    let solve_opts = PlanarIntrinsicsSolveOptions {
        fix_poses: vec![0],
        ..Default::default()
    };
    let backend_opts = BackendSolveOptions::default();

    println!("Running non-linear optimization...");
    let optim_result =
        optimize_planar_intrinsics_from_init(&views, &init_result, &solve_opts, &backend_opts)?;

    println!("✓ Imperative API completed");

    Ok(CalibrationResult {
        fx: optim_result.intrinsics.fx,
        fy: optim_result.intrinsics.fy,
        cx: optim_result.intrinsics.cx,
        cy: optim_result.intrinsics.cy,
        final_cost: optim_result.final_cost,
    })
}

fn generate_synthetic_data() -> Vec<PlanarViewData> {
    // Ground truth camera
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

    // Checkerboard pattern
    let nx = 5;
    let ny = 4;
    let spacing = 0.05;
    let mut board_points = Vec::new();
    for j in 0..ny {
        for i in 0..nx {
            board_points.push(Pt3::new(i as f64 * spacing, j as f64 * spacing, 0.0));
        }
    }

    // Generate views
    let mut views = Vec::new();
    for view_idx in 0..5 {
        let angle = 0.15 * (view_idx as f64) - 0.3;
        let axis = Vector3::new(0.0, 1.0, 0.0);
        let rotation = UnitQuaternion::from_scaled_axis(axis * angle);
        let translation = Vector3::new(0.0, 0.0, 0.5 + 0.1 * view_idx as f64);
        let pose = calib_core::Iso3::from_parts(translation.into(), rotation);

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

    views
}
