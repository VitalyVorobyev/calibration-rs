# Quickstart

This chapter provides copy-paste examples to get you calibrating cameras quickly.

## Prerequisites

- Rust toolchain (1.70+): [https://rustup.rs](https://rustup.rs)
- For building docs: `cargo install mdbook`

## Installation

Add `calib` to your `Cargo.toml`:

```toml
[dependencies]
calib = { git = "https://github.com/VitalyVorobyev/calibration-rs" }
anyhow = "1"  # For error handling in examples
```

## Example 1: Synthetic Planar Calibration

This example generates synthetic calibration data and runs the full pipeline:

```rust
use calib::prelude::*;
use calib::synthetic::planar;

fn main() -> anyhow::Result<()> {
    // 1. Define ground truth camera
    let k_gt = FxFyCxCySkew {
        fx: 800.0,
        fy: 800.0,
        cx: 640.0,
        cy: 360.0,
        skew: 0.0,
    };
    let dist_gt = BrownConrady5 {
        k1: -0.1,
        k2: 0.01,
        k3: 0.0,
        p1: 0.001,
        p2: -0.001,
        iters: 8,
    };
    let camera_gt = Camera::new(Pinhole, dist_gt, IdentitySensor, k_gt);

    // 2. Generate synthetic observations
    let board_points = planar::grid_points(6, 5, 0.04);  // 6x5 grid, 40mm squares
    let poses = planar::poses_yaw_y_z(5, -0.3, 0.15, 0.5, 0.1);  // 5 views
    let views = planar::project_views_all(&camera_gt, &board_points, &poses)?;

    println!("Generated {} views with {} points each", views.len(), board_points.len());

    // 3. Run calibration pipeline
    let input = PlanarIntrinsicsInput { views };
    let config = PlanarIntrinsicsConfig::default();
    let report = calib::pipeline::run_planar_intrinsics(&input, &config)?;

    // 4. Extract results
    if let IntrinsicsParams::FxFyCxCySkew { params } = &report.camera.intrinsics {
        println!("\nEstimated intrinsics:");
        println!("  fx = {:.2} (gt: {:.2}, err: {:.2}%)",
            params.fx, k_gt.fx, 100.0 * (params.fx - k_gt.fx).abs() / k_gt.fx);
        println!("  fy = {:.2} (gt: {:.2}, err: {:.2}%)",
            params.fy, k_gt.fy, 100.0 * (params.fy - k_gt.fy).abs() / k_gt.fy);
        println!("  cx = {:.2} (gt: {:.2})", params.cx, k_gt.cx);
        println!("  cy = {:.2} (gt: {:.2})", params.cy, k_gt.cy);
    }

    println!("\nFinal cost: {:.2e}", report.final_cost);
    Ok(())
}
```

## Example 2: Session API with Checkpointing

The session API provides state management and JSON checkpointing:

```rust
use calib::session::{CalibrationSession, PlanarIntrinsicsProblem, PlanarIntrinsicsObservations};
use calib::prelude::*;
use calib::synthetic::planar;

fn main() -> anyhow::Result<()> {
    // Generate synthetic data
    let k = FxFyCxCySkew { fx: 800.0, fy: 800.0, cx: 640.0, cy: 360.0, skew: 0.0 };
    let dist = BrownConrady5 { k1: 0.0, k2: 0.0, k3: 0.0, p1: 0.0, p2: 0.0, iters: 8 };
    let camera = Camera::new(Pinhole, dist, IdentitySensor, k);

    let board = planar::grid_points(6, 5, 0.04);
    let poses = planar::poses_yaw_y_z(5, -0.3, 0.15, 0.5, 0.1);
    let views = planar::project_views_all(&camera, &board, &poses)?;

    // Create session
    let mut session = CalibrationSession::<PlanarIntrinsicsProblem>::new_with_description(
        "My calibration session".to_string()
    );

    // Set observations
    session.set_observations(PlanarIntrinsicsObservations { views });
    println!("Stage: {:?}", session.stage());  // Uninitialized

    // Initialize with linear solver
    session.initialize(Default::default())?;
    println!("Stage: {:?}", session.stage());  // Initialized

    // Save checkpoint (can resume later)
    let checkpoint = session.to_json()?;
    std::fs::write("checkpoint.json", &checkpoint)?;
    println!("Checkpoint saved ({} bytes)", checkpoint.len());

    // Optimize with non-linear refinement
    session.optimize(Default::default())?;
    println!("Stage: {:?}", session.stage());  // Optimized

    // Export final results
    let report = session.export()?;
    println!("Final cost: {:.2e}", report.report.final_cost);

    Ok(())
}
```

### Resuming from Checkpoint

```rust
use calib::session::{CalibrationSession, PlanarIntrinsicsProblem};

fn main() -> anyhow::Result<()> {
    // Load checkpoint
    let json = std::fs::read_to_string("checkpoint.json")?;
    let mut session = CalibrationSession::<PlanarIntrinsicsProblem>::from_json(&json)?;

    println!("Resumed at stage: {:?}", session.stage());

    // Continue from where we left off
    if session.stage() == calib::session::SessionStage::Initialized {
        session.optimize(Default::default())?;
    }

    let report = session.export()?;
    println!("Final cost: {:.2e}", report.report.final_cost);

    Ok(())
}
```

## Example 3: Custom Workflow with Inspection

For maximum control, use the imperative functions API:

```rust
use calib::helpers::{initialize_planar_intrinsics, optimize_planar_intrinsics_from_init};
use calib::linear::iterative_intrinsics::IterativeIntrinsicsOptions;
use calib::linear::distortion_fit::DistortionFitOptions;
use calib::optim::planar_intrinsics::PlanarIntrinsicsSolveOptions;
use calib::optim::backend::BackendSolveOptions;
use calib::prelude::*;
use calib::synthetic::planar;

fn main() -> anyhow::Result<()> {
    // Generate views
    let k = FxFyCxCySkew { fx: 800.0, fy: 800.0, cx: 640.0, cy: 360.0, skew: 0.0 };
    let dist = BrownConrady5 { k1: -0.05, k2: 0.01, k3: 0.0, p1: 0.0, p2: 0.0, iters: 8 };
    let camera = Camera::new(Pinhole, dist, IdentitySensor, k);

    let board = planar::grid_points(6, 5, 0.04);
    let poses = planar::poses_yaw_y_z(5, -0.3, 0.15, 0.5, 0.1);
    let views = planar::project_views_all(&camera, &board, &poses)?;

    // Step 1: Linear initialization
    let init_opts = IterativeIntrinsicsOptions {
        iterations: 2,
        distortion_opts: DistortionFitOptions {
            fix_k3: true,
            fix_tangential: false,
            iters: 8,
        },
        zero_skew: true,
    };

    let init = initialize_planar_intrinsics(&views, &init_opts)?;

    println!("=== Linear Initialization ===");
    println!("fx = {:.2} (gt: {:.2})", init.intrinsics.fx, k.fx);
    println!("fy = {:.2} (gt: {:.2})", init.intrinsics.fy, k.fy);
    println!("k1 = {:.4} (gt: {:.4})", init.distortion.k1, dist.k1);
    println!("k2 = {:.4} (gt: {:.4})", init.distortion.k2, dist.k2);

    // Check initialization quality before proceeding
    let fx_error = (init.intrinsics.fx - k.fx).abs() / k.fx;
    if fx_error > 0.5 {
        println!("\nWarning: Large initialization error ({:.0}%), check input data", fx_error * 100.0);
    }

    // Step 2: Non-linear refinement
    let solve_opts = PlanarIntrinsicsSolveOptions::default();
    let backend_opts = BackendSolveOptions {
        max_iters: 100,
        verbosity: 1,  // Print progress
        ..Default::default()
    };

    let result = optimize_planar_intrinsics_from_init(&views, &init, &solve_opts, &backend_opts)?;

    println!("\n=== Non-linear Optimization ===");
    println!("fx = {:.2} (gt: {:.2})", result.intrinsics.fx, k.fx);
    println!("fy = {:.2} (gt: {:.2})", result.intrinsics.fy, k.fy);
    println!("Mean reprojection error: {:.4} px", result.mean_reproj_error);

    Ok(())
}
```

## CLI Usage

For batch processing without writing code:

```bash
# Create input JSON (see CLI chapter for format)
echo '{"views": [...]}' > views.json

# Run calibration
cargo run -p calib-cli -- --input views.json > report.json

# With custom config
cargo run -p calib-cli -- --input views.json --config config.json > report.json
```

## Next Steps

- **[Core Concepts](concepts.md)**: Understand coordinate conventions and camera models
- **[Linear Calibration](linear.md)**: Deep dive into initialization algorithms
- **[Non-linear Optimization](nonlinear.md)**: Configure robust losses and parameter fixing
- **[CLI](cli.md)**: Full CLI documentation with JSON schemas
