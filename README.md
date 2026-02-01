# calibration-rs

[![CI](https://github.com/VitalyVorobyev/calibration-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/VitalyVorobyev/calibration-rs/actions/workflows/ci.yml)
[![Docs](https://github.com/VitalyVorobyev/calibration-rs/actions/workflows/publish-docs.yml/badge.svg)](https://vitalyvorobyev.github.io/calibration/)
[![Audit](https://github.com/VitalyVorobyev/calibration-rs/actions/workflows/audit.yml/badge.svg)](https://github.com/VitalyVorobyev/calibration-rs/actions/workflows/audit.yml)

A Rust workspace for end-to-end camera calibration: math primitives, linear solvers, non-linear
refinement, pipelines, and a CLI. Supports perspective cameras, laserline calibration, and multi-camera rigs.

## Status

- Stable foundation: core math types, camera models, deterministic RANSAC, and linear solvers.
- Working pipelines: planar intrinsics with Brown-Conrady distortion, stepwise hand-eye for
  single-camera setups, and single laserline device (camera + laser plane).
- Optimization: planar intrinsics, hand-eye, rig extrinsics, and laserline bundle problems with a
  Levenberg-Marquardt backend.
- In progress: broader pipeline coverage (rig), more CLI commands, and additional
  real-data validation.
- API stability: `vision-calibration` is the compatibility boundary; lower crates may evolve.

## Architecture

```
                           ┌─────────────────────────┐
                           │         vision-calibration           │  ◄── Stable API facade
                           │    (public interface)   │
                           └───────────┬─────────────┘
                                       │
              ┌────────────────────────┼────────────────────────┐
              │                        │                        │
              ▼                        ▼                        ▼
┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────────┐
│   vision-calibration-pipeline    │  │    vision-calibration-optim      │  │    vision-calibration-linear     │
│  Session API, JSON  │  │   Non-linear BA     │  │   Linear solvers    │
│   I/O, workflows    │  │   LM optimization   │  │   Initialization    │
└─────────┬───────────┘  └─────────┬───────────┘  └─────────┬───────────┘
          │                        │                        │
          └────────────────────────┼────────────────────────┘
                                   │
                                   ▼
                       ┌─────────────────────┐
                       │     vision-calibration-core      │  ◄── Math types, camera
                       │   Types, models,    │      models, RANSAC
                       │       RANSAC        │
                       └─────────────────────┘
```

## Crate Summary

| Crate | Description |
|-------|-------------|
| **vision-calibration** | Facade re-exporting all sub-crates for a stable API surface |
| **vision-calibration-core** | Math types (nalgebra), composable camera models, RANSAC, synthetic data |
| **vision-calibration-linear** | Closed-form solvers: homography, Zhang, PnP, epipolar, hand-eye, laserline |
| **vision-calibration-optim** | Non-linear LM refinement: planar intrinsics, rig, hand-eye, laserline |
| **vision-calibration-pipeline** | End-to-end workflows, session API, JSON I/O |

## Quickstart

Add the facade crate to your `Cargo.toml`:

```toml
vision-calibration = { git = "https://github.com/VitalyVorobyev/calibration-rs" }
```

Use the high-level pipeline API:

```rust
use vision_calibration::pipeline::{run_planar_intrinsics, CorrespondenceView, PlanarIntrinsicsConfig, PlanarIntrinsicsInput};
use vision_calibration::core::{IntrinsicsParams, Pt3, Vec2};

fn main() {
    // Populate per-view correspondences (normally from a detector)
    let board = vec![
        Pt3::new(0.0, 0.0, 0.0),
        Pt3::new(0.1, 0.0, 0.0),
        Pt3::new(0.1, 0.1, 0.0),
        Pt3::new(0.0, 0.1, 0.0),
    ];
    let view = CorrespondenceView {
        points_3d: board.clone(),
        points_2d: vec![
            Vec2::new(100.0, 120.0),
            Vec2::new(180.0, 118.0),
            Vec2::new(182.0, 192.0),
            Vec2::new(98.0, 196.0),
        ],
        weights: None,
    };
    let input = PlanarIntrinsicsInput { views: vec![view] };
    let config = PlanarIntrinsicsConfig::default();

    let report = run_planar_intrinsics(&input, &config).expect("planar intrinsics failed");
    println!("Estimated camera config: {:?}", report.camera);

    if let IntrinsicsParams::FxFyCxCySkew { params } = &report.camera.intrinsics {
        println!(
            "Estimated intrinsics: fx={} fy={} cx={} cy={} skew={}",
            params.fx, params.fy, params.cx, params.cy, params.skew
        );
    }
}
```

For checkpointed workflows, see `vision-calibration::session` and `vision-calibration::pipeline::session`.

### Laserline device pipeline (camera + laser plane)

```rust
use vision_calibration::prelude::*;
use vision_calibration::laserline_device::{run_calibration, LaserlineDeviceProblem};

let mut session = CalibrationSession::<LaserlineDeviceProblem>::new();
session.set_input(views)?;
run_calibration(&mut session, None)?;
let export = session.export()?;
println!("Mean reproj error: {:.3}px", export.stats.mean_reproj_error);
```

### Synthetic data generation

For examples, tests, and benchmarking you can generate deterministic synthetic correspondences:

```rust
use vision_calibration::synthetic::planar;
use vision_calibration::core::{BrownConrady5, Camera, FxFyCxCySkew, IdentitySensor, Pinhole};

fn main() -> anyhow::Result<()> {
    let k = FxFyCxCySkew { fx: 800.0, fy: 800.0, cx: 640.0, cy: 360.0, skew: 0.0 };
    let dist = BrownConrady5 { k1: 0.0, k2: 0.0, k3: 0.0, p1: 0.0, p2: 0.0, iters: 8 };
    let cam = Camera::new(Pinhole, dist, IdentitySensor, k);

    let board = planar::grid_points(6, 5, 0.04);
    let poses = planar::poses_yaw_y_z(5, -0.3, 0.15, 0.5, 0.1);
    let views = planar::project_views_all(&cam, &board, &poses)?;
    println!("generated {} views", views.len());
    Ok(())
}
```

### Hand-eye calibration (stepwise)

Use the stepwise helpers in `vision-calibration::pipeline::handeye_single` (see
`crates/vision-calibration/examples/handeyesingle.rs` and `crates/vision-calibration/examples/handeye_session.rs`):

```rust
use vision_calibration::pipeline::handeye_single::{run_handeye_single, HandEyeSingleOptions, HandEyeView};

fn main() {
    let views: Vec<HandEyeView> = /* load 2D/3D views + robot poses */;
    let report = run_handeye_single(&views, &HandEyeSingleOptions::default())
        .expect("hand-eye failed");

    println!(
        "final reproj error: {:.3} px",
        report.handeye_optimized.mean_reproj_error
    );
}
```

For a linear-algorithm overview and usage notes, see `crates/vision-calibration-linear/README.md`.

### Hand-eye calibration (session)

Use the dedicated session problem for a compact workflow (see
`crates/vision-calibration/examples/handeye_session.rs`):

```rust
use vision_calibration::session::{
    CalibrationSession, HandEyeModeConfig, HandEyeSingleInitOptions, HandEyeSingleObservations,
    HandEyeSingleOptimOptions, HandEyeSingleProblem,
};
use vision_calibration::pipeline::handeye_single::HandEyeView;

fn main() -> anyhow::Result<()> {
    let views: Vec<HandEyeView> = /* load 2D/3D views + robot poses */;

    let mut session = CalibrationSession::<HandEyeSingleProblem>::new();
    session.set_observations(HandEyeSingleObservations {
        views,
        mode: HandEyeModeConfig::EyeInHand,
    });

    session.initialize(HandEyeSingleInitOptions::default())?;
    session.optimize(HandEyeSingleOptimOptions::default())?;
    let report = session.export()?;

    println!(
        "final reproj error: {:.3} px",
        report.handeye_optimized.mean_reproj_error
    );
    Ok(())
}
```

## Camera model

`vision-calibration-core` models cameras as a composable pipeline:

```
pixel = K(sensor(distortion(projection(dir))))
```

Where:
- `projection` maps a camera-frame direction to normalized coordinates (e.g., pinhole).
- `distortion` warps normalized coordinates (Brown-Conrady radial and tangential).
- `sensor` applies a homography (identity or Scheimpflug/tilt).
- `K` maps sensor coordinates to pixels (`fx`, `fy`, `cx`, `cy`, `skew`).

`SensorConfig::Scheimpflug` follows OpenCV's tilted sensor model (`tau_x`, `tau_y`), implemented as
the same homography computed by OpenCV's `computeTiltProjectionMatrix`.

## Docs

- API docs and book: https://vitalyvorobyev.github.io/calibration/
- Book sources: `book/`
- Examples: `crates/vision-calibration/examples/` and `crates/vision-calibration-pipeline/examples/`

## Design principles

- Correctness and numerical stability with explicit failure modes.
- Deterministic outputs (seeded RNGs, stable ordering).
- Performance-aware implementations (fixed-size math, minimal allocations).
- API stability at the `vision-calibration` crate boundary and JSON schemas.

## Roadmap (near term)

- Expand pipeline coverage for rig extrinsics and laserline bundle.
- Add JSON schemas and CLI flows for additional pipelines.
- Extend validation on real datasets and regression fixtures.

## Development

- Format: `cargo fmt --all`
- Lint: `cargo clippy --workspace --all-targets --all-features -- -D warnings`
- Test: `cargo test --workspace --all-features`
- Docs: `cargo doc --workspace --no-deps`

The project is in active development. Contributions welcome; see `book/` for the evolving guide.
