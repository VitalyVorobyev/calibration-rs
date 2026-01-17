# calibration-rs

[![CI](https://github.com/VitalyVorobyev/calibration-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/VitalyVorobyev/calibration-rs/actions/workflows/ci.yml)
[![Docs](https://github.com/VitalyVorobyev/calibration-rs/actions/workflows/publish-docs.yml/badge.svg)](https://vitalyvorobyev.github.io/calibration/)
[![Audit](https://github.com/VitalyVorobyev/calibration-rs/actions/workflows/audit.yml/badge.svg)](https://github.com/VitalyVorobyev/calibration-rs/actions/workflows/audit.yml)

A Rust workspace for end-to-end camera calibration: math primitives, linear solvers, non-linear
refinement, pipelines, and a CLI. Supports perspective and linescan sensors and multi-camera rigs.

## Status

- Stable foundation: core math types, camera models, deterministic RANSAC, and linear solvers.
- Working pipelines: planar intrinsics with Brown-Conrady distortion and stepwise hand-eye for
  single-camera setups.
- Optimization: planar intrinsics, hand-eye, rig extrinsics, and linescan bundle problems with a
  Levenberg-Marquardt backend.
- In progress: broader pipeline coverage (rig and linescan), more CLI commands, and additional
  real-data validation.
- API stability: `calib` is the compatibility boundary; lower crates may evolve.

## Crate layout

- `calib`: facade re-exporting all sub-crates for a stable API surface
- `calib-core`: math types, camera models, deterministic RANSAC
- `calib-linear`: closed-form solvers (homography, PnP, epipolar, triangulation, hand-eye, rig)
- `calib-optim`: non-linear refinement (planar intrinsics, rig extrinsics, hand-eye, linescan)
- `calib-pipeline`: end-to-end workflows, session API, JSON I/O
- `calib-cli`: command-line wrapper for batch planar intrinsics

## Quickstart

Add the facade crate to your `Cargo.toml`:

```toml
calib = { git = "https://github.com/VitalyVorobyev/calibration-rs" }
```

Use the high-level pipeline API:

```rust
use calib::pipeline::{run_planar_intrinsics, CameraViewData, PlanarIntrinsicsConfig, PlanarIntrinsicsInput};
use calib::core::{IntrinsicsParams, Pt3, Vec2};

fn main() {
    // Populate per-view correspondences (normally from a detector)
    let board = vec![
        Pt3::new(0.0, 0.0, 0.0),
        Pt3::new(0.1, 0.0, 0.0),
        Pt3::new(0.1, 0.1, 0.0),
        Pt3::new(0.0, 0.1, 0.0),
    ];
    let view = CameraViewData {
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

For checkpointed workflows, see `calib::session` and `calib::pipeline::session`.

### Hand-eye calibration (stepwise)

Use the stepwise helpers in `calib::pipeline::handeye_single` (see
`crates/calib/examples/handeyesingle.rs` and `crates/calib/examples/handeye_session.rs`):

```rust
use calib::pipeline::handeye_single::{run_handeye_single, HandEyeSingleOptions, HandEyeView};

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

For a linear-algorithm overview and usage notes, see `crates/calib-linear/README.md`.

### Hand-eye calibration (session)

Use the dedicated session problem for a compact workflow (see
`crates/calib/examples/handeye_session.rs`):

```rust
use calib::session::{
    CalibrationSession, HandEyeModeConfig, HandEyeSingleInitOptions, HandEyeSingleObservations,
    HandEyeSingleOptimOptions, HandEyeSingleProblem,
};
use calib::pipeline::handeye_single::HandEyeView;

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

`calib-core` models cameras as a composable pipeline:

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

## CLI usage

```bash
cargo run -p calib-cli -- --input views.json --config config.json > report.json
```

## Docs

- API docs and book: https://vitalyvorobyev.github.io/calibration/
- Book sources: `book/`
- Examples: `crates/calib/examples/` and `crates/calib-pipeline/examples/`

## Design principles

- Correctness and numerical stability with explicit failure modes.
- Deterministic outputs (seeded RNGs, stable ordering).
- Performance-aware implementations (fixed-size math, minimal allocations).
- API stability at the `calib` crate boundary and JSON schemas.

## Roadmap (near term)

- Expand pipeline coverage for rig extrinsics and linescan bundle.
- Add JSON schemas and CLI flows for additional pipelines.
- Extend validation on real datasets and regression fixtures.

## Development

- Format: `cargo fmt --all`
- Lint: `cargo clippy --workspace --all-targets --all-features -- -D warnings`
- Test: `cargo test --workspace --all-features`
- Docs: `cargo doc --workspace --no-deps`

The project is in active development. Contributions welcome; see `book/` for the evolving guide.
