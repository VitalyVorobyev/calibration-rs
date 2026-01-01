# calibration-rs

A Rust toolbox for calibrating vision sensors (perspective and linescan) and multi-camera rigs. The project aims to provide modern algorithms, clear abstractions, and ergonomic APIs for both research and production use. Linear initialization blocks are available today; non-linear refinement and full pipelines are being built on top.

## Crate layout
- `calib`: convenience facade that re-exports all sub-crates.
- `calib-core`: math aliases, composable camera models (projection, distortion, sensor), and a generic RANSAC engine.
- `calib-linear`: classic closed-form solvers (homography, planar pose, Zhang intrinsics, epipolar geometry, rig extrinsics, hand–eye).
- `calib-optim`: non-linear least-squares traits and backends (currently LM), robust kernels, and problem definitions.
- `calib-pipeline`: ready-to-use calibration pipelines; currently planar intrinsics (Zhang-style) with LM refinement.
- `calib-cli`: small CLI wrapper around `calib-pipeline` for batch / scripting workflows.

## Quickstart
Add the workspace or the top-level crate to your `Cargo.toml`:

```toml
calibration = { git = "https://github.com/VitalyVorobyev/calibration-rs", package = "calib" }
```

Use the high-level pipeline API:

```rust
use calib::pipeline::{run_planar_intrinsics, PlanarIntrinsicsConfig, PlanarIntrinsicsInput, PlanarViewData};
use calib::core::{IntrinsicsConfig, Pt3, Vec2};

fn main() {
    // Populate per-view correspondences (normally from a detector)
    let board = vec![
        Pt3::new(0.0, 0.0, 0.0),
        Pt3::new(0.1, 0.0, 0.0),
        Pt3::new(0.1, 0.1, 0.0),
        Pt3::new(0.0, 0.1, 0.0),
    ];
    let view = PlanarViewData {
        points_3d: board.clone(),
        points_2d: vec![Vec2::new(100.0, 120.0), Vec2::new(180.0, 118.0), Vec2::new(182.0, 192.0), Vec2::new(98.0, 196.0)],
    };
    let input = PlanarIntrinsicsInput { views: vec![view] };
    let config = PlanarIntrinsicsConfig::default();

    let report = run_planar_intrinsics(&input, &config).expect("planar intrinsics failed");
    println!("Estimated camera config: {:?}", report.camera);

    if let IntrinsicsConfig::FxFyCxCySkew { fx, fy, cx, cy, skew } = &report.camera.intrinsics {
        println!("Estimated intrinsics: fx={fx} fy={fy} cx={cx} cy={cy} skew={skew}");
    }
}
```

## Camera model
`calib-core` models cameras as a composable pipeline:

```
pixel = K ∘ sensor ∘ distortion ∘ projection(dir)
```

Where:
- `projection` maps a camera-frame direction to normalized coordinates (e.g., pinhole).
- `distortion` warps normalized coordinates (Brown–Conrady radial/tangential).
- `sensor` applies a homography (identity or Scheimpflug/tilt).
- `K` maps sensor coordinates to pixels (`fx`, `fy`, `cx`, `cy`, `skew`).

`SensorConfig::Scheimpflug` follows OpenCV’s tilted sensor model (`tau_x`, `tau_y`), implemented as the same homography computed by OpenCV’s `computeTiltProjectionMatrix`.

CLI usage for batch jobs:

```bash
cargo run -p calib-cli -- --input views.json --config config.json > report.json
```

## Design principles
- Modern, correct algorithms with clear numerical assumptions.
- Strong separation between math primitives (`calib-core`), initialization (`calib-linear`), and refinement (`calib-optim` / `calib-pipeline`).
- Testable components with synthetic checks and JSON roundtrips.
- Ergonomics first: simple data structures, serde support, and a stable public surface.

## Project roadmap (high level)
- Short term: polish linear blocks (hand–eye variants, robust epipolar estimation), improve documentation, ship examples and datasets for planar intrinsics.
- Near term: extend optimization backends (trust region, Dogleg), add bundle-adjustment style problems, and multi-camera rig refinement.
- Medium term: linescan-specific models, rolling-shutter support, and calibration report generation.
- Longer term: richer pipelines (stereo, LiDAR-camera, IMU-camera), dataset IO crate, benchmarking harness, and C/FFI bindings.

## Development
- Run tests: `cargo test`
- Format/lint: `cargo fmt && cargo clippy`
- Docs: `cargo doc --workspace --no-deps`
- CLI smoke test: `cargo run -p calib-cli -- --help`

The project is in early development—APIs may change. Contributions welcome; see the Rust book outline in `book/` for the evolving guide.
