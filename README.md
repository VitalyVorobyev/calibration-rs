# calibration-rs

A Rust toolbox for calibrating vision sensors (perspective and linescan) and multi-camera rigs. The project provides modern algorithms, clear abstractions, and ergonomic APIs for both research and production use, with a complete pipeline from linear initialization through non-linear refinement.

## Current Status

✅ **Production-ready components**:
- **calib-linear**: Feature-complete with 19 comprehensive tests using real stereo data
- **calib-optim**: Planar intrinsics with Brown-Conrady distortion optimization, validated on real data

🚧 **In active development**:
- calib-pipeline: Basic planar intrinsics pipeline functional
- Multi-camera rig calibration
- Bundle adjustment

## Crate Layout

- **`calib`**: Convenience facade that re-exports all sub-crates
- **`calib-core`**: Math types, composable camera models (projection, distortion, sensor), generic RANSAC engine
- **`calib-linear`**: ✅ Closed-form solvers (homography, planar pose, Zhang intrinsics, epipolar geometry, PnP DLT/P3P/EPnP, triangulation, hand-eye)
- **`calib-optim`**: ✅ Non-linear optimization with backend-agnostic IR, autodiff support, Brown-Conrady distortion (k1-k3, p1-p2)
- **`calib-pipeline`**: Ready-to-use calibration workflows with JSON I/O
- **`calib-cli`**: Command-line interface for batch processing

## Quickstart
Add the workspace or the top-level crate to your `Cargo.toml`:

```toml
calibration = { git = "https://github.com/VitalyVorobyev/calibration-rs", package = "calib" }
```

Use the high-level pipeline API:

```rust
use calib::pipeline::{run_planar_intrinsics, PlanarIntrinsicsConfig, PlanarIntrinsicsInput, PlanarViewData};
use calib::core::{IntrinsicsParams, Pt3, Vec2};

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

### Hand-eye calibration (stepwise)
Use the stepwise helpers in `calib::pipeline::handeye_single` (see
`crates/calib/examples/handeyesingle.rs` and `crates/calib/examples/handeye_session.rs`):

```rust
use calib::pipeline::handeye_single::{
    run_handeye_single, BackendSolveOptions, HandEyeMode, HandEyeSolveOptions, HandEyeView,
    IterativeIntrinsicsOptions, PlanarIntrinsicsSolveOptions, PoseRansacOptions,
};

fn main() {
    let views: Vec<HandEyeView> = /* load 2D/3D views + robot poses */;
    let report = run_handeye_single(
        &views,
        &IterativeIntrinsicsOptions::default(),
        &PlanarIntrinsicsSolveOptions::default(),
        &BackendSolveOptions::default(),
        &PoseRansacOptions::default(),
        HandEyeMode::EyeInHand,
        &HandEyeSolveOptions::default(),
        &BackendSolveOptions::default(),
    )
    .expect("hand-eye failed");

    println!(
        "final reproj error: {:.3} px",
        report.handeye_optimized.mean_reproj_error
    );
}
```

For a linear-algorithm overview and usage notes, see `crates/calib-linear/README.md`.

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

## Recent Updates

### December 2024 - January 2025
- ✅ **Brown-Conrady distortion optimization** (k1, k2, k3, p1, p2) with autodiff
- ✅ **Selective parameter fixing** for robust convergence
- ✅ **Real data validation** using stereo chessboard dataset
- ✅ **Integration tests** demonstrating 18-20% reprojection error improvement
- ✅ **Comprehensive documentation** with examples and API docs

## Implementation Status

| Component | Status | Tests | Notes |
|-----------|--------|-------|-------|
| calib-core | ✅ Complete | 4 passing | Composable camera models, RANSAC |
| calib-linear | ✅ Complete | 19 passing | All solvers validated on real stereo data |
| calib-optim | ✅ Functional | 13 passing | Planar intrinsics + distortion working |
| calib-pipeline | 🟡 Basic | 4 passing | Planar intrinsics pipeline functional |
| calib-cli | 🟡 Basic | N/A | Command-line wrapper |

**Total: 40 tests passing** across the workspace.

## Project Roadmap

### Short Term (Q1 2025)
- Polish existing documentation and examples
- Add more robust epipolar estimation variants
- Extend hand-eye calibration options

### Medium Term (Q2-Q3 2025)
- Bundle adjustment for multi-view optimization
- Multi-camera rig calibration refinement
- Additional optimization backends (trust region, Dogleg)
- Linescan camera models

### Long Term
- Rolling-shutter support
- Stereo/LiDAR-camera/IMU-camera pipelines
- Dataset I/O crate for common formats
- Benchmarking harness
- C/FFI bindings for integration with other languages

## Development
- Run tests: `cargo test`
- Format/lint: `cargo fmt && cargo clippy`
- Docs: `cargo doc --workspace --no-deps`
- CLI smoke test: `cargo run -p calib-cli -- --help`

The project is in early development—APIs may change. Contributions welcome; see the Rust book outline in `book/` for the evolving guide.
