# vision-calibration

[![CI](https://github.com/VitalyVorobyev/calibration-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/VitalyVorobyev/calibration-rs/actions/workflows/ci.yml)
[![Docs](https://github.com/VitalyVorobyev/calibration-rs/actions/workflows/publish-docs.yml/badge.svg)](https://vitalyvorobyev.github.io/calibration/)
[![Audit](https://github.com/VitalyVorobyev/calibration-rs/actions/workflows/audit.yml/badge.svg)](https://github.com/VitalyVorobyev/calibration-rs/actions/workflows/audit.yml)

A Rust workspace for end-to-end camera calibration: math primitives, linear solvers, non-linear
refinement, and session-based pipelines. Supports perspective cameras, laserline calibration,
multi-camera rigs, and hand-eye calibration.

## Architecture

```
                           ┌─────────────────────────┐
                           │    vision-calibration   │  ◄── Unified API facade
                           │    (public interface)   │
                           └───────────┬─────────────┘
                                       │
              ┌────────────────────────┼────────────────────────┐
              │                        │                        │
              ▼                        ▼                        ▼
┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────────┐
│     vc-pipeline     │  │      vc-optim       │  │      vc-linear      │
│  Session API, JSON  │  │   Non-linear BA     │  │   Linear solvers    │
│   I/O, workflows    │  │   LM optimization   │  │   Initialization    │
└─────────┬───────────┘  └─────────┬───────────┘  └─────────┬───────────┘
          │                        │                        │
          └────────────────────────┼────────────────────────┘
                                   │
                                   ▼
                       ┌─────────────────────┐
                       │       vc-core       │  ◄── Math types, camera
                       │   Types, models,    │      models, RANSAC
                       │       RANSAC        │
                       └─────────────────────┘
```

## Crate Summary

| Crate | Description |
|-------|-------------|
| **vision-calibration** | Facade re-exporting all sub-crates for a unified API surface |
| **vision-calibration-core** | Math types (nalgebra), composable camera models, RANSAC, synthetic data |
| **vision-calibration-linear** | Closed-form solvers: homography, Zhang, PnP, epipolar, hand-eye, laserline |
| **vision-calibration-optim** | Non-linear LM refinement: planar intrinsics, rig, hand-eye, laserline |
| **vision-calibration-pipeline** | Session API, step functions, JSON checkpointing |
| **vision-calibration-py** | Python bindings (PyO3/maturin) for all high-level workflows |

## Quick Start

Add the facade crate to your `Cargo.toml`:

```toml
vision-calibration = { git = "https://github.com/VitalyVorobyev/calibration-rs" }
```

The facade is module-first: prefer `vision_calibration::<workflow>::...` namespaces
or `vision_calibration::prelude::*` rather than relying on broad top-level symbols.

### Python Package

Build and install the local Python package:

```bash
maturin develop -m crates/vision-calibration-py/Cargo.toml
```

Use the Python API:

```python
import vision_calibration as vc

print(vc.__version__)

obs = vc.Observation(
    points_3d=[(0.0, 0.0, 0.0), (0.1, 0.0, 0.0), (0.1, 0.1, 0.0), (0.0, 0.1, 0.0)],
    points_2d=[(100.0, 100.0), (200.0, 100.0), (200.0, 200.0), (100.0, 200.0)],
)
dataset = vc.PlanarDataset(views=[vc.PlanarView(observation=obs)] * 3)
config = vc.PlanarCalibrationConfig(max_iters=80, robust_loss=vc.robust_huber(1.0))
result = vc.run_planar_intrinsics(dataset, config)
print(result.mean_reproj_error)
```

### Planar Intrinsics Calibration

```rust,no_run
use vision_calibration::prelude::*;
use vision_calibration::planar_intrinsics::{step_init, step_optimize};

fn main() -> anyhow::Result<()> {
    let dataset: PlanarDataset = todo!("load calibration data");

    let mut session = CalibrationSession::<PlanarIntrinsicsProblem>::new();
    session.set_input(dataset)?;

    step_init(&mut session, None)?;
    step_optimize(&mut session, None)?;

    let result = session.export()?;
    println!("Camera: {:?}", result.params.camera);
    Ok(())
}
```

### Laserline Device Calibration

```rust,no_run
use vision_calibration::prelude::*;
use vision_calibration::laserline_device::run_calibration;

fn main() -> anyhow::Result<()> {
    let input = todo!("load laserline calibration data");

    let mut session = CalibrationSession::<LaserlineDeviceProblem>::new();
    session.set_input(input)?;
    run_calibration(&mut session, None)?;

    let export = session.export()?;
    Ok(())
}
```

### Single-Camera Hand-Eye Calibration

```rust,no_run
use vision_calibration::prelude::*;
use vision_calibration::single_cam_handeye::{
    SingleCamHandeyeInput, SingleCamHandeyeView, HandeyeMeta,
    step_intrinsics_init, step_intrinsics_optimize,
    step_handeye_init, step_handeye_optimize,
};

fn main() -> anyhow::Result<()> {
    let input: SingleCamHandeyeInput = todo!("load hand-eye data");

    let mut session = CalibrationSession::<SingleCamHandeyeProblem>::new();
    session.set_input(input)?;

    // 4-step calibration: intrinsics init/optimize, then hand-eye init/optimize
    step_intrinsics_init(&mut session, None)?;
    step_intrinsics_optimize(&mut session, None)?;
    step_handeye_init(&mut session, None)?;
    step_handeye_optimize(&mut session, None)?;

    let export = session.export()?;
    println!("Reprojection error: {:.4} px", export.mean_reproj_error);
    Ok(())
}
```

### Synthetic Data Generation

For tests and benchmarking you can generate deterministic synthetic correspondences:

```rust,no_run
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

## Session API

All calibration workflows use the `CalibrationSession` state container with problem-specific
step functions. Each problem type defines its own sequence of steps:

| Problem Type | Steps |
|---|---|
| `PlanarIntrinsicsProblem` | `step_init` → `step_optimize` |
| `SingleCamHandeyeProblem` | `step_intrinsics_init` → `step_intrinsics_optimize` → `step_handeye_init` → `step_handeye_optimize` |
| `RigExtrinsicsProblem` | `step_intrinsics_init_all` → `step_intrinsics_optimize_all` → `step_rig_init` → `step_rig_optimize` |
| `RigHandeyeProblem` | 6 steps: intrinsics (×2) → rig (×2) → hand-eye (×2) |
| `LaserlineDeviceProblem` | `step_init` → `step_optimize` |

Each problem type also provides a `run_calibration` convenience function that runs all steps.
Sessions support JSON serialization for checkpointing and resuming.

For larger workflows, configs are grouped by responsibility (e.g. `init`, `solver`,
`optimize`, `handeye_ba`) instead of large flat option bags.

## Examples

Run examples with:

```bash
cargo run -p vision-calibration --example planar_synthetic    # Synthetic planar intrinsics
cargo run -p vision-calibration --example planar_real         # Real stereo images
cargo run -p vision-calibration --example stereo_session      # Stereo rig extrinsics
cargo run -p vision-calibration --example stereo_charuco_session  # Stereo ChArUco rig extrinsics
cargo run -p vision-calibration --example handeye_synthetic   # Single-camera hand-eye
cargo run -p vision-calibration --example handeye_session     # KUKA robot data
cargo run -p vision-calibration --example rig_handeye_synthetic  # Multi-camera rig hand-eye
```

Python counterparts (requires local package installed into `./.venv`):

```bash
./.venv/bin/python crates/vision-calibration-py/examples/planar_synthetic.py
./.venv/bin/python crates/vision-calibration-py/examples/planar_real.py
./.venv/bin/python crates/vision-calibration-py/examples/stereo_session.py
./.venv/bin/python crates/vision-calibration-py/examples/stereo_charuco_session.py
./.venv/bin/python crates/vision-calibration-py/examples/handeye_synthetic.py
./.venv/bin/python crates/vision-calibration-py/examples/handeye_session.py
./.venv/bin/python crates/vision-calibration-py/examples/rig_handeye_synthetic.py
./.venv/bin/python crates/vision-calibration-py/examples/laserline_device_session.py
```

Notes:

- Python examples are in `crates/vision-calibration-py/examples/`.
- Real-image examples use optional Python deps:
  `./.venv/bin/python -m pip install "vision-calibration[examples]"`.

## Camera Model

`vision-calibration-core` models cameras as a composable pipeline:

```
pixel = K(sensor(distortion(projection(dir))))
```

Where:
- `projection` maps a camera-frame direction to normalized coordinates (e.g., pinhole).
- `distortion` warps normalized coordinates (Brown-Conrady radial and tangential).
- `sensor` applies a homography (identity or Scheimpflug/tilt).
- `K` maps sensor coordinates to pixels (`fx`, `fy`, `cx`, `cy`, `skew`).

## Docs

- API docs and book: https://vitalyvorobyev.github.io/calibration/
- Book sources: `book/`
- Examples: `crates/vision-calibration/examples/`

## Design Principles

- Correctness and numerical stability with explicit failure modes.
- Deterministic outputs (seeded RNGs, stable ordering).
- Performance-aware implementations (fixed-size math, minimal allocations).
- API stability at the `vision-calibration` crate boundary and JSON schemas.

## Development

```bash
cargo fmt --all                                              # Format
cargo clippy --workspace --all-targets --all-features        # Lint
cargo test --workspace --all-features                        # Test
cargo doc --workspace --no-deps                              # Build docs
python -m compileall crates/vision-calibration-py/python/vision_calibration
```
