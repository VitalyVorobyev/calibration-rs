# vision-calibration

[![CI](https://github.com/VitalyVorobyev/calibration-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/VitalyVorobyev/calibration-rs/actions/workflows/ci.yml)
[![Docs](https://github.com/VitalyVorobyev/calibration-rs/actions/workflows/publish-docs.yml/badge.svg)](https://vitalyvorobyev.github.io/calibration/)
[![Audit](https://github.com/VitalyVorobyev/calibration-rs/actions/workflows/audit.yml/badge.svg)](https://github.com/VitalyVorobyev/calibration-rs/actions/workflows/audit.yml)

End-to-end camera calibration in Rust. Supports planar intrinsics, Scheimpflug tilt, multi-camera rigs, hand-eye calibration, and laserline devices.

Also includes standalone multi-view geometry crates (`vision-geometry`, `vision-mvg`) for epipolar geometry, homography estimation, triangulation, and robust pose recovery.

## Install

### Rust

```toml
# Calibration workflows
vision-calibration = "0.2"

# Multi-view geometry (independent of calibration)
vision-mvg = "0.2"

# Low-level geometric solvers
vision-geometry = "0.2"
```

### Python

```bash
pip install vision-calibration
```

## Quick Start (Rust)

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

## Quick Start (Python)

```python
import vision_calibration as vc

obs = vc.Observation(
    points_3d=[(0.0, 0.0, 0.0), (0.1, 0.0, 0.0), (0.1, 0.1, 0.0), (0.0, 0.1, 0.0)],
    points_2d=[(100.0, 100.0), (200.0, 100.0), (200.0, 200.0), (100.0, 200.0)],
)
dataset = vc.PlanarDataset(views=[vc.PlanarView(observation=obs)] * 3)
result = vc.run_planar_intrinsics(dataset)
print(result.mean_reproj_error)
```

## Calibration Workflows

| Workflow | Description |
|----------|-------------|
| Planar intrinsics | Zhang's method with bundle adjustment |
| Scheimpflug intrinsics | Tilt-lens cameras |
| Single-camera hand-eye | Eye-in-hand with robot poses |
| Multi-camera rig | Extrinsic calibration across cameras |
| Rig hand-eye | Full rig + robot calibration |
| Laserline device | Laser triangulation sensors |

## Multi-View Geometry

The `vision-mvg` and `vision-geometry` crates can be used independently of the calibration pipeline:

- Fundamental and essential matrix estimation (7-point, 8-point, 5-point)
- Homography estimation and decomposition
- Robust pose recovery with RANSAC
- Linear triangulation
- Camera matrix DLT and RQ decomposition

## Examples

```bash
cargo run -p vision-calibration --example planar_synthetic
cargo run -p vision-calibration --example handeye_synthetic
cargo run -p vision-calibration --example stereo_session
```

See `crates/vision-calibration/examples/` for all Rust examples and `crates/vision-calibration-py/examples/` for Python.

## Docs

- [API reference](https://vitalyvorobyev.github.io/calibration/)
- [Contributing & architecture](CONTRIBUTING.md)

## Diligence Statement

This project is developed with AI coding assistants (`Codex` and `Claude Code`) as implementation tools.
Not every code path is manually line-reviewed by a human before merge. The project author is an expert in
computer vision, validates algorithmic behavior and numerical results, and enforces quality gates
(`fmt`/`clippy`/tests/docs/Python checks) before release.

## License

MIT
