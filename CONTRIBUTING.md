# Contributing to vision-calibration

## Architecture

```
                           ┌─────────────────────────┐
                           │    vision-calibration    │  <- Unified API facade
                           │    (public interface)    │
                           └───────────┬──────────────┘
                                       │
              ┌────────────────────────┼────────────────────────┐
              │                        │                        │
              v                        v                        v
┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────────┐
│     vc-pipeline     │  │      vc-optim       │  │      vc-linear      │
│  Session API, JSON  │  │   Non-linear BA     │  │   Linear solvers    │
│   I/O, workflows    │  │   LM optimization   │  │   Initialization    │
└─────────┬───────────┘  └─────────┬───────────┘  └────────┬────────────┘
          │                        │                       │
          └────────────────────────┼───────────────────────┘
                                   │
                                   v
                       ┌─────────────────────┐
                       │       vc-core       │  <- Math types, camera
                       │   Types, models,    │     models, RANSAC
                       │       RANSAC        │
                       └─────────────────────┘

          ┌──────────────────┐
          │    vision-mvg    │  <- Multi-view geometry pipelines
          │  Pose recovery,  │     (robust estimation, cheirality)
          │ robust estimation│
          └────────┬─────────┘
                   │
                   v
          ┌──────────────────┐
          │ vision-geometry  │  <- Low-level geometric solvers
          │ Epipolar, homo-  │     (deterministic, allocation-light)
          │ graphy, triang.  │
          └──────────────────┘
```

### Crate Summary

| Crate | Description |
|-------|-------------|
| **vision-calibration** | Facade re-exporting all sub-crates for a unified API surface |
| **vision-calibration-core** | Math types (nalgebra), composable camera models, RANSAC, synthetic data |
| **vision-calibration-linear** | Closed-form solvers: Zhang, PnP, hand-eye, laserline init |
| **vision-calibration-optim** | Non-linear LM refinement: planar intrinsics, rig, hand-eye, laserline |
| **vision-calibration-pipeline** | Session API, step functions, JSON checkpointing |
| **vision-calibration-py** | Python bindings (PyO3/maturin) for all high-level workflows |
| **vision-geometry** | Shared geometric solvers: epipolar, homography, triangulation, camera matrix |
| **vision-mvg** | Multi-view geometry: pose recovery, robust estimation, cheirality, residuals |

### Layering Rules

- `vision-calibration-linear` and `vision-calibration-optim` are peers -- they depend on `core` but not each other.
- `vision-geometry` contains low-level, deterministic solvers shared across the workspace.
- `vision-mvg` builds on `vision-geometry` for higher-level multi-view pipelines.

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

## Session Framework

All calibration workflows use `CalibrationSession<P: ProblemType>` with external step functions:

```rust
let mut session = CalibrationSession::<PlanarIntrinsicsProblem>::new();
session.set_input(data)?;
step_init(&mut session, None)?;
step_optimize(&mut session, None)?;
let result = session.export()?;
```

| Problem Type | Steps |
|---|---|
| `PlanarIntrinsicsProblem` | `step_init` -> `step_optimize` |
| `ScheimpflugIntrinsicsProblem` | `step_init` -> `step_optimize` (via `run_calibration`) |
| `SingleCamHandeyeProblem` | `step_intrinsics_init` -> `step_intrinsics_optimize` -> `step_handeye_init` -> `step_handeye_optimize` |
| `RigExtrinsicsProblem` | `step_intrinsics_init_all` -> `step_intrinsics_optimize_all` -> `step_rig_init` -> `step_rig_optimize` |
| `RigHandeyeProblem` | 6 steps: intrinsics (x2) -> rig (x2) -> hand-eye (x2) |
| `LaserlineDeviceProblem` | `step_init` -> `step_optimize` |

Each problem type also provides a `run_calibration` convenience function. Sessions support JSON serialization for checkpointing.

## Conventions

- **Poses**: `frame_se3_frame` naming. `T_C_W` = world-to-camera.
- **SE3 storage**: `[qx, qy, qz, qw, tx, ty, tz]`
- **Autodiff**: use `.clone()` liberally, `T::from_f64().unwrap()` for constants, generic `fn residual<T: RealField>()`.
- **Parameters**: grouped config structs by stage, not flat boolean bags.
- **Error handling**: `Result` for public APIs, `assert!` only for internal invariants.

## Development

```bash
cargo build --workspace              # Build
cargo test --workspace               # Test
cargo test -p vision-calibration-core  # Test one crate
cargo fmt --all                      # Format
cargo clippy --workspace --all-targets --all-features -- -D warnings  # Lint
cargo doc --workspace --no-deps      # Docs
```

### Quality Gates

Before committing:

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --all-features
cargo doc --workspace --no-deps
python3 -m compileall crates/vision-calibration-py/python/vision_calibration
```

### Testing Guidelines

- Synthetic ground-truth tests for algorithms
- JSON roundtrip for config/export types
- Loose tolerances for linear init (~5%), tight for optimization (<1%)
- Distortion k3 fixed by default (`fix_k3: true`)
- Hartley normalization for DLT, robust loss functions for outliers

### Adding a New Problem Type

1. Create module in `vision-calibration-pipeline/src/<name>/` with `mod.rs`, `problem.rs`, `state.rs`, `steps.rs`
2. Implement `ProblemType` trait (Config, Input, State, Output, Export)
3. Write step functions and `run_calibration` convenience wrapper
4. Re-export from facade crate in `vision-calibration/src/lib.rs`
5. Add Python binding in `vision-calibration-py`

## Design Decisions

ADRs are in `docs/adrs/`. Key decisions:

- **ADR 0005** -- Composable camera model
- **ADR 0006** -- Workspace layering
- **ADR 0007** -- Session framework
- **ADR 0008** -- Optimization IR
- **ADR 0009** -- Naming conventions

## Docs

- API docs: https://vitalyvorobyev.github.io/calibration/
- Book sources: `book/`
- Examples: `crates/vision-calibration/examples/`
