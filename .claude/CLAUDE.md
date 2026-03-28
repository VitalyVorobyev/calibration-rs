# CLAUDE.md

## Commands

```bash
cargo build --workspace              # Build
cargo test --workspace               # Test
cargo test -p vision-calibration-core  # Test one crate
cargo fmt --all                      # Format
cargo clippy --workspace --all-targets --all-features -- -D warnings  # Lint
cargo doc --workspace --no-deps      # Docs
```

## Architecture

7-crate layered workspace. See ADR 0006.

```
vision-calibration (facade) → vision-calibration-pipeline (sessions, workflows)
                                    ↓
                    vision-calibration-optim + vision-calibration-linear  (peers, no cross-dep)
                                    ↓               ↓
                            vision-calibration-core   vision-mvg → vision-geometry
                            (types, models, RANSAC)   (MVG pipelines)  (low-level solvers)
```

Plus `vision-calibration-py` (PyO3 bindings, depends on facade only).

**Key rules**:
- linear and optim are peers — they depend on core but not each other.
- `vision-geometry` has low-level deterministic solvers (epipolar, homography, triangulation, camera matrix).
- `vision-mvg` builds on vision-geometry for multi-view pipelines (pose recovery, robust estimation, cheirality).
- vision-calibration-linear depends on vision-geometry for shared solvers.

## Camera Model (ADR 0005)

Composable pipeline: `pixel = K(sensor(distortion(projection(dir))))`. Each stage is a generic type parameter on `Camera<P, D, S, K>`.

## Session Framework (ADR 0007)

All workflows use `CalibrationSession<P: ProblemType>` with external step functions. Pattern:

```rust
let mut session = CalibrationSession::<PlanarIntrinsicsProblem>::new();
session.set_input(data)?;
step_init(&mut session, None)?;
step_optimize(&mut session, None)?;
let result = session.export()?;
```

Six problem types: `PlanarIntrinsics`, `ScheimpflugIntrinsics`, `SingleCamHandeye`, `RigExtrinsics`, `RigHandeye`, `LaserlineDevice`.

## Optimization IR (ADR 0008)

Problems defined as `ProblemIR` (param blocks + residual blocks), compiled to solver backends. Factor functions are generic over `T: RealField` for autodiff.

## Conventions (ADR 0009)

- **Poses**: `frame_se3_frame` naming. `T_C_W` = world-to-camera.
- **SE3 storage**: `[qx, qy, qz, qw, tx, ty, tz]`
- **Autodiff**: use `.clone()` liberally, `T::from_f64().unwrap()` for constants, generic `fn residual<T: RealField>()`.

## Key Development Rules

- **Testing**: synthetic ground-truth tests for algorithms, JSON roundtrip for config/export types, loose tolerances for linear init (~5%), tight for optimization (<1%).
- **Distortion**: k3 fixed by default (`fix_k3: true`) — only enable for wide-angle or high-quality data.
- **Numerics**: Hartley normalization for DLT, robust loss functions for outliers, Lie group manifolds for rotations.
- **Error handling**: `Result` for public APIs, `assert!` only for internal invariants.
- **Parameters**: grouped config structs by stage, not flat boolean bags.

## Adding a New Problem Type

1. Create module in `vision-calibration-pipeline/src/<name>/` with `mod.rs`, `problem.rs`, `state.rs`, `steps.rs`
2. Implement `ProblemType` trait (Config, Input, State, Output, Export)
3. Write step functions and `run_calibration` convenience wrapper
4. Re-export from facade crate in `vision-calibration/src/lib.rs`
5. Add Python binding in `vision-calibration-py`

## Quality Gates

Before committing:

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --all-features
cargo doc --workspace --no-deps  # check for warnings
python3 -m compileall crates/vision-calibration-py/python/vision_calibration
```

## Planning

- ADRs in `docs/adrs/` — design decisions (see README there)
- Backlog in `docs/backlog.md` — task tracking with `M<n>-T<nn>` IDs
- Reports in `docs/report/` — per-task completion records
- Automated workflow: `/orchestrate`, `/architect`, `/implement`, `/review`, `/gate-check`
