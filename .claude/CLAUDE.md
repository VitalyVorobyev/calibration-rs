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

6-crate workspace (~24k LoC Rust). See ADR 0006.

```
vision-calibration (facade) → vision-calibration-pipeline (sessions, workflows)
                                    ↓
                    vision-calibration-optim + vision-calibration-linear  (peers, no cross-dep)
                                    ↓
                            vision-calibration-core (types, models, RANSAC)
```

Plus `vision-calibration-py` (PyO3 bindings, depends on facade only).

**Key rule**: linear and optim are peers — they depend on core but not each other.

### Feature Flags

- `vision-calibration-core`: optional `tracing` feature enables `tracing` crate instrumentation (off by default).
- All other crates: `default = []`, no public feature flags.

### Python Bindings

Built with [maturin](https://www.maturin.rs/) + PyO3 0.28 (`abi3-py310`, cdylib `_vision_calibration`). Dev build:

```bash
maturin develop -m crates/vision-calibration-py/Cargo.toml
```

Published to PyPI via the `release-pypi.yml` GitHub Actions workflow.

## Camera Model (ADR 0005)

Composable pipeline: `pixel = K(sensor(distortion(projection(dir))))`. Defined as `Camera<S, P, D, Sm, K>` where `S` is the scalar (`RealField + Copy`), `P` projection, `D` distortion, `Sm` sensor, `K` intrinsics.

## Session Framework (ADR 0007)

All workflows use `CalibrationSession<P: ProblemType>` with external step functions. Pattern:

```rust
let mut session = CalibrationSession::<PlanarIntrinsicsProblem>::new();
session.set_input(data)?;
step_init(&mut session, None)?;
step_optimize(&mut session, None)?;
let result = session.export()?;
```

Nine problem types: `PlanarIntrinsics`, `ScheimpflugIntrinsics`, `SingleCamHandeye`, `RigExtrinsics`, `RigHandeye`, `LaserlineDevice`, plus the Scheimpflug family — `RigScheimpflugExtrinsics`, `RigScheimpflugHandeye` (EyeInHand), and `RigLaserlineDevice`.

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

## MSRV

Workspace MSRV: **1.88**. Some transitive deps (`fixed`, `kiddo`) are
pinned in `Cargo.lock` below their latest release to stay compatible.
**Do not run `cargo update` without reading `docs/MSRV.md`** — it will
silently bump deps past 1.88 and break the `MSRV (1.88)` CI job. The
job uses `--locked` so drift fails at PR time, but the lockfile must
be re-pinned manually after any update.

## Planning

- 10 ADRs (0001–0010) in `docs/adrs/` — design decisions (see README there)
- Automated workflow skills: `/orchestrate`, `/architect`, `/implement`, `/review`, `/gate-check`

## Strategic Roadmap (>40 weeks)

We work to a multi-quarter, four-track plan summarized in `docs/ROADMAP.md`. The
load-bearing path is **A1 → A2 → B5**:

- **A — Calibration core** (puzzle-rig-anchored). A1 = manual init (this is PR #32,
  superseding PR #27); A2 = per-feature residuals on every `*Export`; A3 = Zhang init
  robustness; A4 = EyeToHand for Scheimpflug; A5 = Python parity; A6 = rig_family
  refactor.
- **B — Tauri 2 + React + TypeScript desktop app**. B0 scaffold → B1 file load →
  B2 detection wrap → B3 calibration runner → B4 3D rig viewer → **B5 diagnose mode
  (the MVP)** → B6 polish.
- **C — MVG** (postponed until B5). C1 PR #28 land → C2 N-view triangulation →
  C3 BA frozen-intrinsics → C4 Scheimpflug-aware rectification → C5 dense matcher
  (opencv-rust SGBM, feature-flagged).
- **D — Earn v1.0** (continuous ratchet). Typed errors → doc-warning-free → Python
  parity audit → v1.0 release.

We are pre-1.0; breaking changes are acceptable.
