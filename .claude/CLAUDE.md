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

5-crate layered workspace (~7.2k LoC). See ADR 0006.

```
vision-calibration (facade) → vision-calibration-pipeline (sessions, workflows)
                                    ↓
                    vision-calibration-optim + vision-calibration-linear  (peers, no cross-dep)
                                    ↓
                            vision-calibration-core (types, models, RANSAC)
```

Plus `vision-calibration-py` (PyO3 bindings, depends on facade only).

**Key rule**: linear and optim are peers — they depend on core but not each other.

## Camera Model (ADR 0005)

Composable pipeline: `pixel = K(sensor(distortion(projection(dir))))`. Each stage is a generic type parameter on `Camera<P, D, S, K>`.

## Session Framework (ADR 0007)

All workflows use `CalibrationSession<P: ProblemType>` with external step functions.

**Standard pipeline** (auto-initialization):
```rust
let mut session = CalibrationSession::<PlanarIntrinsicsProblem>::new();
session.set_input(data)?;
step_init(&mut session, None)?;
step_optimize(&mut session, None)?;
let result = session.export()?;
```

**Expert pipeline** (manual initialization seeds, ADR 0011):
```rust
step_set_init(&mut session, PlanarManualInit {
    intrinsics: Some(nominal_k),   // from datasheet
    ..Default::default()           // auto-init distortion and poses
}, None)?;
step_optimize(&mut session, None)?;
```

`step_init` is always a one-liner delegating to `step_set_init` with all-`None` defaults.
`step_optimize` is unchanged — its precondition (state is fully initialized) is preserved.

Six problem types: `PlanarIntrinsics`, `ScheimpflugIntrinsics`, `SingleCamHandeye`, `RigExtrinsics`, `RigHandeye`, `LaserlineDevice`.

## Manual Initialization Pattern (ADR 0011)

Each problem type exposes a `XxxManualInit` struct and `step_set_init` function:

```rust
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct PlanarManualInit {
    pub intrinsics: Option<FxFyCxCySkew<Real>>,
    pub distortion: Option<BrownConrady5<Real>>,
    pub poses:      Option<Vec<Iso3>>,
}
```

Rules:
- All fields `Option<T>` — `None` = auto-initialize, `Some` = use directly.
- **No `#[non_exhaustive]`** — breaks `..Default::default()` ergonomics.
- **Serde derives required** — needed for Python binding deserialization via `pythonize`.
- **Intrinsics-pose coupling**: when `intrinsics` is `Some` but `poses` is `None`, poses are recovered using the **manual** intrinsics (not auto-estimated).
- **Sensor not in ManualInit** for `LaserlineDevice` — sensor model is always taken from `session.config` (hardware property).
- **RigExtrinsics coupling**: `cam_se3_rig` and `rig_se3_target` must be both-or-neither.

Log the init source in `log_success_with_notes`: `"(auto)"` vs `"(manual: intrinsics, poses)"`.

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
3. Write step functions in `steps.rs`:
   - `XxxManualInit` struct (all-`Option<T>`, derive `Debug + Clone + Default + Serialize + Deserialize`)
   - `step_set_init(session, XxxManualInit, opts)` — seeds provided fields, auto-initializes the rest
   - `step_init(session, opts)` — delegates: `step_set_init(session, XxxManualInit::default(), opts)`
   - `step_optimize(session, opts)` — unchanged, operates on whatever `step_set_init` wrote to state
   - `run_calibration(session, config)` — convenience: `step_init` → `step_optimize`
4. Re-export all types and step functions from `mod.rs` and facade `vision-calibration/src/lib.rs`
5. Add Python binding:
   - Native Rust function in `vision-calibration-py/src/lib.rs` (uses `run_problem` helper)
   - `run_<name>_with_init` native function accepting a manual init payload
   - Python dataclass with `to_payload()` in `models.py`
   - High-level typed wrapper in `_api.py`
   - Exports in `__init__.py`

## Test Expectations per Problem Type

For each new `step_set_init`, write 3 tests:
1. **Exact seeds**: GT params → `step_set_init` → `step_optimize` → reproj < 1e-3 px
2. **Perturbed seeds**: ~10% off GT → converges to reproj < 1.0 px
3. **Default equivalence**: `step_set_init(default)` == `step_init` (regression guard)

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
