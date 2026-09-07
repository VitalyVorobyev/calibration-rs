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

## Desktop app (`app/`) commands

Track B Tauri 2 + React + TypeScript viewer. **Always use bun**, never
`npm`/`pnpm`/`yarn`. The committed lockfile is `bun.lock`, and
`tauri.conf.json` `beforeDevCommand`/`beforeBuildCommand` must invoke
`bun run …`.

```bash
cd app
bun install            # First-time setup
bun run tauri dev      # Launch the desktop app (NOT `bun run dev` — that
                       #   starts Vite only and the Tauri APIs are absent)
bun run build          # TS compile + Vite build (frontend only)
bun run tauri build    # Bundle the desktop app
```

## Engineering principles & critical review

Hold every change to a high design bar, and actively keep this workspace
from drifting back into undisciplined, copy-paste "agentic slop." Apply
these principles by default — not only when explicitly asked:

- **SOLID** — one responsibility per type/module; extend behavior through
  the detector / refiner / orientation traits, not by editing parallel
  match arms; depend on trait abstractions (`DenseDetector`, the refiner
  traits), not concretions.
- **DRY / single source of truth** — one canonical definition per concept.
  Lower config to core params in exactly one place; never let a
  threshold's or parameter's meaning be duplicated or diverge across
  crates.
- **KISS & YAGNI** — choose the simplest design that meets the actual
  requirement; do not add config knobs, enum variants, or abstraction
  layers for hypothetical future needs.
- **Make illegal states unrepresentable** — push invariants into the type
  system (enum-with-payload over a discriminator + parallel fields;
  `Option` / newtypes over magic sentinels) so misuse fails to compile.
- **Least astonishment & orthogonality** — APIs do what their names say;
  keep independent concerns independent (orientation is a cross-cutting
  stage, not a sub-mode of one detector).
- **Minimal, honest public surface** — expose only what callers need; keep
  diagnostics and internals out of the stable API (see *Public surface
  hygiene*).

**Be critical of every proposal — including the user's.** Treat a request
as the start of a design discussion, not an instruction to implement
verbatim. Before coding, review it against the principles above and the
existing architecture; if it would introduce duplication, leak internals,
add a dominated alternative, widen the public surface needlessly, or
otherwise degrade the design, **say so and offer the cleaner alternative**
rather than building it as-stated. When the better design needs a breaking
change and the crate is pre-1.0, prefer the better design (see *Decisive
cleanup*). Every change should leave the workspace's design and style
better than it found them.

## Architecture

Workspace of 11 crates + `xtask` (dev tooling). Nine ship to crates.io in
dependency order — `vision-calibration-core` → `vision-geometry` →
`vision-calibration-dataset`/`vision-calibration-detect` →
`vision-calibration-linear`/`vision-calibration-optim` → `vision-mvg` →
`vision-calibration-pipeline` → `vision-calibration` (facade — see
`release.yml`'s publish DAG). `vision-calibration-py` ships to PyPI instead
(crates.io `publish = false`); `vision-calibration-bench` is
workspace-internal (`publish = false`); `vision-calibration-examples-private`
lives *outside* this workspace entirely (its own `[workspace]`), path-pinning
the published crates for private end-to-end examples. See ADR 0006 (amended
2026-07-04 for the geometry/mvg edges below).

```
vision-calibration (facade) → vision-calibration-pipeline (sessions, workflows)
                                    ↓
                    vision-calibration-optim + vision-calibration-linear  (peers, no cross-dep)
                                    ↓
                            vision-calibration-core (types, models, RANSAC)

vision-geometry (two-view solvers: homography, epipolar, camera_matrix,
triangulation) ← depended on by linear, pipeline, and the facade
        ↓
vision-mvg (N-view MVG: bundle adjust, rectification, dense stereo) ← facade
```

Plus `vision-calibration-py` (PyO3 bindings, depends on facade only).

**Key rule**: linear and optim are peers — they depend on core but not each other.

### Feature Flags

- `vision-calibration-core`: optional `tracing` feature enables `tracing` crate instrumentation (off by default).
- All other crates: `default = []`, no public feature flags.

### Python Bindings

Built with [maturin](https://www.maturin.rs/) + PyO3 0.29 (`abi3-py310`, cdylib `_vision_calibration`). Dev build:

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

Eight problem types (post A6 sensor-axis collapse, see [ADR 0013](../docs/adrs/0013-rig-family-sensor-axis-refactor.md)): `PlanarIntrinsics`, `ScheimpflugIntrinsics`, `SingleCamHandeye`, `LaserlineDevice`, `RigExtrinsics`, `RigHandeye`, `RigLaserlineDevice`, and `RigHandeyeLaserline`. The two rig problem types (`RigExtrinsics`, `RigHandeye`) cover both pinhole and Scheimpflug rigs via `RigExtrinsicsConfig::sensor` / `RigHandeyeConfig::sensor` (`SensorMode::Pinhole` | `SensorMode::Scheimpflug { … }`). `RigLaserlineDevice` accepts a frozen rig hand-eye export (pinhole or Scheimpflug) as upstream calibration; `RigHandeyeLaserline` solves the same chain jointly.

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

Workspace MSRV: **1.93**. CI gate: `MSRV (1.93)` in
`.github/workflows/ci.yml`. History and bump policy live in
[`docs/MSRV.md`](../docs/MSRV.md).

## Releasing

[`docs/RELEASE-RUNBOOK.md`](../docs/RELEASE-RUNBOOK.md) is the source of
truth: the four version sources that must move in one commit, the crates.io
publish DAG, the per-crate Trusted Publishing setup, the pre-tag checklist,
and how to read a `cargo publish --dry-run` failure during a lockstep bump.
Read it before touching a version string — it also records why each rule
exists, which is what stops the next release repeating an old failure.

## Planning

- **Backlog workflow** — AGENTS.md §11 is the source of truth: one task per
  commit, `docs/backlog.md` holds only open and parked work, and a completed
  task's dated note moves to `docs/backlog-archive.md` as the durable record.
- ADRs in `docs/adrs/` — design decisions (see README there). 0011 covers
  manual init, 0012 covers per-feature residuals, 0013 covers the
  `rig_family` sensor-axis refactor (the Scheimpflug rig modules collapse).
- Tutorials in `docs/tutorials/` — hands-on onboarding for new users. New
  features should ship with a tutorial entry.
- Automated workflow skills: `/orchestrate`, `/architect`, `/implement`, `/review`, `/gate-check`

## Strategic Roadmap

[`docs/ROADMAP.md`](../docs/ROADMAP.md) is authoritative — read it rather
than working from a summary here, which is exactly what goes stale. Open
tasks live in [`docs/backlog.md`](../docs/backlog.md); completed ones, with
their durable completion notes, in
[`docs/backlog-archive.md`](../docs/backlog-archive.md).

We are pre-1.0; breaking changes are acceptable.
