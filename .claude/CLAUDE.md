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

Workspace MSRV: **1.93** (raised from 1.88 on 2026-05-23). No
transitive deps are pinned for MSRV reasons; `cargo update` is safe.
CI gate: `MSRV (1.93)` in `.github/workflows/ci.yml`. See
`docs/MSRV.md` for history and bump policy.

## Releasing — version-source lockstep

Four version sources must move together on every bump. The PR that
prepares a release (the one that the release tag will point at) must
update **all four** in the same commit, or the release tag will wedge
in CI:

1. `Cargo.toml` `workspace.package.version` (line ~21).
2. `Cargo.toml` `[workspace.dependencies]` path-dep pins for the nine
   publishable workspace crates (`vision-calibration*` plus
   `vision-geometry` and `vision-mvg`, lines ~33–45).
3. `crates/vision-calibration-py/pyproject.toml` `project.version` —
   the one that `release-pypi.yml`'s `Verify tag/version sync` job
   reads directly. **Not** wired into `[workspace.package]`.
4. `crates/vision-calibration-examples-private/Cargo.toml` (1 package
   version + 6 path-dep pins — the count grows as it gains deps; grep,
   don't trust this number). Out of the publish set but still
   compiled in CI.

**Publish set (2026-06-17; DAG corrected 2026-07-04 to match
`release.yml`):** `vision-geometry` and `vision-mvg` joined the publish
set — nine publishable crates total. Crates.io publish order follows the
dependency DAG: `vision-calibration-core` → `vision-geometry` →
`vision-calibration-dataset` → `vision-calibration-detect` →
`vision-calibration-linear` → `vision-calibration-optim` →
`vision-mvg` → `vision-calibration-pipeline` → `vision-calibration`.
(`vision-calibration-py` is `publish = false` on crates.io — it ships to
PyPI via `release-pypi.yml`.) `vision-geometry`/`vision-mvg` have never been
published, so their first crates.io version is the current workspace version
(a fresh `0.x` crate may be published at `0.6.0`); the already-published crates
only need re-publishing on the next workspace-wide version bump.

`Cargo.lock` refreshes by running `cargo build --workspace` once after
the `.toml` edits — only the workspace crate `version` strings change
(no dep resolution).

**Why this section exists.** `0.4.0` and `0.5.0` shipped to crates.io
but never reached PyPI: each release missed the `pyproject.toml` bump,
which tripped the `release-pypi.yml` verify gate and skipped the
wheel build / upload. `0.5.1` repaired this; see the fix commit for
the full failure map.

`0.6.0` exists because of a second trap: PR #67 added the public
`vision_calibration_core::linalg` module to the already-published
`core@0.5.1` **without** a version bump, so local `core@0.5.1` diverged
from the immutable crates.io `core@0.5.1`. Publishing `vision-geometry`
(which re-exports `core::linalg`) then failed `--dry-run` with `E0432`,
because publish strips path deps and resolved the *old* registry
`core@0.5.1` that has no `linalg`. **Lesson:** adding public API to an
already-published crate is a release event — it must ride a
workspace-wide version bump, never land at the same version. The whole
post-v0.5.1 DAG (PRs #66–#70) was unpublished, so `0.6.0` re-releases
all nine crates together in DAG order.

**Pre-tag local check** (catches the four most common release breakages
before the tag is pushed):

```bash
# All four version sources must print the same vX.Y.Z.
grep -RHn 'version = "' Cargo.toml \
    crates/vision-calibration-py/pyproject.toml \
    crates/vision-calibration-examples-private/Cargo.toml \
  | grep -v 'edition\|rust-version'

# The publish-docs.yml gate (RUSTDOCFLAGS=-D warnings) only runs on
# push-to-main, not on PRs. Run it locally before tagging — it catches
# broken intra-doc-links that the ci.yml `cargo doc` step swallows.
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --all-features --no-deps

# Confirm the pyo3 build still passes — the release-pypi.yml verify
# job also rebuilds the extension before the wheel jobs fan out.
maturin develop -m crates/vision-calibration-py/Cargo.toml
python -m unittest discover -s crates/vision-calibration-py/tests -p "test_*.py"
```

## Planning

- **Backlog** in `docs/backlog.md` is the source-of-truth task tracker
  (AGENTS.md §11), and holds **only open and parked work**. On completing a
  task, mark it `[x]` with a dated, informative one-paragraph completion note —
  that note is the durable record — then **move the entry** to
  `docs/backlog-archive.md`. **No per-task `docs/report/`
  files** (retired; historical entries archived under
  `docs/internal/archive/report/` for reference only). Maintaining the backlog +
  the docs next to the code matters more than any separate paper trail.
- ADRs in `docs/adrs/` — design decisions (see README there). 0011 covers
  manual init, 0012 covers per-feature residuals, 0013 covers the
  `rig_family` sensor-axis refactor (the Scheimpflug rig modules collapse).
- Tutorials in `docs/tutorials/` — hands-on onboarding for new users. New
  features should ship with a tutorial entry.
- Automated workflow skills: `/orchestrate`, `/architect`, `/implement`, `/review`, `/gate-check`

## Strategic Roadmap

`docs/ROADMAP.md` is authoritative — read it rather than trusting a summary
here. As of 2026-08-11: Tracks **A** (calibration core), **S** (device-spec →
seed init), **Q** (soundness proofs + regression gates), **R** (API/config
revision), **C** (MVG, including the pure-Rust dense matcher) and the benchmark
harness are **done**; **O** is closed won't-do. Live work is **B** (desktop app
elevation), **D** (the v1.0 gate), and two externally-blocked items —
`D4-NALGEBRA-035` (waiting on `tiny-solver`) and `D5-CHARUCO-LABELS` (waiting on
`calib-targets`). Open tasks live in `docs/backlog.md`; completed ones in
`docs/backlog-archive.md`.

We are pre-1.0; breaking changes are acceptable.
