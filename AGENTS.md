# AGENTS.md — calibration-rs

The working rules for this repository, for people and agents alike. Design
reasons live in the [ADRs](docs/adrs/README.md), open work in the
[backlog](docs/backlog.md), direction in the [ROADMAP](docs/ROADMAP.md); the
[developer docs map](docs/README.md) lists the rest.

## 1) Project

A Rust workspace for **end-to-end camera calibration**: math primitives and
camera models, closed-form initialization, non-linear refinement with
parameter uncertainty, calibration workflows, multiple-view geometry, a facade
crate, Python bindings and a desktop app. The published crates, the PyPI
package and the unpublished tools (`vision-calibration-bench`, `xtask`) are the
workspace members in the root `Cargo.toml`; `vision-calibration-examples-private`
lives outside the workspace and path-pins the published crates.

Priorities, in order: **correctness and numerical stability**, **determinism**
(same inputs + seed → same outputs), **clarity of use**, **performance**, and
**API stability** of the facade and the JSON schemas.

## 2) Layering

The workspace is a strict layered DAG ([ADR 0006](docs/adrs/0006-layered-crate-architecture.md)).

| Crate | May depend on | Holds |
|---|---|---|
| `vision-calibration-core` | — | math aliases, `linalg` numerics, camera models, RANSAC, synthetic generators |
| `vision-geometry` | core | deterministic two-view solvers |
| `vision-calibration-linear` | core, geometry | closed-form / linear initialization |
| `vision-calibration-optim` | core | NLLS IR, factors, robust kernels, solver backends, covariance |
| `vision-mvg` | core, geometry, optim | MVG pipelines: robust pose, triangulation, BA, rectification, dense stereo |
| `vision-calibration-dataset` | — | `DatasetSpec` manifest, validator, folder sniffer |
| `vision-calibration-detect` | — | target detectors (`DetectorConfig`), detection cache; plain arrays at its boundary |
| `vision-calibration-pipeline` | core, linear, optim, geometry, dataset, detect | sessions, problem types, `dataset_runner`, analysis |
| `vision-calibration` | all of the above | the facade: re-exports only, no logic |
| `vision-calibration-cli` | the facade | the `vcal` program: a thin command layer over the facade |
| `vision-calibration-py` | the facade, the CLI | PyO3/maturin bindings; the `vcal` console script runs the CLI |
| `vision-calibration-bench` | facade, pipeline, core, optim, dataset, linear | benchmarks, acceptance, the public regression suite |

`linear` and `optim` are peers (no edge either way). The pipeline does not
depend on `vision-mvg`; the facade re-exports it.

**API exposure.** The facade is the compatibility boundary: module-first, each
symbol reachable by one path (not at module, top level and prelude at once).
What workspace crates need from each other but users do not is
`#[doc(hidden)]` and outside the semver promise. Lower crates are sharp tools:
small, documented APIs.

## 3) Quality gates

Run before opening a PR; CI runs the same:

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --all-features
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --all-features --no-deps
cargo xtask check-docs                    # user-facing docs (§9)
cargo xtask emit-schemas --check          # when config, input, export or suite-file types change
python3 -m compileall crates/vision-calibration-py/python/vision_calibration
python3 scripts/check_pyi_coverage.py --check      # when the Python surface changes
python3 scripts/check_binding_parity.py --check    # when a public capability changes
```

CI also runs heavier jobs worth running locally when your change touches
them: the MSRV check (`--locked` on the `rust-version` toolchain), the
`wasm32` build of core and dataset, the LFS image tests, the regression suite
(`calib-bench suite check` on both backends, `suite compare`), the uncertainty
gates (Monte-Carlo, robot-error, next-view), the Python runtime tests, and the
app jobs (§12). No new warnings; `#[allow(...)]` only with a stated reason.

## 4) Coding conventions

* **Naming** — problem types `<Name>Problem` with `<Name>Config` /
  `<Name>Export`; steps, options, results and poses follow
  [ADR 0024](docs/adrs/0024-api-vocabulary.md) (`step_<action>`,
  `<Stage><Action>Options`, `Init` never `Initialize`, `<frame>_se3_<frame>`,
  [ADR 0009](docs/adrs/0009-coordinate-and-pose-conventions.md)).
* **Public types** — `Config` and `Export` types are `#[non_exhaustive]`;
  config-field rustdoc gives units, valid range and why the default is what
  it is.
* **Config shape** — grouped by stage, shared stage structs reused, one name
  per concept; field semantics explicit and mode-safe.
* **Numerics** — `vision_calibration_core::Real` (`f64`) throughout;
  normalize where the algorithm requires it (DLT, 8-point); detect degenerate
  configurations and return an error.
* **Errors** — `Result` for user-facing APIs; `assert!` only for internal
  invariants; no panics in pipeline/CLI paths on invalid input. Outside tests,
  an `unwrap()` / `expect()` says why it cannot fail;
  `T::from_f64(c).unwrap()` for factor constants is the exception.
* **Optimization** — residual kernels generic over `T: RealField` for
  autodiff; rotations on Lie-group manifolds; document any finite-difference
  step size.
* **Hot paths** — no per-point heap allocations in tight loops; reuse
  buffers; fixed-size matrices for tiny systems; no repeated `svd`/`sqrt` in
  inner loops. If performance could change meaningfully, add a benchmark or
  state the rationale.
* **Design** — SOLID, one source of truth per concept, the simplest design
  that meets the actual requirement, illegal states unrepresentable
  (enum-with-payload over discriminator + parallel fields).

## 5) Results contract

What every result promises, whichever workflow produced it:

* **Deterministic** — explicit RNG seeds (never `thread_rng` in
  algorithms); deterministic ordering (no `HashMap` iteration order in public
  results); no wall-clock or `Instant` values in results or exports.
* **No silent defaults** — nothing that changes a result's meaning (frame
  and pose conventions, image pairings, model choices) is guessed or
  auto-corrected: ambiguity and unsupported combinations are typed errors,
  raised before any expensive work. Tuning parameters with safe defaults are
  not ambiguity.
* **Uncertainty is part of the result**
  ([ADR 0026](docs/adrs/0026-parameter-uncertainty.md)). A refiner or problem
  type returns the covariance of what it estimates, or a typed reason it is
  unavailable; a quantity derived from a calibration propagates it. A change
  that alters results is measured first and stays opt-in until the evidence
  is in.
* **The export is the wire contract**
  ([ADR 0012](docs/adrs/0012-export-contract.md)): JSON with a published
  schema.

## 6) Testing

Every algorithmic change ships with tests:

* synthetic ground-truth tests for new solvers and refiners (the proof-pack
  standard in [`docs/notes/`](docs/notes/README.md));
* edge cases: noise, partial observations, degenerate configurations;
* Monte-Carlo agreement for every new or changed σ;
* JSON round-trips for new config/input/output structs;
* regression tests for pipeline outputs (within tolerance);
* for a numerical bug fix, a test that fails before the fix and passes after.

Tolerances: ~5 % for linear init, < 1 % after optimization.

## 7) Dependencies and MSRV

* Core stays lightweight; heavy dependencies elsewhere go behind features.
  New dependencies need a reason and a compatible license.
* `nalgebra` is the one public dependency, re-exported as
  `vision_calibration::nalgebra`. Its version, with `faer` / `faer-ext`,
  follows what both solver backends are built against, so a bump is a
  breaking release of ours ([ADR 0006](docs/adrs/0006-layered-crate-architecture.md)).
  `serde` derives appear on every config and export type; nothing else
  reaches a facade signature.
* **MSRV** — `rust-version` in the root `Cargo.toml` is the single source;
  the `MSRV` CI job checks every target and feature on it with `--locked`.
  Raise it when a dependency needs a newer toolchain or a stable feature
  simplifies the code, rather than pinning dependencies in `Cargo.lock`. Bump
  `rust-version` and the CI job in the same PR.

## 8) Versioning

All published crates and the PyPI package share one version.

* **Before 1.0** — a release that breaks the public API, a JSON schema or the
  meaning of a result bumps the minor (`0.x.0`) and lists migration notes in
  the CHANGELOG. Every other release, additive features included, is a patch
  (`0.x.y`). Cargo treats `0.x.y` as compatible with `0.x.0`, so a break in a
  patch reaches users silently.
* **From 1.0** — standard SemVer: breaking → major, additive → minor, fixes →
  patch.
* `cargo semver-checks` runs before every tag (the release workflow enforces
  it). Items under `#[doc(hidden)]` are exempt.
* The procedure is [`docs/RELEASE-RUNBOOK.md`](docs/RELEASE-RUNBOOK.md); read
  it before touching any version string.

## 9) Documentation

* When public types, config parameters or algorithm behavior change, update
  the rustdoc, the README and/or the book, a minimal usage example, and for
  Python `python/vision_calibration/types.py` and `__init__.pyi`.
* **Rust, Python and the book move together.** A new public capability lands
  with its Python binding and its book coverage in the same PR. A capability
  that is deliberately Rust-only is recorded in the binding parity check, with
  the reason.
* **User-facing docs never cite internal material.** The READMEs, `book/src`,
  the CHANGELOG, published rustdoc and the Python package do not mention ADRs,
  `docs/` paths, backlog IDs, PR numbers, private datasets or agent files, and
  state the current design without narrating history. `cargo xtask
  check-docs` enforces the mechanical part. Dev-facing docs (this file,
  `.claude/`, `docs/`, `app/DEVELOPING.md`) may cite anything.

## 10) Extension recipes

* **A problem type** — [ADR 0007](docs/adrs/0007-session-framework.md) and
  the book's "Adding a New Pipeline Problem Type": a module
  `vision-calibration-pipeline/src/<name>/` (`mod.rs`, `problem.rs`,
  `state.rs`, `steps.rs`) implementing `ProblemType`, step functions plus a
  `run_calibration` wrapper, a facade re-export, and a Python binding. Its
  camera comes from `camera: CameraConfig` (a rig adds `camera_overrides`),
  is packed through optim's `params::camera`, evaluated through
  `CameraBlocks`, and exported as `CameraParams`, so every camera model works
  without model-specific code.
* **A camera model** — [ADR 0005](docs/adrs/0005-composable-camera-model.md)
  and [ADR 0013](docs/adrs/0013-sensor-variants.md): a variant of the
  projection, distortion or sensor kind in core with its parameter slots; it
  then reaches every workflow. It ships with a math note, a synthetic matrix
  test across every problem type, the Python class and a book chapter.
* **A residual (factor)** — a `FactorKind` variant in optim, its generic
  residual, and an arm in both backends and the covariance family map; the
  book's "Adding a New Optimization Problem" chapter walks through it. A
  residual a user needs is not a variant: it comes through the public
  `optim::problem` builder ([ADR 0027](docs/adrs/0027-public-optimization-extension.md),
  the book's "Your Own Residual").
* **An MVG pipeline** — `vision-mvg`, within the scope ceiling of ADR 0006
  (geometry over calibrated rigs, no SfM).

## 11) PRs, commits and the backlog

* One focused change per PR: summary, tests run, perf notes. Behavior
  changes are stated explicitly, with a config/flag or migration notes. When
  trade-offs conflict, preserve correctness.
* Commit prefixes: `feat:`, `fix:`, `refactor:`, `perf:`, `docs:`, `test:`,
  `chore:`; a backlog task is `feat(backlog): <task-id> <short description>`
  (likewise `fix(backlog)`, `docs(backlog)`).
* **The backlog** (`docs/backlog.md`) holds only open (`[ ]`) and parked
  (`[~]`) work; a parked entry states why and what would reopen it. One task
  per commit, unless two cannot build alone (say so in the message).
  Completing a task means, in the same commit: delete its entry (and an empty
  track heading), add the user-visible change to `CHANGELOG.md` under
  `[Unreleased]`, and update the docs next to the code (rustdoc, ADRs, the
  book). No per-task report files; the commit and PR carry the rest.
* `privatedata/` (git-ignored) never reaches a commit, PR or report beyond
  aggregate numbers.
* The sibling crates `../calib-targets-rs` and `../vision-metrology` are
  separate repositories: an edit there is its own commit in that repo, called
  out as a cross-repo change.

## 12) Desktop app (`app/`)

Tauri 2 + React 19 + TypeScript; the developer guide is
[`app/DEVELOPING.md`](app/DEVELOPING.md), the design
[ADR 0014](docs/adrs/0014-desktop-app.md).

* **Always `bun`**, never `npm`/`pnpm`/`yarn`; `bun run tauri dev` launches
  the app (`bun run dev` alone is Vite without Tauri IPC).
* `app/src-tauri` is **excluded from the root workspace** (`exclude =
  ["app"]`) and pins its own `Cargo.lock`; root `cargo test --workspace` does
  not cover it.
* UI comes from `@vitavision/ui`, `@vitavision/workbench` and
  `@vitavision/three(-react)`; no app-local component kit, colour only from
  design tokens. No `app/src` module over ~400 lines (generated files
  excepted). Source images load over the asset protocol, never as IPC text.
* Gates (from `app/`): `bun run lint`, `bun run format:check`,
  `bun run typecheck`, `bun run test`, `bun run build`; from `app/src-tauri`:
  `cargo fmt --check`, `cargo clippy --all-targets -- -D warnings`,
  `cargo test`.
