# AGENTS.md — calibration-rs

A multi-crate Rust workspace for **end-to-end camera calibration**: math
primitives and linear solvers, non-linear refinement, pipelines, a facade API,
Python bindings and a desktop app.

Twelve workspace members — nine crates.io packages, the PyPI extension, the
unpublished benchmark crate and `xtask` — plus
`vision-calibration-examples-private`, which lives outside the workspace (its
own `[workspace]`) and path-pins the published crates:

* **`vision-calibration-core`** — math aliases (+ `linalg` numerics), composable camera models, RANSAC, synthetic-data helpers.
* **`vision-calibration-linear`** — closed-form / linear initialisation (homography, PnP, epipolar, rig extrinsics, hand–eye).
* **`vision-calibration-optim`** — non-linear least-squares IR, robust kernels, one Levenberg–Marquardt loop over two solver backends (tiny-solver, factrs).
* **`vision-geometry`** — deterministic two-view solvers (epipolar, homography, triangulation, camera matrix).
* **`vision-mvg`** — MVG pipelines: robust pose recovery, N-view triangulation, bundle adjustment (`refine` feature), Scheimpflug-aware rectification, dense stereo.
* **`vision-calibration-dataset`** — `DatasetSpec` manifest, validator, folder sniffer.
* **`vision-calibration-detect`** — target detectors (chessboard / ChArUco / puzzleboard / ring-grid) behind the sealed `Detector` trait + detection cache.
* **`vision-calibration-pipeline`** — session framework, the seven problem types, `dataset_runner`.
* **`vision-calibration`** — facade re-exporting the above; the compatibility boundary.
* **`vision-calibration-py`** — PyO3/maturin Python package (PyPI).
* **`vision-calibration-bench`** (`publish = false`) — registry-driven dataset benchmarks + regression records, and the public regression suite in `regression/` (`calib-bench suite`).
* **`xtask`** — `cargo xtask emit-schemas`, `cargo xtask check-docs`.
* **`vision-calibration-examples-private`** (*outside* the workspace) — private-dataset acceptance runners.

Priorities, in order: **correctness and numerical stability**, **determinism**
(same inputs + seed → same outputs), **performance**, and **API stability** of
the facade and the JSON schemas.

---

## 1) Layering rules (most important)

### Dependency direction

* `vision-calibration-core` **must not depend on** any other workspace crate.
* `vision-calibration-linear`, `vision-calibration-optim`, and `vision-geometry`
  **may depend on** `vision-calibration-core`. `linear` and `optim` are peers
  (no cross-dep); `linear` may also use `vision-geometry`.
* `vision-mvg` **may depend on** `vision-geometry` and `vision-calibration-core`.
* `vision-calibration-dataset` and `vision-calibration-detect` are the
  manifest/detector layer feeding the pipeline's `dataset_runner`.
  `detect` stays nalgebra-free at its boundary (plain arrays), so detector
  crates may use a different nalgebra than the solver stack.
* `vision-calibration-pipeline` **may depend on** core, linear, optim,
  geometry, dataset, and detect (not `vision-mvg` — the facade re-exports
  the MVG surface directly).
* `vision-calibration` is the facade: re-exports only, no logic.
* `vision-calibration-py` **may depend on** `vision-calibration` (preferred) and Python binding tooling crates.
* `vision-calibration-bench` / `vision-calibration-examples-private`
  (unpublished) consume the facade + pipeline for dataset runs. `bench` also
  reaches core/optim/dataset/linear directly; that is deliberate and confined to it.

### Where code goes

* **Math types, `linalg` numerics, camera models, RANSAC, synthetic generators** → core
* **Closed-form/linear solvers** → linear
* **NLLS IR, robust kernels, solver backends** → optim
* **Deterministic two-view solvers** → `vision-geometry`
* **MVG pipelines** → `vision-mvg`
* **Dataset manifests / sniffing** → dataset; **target detectors** → detect
* **Sessions, problem types, dataset runner** → pipeline
* **Public re-exports** → `vision-calibration`
* **Python bindings and packaging** → `vision-calibration-py`

### API exposure

* The facade is the compatibility boundary; keep its surface stable and
  module-first. Do not expose the same symbol at module, top-level and prelude
  at once.
* Lower crates are sharp tools: small, documented APIs; breaking changes go in
  the CHANGELOG with migration notes.

### Adding a problem type

The design is [ADR 0007](docs/adrs/0007-session-framework.md); the walk-through
is the book's "Adding a New Pipeline Problem Type" chapter. In short: a module
`vision-calibration-pipeline/src/<name>/` (`mod.rs`, `problem.rs`, `state.rs`,
`steps.rs`) implementing `ProblemType`, step functions plus a `run_calibration`
wrapper, a facade re-export, and a Python binding. A camera it calibrates
comes from `camera: CameraConfig` (a rig adds `camera_overrides`), is packed
through optim's `params::camera` helper, evaluated through `CameraBlocks`, and
exported as `CameraParams`, so every camera model works without model-specific
code.

---

## 2) Goals and non-goals

Goals: reliable calibration for perspective cameras and laserline systems;
clear separation of **initialisation**, **refinement** and **orchestration**;
pluggable optimization backends; JSON-serializable configs, inputs and outputs
for reproducible runs.

Non-goals unless requested: heavy ML dependencies in default builds,
non-deterministic outputs, bulky dependencies in core.

---

## 3) Quality gates

The canonical list; CI runs the same checks. Before opening a PR:

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --all-features
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps
cargo xtask check-docs                    # user-facing docs (§7)
cargo xtask emit-schemas --check          # when config, input, export or suite-file types change
python3 -m compileall crates/vision-calibration-py/python/vision_calibration
python3 scripts/check_pyi_coverage.py --check      # when the Python surface changes
python3 scripts/check_binding_parity.py --check    # when a public capability changes
```

Also check minimal builds where relevant (`cargo test -p vision-calibration-core`,
`-p vision-calibration`, `-p vision-calibration-py`). App changes have their
own gates (§10). No new warnings; `#[allow(...)]` only with a stated reason.

---

## 4) Coding conventions

* **Determinism** — explicit RNG seeds (never `thread_rng` in algorithms);
  deterministic ordering in outputs (no `HashMap` iteration order in public
  results); no wall-clock or `Instant` values in results or exports (they
  break golden comparisons).
* **Naming** — problem types `<Name>Problem` with `<Name>Config` /
  `<Name>Export`; steps, options, results and poses follow
  [ADR 0024](docs/adrs/0024-api-vocabulary.md) (`step_<action>`,
  `<Stage><Action>Options`, `Init` never `Initialize`, `<frame>_se3_<frame>`).
* **Public types** — `Config` and `Export` types are `#[non_exhaustive]`;
  config-field rustdoc gives units, valid range and why the default is what
  it is.
* **Numerics** — `vision_calibration_core::Real` (`f64`) throughout;
  normalize where the algorithm requires it (DLT, 8-point); detect degenerate
  configurations and return an error.
* **Errors** — `Result` for user-facing APIs; `assert!` only for internal
  invariants; no panics in pipeline/CLI paths on invalid input. Outside tests,
  an `unwrap()` / `expect()` says why it cannot fail (comment or message);
  `T::from_f64(c).unwrap()` for factor constants is the exception.
* **No silent defaults** — nothing that changes a result's meaning (frame
  and pose conventions, image pairings, model choices) is guessed or
  auto-corrected: ambiguity and unsupported combinations are typed errors,
  raised before any expensive work. Tuning parameters with safe defaults are
  not ambiguity.
* **Config shape** — grouped by stage, shared stage structs reused, one
  name per concept (ADR 0024); field semantics explicit and mode-safe.
* **Hot paths** — no per-point heap allocations in tight loops; reuse
  buffers; fixed-size matrices for tiny systems; no repeated `svd`/`sqrt` in
  inner loops. If performance could change meaningfully, add a benchmark or
  state the rationale.
* **Optimization** — residual kernels generic over `T: RealField` for
  autodiff; rotations on Lie-group manifolds; document any finite-difference
  step size.

---

## 5) Testing policy

Every algorithmic change ships with tests:

* synthetic ground-truth tests for new solvers and refiners;
* edge cases: noise, partial observations, degenerate configurations;
* JSON round-trips for new config/input/output structs;
* regression tests for pipeline outputs (within tolerance);
* for a numerical bug fix, a test that fails before the fix and passes after.

---

## 6) Dependencies

* Core stays lightweight; heavy dependencies elsewhere go behind features.
* New dependencies need a reason and a compatible license.
* `nalgebra` / `faer` / `faer-ext` are pinned by the solver backends, which
  exchange those types with optim (see the root `Cargo.toml` comment).
* **Public dependencies.** `nalgebra` is the one deliberate public
  dependency: the math aliases (`Iso3`, `Pt2`, …) are nalgebra types, and the
  facade re-exports the crate as `vision_calibration::nalgebra` so callers
  use the matching version. A nalgebra major bump is a breaking release of
  ours. `serde` derives appear on every config and export type. Nothing else
  reaches a public signature of the facade; `image` and `serde_json` are
  confined to the detector and dataset-runner boundaries.
* **MSRV** — `rust-version` in the root `Cargo.toml` is the single source;
  the `MSRV (…)` CI job checks every target and feature on it with
  `--locked`. Raise it when a dependency we track needs a newer toolchain
  or a stable feature simplifies the code, rather than pinning
  dependencies under the floor in `Cargo.lock`. Bump `rust-version` and
  the CI job in the same PR.

---

## 7) Documentation

When public types, config parameters or algorithm behavior change, update the
rustdoc, the README and/or the book, a minimal usage example, and for Python
`python/vision_calibration/types.py` and `__init__.pyi`.

**Rust, Python and the book move together.** A new public capability lands
with its Python binding and its book coverage in the same PR. A capability
that is deliberately Rust-only is recorded as such in the binding parity
check, with the reason.

**User-facing docs never cite internal material.** User-facing means the
READMEs (root, crates, `app/README.md`), `book/src`, the
CHANGELOG, published crates' rustdoc, and the Python package. They must not
mention ADRs, `docs/notes` / `docs/internal` / backlog / roadmap paths,
backlog or track IDs, PR numbers, private datasets, or agent files, and they
state the current design without narrating history. Rationale a reader needs
is stated inline. `cargo xtask check-docs` enforces the mechanical part.
Dev-facing docs (`AGENTS.md`, `.claude/`, `docs/adrs`, `docs/notes`,
`docs/ROADMAP.md`, `docs/backlog.md`, `app/DEVELOPING.md`) may cite anything.

---

## 8) PRs and commits

* One focused change per PR: summary, tests run, perf notes.
* Behavior changes are stated explicitly, with a config/flag or migration
  notes.
* Commit prefixes: `feat:`, `fix:`, `refactor:`, `perf:`, `docs:`, `test:`,
  `chore:`.
* When trade-offs conflict, preserve correctness; make new behavior opt-in and
  justify it with tests or a benchmark.
* `privatedata/` (git-ignored) never reaches a commit, PR or report beyond
  aggregate numbers (counts, error magnitudes).
* The sibling crates `../calib-targets-rs` and `../vision-metrology` are
  separate git repositories: an edit there is its own commit in that repo,
  called out as a cross-repo change.

---

## 9) Backlog workflow (mandatory)

The priority is a current **backlog** and current **documentation**, not a
paper trail. History lives in `CHANGELOG.md`, PR descriptions and git.

* `docs/backlog.md` is the source of truth for what is left. It holds **only
  open (`[ ]`) and parked (`[~]`) work**; a parked entry states why and what
  would reopen it.
* One backlog task at a time (one commit per task), unless tasks are so
  tightly coupled that neither builds alone; then say so in the commit
  message.
* Completing a task means, in the same commit:
  1. **Delete** its entry from `docs/backlog.md` (and the track heading once
     it is empty).
  2. Add a user-visible change to `CHANGELOG.md` under `[Unreleased]`.
  3. Update the documentation next to the code: rustdoc, ADRs (`docs/adrs/`)
     for design decisions, and the book (`book/`) for user-facing features.
* The commit message and PR description carry the rest (what landed, why,
  tests run). Do not write per-task report files.
* Documentation states the current design. Do not narrate history and do not
  cite backlog/track IDs in code comments or user-facing docs.
* Commit message format: `feat(backlog): <task-id> <short description>`
  (likewise `fix(backlog)`, `docs(backlog)`).

---

## 10) Desktop app (`app/`)

* Tauri 2 + React 19 + TypeScript; frontend in `app/src/` (the
  `@vitavision/workbench` shell in `shell/`, the project model in
  `project/`, five workspaces under `workspaces/`: Run, Diagnose, 3D,
  Epipolar, Depth), Rust Tauri backend in `app/src-tauri/src/`. No `app/src`
  module over ~400 lines (generated files excepted).
* Source images load over Tauri's asset protocol, scoped to folders the user
  opened (`assets.rs`); never send image bytes over IPC as text.
* **Always `bun`**, never `npm`/`pnpm`/`yarn`; the lockfile is `bun.lock`, and
  `tauri.conf.json`'s `beforeDevCommand`/`beforeBuildCommand` call `bun run …`.
  `bun run tauri dev` launches the app (`bun run dev` alone is Vite-only, with
  no Tauri IPC).
* UI comes from `@vitavision/ui` (components, tokens, fonts), the frame from
  `@vitavision/workbench` and the 3D viewer from `@vitavision/three` /
  `@vitavision/three-react`; no app-local component kit. ESLint's `tokensOnly` rule keeps colour in `app/src` on the design
  tokens.
* Gates (from `app/`): `bun run lint`, `bun run format:check`,
  `bun run typecheck`, `bun run test`, `bun run build`; and from
  `app/src-tauri`: `cargo fmt --check`,
  `cargo clippy --all-targets -- -D warnings`, `cargo test`.
* `app/src-tauri` is **excluded from the root Cargo workspace**
  (`exclude = ["app"]`) and pins its own `Cargo.lock`; root
  `cargo build|test --workspace` does **not** cover it.
* Developer guide: `app/DEVELOPING.md`; design rationale:
  `docs/adrs/0014-desktop-app.md`.
* The workspaces are being reorganized around the project as Project →
  Dataset → Detect → Calibrate → Diagnose → Use (backlog, App track). New
  views go into that structure rather than into the current workspaces.
