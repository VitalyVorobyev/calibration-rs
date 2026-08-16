# AGENTS.md — calibration-rs

This repository is a multi-crate Rust workspace for **end-to-end camera calibration**:
from math primitives and linear solvers to non-linear refinement, pipelines, facade APIs,
and Python bindings.

Crates — eleven workspace members (nine crates.io-publishable, the PyPI
extension, and the unpublished benchmark crate), plus
`vision-calibration-examples-private`, which lives outside this workspace
entirely (its own `[workspace]`) and path-pins the published crates:

* **`vision-calibration-core`** — math aliases (+ `linalg` numerics), composable camera models, and a generic RANSAC engine.
* **`vision-calibration-linear`** — closed-form / linear initialisation blocks (homography, PnP, epipolar, rig extrinsics, hand–eye).
* **`vision-calibration-optim`** — non-linear least squares IR, robust kernels, and solver backends (tiny-solver LM).
* **`vision-geometry`** — deterministic two-view solvers (epipolar, homography, triangulation, camera matrix).
* **`vision-mvg`** — MVG pipelines: robust pose recovery, N-view triangulation, bundle adjustment (`refine` feature), Scheimpflug-aware rectification, dense stereo.
* **`vision-calibration-dataset`** — `DatasetSpec` manifest, validator, folder sniffer.
* **`vision-calibration-detect`** — target detectors (chessboard / ChArUco / puzzleboard / ring-grid) behind the sealed `Detector` trait + detection cache.
* **`vision-calibration-pipeline`** — session framework, the eight problem types, `dataset_runner`.
* **`vision-calibration`** — facade crate re-exporting the above for a stable, ergonomic API.
* **`vision-calibration-py`** — PyO3/maturin Python extension crate exposing high-level workflows (PyPI).
* **`vision-calibration-bench`** (workspace member, `publish = false`) — registry-driven dataset benchmarks + regression records.
* **`vision-calibration-examples-private`** (*outside* the workspace) — private-dataset acceptance runners (rtv3d family).

The codebase prioritizes:

* **Correctness & numerical stability**
* **Determinism** (same inputs + seed → same outputs)
* **Performance** (avoid unnecessary allocations; efficient linear algebra)
* **API stability** in the top-level `vision-calibration` crate and JSON schemas

If you are an automated agent (Codex, etc.), follow these rules strictly.

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
* `vision-calibration-pipeline` **may depend on** core, linear, optim,
  geometry, dataset, and detect (not `vision-mvg` — the facade re-exports
  the MVG surface directly).
* `vision-calibration` is top-level entry points (facade only, no logic).
* `vision-calibration-py` **may depend on** `vision-calibration` (preferred) and Python binding tooling crates.
* `vision-calibration-bench` / `vision-calibration-examples-private`
  (unpublished) consume the facade + pipeline for dataset runs. `bench` also
  reaches core/optim/dataset directly; that is deliberate and confined to it.

### Where code goes

* **Math types, `linalg` numerics, camera models, RANSAC** → `vision-calibration-core`
* **Closed-form/linear solvers** → `vision-calibration-linear`
* **NLLS IR, robust kernels, solver backends** → `vision-calibration-optim`
* **Deterministic two-view solvers** → `vision-geometry`
* **MVG pipelines (pose recovery, triangulation, BA, rectification, dense stereo)** → `vision-mvg`
* **Dataset manifests / sniffing** → `vision-calibration-dataset`; **target detectors** → `vision-calibration-detect`
* **Sessions, problem types, dataset runner** → `vision-calibration-pipeline`
* **Public re-exports/docs** → `vision-calibration`
* **Python module bindings, Python package glue, and wheel packaging** → `vision-calibration-py`

### API exposure

* `vision-calibration` is the compatibility boundary. Keep its public surface stable.
* Keep facade APIs module-first; avoid duplicating the same symbols at module, top-level, and prelude simultaneously.
* Lower crates are “sharp tools”: keep APIs small and documented; avoid breaking changes without semver notes.

---

## 2) Project goals and non-goals

### Goals

* Reliable, end-to-end camera calibration for perspective cameras and laserline systems.
* Clear separation between **initialisation**, **refinement**, and **pipeline orchestration**.
* Pluggable optimization backends and robust estimation where needed.
* JSON-serializable configs/inputs/outputs for reproducible runs.

### Non-goals (unless explicitly requested)

* Heavy ML dependencies in default builds.
* Non-deterministic outputs.
* Bulky dependencies in `vision-calibration-core`.

---

## 3) Build, test, and quality gates

Before opening a PR, run:

* `cargo fmt --all`
* `cargo clippy --workspace --all-targets --all-features -- -D warnings`
* `cargo test --workspace --all-features`

Also check minimal builds where relevant:

* `cargo test -p vision-calibration-core`
* `cargo test -p vision-calibration`
* `cargo test -p vision-calibration-py`

When Python package files are modified, also run:

* `python -m compileall crates/vision-calibration-py/python/vision_calibration`

**Do not** introduce new warnings. Avoid `#[allow(...)]` unless justified.

---

## 4) Coding conventions

### Determinism

* Use explicit RNG seeds; do not use `thread_rng` in algorithms.
* Preserve deterministic ordering in outputs (avoid `HashMap` iteration order for public results).

### Numerics

* Use `vision_calibration_core::Real` (`f64`) consistently.
* Normalize inputs where algorithms require it (e.g., DLT/8-point).
* Guard against degenerate configurations and report errors explicitly.

### Error handling

* Prefer `Result` for user-facing APIs; reserve `assert!` for internal invariants.
* Avoid panics in pipeline/CLI paths when input validation can fail.

### Configuration shape

* Prefer grouped config structs by stage/responsibility (`init`, `solver`, `optimize`, `ba`) over flat boolean-heavy bags.
* Keep field semantics explicit and mode-safe (especially for frame/mode-dependent transforms).

### Allocations / hot paths

* Avoid per-point heap allocations in tight loops.
* Reuse buffers where possible.
* Prefer fixed-size matrices for tiny systems.

### Optimization

* Prefer analytic Jacobians; if finite differences are used, document step size and scaling.
* Keep parameterizations well-conditioned (e.g., axis-angle or Lie algebra for rotations).

---

## 5) Performance rules

When modifying core solvers or pipelines:

* Avoid repeated expensive ops in inner loops (`svd`, `sqrt`, normalizations) unless required.
* Keep memory access contiguous and cache-friendly.
* If performance could change meaningfully, add a micro-benchmark or document rationale.

---

## 6) Testing policy

Every algorithmic change must include tests.

Minimum expectations:

* Synthetic correctness tests for new solvers/refiners.
* Edge cases: noisy data, partial observations, and degenerate configurations.
* JSON roundtrip tests for any new config/input/output structs.
* Regression tests for pipeline outputs (within tolerance).

---

## 7) Documentation expectations

When adding/changing:

* public types
* configuration parameters / thresholds
* algorithm behavior

You must update:

* rustdoc for affected items
* README and/or `book/` docs
* a minimal example snippet showing the new usage
* for Python bindings: `python/vision_calibration/types.py` and `python/vision_calibration/__init__.pyi`

---

## 8) Dependency policy

* `vision-calibration-core`: keep dependencies minimal and lightweight.
* Other crates may add ergonomic dependencies, but prefer feature flags for heavy deps.
* Any new dependency must be justified and license-compatible.

---

## 9) PR/commit expectations (for agents)

* Keep PRs focused (one feature/fix at a time).
* Include: summary, tests run, and any perf notes.
* If behavior changes: state it explicitly and provide a config/flag or migration notes.

Suggested commit prefixes:

* `feat:`, `fix:`, `refactor:`, `perf:`, `docs:`, `test:`

---

## 10) If you’re unsure

When trade-offs conflict (speed vs accuracy, stability vs cleanup):

* Preserve correctness.
* Add configuration/feature flags for opt-in behavior.
* Add tests and (if needed) a benchmark to justify the change.

---

## 11) Backlog implementation workflow (mandatory)

Backlog execution must be traceable task-by-task. The priority is a current
**backlog** and current **documentation** — not a per-task paper trail.

* Source of truth for execution status is `docs/backlog.md`. It holds **only open (`[ ]`) and parked (`[~]`) work**, so its length tracks what is left rather than what has been done. Completion notes live in `docs/backlog-archive.md`, grouped by track.
* Implement one backlog task at a time (do not batch multiple tasks into one commit), unless tasks are tightly coupled and cannot be merged independently while keeping the workspace buildable. In that case, document the coupling explicitly in the backlog note and commit message.
* Every completed task must include both of the following:
  1. **Backlog update**: mark the task `[x]` with a completion note (date, a one-paragraph summary of what landed, optionally commit id), then **move the entry** out of `docs/backlog.md` into `docs/backlog-archive.md` under its track heading. That note is the durable record — keep it informative. When a track's last open item closes, drop the whole track from the backlog and add a one-line entry under *Closed tracks*.
  2. **Dedicated commit**: commit only that task’s code/docs/tests updates.
* Keep the **documentation that lives next to the code** current as part of the task: module/rustdoc, ADRs (`docs/adrs/`) for design decisions, and tutorials (`docs/tutorials/`) for new user-facing features. Update what the change touches; do **not** write a separate per-task report file (`docs/report/` is retired — historical entries are archived under `docs/internal/archive/report/` for reference only).
* Recommended commit message format:
  * `feat(backlog): <task-id> <short description>`
  * `fix(backlog): <task-id> <short description>`
  * `docs(backlog): <task-id> <short description>`

---

## 12) Desktop app (`app/`)

* Tauri 2 + React 19 + TypeScript desktop app; frontend in `app/src/`
  (five workspaces under `app/src/workspaces/`: Run, Diagnose, 3D, Epipolar,
  Depth), Rust Tauri backend in `app/src-tauri/src/`.
* **Always use `bun`**, never `npm`/`pnpm`/`yarn`. Commands (run from `app/`):
  `bun install`, `bun run tauri dev` (launches the app — `bun run dev` alone
  is Vite-only and the Tauri IPC surface is absent), `bun run build`,
  `bun run tauri build`.
* `app/src-tauri` is **excluded from the root Cargo workspace**
  (`/Cargo.toml`'s `exclude = ["app"]`) and pins its own `Cargo.lock`.
  `cargo build|test --workspace` at the repo root does **not** cover it;
  build/test it from `app/` instead.
* See `app/README.md` and `docs/adrs/0014-tauri-desktop-app.md`.
