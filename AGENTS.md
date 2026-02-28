# AGENTS.md — calibration-rs

This repository is a multi-crate Rust workspace for **end-to-end camera calibration**:
from math primitives and linear solvers to non-linear refinement, pipelines, facade APIs,
and Python bindings.

Crates:

* **`vision-calibration-core`** — math aliases, composable camera models, and a generic RANSAC engine.
* **`vision-calibration-linear`** — closed-form / linear initialisation blocks (homography, PnP, epipolar, rig extrinsics, hand–eye).
* **`vision-calibration-optim`** — non-linear least squares traits, robust kernels, and solver backends (LM today).
* **`vision-calibration-pipeline`** — end-to-end calibration pipelines (currently planar intrinsics).
* **`vision-calibration`** — facade crate re-exporting the above for a stable, ergonomic API.
* **`vision-calibration-py`** — PyO3/maturin Python extension crate exposing high-level workflows.

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
* `vision-calibration-linear` and `vision-calibration-optim` **may depend on** `vision-calibration-core`.
* `vision-calibration-pipeline` **may depend on** `vision-calibration-core`, `vision-calibration-linear`, and `vision-calibration-optim`.
* `vision-calibration` is top-level entry points.
* `vision-calibration-py` **may depend on** `vision-calibration` (preferred) and Python binding tooling crates.

### Where code goes

* **Math types, camera models, RANSAC** → `vision-calibration-core`
* **Closed-form/linear solvers** → `vision-calibration-linear`
* **NLLS traits, robust kernels, solver backends** → `vision-calibration-optim`
* **Full pipelines and reports** → `vision-calibration-pipeline`
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

* Preserve correctness + backwards compatibility first.
* Add configuration/feature flags for opt-in behavior.
* Add tests and (if needed) a benchmark to justify the change.
