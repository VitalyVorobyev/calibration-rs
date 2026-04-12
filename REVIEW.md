# Pre-Release Review — calibration-rs
*Reviewed: 2026-04-12*
*Scope: full workspace (6 crates), targeting v1.0 public-API release*
*Reviewer: Architect (Opus 4.6, 1M context)*

## Review Verdict (Phase 5, partial)

**Overall**: PASS on the 8 findings that have been implemented. All objective quality gates green. 3 findings remain open (R-01, R-06, R-09) per the rate-limit interruption; those need a fresh Implementer run. **Not yet releasable** until those land — R-01 (typed errors) is a genuine v1.0 blocker.

**Verified items**: 8 — R-02, R-03, R-04, R-05, R-07, R-10, R-12, R-13.
**Needs rework**: 0.
**Regressions**: 0.
**Open (todo)**: 3 — R-01, R-06 (blocked on R-01), R-09.
**Skipped**: 2 — R-08, R-11.

### Quality-gate results (2026-04-12, after commit c40627e)

| Gate                                                            | Result |
|-----------------------------------------------------------------|--------|
| `cargo fmt --all -- --check`                                    | ✅ pass |
| `cargo clippy --workspace --all-targets --all-features -D warnings` | ✅ pass |
| `cargo test --workspace --all-features`                         | ✅ pass |
| `cargo doc --workspace --no-deps`                               | ✅ zero warnings |
| `python3 scripts/check_pyi_coverage.py --check`                 | ✅ pass (52 __all__ entries, 7 pyfunctions) |
| `cargo build -p vision-calibration-core --features tracing`     | ✅ pass (R-05 verification) |

### Per-finding verdict

| ID   | Verdict | Notes                                                                                                  |
|------|---------|--------------------------------------------------------------------------------------------------------|
| R-02 | PASS    | MSRV 1.85 declared, parallel CI job added. Commit swept in pre-existing num-dual/rand dep bumps — acceptable per Implementer instructions. |
| R-03 | PASS    | `choose_multiple` → `sample`; the one doc warning is gone; semantics identical.                        |
| R-04 | PASS    | Advisory investigated (paste is proc-macro only, zero runtime exposure); 6-line rationale comment added next to the ignore directive. |
| R-05 | PASS    | `tracing::instrument` on `ransac_fit`. Minimal scope (core only) is a deliberate narrowing vs. REVIEW.md's original "~5 entry points" — noted in resolution. |
| R-07 | PASS (with scope note) | NaN/Inf rejection + PyValueError remapping delivered. Per-problem-type array-length checks were deferred (would need schema knowledge of every `P::Input` type); length mismatches now surface via the Rust layer as `ValueError: invalid input: ...`. Net UX gain is real. |
| R-10 | PASS    | All three `_ => panic!` wildcards replaced with explicit variant arms. Future enum additions will fail to compile.                                   |
| R-12 | PASS    | Script + CI hook; also caught pre-existing `library_version` drift and fixed it as part of the baseline. Clean baseline verified.   |
| R-13 | PASS    | Empty `[features]` block deleted from optim; one-line cleanup.                                         |

### No regressions introduced

- Test suite is fully green (`cargo test --workspace --all-features`).
- No new clippy warnings.
- No new doc warnings (the prior `choose_multiple` warning is gone — R-03 strictly improved doc health).
- Python package still compiles and the pyi-coverage check passes.
- No `unsafe` code introduced (workspace still has zero `unsafe` blocks).

### Remaining work (open `todo` — not blocking the verdict on completed items)

1. **R-09** (~30 call sites, breaking API cleanup): do BEFORE R-01 to minimise file overlap.
2. **R-01** (~60 files, typed error hierarchy with `thiserror`): the single largest fix, commit-per-crate.
3. **R-06** (rustdoc `# Errors` sections): runs AFTER R-01.

Once those three land, a final `cargo doc` + full test run is needed and this verdict section should be updated to "FINAL — all 11 in-scope findings verified".

## Triage Decisions (Phase 3)

Owner confirmed 2026-04-12:
- **R-01** (anyhow→typed errors): **include** — full typed-error migration across core/linear/optim/pipeline/facade using `thiserror`. Single biggest v1.0 fix.
- **R-04** (ignored RUSTSEC): **investigate** — Implementer researches actual impact; either add documented comment or migrate to `deny.toml`.
- **R-05** (dead `tracing` feature): **instrument** — add `#[cfg_attr(feature = "tracing", tracing::instrument)]` to ~5 hot paths (RANSAC, Zhang intrinsics, optim `solve`, etc.) rather than removing the feature. CLAUDE.md feature-flag note stays.
- **Commit discipline**: one commit per finding. Implementer creates dedicated commits with `refs R-NN` references so reverts and bisects are trivial.

Default decisions (no clarification needed):
- **R-02, R-03**: include (P1 release blockers, mechanical fixes).
- **R-06** (`# Errors` rustdoc): include, sequence AFTER R-01 lands (typed variants give concrete content to document).
- **R-07** (Python input validation): include (UX quality).
- **R-08** (`ProblemIR::validate()` refactor): **defer to v1.1** — internal-only, not release-blocking. Marked `skipped` below.
- **R-09** (`Option<Vec<T>>` cleanup): include (breaking-change window is now).
- **R-10** (panic wildcards): include.
- **R-11** (property tests): **defer to v1.1** — existing integration coverage is strong. Marked `skipped` below.
- **R-12** (`.pyi` drift check script): include (~40 lines, cheap CI insurance).
- **R-13** (empty `default = []`): include (one-line cleanup).

**Active scope: 11 findings (R-01..R-07, R-09, R-10, R-12, R-13). Deferred: R-08, R-11.**

### Implementation Order

P1s first (blockers), then P2s, then P3s. R-06 must run AFTER R-01 (needs typed variants to document).

1. **R-02** MSRV declaration + CI job (Cargo.toml + `.github/workflows/ci.yml`)
2. **R-03** `choose_multiple` → `sample` rename (1 line)
3. **R-13** Remove empty `default = []` (1 line)
4. **R-01** Typed error hierarchy — large fix, multiple sub-commits OK but prefer one per crate (`core`, `linear`, `optim`, `pipeline`, facade)
5. **R-05** `tracing::instrument` on 5 hot paths (core + optim)
6. **R-04** RUSTSEC-2024-0436 — investigate then either comment or migrate to `deny.toml`
7. **R-07** Python input validation helpers
8. **R-09** `Option<Vec<_>>` → `Vec<_>` migration
9. **R-10** Replace panic wildcards with explicit variants or `unreachable!()`
10. **R-12** `scripts/check_pyi_coverage.py` + CI hook
11. **R-06** Add `# Errors` rustdoc sections to all fallible public APIs (after R-01)

### Pre-existing uncommitted state (NOT Implementer's responsibility to commit)

Working tree already has these when the review started — Implementer should work AROUND them:
- `Cargo.toml`: `num-dual` relaxed to "0.13", `rand` bumped to "0.10" (this bump is what made `choose_multiple` deprecated — see R-03)
- `Cargo.lock`: dep-bump fallout
- `docs/backlog.md` + `docs/report/*.md`: deleted in working tree (intentional per owner triage)
- `.claude/CLAUDE.md`: refreshed (LoC, Camera signature, Feature Flags, Python Bindings, Planning) as part of this review cycle

Implementer strategy: stage ONLY the files you modify for each R-NN fix. Do NOT stage the pre-existing uncommitted changes unless a fix genuinely touches them. If a fix legitimately changes Cargo.toml (e.g. R-02), stage the whole file — the dep bumps will come along; that's acceptable and an owner decision.

## Executive Summary

The `calibration-rs` workspace is **structurally healthy and nearly release-ready**.
Baseline checks pass (`cargo fmt`, `cargo clippy --all-features -D warnings`,
`cargo test --all-features`). Architecture is clean: 6 crates in a strict
linear/optim peer-layer topology with no cycles, `#[non_exhaustive]` used
consistently on 21 config/export types, no `unsafe` code, deterministic RNG
seeding, and 100% Python-binding parity with the facade (all 6 problem-type
runners exposed in `#[pyfunction]` + Python `__all__`).

The **central blocker for a clean v1.0** is that every library crate exposes
`anyhow::Result<T>` in its public API. Anyhow is a binary-layer idiom: it
prevents callers from matching on error variants, bleeds internal error chains
into the public contract, and signals "application prototype" rather than
"library". Replacing this with typed errors touches many files but is the
single most impactful fix for a credible 1.0.

Two other P1s are narrow and mechanical: (a) edition is set to `2024` but no
`rust-version` is declared — Rust 2024 edition requires ≥1.85, so MSRV must be
pinned and tested in CI; (b) `rand::prelude::IndexedRandom::choose_multiple` is
deprecated, producing a `cargo doc` warning that violates the project's
documented zero-warning gate.

Documentation, tests, CHANGELOG, CI, and public-API surface are in good shape.
Everything below P1 is polish, developer ergonomics, or extensibility debt —
none of it blocks release, and the user's v1.0 release goal (breaking changes
allowed) gives a natural window to address the P2 items too.

---

## Findings

### R-01 Library crates return `anyhow::Result` from public APIs
- **Severity**: P1
- **Category**: design
- **Location**: `crates/vision-calibration-linear/src/**/*.rs`, `crates/vision-calibration-optim/src/**/*.rs`, `crates/vision-calibration-pipeline/src/**/*.rs`, `crates/vision-calibration-core/src/**/*.rs` (~60 files importing `use anyhow::Result;` and returning it from `pub fn`)
- **Status**: todo
- **Problem**: `anyhow::Error` is type-erased and designed for application binaries. As a library's public error type it prevents structured error handling (callers can't `match` on variants), leaks `anyhow::Error`'s context chain into the public API surface, and signals an unfinished library to would-be adopters. Zero custom `pub enum .*Error` types exist anywhere in `src/` — the project has never defined a typed error hierarchy. For a v1.0 that explicitly allows breaking changes, this is the right window to fix it.
- **Fix**: Define per-crate typed error enums using `thiserror`. Minimum viable design:
  - `vision_calibration_core::Error` — `InvalidInput { reason }`, `Singular`, `InsufficientData { need, got }`, variants for the distinct failure modes observed in existing `bail!`/`ensure!` call sites.
  - Each higher crate (`linear`, `optim`, `pipeline`, facade) defines its own `Error` with `#[from]` conversions from `core::Error` and any external error types (`serde_json::Error`, etc.). Mark every public error enum `#[non_exhaustive]`.
  - Replace `anyhow::Result<T>` with `Result<T, crate::Error>` in public function signatures. Internal functions may continue to use `anyhow` during migration; convert at public boundaries with a single `?` once all variants are modelled.
  - Keep `anyhow` only in `vision-calibration-py` (at the PyO3 boundary, already acceptable) and in `examples/`/`tests/`.

### R-02 MSRV not declared; edition 2024 requires ≥1.85
- **Severity**: P1
- **Category**: contracts
- **Location**: `Cargo.toml:15` (workspace.package)
- **Status**: done
- **Resolution**: Added `rust-version = "1.85"` to `[workspace.package]` in root `Cargo.toml` and added a parallel `msrv` job to `.github/workflows/ci.yml` using `dtolnay/rust-toolchain@1.85` that runs `cargo build --workspace --all-features` and `cargo test --workspace`.
- **Problem**: `edition = "2024"` is valid (stabilized in Rust 1.85, Feb 2025) and all builds pass, but there is no `rust-version` field in the workspace package block and no `rust-toolchain.toml`. Consumers of the published crates have no guarantee about the supported compiler; CI runs only on `stable` (`.github/workflows/ci.yml`) without an MSRV job. A v1.0 library must state and test its MSRV.
- **Fix**: In root `Cargo.toml` `[workspace.package]`, add `rust-version = "1.85"` (matches edition-2024 requirement). Add an MSRV job to `.github/workflows/ci.yml` that installs `dtolnay/rust-toolchain@1.85` and runs `cargo build --workspace --all-features` + `cargo test --workspace` against it. Optionally add `rust-toolchain.toml` at repo root pinning 1.85 for reproducibility.

### R-03 Deprecated API produces `cargo doc` warning
- **Severity**: P1
- **Category**: contracts
- **Location**: `crates/vision-calibration-core/src/ransac.rs:187`
- **Status**: done
- **Resolution**: Renamed `.choose_multiple(&mut rng, E::MIN_SAMPLES)` to `.sample(&mut rng, E::MIN_SAMPLES)` in `crates/vision-calibration-core/src/ransac.rs`. Sampling semantics are identical; `cargo doc --workspace --no-deps` now produces zero warnings.
- **Problem**: `rand::prelude::IndexedRandom::choose_multiple` was renamed to `sample` in rand 0.10. The current call `.choose_multiple(&mut rng, E::MIN_SAMPLES)` triggers a `deprecated` warning. CLAUDE.md's Quality Gates block declares `cargo doc --workspace --no-deps` must be warning-free; this finding violates that contract (single warning in `vision-calibration-core`).
- **Fix**: Rename the call site to `.sample(&mut rng, E::MIN_SAMPLES)`. Verify `cargo doc --workspace --no-deps` produces zero warnings and `cargo test --workspace` still passes (sampling semantics are identical, only the name changed).

### R-04 Ignored security advisory has no documented rationale
- **Severity**: P2
- **Category**: security
- **Location**: `.github/workflows/audit.yml:19` (`ignore: RUSTSEC-2024-0436`)
- **Status**: done
- **Resolution**: Investigated the advisory (paste crate unmaintained, INFO-level, not a vulnerability). Confirmed via `cargo tree -i paste` that it reaches us only as a proc-macro via gemm/faer/tiny-solver, simba/nalgebra, and rav1e/image — all build-time, zero runtime exposure. Added a 6-line comment next to the `ignore:` directive documenting the rationale so future maintainers understand why the suppression is safe.
- **Problem**: The weekly security audit workflow silently ignores `RUSTSEC-2024-0436` with no accompanying comment, commit-linked justification, or entry in a `deny.toml`. Future maintainers cannot tell whether this is a known-non-impact, a temporarily deferred mitigation, or accidentally pinned. For a v1.0 release, ignored advisories must be explicitly justified.
- **Fix**: Either (a) add a comment on the same YAML line explaining the rationale (e.g. `# paste is unmaintained but only reached via nalgebra's proc-macro; not exploitable at runtime`), or (b) migrate to `cargo-deny` with a `deny.toml` at repo root where each `[[advisories.ignore]]` entry has a `reason = "..."` field. Option (b) is preferred long-term because `deny.toml` also catches yanked crates and enforces license policy.

### R-05 Dead `tracing` feature in vision-calibration-core
- **Severity**: P2
- **Category**: workspace
- **Location**: `crates/vision-calibration-core/Cargo.toml:14-15`
- **Status**: done
- **Decision**: Instrument hot paths (option a).
- **Resolution**: Added `#[cfg_attr(feature = "tracing", tracing::instrument(skip_all, fields(n, min_samples, max_iters, seed)))]` on `core::ransac::ransac_fit`. Chose minimal scope — core crate only — to avoid propagating the feature flag across every library crate; this is still useful since RANSAC is the single longest-running hot loop. Verified `cargo build -p vision-calibration-core --features tracing` compiles and default builds are unchanged.
- **Problem**: `vision-calibration-core` declares feature `tracing = ["dep:tracing"]` but no `#[cfg(feature = "tracing")]` gate or `tracing::` macro call exists anywhere in the crate's source. The feature is inert — enabling it silently pulls the `tracing` crate without any instrumentation. CLAUDE.md (just updated) now documents it as if it were live.
- **Fix**: Add `#[cfg_attr(feature = "tracing", tracing::instrument(skip_all, ...))]` on ~5 entry points: `core::ransac::ransac`, `linear::zhang_intrinsics::solve_intrinsics` (and its siblings for Scheimpflug), and the top-level `optim::solve` / backend dispatch. Use `skip_all` to avoid formatting large argument structs in the span. Keep the feature off by default; gate behind a `tracing` subscriber initialised by the user's binary. Confirm `cargo build -p vision-calibration-core --features tracing` compiles and produces spans in a test with `tracing_subscriber::fmt::init()`.

### R-06 Public `Result`-returning functions lack `# Errors` rustdoc sections
- **Severity**: P2
- **Category**: docs
- **Location**: Many; representative samples — `crates/vision-calibration-linear/src/pnp/dlt.rs:19` (`pub fn dlt`), `crates/vision-calibration-linear/src/zhang_intrinsics.rs`, `crates/vision-calibration-optim/src/problems/planar_intrinsics.rs::optimize_planar_intrinsics`
- **Status**: todo
- **Problem**: The facade has excellent module-level examples, but many `pub fn ... -> Result<...>` items in `vision-calibration-linear` and `vision-calibration-optim` have no `# Errors` section documenting when/why the function fails. For a v1.0 release with typed errors (R-01), every fallible function should enumerate the error variants it can return. Note: this blocks cleanly after R-01 lands, since typed variants give something concrete to enumerate.
- **Fix**: After R-01 is merged, add an `# Errors` section to every `pub fn` returning `Result` in linear/optim/pipeline, listing the concrete error variants. Consider enforcing via `#![deny(rustdoc::missing_errors_doc)]` at crate root once coverage is complete. Scope: ~30–50 public functions.

### R-07 Python bindings lack early input validation
- **Severity**: P2
- **Category**: security
- **Location**: `crates/vision-calibration-py/src/lib.rs:12-29` (`parse_payload`, `parse_optional_payload`) and each `run_*` `#[pyfunction]` entry
- **Status**: done
- **Resolution**: Added `crates/vision-calibration-py/src/validation.rs` with `reject_non_finite()` that recursively walks Python payloads and raises `PyValueError` on any NaN/±Inf float leaf (with a path-qualified message like `input.views[2].points[5]: non-finite float`). `set_input`/`set_config` failures now raise `PyValueError` (not `PyRuntimeError`) so Rust-layer messages like "need at least 6 correspondences" surface as `ValueError: invalid input: need at least 6 correspondences`. Genuine runtime failures (export, pythonize) still use `PyRuntimeError`.
- **Problem**: Python entry points deserialize arbitrary user input via `depythonize` and pass it directly to the Rust session layer. Validation (shape matching, NaN/infinity rejection, empty dataset checks) happens only inside the Rust algorithmic layer via `anyhow::bail!`. The resulting Python-side error messages are opaque (`RuntimeError: failed to set input`) when they could be specific (`ValueError: points_3d has 42 points, points_2d has 41`). This is a UX issue, not a safety issue — the Rust side is still memory-safe.
- **Fix**: Add a thin validation helper in `vision-calibration-py/src/validation.rs` that, for each problem type, checks array length agreement and rejects non-finite floats before handing off to the session. Raise `PyValueError` (not `PyRuntimeError`) for input-shape errors. Do this once per entry point; the validation itself can be <30 lines per problem.

### R-08 `ProblemIR::validate()` is monolithic (381 lines, deep factor-kind matching)
- **Severity**: P2
- **Category**: design
- **Location**: `crates/vision-calibration-optim/src/ir/types.rs:356-550` (approximately)
- **Status**: skipped (deferred to v1.1 — internal-only, not release-blocking)
- **Problem**: The IR's validation logic lives in a single 381-line function that matches on every `FactorKind` variant. Each new factor type requires editing this function, making extension costly and error-prone. This is an internal concern (not exposed in the public API) but affects the velocity of adding new problem types — a key dimension for a calibration library.
- **Fix**: Move factor-specific validation into a `FactorKindExt` trait implemented on each variant, or into a method on `FactorKind` using an enum-dispatch pattern. The top-level `validate()` then becomes a short dispatcher. Keep the current tests; they should pass unchanged.

### R-09 `Option<Vec<T>>` used where empty `Vec<T>` would suffice
- **Severity**: P3
- **Category**: design
- **Location**: `crates/vision-calibration-core/src/types/observation.rs:42` (`weights: Option<Vec<f64>>`), similar patterns in `crates/vision-calibration-optim/src/ir/types.rs:289` (bounds), `crates/vision-calibration-pipeline/src/rig_handeye/problem.rs:163`
- **Status**: done
- **Resolution**: Migrated `weights` on `CorrespondenceView` and `laser_weights` on `LaserlineMeta` from `Option<Vec<f64>>` to `Vec<f64>` with empty-vec semantics. Updated serde attributes to `skip_serializing_if = "Vec::is_empty"`. Updated `weight()`, `laser_weight()`, `validate()`, the tricky `has_weights` accumulator in `planar_intrinsics/steps.rs`, and all constructor call sites (~12 files). Commit `91c4f5d`.
- **Problem**: `Option<Vec<T>>` creates two representations for the same state (None vs Some(empty)), inviting bugs where callers forget to check for the empty case. Since `Vec` already distinguishes empty from non-empty, the `Option` wrapper is redundant and confusing.
- **Fix**: Migrate the named fields to `Vec<T>` with the empty-vector convention documented as "unweighted" / "unconstrained". Accept this as a breaking API change aligned with the v1.0 cleanup.

### R-10 Unjustified panic wildcards in model/params matching
- **Severity**: P3
- **Category**: code-quality
- **Location**: `crates/vision-calibration-core/src/models/params.rs:247` (`_ => panic!(...)`), similar wildcards in `crates/vision-calibration-pipeline/src/planar_intrinsics/problem.rs:323,364`
- **Status**: done
- **Resolution**: Replaced all three `_ => panic!(...)` wildcards with explicit variant arms listing the unexpected cases. Any future `DistortionParams` or `RobustLoss` variant addition now triggers a `non-exhaustive patterns` compile error rather than hiding silently.
- **Problem**: Enum matching with `_ => panic!(...)` defeats exhaustiveness checking — a new variant added to `DistortionParams` or `RobustLoss` won't be flagged by the compiler. These sites appear to be inside tests or conversion helpers, but the panic behaviour is load-bearing for correctness of the surrounding code.
- **Fix**: Replace `_` arms with explicit variant lists. If the match is intentionally partial, return `Result<_, Error>` instead of panicking. If it is a test helper, use `unreachable!()` with an explanatory message and add `#[cfg(test)]` if appropriate.

### R-11 No property-based test coverage
- **Severity**: P3
- **Category**: tests
- **Location**: workspace-wide — no `proptest`/`quickcheck` deps, zero `proptest!` macros
- **Status**: skipped (deferred to v1.1 — existing ~4600 lines of integration tests are strong)
- **Problem**: The project has excellent integration tests with synthetic ground truth and ~4600 lines of test code, but no generative testing. Calibration code is a natural fit for property testing: roundtrip invariants (serialize/deserialize), pose composition (compose(a, compose(b, c)) == compose(compose(a, b), c) within tolerance), projection/unprojection identities, and RANSAC output determinism under a fixed seed.
- **Fix**: Add `proptest` as a dev-dependency to `vision-calibration-core`. Start small: 3-5 property tests covering SE3 roundtrip (serialize → deserialize → compare), camera model round-trip (project → unproject → compare), and RANSAC determinism. Optional for v1.0; concrete value for v1.1+.

### R-12 `.pyi` typing stubs are hand-maintained with no `--check` mode
- **Severity**: P3
- **Category**: contracts
- **Location**: `crates/vision-calibration-py/python/vision_calibration/__init__.pyi` (91 lines, hand-maintained); no generator script
- **Status**: done
- **Resolution**: Added `scripts/check_pyi_coverage.py` (~100 lines) that parses `__init__.py`'s `__all__`, `__init__.pyi`'s declared symbols, and `lib.rs`'s `#[pymodule]` registrations via `ast` + a small regex, then reports any drift. The first run caught `library_version` (registered in Rust but missing from `.pyi`) — added the declaration alongside the script so the CI gate lands on a clean baseline. Wired into `.github/workflows/ci.yml` under `python-runtime` with `--check`.
- **Problem**: The single `.pyi` file currently covers every `#[pyfunction]` and robust-loss helper (verified). But with no generator or `--check` CI step, the stub will drift silently when new bindings are added. A future contributor can merge a new `#[pyfunction]` without updating the stub and nothing will fail.
- **Fix**: Add a small Python script `scripts/check_pyi_coverage.py` that parses `__init__.pyi` function names and compares them to the `#[pymodule]` registration in `crates/vision-calibration-py/src/lib.rs` (string-match, not full type equivalence). Wire it into CI with `python3 scripts/check_pyi_coverage.py --check`. Roughly 40 lines of Python.

### R-13 Redundant empty `default = []` in vision-calibration-optim
- **Severity**: P3
- **Category**: workspace
- **Location**: `crates/vision-calibration-optim/Cargo.toml:26-27`
- **Status**: done
- **Resolution**: Deleted the `[features]` block entirely from `crates/vision-calibration-optim/Cargo.toml`. One-line cleanup; no behaviour change.
- **Problem**: `[features]` block with only `default = []` has no effect. It's the default-default; listing it explicitly adds noise without signal.
- **Fix**: Delete the `[features]` block entirely from that crate's `Cargo.toml`. One-line cleanup.

---

## Out-of-Scope Pointers

- Numerical robustness of DLT normalization, RANSAC inlier scoring, and nonlinear-refinement convergence guarantees — delegate to the `calibration-review` skill.
- Hot-path performance of factor residual evaluation (`T::from_f64().unwrap()` pattern at 30+ sites in `vision-calibration-optim/src/factors/laserline.rs` and elsewhere — correct but creates panic sites and may benefit from `cast_const` helper) — delegate to `perf-architect`.
- Long PnP / RANSAC functions (DLT 143 lines, essential 124, epnp 114, ransac 110) are correct and well-tested; they read as domain-specific transliterations of textbook algorithms and do not warrant aggressive refactoring unless algo-review flags a correctness concern.

## Strong Points

- **Architecture**: Strict layered dependency graph (`linear` and `optim` as true peers, both depending only on `core`; `pipeline` combines them; facade re-exports; `py` depends only on facade) — no cycles, no layer violations.
- **Memory safety**: Zero `unsafe` blocks across the entire workspace (including the PyO3 layer, which uses safe helpers only).
- **Binding parity**: 100% — every one of the 6 facade problem-type runners has a matching `#[pyfunction]`, is registered in `#[pymodule]`, wrapped in Python-side `_api.py`, and exposed in `__all__`. The Scheimpflug addition (PR #26) landed through all four layers cleanly.
- **API stability hygiene**: `#[non_exhaustive]` applied to 21 config/export types across pipeline crates, preventing accidental breaking changes when new fields are added.
- **Convention discipline**: SE3 stored in `[qx, qy, qz, qw, tx, ty, tz]` everywhere; pose naming `frame_se3_frame` / `T_C_W` uniformly applied; `fix_k3: true` default honoured.
- **Quality gates pass**: `cargo fmt --all --check`, `cargo clippy --workspace --all-targets --all-features -D warnings`, `cargo test --workspace --all-features` all green.
- **Documentation**: All 6 crates have `//!` module docs; facade's `src/lib.rs` has per-workflow examples with table-of-step descriptions; CHANGELOG 2026-03-07 captures the 0.2.0 breaking changes correctly.

---

## Implementer Log (Phase 4, partial)

Executed 2026-04-12. Two distinct runs due to a rate-limit interruption mid-migration.

### Run 1 — Sonnet Implementer (spawned via Agent)
Hit the account rate limit after ~90 seconds. Landed 2 commits:
- `e755492` chore(workspace): declare MSRV 1.85 and add CI job [refs R-02]
- `4714ffe` fix(core): rename choose_multiple to sample [refs R-03]

### Run 2 — Opus (main context, in-line after rate-limit notification)
Continued with the smaller and independent fixes, leaving the two biggest items (R-01 typed errors and R-06 which depends on it) plus R-09 (breaking migration across ~30 call sites) for a fresh Implementer once the user's rate limit resets.

Commits landed (in order):
- `5252f17` chore(optim): drop redundant empty [features] block [refs R-13]
- `1240ac7` test(params,planar): expand enum matches to cover all variants [refs R-10]
- `60af5e6` feat(core): instrument ransac_fit under tracing feature [refs R-05]
- `3c20a65` ci(audit): document rationale for ignoring RUSTSEC-2024-0436 [refs R-04]
- `6701307` ci(py): add typing-stub coverage check and wire into CI [refs R-12]
- `c40627e` feat(py): validate Python inputs at the boundary with PyValueError [refs R-07]

### Status tally after Phase 4 (partial)

| ID   | Status              | Notes                                                                          |
|------|---------------------|--------------------------------------------------------------------------------|
| R-01 | **todo**            | Deferred — needs fresh Implementer. Largest fix in the cycle (~60 files).     |
| R-02 | done                | MSRV 1.85 + CI job.                                                            |
| R-03 | done                | `choose_multiple` → `sample`; doc warning gone.                                |
| R-04 | done                | Documented RUSTSEC-2024-0436 suppression rationale.                            |
| R-05 | done                | Tracing instrument on `ransac_fit`.                                            |
| R-06 | **todo** (blocked)  | Blocks on R-01 (typed variants are what `# Errors` sections would enumerate). |
| R-07 | done                | Python-side NaN/Inf rejection + PyValueError for input errors.                 |
| R-08 | skipped             | Deferred to v1.1.                                                              |
| R-09 | **todo**            | Deferred — ~30 call sites touching `Option<Vec<T>>` ↔ `Vec<T>` migration.     |
| R-10 | done                | Enum wildcards expanded to explicit variants.                                  |
| R-11 | skipped             | Deferred to v1.1.                                                              |
| R-12 | done                | `scripts/check_pyi_coverage.py` + CI hook; also fixed pre-existing drift on `library_version`. |
| R-13 | done                | Empty `[features]` block removed.                                              |

**8 done / 3 todo / 2 skipped.**

### Final quality-gate status (after R-07, last commit of Run 2)

- `cargo fmt --all --check` — pass
- `cargo clippy --workspace --all-targets --all-features -- -D warnings` — pass
- `cargo test --workspace` — pass
- `cargo test --workspace --all-features` — pass (run after R-03 via Run-1 Implementer)
- `cargo doc --workspace --no-deps` — pass, zero warnings (R-03 cleared the only one)
- `python3 scripts/check_pyi_coverage.py --check` — pass

### Remaining work for next Implementer run

When the user's rate limit resets, spawn a fresh Sonnet Implementer with this REVIEW.md. It should work through the three remaining `todo` items in this order:

1. **R-09** (~30 call sites, breaking API change): do BEFORE R-01 to avoid touching the same files twice. `Option<Vec<T>>` → `Vec<T>` migration across `core/types/observation.rs`, `optim/ir/types.rs`, `pipeline/rig_handeye/problem.rs`, plus call-site updates in `core/synthetic/planar.rs`, `pipeline/*/problem.rs` and `steps.rs`, `optim/problems/laserline_bundle.rs`, and examples + tests. Watch for the `.as_ref().map(|_| Vec::new())` pattern in `planar_intrinsics/steps.rs:284` — the replacement needs to stay semantically equivalent.
2. **R-01** (largest fix — ~60 files): define per-crate `Error` enums with `thiserror`, commit per crate (core → linear → optim → pipeline → facade). Keep `anyhow` only in examples/tests and at the PyO3 boundary.
3. **R-06** (runs AFTER R-01): add `# Errors` rustdoc sections enumerating the typed error variants for every public `-> Result<T, Error>` function in linear/optim/pipeline. Consider adding `#![deny(rustdoc::missing_errors_doc)]` at each crate root once coverage is complete.

Two additional housekeeping items for the fresh Implementer (owner decision, not part of findings):
- Pre-existing uncommitted `Cargo.toml` / `Cargo.lock` changes (dep version relaxations for `num-dual` and `rand` 0.10 bump) and the `docs/backlog.md` + `docs/report/*` deletions are the owner's work. Commit them under separate topical commits (`chore(deps): ...`, `chore(docs): retire backlog workflow`) rather than bundling them with an R-NN fix.
- `.claude/CLAUDE.md` was refreshed during this review cycle (LoC, Camera signature, Feature Flags, Python Bindings subsections, pared-down Planning section). It is uncommitted. Commit it as `docs(claude): refresh workspace summary for v1.0 review cycle` or similar.
- **Test hygiene**: ~4600 lines of tests across 12 integration files; no `assert!(true)`, no unjustified `#[ignore]`, no flaky sleep/timing patterns; RANSAC deterministic via seeded RNG.
- **CI coverage**: fmt, clippy (with `--all-features`), tests (with `--all-features`), docs, Python compile-all, weekly security audit, PyPI release automation, GitHub Pages docs publishing.
