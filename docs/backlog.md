# Backlog and Milestones

Planning model:

- Architecture decisions live in `docs/adrs/`.
- Execution tracking lives in this backlog.
- Automated workflow: `/orchestrate`, `/architect`, `/implement`, `/review`, `/gate-check`.

Execution workflow:

- Each implemented task must produce:
  1. A concise report in `docs/report/`
  2. A status update in this backlog
  3. A dedicated git commit
- Task IDs use `M<milestone>-T<nn>` (example: `M1-T03`).

---

## Completed Milestones

<details>
<summary>M0: Workflow Enablement (Done)</summary>

- [x] `M0-T01` Define mandatory backlog workflow. (Done: 2026-03-07)
- [x] `M0-T02` Allow coupled-task bundling. (Done: 2026-03-07)
</details>

<details>
<summary>M1: Scheimpflug Pipeline Conformance (Done)</summary>

- [x] `M1-T01`..`M1-T06` Scheimpflug ProblemType migration. (Done: 2026-03-07)
</details>

<details>
<summary>M3: Python and Facade Alignment (Done)</summary>

- [x] `M3-T01`..`M3-T03` Python bindings unified via run_problem. (Done: 2026-03-07)
</details>

<details>
<summary>M4: Documentation and Release Readiness (Done)</summary>

- [x] `M4-T01`..`M4-T04` Docs, examples, CI gates, release notes. (Done: 2026-03-07)
</details>

---

## Active Milestones — v1.0 Public API Release

### M5: Facade API Cleanup

Goal: Clean, module-first public API with consistent naming. Breaking changes allowed.

ADR links: 0003, 0006, 0007, 0010

- [x] `M5-T01` Remove flat re-exports from `vision-calibration-pipeline/src/lib.rs`. Keep only module declarations and the `session` module re-export. Each problem type accessed via `crate::planar_intrinsics::*`, not top-level. (Done: 2026-03-07)
- [x] `M5-T02` Clean up facade `vision-calibration/src/lib.rs`: remove `core` glob re-export (`pub use vision_calibration_core::*`), replace with explicit type list. Remove `handeye` escape-hatch module. (Done: 2026-03-07)
- [x] `M5-T03` Standardize step option naming via ADR 0010: use `<Stage><Action>Options` with explicit stage in every module and full action words (`Optimize`, not `Optim`), with hard renames (no compatibility aliases). (Done: 2026-03-07)
- [x] `M5-T04` Standardize config type naming to `<ProblemName>Config` for top-level problem configs: `PlanarConfig` -> `PlanarIntrinsicsConfig`, `ScheimpflugIntrinsicsCalibrationConfig` -> `ScheimpflugIntrinsicsConfig`. Nested config audit result: keep existing stage-grouped nested configs where already present (for example `LaserlineDevice*Config`, `RigHandeye*Config`) and keep flat configs where they remain clear. (Done: 2026-03-07)
- [x] `M5-T05` Standardize export types: all problem modules now expose distinct `<ProblemName>Export` structs and include consistent top-level `mean_reproj_error` + `per_cam_reproj_errors` fields (single-camera problems expose one-element `per_cam_reproj_errors`). (Done: 2026-03-07)
- [x] `M5-T06` Remove `run_calibration_direct` from Scheimpflug. All problem types should only have session-based API. (Done: 2026-03-07)
- [x] `M5-T07` Audit and trim `prelude` — it should contain only the types needed for the "hello world" calibration, not all problem types. (Done: 2026-03-07)

Acceptance criteria:
- Pipeline lib.rs has no flat re-exports (only `pub mod` + session re-exports).
- Facade uses explicit re-exports, no globs.
- Each problem module is self-contained — no naming collisions when using multiple modules.
- `cargo doc` shows clean, navigable module hierarchy.

### M6: Planar Family Consolidation

Goal: Unify shared logic between standard planar and Scheimpflug intrinsics. (Formerly M2)

ADR links: 0002, 0005

- [x] `M6-T01` Extract shared planar initialization logic (Zhang's method, pose recovery) into internal helper functions in pipeline. (Done: 2026-03-07)
- [ ] `M6-T02` Extract shared optimization setup (param blocks, residual construction) into shared code in optim.
- [ ] `M6-T03` Ensure both planar and Scheimpflug step functions call the shared helpers.
- [ ] `M6-T04` Document the planar family relationship in ADR 0002 update.

Acceptance criteria:
- Shared logic exists in one place.
- Both problem types still have separate `ProblemType` implementations.
- No duplicated algorithm code between the two modules.

### M7: Rustdoc and API Documentation

Goal: Every public type and function has rustdoc with examples.

- [ ] `M7-T01` Audit all public items in `vision-calibration-core` — add missing rustdoc.
- [ ] `M7-T02` Audit all public items in `vision-calibration-linear` — add missing rustdoc.
- [ ] `M7-T03` Audit all public items in `vision-calibration-optim` — add missing rustdoc.
- [ ] `M7-T04` Audit all public items in `vision-calibration-pipeline` — add missing rustdoc.
- [ ] `M7-T05` Audit facade crate rustdoc — ensure module-level docs guide users to the right starting point.
- [ ] `M7-T06` Add `#[doc(hidden)]` to internal implementation details that leak through re-exports.
- [ ] `M7-T07` Update book to match finalized API surface.

Acceptance criteria:
- `cargo doc --workspace --no-deps` produces no warnings.
- Every public type has at least a one-line doc comment.
- Every problem module has a usage example in its module doc.

### M8: API Hardening

Goal: Make the API resilient to accidental breaking changes.

- [ ] `M8-T01` Add `#[non_exhaustive]` to all public config, export, and error enums/structs.
- [ ] `M8-T02` Review `Serialize`/`Deserialize` derives — ensure all JSON-facing types have them, internal types don't.
- [ ] `M8-T03` Add integration tests that exercise the facade API as an external user would (compile-only tests with `use vision_calibration::*`).
- [ ] `M8-T04` Pin JSON schema versions in session metadata and add schema version validation on deserialization.
- [ ] `M8-T05` Add `deny(missing_docs)` to facade crate.

Acceptance criteria:
- All public config/export/error types are `#[non_exhaustive]`.
- Facade crate compiles with `deny(missing_docs)`.
- Schema version mismatch produces a clear error.

### M9: Python API Parity

Goal: Python bindings match Rust API capabilities.

- [ ] `M9-T01` Audit Python type stubs (`__init__.pyi`) against current Rust exports.
- [ ] `M9-T02` Add step-by-step Python API (not just `run_all`) for at least `PlanarIntrinsicsProblem`.
- [ ] `M9-T03` Add Python-side config validation with clear error messages.
- [ ] `M9-T04` Update Python examples to match finalized API.
- [ ] `M9-T05` Add Python integration tests covering all 6 problem types.

Acceptance criteria:
- Python stubs match Rust serde contracts.
- At least one problem type supports step-by-step Python API.
- All Python examples run without errors.

---

## Standard Gate Checklist (Per Milestone Completion)

- [ ] `cargo fmt --all -- --check`
- [ ] `cargo clippy --workspace --all-targets --all-features -- -D warnings`
- [ ] `cargo test --workspace --all-features`
- [ ] `cargo doc --workspace --no-deps` (no warnings)
- [ ] `python3 -m compileall crates/vision-calibration-py/python/vision_calibration`
