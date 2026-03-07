# Backlog and Milestones

Planning model:

- Architecture decisions live in `docs/adrs/`.
- Execution tracking lives in this backlog.
- `IMPLEMENTATION_PLAN.md` is removed by ADR 0004 and must not be restored.

Execution workflow:

- Each implemented task must produce:
  1. A concise report in `docs/report/`
  2. A status update in this backlog
  3. A dedicated git commit
- Task IDs use `M<milestone>-T<nn>` (example: `M1-T03`).

## Review Findings Snapshot

Findings from project-structure review (2026-03-07):

1. `major` — Scheimpflug workflow in pipeline is not modeled as `ProblemType` with `problem/state/steps`.
Location: [scheimpflug_intrinsics.rs](/Users/vitalyvorobyev/vision/calibration-rs/crates/vision-calibration-pipeline/src/scheimpflug_intrinsics.rs:1)
2. `major` — Python binding path for Scheimpflug is special-cased and bypasses generic session wrapper.
Location: [lib.rs](/Users/vitalyvorobyev/vision/calibration-rs/crates/vision-calibration-py/src/lib.rs:219)
3. `major` — Planar and Scheimpflug intrinsics flows duplicate orchestration logic and are not yet unified as a documented workflow family implementation.
Locations: [planar_intrinsics/mod.rs](/Users/vitalyvorobyev/vision/calibration-rs/crates/vision-calibration-pipeline/src/planar_intrinsics/mod.rs:33), [scheimpflug_intrinsics.rs](/Users/vitalyvorobyev/vision/calibration-rs/crates/vision-calibration-pipeline/src/scheimpflug_intrinsics.rs:122)
4. `minor` — Facade surface mixes session workflows and one direct workflow path for planar family.
Location: [lib.rs](/Users/vitalyvorobyev/vision/calibration-rs/crates/vision-calibration/src/lib.rs:76)

## Milestones Before Next Release

### M0: Workflow Enablement

ADR links: 0004

- [x] `M0-T01` Define mandatory backlog workflow in `AGENTS.md`, add `docs/report/` template and naming convention. (Done: 2026-03-07)
- [x] `M0-T02` Allow coupled-task bundling with explicit documentation when independent commits would break build/API continuity. (Done: 2026-03-07)

### M1: Scheimpflug Pipeline Conformance (Release-Blocking)

ADR links: 0001, 0002, 0003

- [x] `M1-T01` Create `crates/vision-calibration-pipeline/src/scheimpflug_intrinsics/` directory with `mod.rs`, `problem.rs`, `state.rs`, `steps.rs`. (Done: 2026-03-07, bundled with M1-T02..T06)
- [x] `M1-T02` Introduce `ScheimpflugIntrinsicsProblem` implementing `ProblemType` with explicit `name`, `schema_version`, input/config validation, export contract. (Done: 2026-03-07)
- [x] `M1-T03` Move current direct-function implementation into step functions (`step_init`, `step_optimize`, optional convenience `run_calibration`). (Done: 2026-03-07)
- [x] `M1-T04` Add `ScheimpflugIntrinsicsState` with initialization/optimization intermediate state and JSON roundtrip tests. (Done: 2026-03-07)
- [x] `M1-T05` Add problem-level tests for validation, config roundtrip, and export behavior. (Done: 2026-03-07; includes facade-level Scheimpflug integration tests)
- [x] `M1-T06` Keep a compatibility path in `vision-calibration::scheimpflug_intrinsics` so external callers are not broken during migration. (Done: 2026-03-07)

Acceptance criteria:

- Scheimpflug follows same pipeline shape as other workflows.
- No direct-function-only pipeline module remains.
- All existing and new Scheimpflug tests pass.

### M2: Planar Family Consolidation

ADR links: 0002

- [ ] `M2-T01` Extract shared planar initialization/pose helper logic into internal reusable code.
- [ ] `M2-T02` Remove avoidable duplication between standard planar and Scheimpflug paths.
- [ ] `M2-T03` Define explicit long-term contract for sensor-model variants (identity vs Scheimpflug) and document migration path.
- [ ] `M2-T04` Evaluate whether optimizer-level unification is feasible without breaking `PlanarIntrinsicsParams`/`PlanarIntrinsicsEstimate` contracts.

Acceptance criteria:

- Shared logic exists in one place.
- Sensor-model behavior differences stay explicit in types/config.
- No hidden mode branching that obscures API contracts.

### M3: Python and Facade Alignment

ADR links: 0003

- [x] `M3-T01` Switch Scheimpflug Python entrypoint to generic `run_problem` path once `ScheimpflugIntrinsicsProblem` exists. (Done: 2026-03-07)
- [x] `M3-T02` Keep `models.py`, `types.py`, and `__init__.pyi` synchronized with final Rust config/export contracts. (Done: 2026-03-07; package contract files updated and synchronized)
- [x] `M3-T03` Add Python tests for session-style Scheimpflug config handling and error mapping consistency. (Done: 2026-03-07)

Acceptance criteria:

- Python bindings use one workflow wiring pattern across all problem types.
- Typed contracts remain consistent with Rust serde payloads.

### M4: Documentation and Release Readiness

ADR links: 0001, 0002, 0003, 0004

- [x] `M4-T01` Update book/docs for finalized Scheimpflug pipeline shape and usage. (Done: 2026-03-07)
- [x] `M4-T02` Add/refresh minimal Rust and Python examples that reflect final contracts. (Done: 2026-03-07)
- [x] `M4-T03` Confirm CI and release workflows enforce required gates (fmt, clippy all-features, tests all-features, python compileall, python runtime tests). (Done: 2026-03-07; clippy follow-up fixes applied)
- [x] `M4-T04` Add release notes/migration notes for any API surface changes. (Done: 2026-03-07)

Acceptance criteria:

- Documentation matches actual API shape.
- Gate commands pass locally and in CI.
- Release notes clearly describe user-visible changes.

## Standard Gate Checklist (Per Milestone Completion)

- [ ] `cargo fmt --all`
- [ ] `cargo clippy --workspace --all-targets --all-features -- -D warnings`
- [ ] `cargo test --workspace --all-features`
- [ ] `cargo test -p vision-calibration-core`
- [ ] `cargo test -p vision-calibration`
- [ ] `cargo test -p vision-calibration-py`
- [ ] `python3 -m compileall crates/vision-calibration-py/python/vision_calibration`
- [ ] Python runtime tests after extension build (`maturin develop` + unittest/pytest suite)
