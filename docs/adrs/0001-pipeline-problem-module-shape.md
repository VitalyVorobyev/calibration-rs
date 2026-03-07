# ADR 0001: Pipeline Problem Module Shape

- Status: Accepted
- Date: 2026-03-07

## Context

`vision-calibration-pipeline` is built around `ProblemType`-driven session orchestration.
Existing workflows (`planar_intrinsics`, `single_cam_handeye`, `rig_extrinsics`, `rig_handeye`, `laserline_device`) follow this shape:

- module directory with `mod.rs`
- `problem.rs` for `ProblemType` + config + export contract
- `state.rs` for intermediate pipeline state
- `steps.rs` for step functions + convenience `run_calibration`

Current Scheimpflug flow is implemented as a single file:
[scheimpflug_intrinsics.rs](/Users/vitalyvorobyev/vision/calibration-rs/crates/vision-calibration-pipeline/src/scheimpflug_intrinsics.rs).
It does not define a `ProblemType` and bypasses session state/checkpointing patterns.

## Decision

All end-to-end workflows in `vision-calibration-pipeline` must be modeled as `ProblemType` modules with `problem.rs`, `state.rs`, and `steps.rs`.

Direct-function pipeline implementations are temporary only and must be migrated to the standard module shape before release.

## Consequences

Positive:

- Consistent lifecycle: validation, invalidation policy, step logging, export behavior.
- Uniform session/checkpoint support across workflows.
- Lower duplication in bindings and facade docs.

Negative:

- Migration cost for current Scheimpflug implementation.
- Temporary compatibility wrappers may be needed while migrating.

## Required Follow-up

- Migrate Scheimpflug pipeline from single-file direct function to `ProblemType` module shape.
- Keep public API compatibility in `vision-calibration` via re-exports/adapters during migration.
