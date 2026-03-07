# ADR 0003: Facade and Python API Consistency

- Status: Accepted
- Date: 2026-03-07

## Context

`vision-calibration` is the compatibility boundary and should expose cohesive workflow contracts.
Python bindings mostly use a generic session wrapper (`run_problem`) for `ProblemType` workflows:
[lib.rs](/Users/vitalyvorobyev/vision/calibration-rs/crates/vision-calibration-py/src/lib.rs).

Scheimpflug currently uses a special-case binding path instead of `run_problem`,
because it is not modeled as a `ProblemType`.

## Decision

Public APIs should map 1:1 with pipeline workflow contracts:

- Rust facade re-exports workflow modules/contracts from pipeline.
- Python bindings use the generic `run_problem` path whenever the workflow is a `ProblemType`.
- Special-case Python wrappers are allowed only for transitional compatibility and must be removed after migration.

## Consequences

Positive:

- Predictable behavior across Rust and Python entry points.
- Fewer bespoke conversion/error-handling paths.
- Easier maintenance and lower schema drift risk.

Negative:

- Requires short-lived compatibility code during migration.

## Required Follow-up

- After Scheimpflug `ProblemType` migration, switch `run_scheimpflug_intrinsics`
  to the generic session-based wrapper path in Python bindings.
