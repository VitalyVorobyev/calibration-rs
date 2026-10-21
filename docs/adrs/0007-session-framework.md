# ADR 0007: Session Framework with External Step Functions

## Context

A calibration runs in stages (initialization, optimization, filtering).
Users need to inspect intermediate state, checkpoint it, seed or replace
individual stages, and drive the same workflow from Rust, Python and the
app. A monolithic `calibrate()` supports none of that.

## Decision

Every end-to-end workflow is a `ProblemType` driven through a mutable
`CalibrationSession<P>` by free step functions.

- `ProblemType` defines `Config`, `Input`, `State`, `Output` and `Export`,
  plus `name()` and `schema_version()` (bumped when the wire format of an
  associated type changes).
- `CalibrationSession<P>` holds config, input, state, output and a log.
  Changing the input clears computed state. The session serializes to JSON
  for checkpointing.
- Step functions are free functions,
  `fn step_x(session: &mut CalibrationSession<P>, opts: Option<Opts>) -> Result<()>`,
  and each problem module adds a `run_calibration` convenience wrapper.
- A problem module lives in `vision-calibration-pipeline/src/<name>/` with
  `problem.rs` (the `ProblemType`, config and export), `state.rs` and
  `steps.rs` (step functions and `run_calibration`). There are no
  direct-function pipelines.
- The facade re-exports the modules. Python drives every problem type
  through one generic `run_problem(input_json, config_json) -> export_json`
  binding, with no per-workflow wrapper.

## Consequences

- A new problem type is mechanical: a module, a `ProblemType`, its steps
  (AGENTS.md §1 lists the steps).
- Steps compose without trait hierarchies, and a caller can run, replace or
  seed any one of them (ADR 0011).
- The mutable session is less idiomatic than a builder, but it matches the
  interactive inspect-and-adjust workflow.
