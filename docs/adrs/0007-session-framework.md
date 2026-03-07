# ADR 0007: Session Framework with External Step Functions

- Status: Accepted
- Date: 2026-03-07 (retroactive)

## Context

Calibration workflows involve multiple sequential stages (initialization, optimization, filtering). Users need to inspect intermediate state, checkpoint progress, and customize individual steps. A monolithic `calibrate()` function doesn't support this.

## Decision

Use a **mutable state container** (`CalibrationSession<P>`) with **external step functions**:

- `ProblemType` trait defines associated types: `Config`, `Input`, `State`, `Output`, `Export`.
- `CalibrationSession<P>` holds config, input, state, output, and metadata.
- Step functions are free functions: `fn step_X(session: &mut CalibrationSession<P>, opts: Option<StepOpts>) -> Result<()>`.
- Each problem module also provides a `run_calibration()` convenience wrapper.

Session features:
- JSON serialization for checkpointing (`to_json` / `from_json`).
- Invalidation policy: input changes clear computed state.
- Audit log: timestamped entries for each step.

## Consequences

- Adding a new problem type is mechanical: implement `ProblemType`, write step functions, create module.
- Step functions compose without inheritance or complex trait hierarchies.
- Python bindings map naturally to `run_problem(input_json, config_json) -> export_json`.
- Trade-off: mutable session pattern is less Rust-idiomatic than a builder/pipeline, but matches the interactive calibration workflow well.
