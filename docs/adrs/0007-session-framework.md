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
  for checkpointing. It also carries the run control its steps report
  progress to and poll for cancellation (ADR 0029); that is a handle, not
  state, and is never serialized.
- Step functions are free functions,
  `fn step_x(session: &mut CalibrationSession<P>, opts: Option<Opts>) -> Result<()>`,
  and each problem module adds a `run_calibration` convenience wrapper.
- A problem module lives in `vision-calibration-pipeline/src/<name>/` with
  `problem.rs` (the `ProblemType`, config and export), `state.rs` and
  `steps.rs` (step functions and `run_calibration`). There are no
  direct-function pipelines.
- Problem types that share stages share their step bodies, not copies of
  them. The rig problem types (`rig_extrinsics`, `rig_handeye`) open with
  the same four rig stages (per-camera intrinsics init and refinement, rig
  init, rig bundle adjustment); they implement the crate-private
  `rig_family::RigStages` trait (input as a rig dataset, per-camera config,
  stage settings, a `RigStagesState` flattened into the problem's state, so
  the session JSON keeps its shape), and their public `step_*` functions
  delegate to the generic stage bodies. The stages' seed and result types
  are defined once and re-exported by each module. The hand-eye problem
  types (`single_cam_handeye`, `rig_handeye`) close with the same two
  hand-eye stages (initialization, bundle adjustment) through
  `rig_family::HandeyeStages`: a single camera is the one-camera rig whose
  frame is the camera's. Their seeds follow one rule: a seeded pose is
  used as given, one seeded pose closes the other through the first view,
  and the closed-form solve runs only with neither seeded; a run clears
  the previous initialization once it starts. Single-camera seed checks
  and step-log provenance are shared the same way through
  `planar_family`. A difference between problem types is a trait item, not
  a forked copy.
- The facade re-exports the modules. Python drives every problem type
  through one generic `run_problem(input_json, config_json) -> export_json`
  binding, with no per-workflow wrapper.
- `ProblemType` is sealed (a crate-private supertrait): the workflows are
  the library's own. A pipeline of a user's own is composed from public
  pieces instead, the closed-form initializers (`init`), a `Problem` and
  `solve` (ADR 0027), so its steps, state and export are not API.

## Consequences

- A new problem type is mechanical: a module, a `ProblemType`, its steps
  (AGENTS.md §10 lists the steps).
- Steps compose without trait hierarchies, and a caller can run, replace or
  seed any one of them (ADR 0011).
- The mutable session is less idiomatic than a builder, but it matches the
  interactive inspect-and-adjust workflow.
