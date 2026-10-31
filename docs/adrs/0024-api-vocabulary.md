# ADR 0024: One API Vocabulary

## Context

Seven problem types expose the same concepts: inputs, configs, outputs,
exports, step options and results, manual seeds, init iterations, parameter
fixing, the pose gauge, the solver budget, robust losses, robot-pose priors.
Spelled differently per problem, every consumer (JSON, the schema-driven
forms, Python, the docs) needs per-problem special cases, and readers cannot
scan the API.

## Decision

Each concept has one name and one shape across the workspace, identical in
Rust, Python, the JSON schemas and the app.

### Problem types

Every problem module exports `<Name>Problem` and its four associated types
under fixed names:

- **Input** is `<Name>Input`. When the data is a shared type the name is a
  type alias (`PlanarIntrinsicsInput = PlanarDataset`,
  `RigHandeyeInput = RigDataset<RobotPoseMeta>`); otherwise it is a struct.
  Aliases cost nothing and let every problem be used from its own module
  without knowing which shared type backs it.
- **Config** is `<Name>Config`; **Export** is `<Name>Export`.
- **Output** (`ProblemType::Output`) is an *estimate*: the final solve's
  `<Solve>Estimate` from `vision-calibration-optim`, re-exported by the
  module (`PlanarIntrinsicsEstimate`, `HandeyeEstimate` for both hand-eye
  problems, `RigLaserlineEstimate`). Only when the export needs more than
  the solve returns does the problem wrap it in its own `<Name>Estimate`
  (`LaserlineDeviceEstimate` adds the residual statistics). Anything the
  config determines is read from the config at export time, not copied into
  the output (the joint rig laser export takes the hand-eye mode from
  `handeye.handeye_init`; a config change clears the output).

### Suffixes

One suffix per role, never shared:

| Suffix | Role | Example |
|---|---|---|
| `Estimate` | result of a solve; a problem's output | `RigExtrinsicsEstimate` |
| `Export` | the stable result a caller keeps | `RigHandeyeExport` |
| `Result` | what a step returns | `IntrinsicsInitResult` |
| `Options` | per-call step overrides; solver options | `RigOptimizeOptions` |
| `Config` | persisted configuration | `HandeyeInitConfig` |
| `ManualInit` | the seed of a `step_*_with_seed` | `RigManualInit` |

`Output` names only the trait's associated type. `dataset_runner` builders
return `<Family>RunResult` (the input plus image paths), with the input in
an `input` field.

### Steps

- A step is `step_<stage>_<action>` (or `step_<action>` in a single-stage
  problem). It takes `opts: Option<<Stage><Action>Options>`, where `None`
  keeps the session config, and returns `Result<<Stage><Action>Result,
  Error>` when it reports values. Its seeded variant
  `step_<stage>_<action>_with_seed(session, manual: <Stage>ManualInit,
  opts)` takes the seed before the options.
- The stage is mandatory in every type name and the action is a full word:
  `IntrinsicsInitOptions`, `RigOptimizeResult`, never a bare `InitOptions`,
  never `Optim` or `Ba`. Single-stage problems name theirs: `Intrinsics`
  (planar), `Device` (camera and laser plane, `laserline_device`), `Plane`
  (the laser planes, `rig_laserline_device`).
- Every step takes options, even when none exist yet: the type is an empty
  `#[non_exhaustive]` struct (`RigInitOptions`, `PlaneInitOptions`), so a
  later override is not a breaking signature change.
- Option types that several problems share are defined once in
  `vision_calibration_pipeline::common` and re-exported by each module
  (`IntrinsicsInitOptions`, `RigOptimizeOptions`, `HandeyeInitOptions`, …).
  Results and seeds are per problem: same name, distinct type in each module
  (`rig_handeye::IntrinsicsInitResult` is not
  `planar_intrinsics::IntrinsicsInitResult`); the facade is module-first, so
  the names never meet.
- `run_calibration(session: &mut CalibrationSession<P>) -> Result<(), Error>`
  has this signature in every problem and runs with the session's config.
  Other convenience runners take `session` first and options as
  `Option<…Options>` (`run_calibration_with_filtering`).

### Casing

`handeye` is one word: modules (`single_cam_handeye`), JSON kinds
(`rig_handeye`), config groups (`handeye_init`) and fields (`handeye_mode`)
all spell it so, and CamelCase is `Handeye` everywhere (`HandeyeMode`,
`HandeyeEstimate`, `ReprojLevel::Handeye`).

### Python

Python names are the Rust names: the workflow types are `<Name>Input`,
`<Name>Config` and `<Name>Export`, and every `*Config`, `*Input`, `*Export`,
`*Estimate`, `*Params` or `*Result` class names a Rust type.
`scripts/check_binding_parity.py` enforces both. Python-only classes exist
only for what Rust spells generically (`PlanarView` for `View<NoMeta>`) or
models natively (the camera classes).

### Config shape

- Configs are grouped by stage (`init`, `intrinsics`, `rig`,
  `handeye_init`, `handeye_optimize`, `solver`, `optimize`, …), never flat
  boolean bags. A group that pairs a stage with an action spells it
  `<stage>_<action>`, matching the step and its options
  (`handeye_optimize` ↔ `step_handeye_optimize` ↔ `HandeyeOptimizeOptions`).
- A group's type is `<Stage><Action>Config` (`HandeyeOptimizeConfig`,
  `DeviceOptimizeConfig`, `JointOptimizeConfig`; the joint rig laser solve
  is the `Joint` stage).
- A field does not repeat its group's stage: `rig.refine_intrinsics`,
  `rig.refine_sensor`, `intrinsics.iterations` (the same name as
  `IntrinsicsInitOptions::iterations`, which overrides it). One name per
  concept wins over this rule: `handeye_init.handeye_mode` keeps the name
  the mode has in every input and export.
- Shared stages are defined once in
  `vision_calibration_pipeline::common::config` and reused:
  `IntrinsicsInitConfig`, `SolverConfig` (`max_iters`, `verbosity`,
  `robust_loss`, `backend`), `RobotPoseConfig`, `HandeyeInitConfig`,
  `RigConfig`.
- A camera is configured by one `camera: CameraConfig { model, fix }`
  group (ADR 0005). Its parameters are fixed by name, per slot
  (`fix.distortion: ["k3"]`), validated against the chosen model. Per-view
  poses are fixed with `fix_poses: Vec<usize>`.
- A rig's gauge is removed by fixing the reference camera
  (`reference_camera_idx`) only; fixing a view pose as well is redundant.
- Iteration counts and verbosity are plain values with concrete
  per-problem defaults, documented on the field; never `Option`.
- Non-laser problems have one `robust_loss`. Laser stages keep separate
  target and laser losses and weights, because corners and laser stripes
  are different residual families; a loss scale is in its residual's unit.
- Every problem type accepts every camera model; rigs add per-camera
  overrides of `camera`.

### Renames are hard

Pre-1.0, a vocabulary fix is a hard rename with no compatibility aliases,
recorded in the CHANGELOG with migration notes.

## Consequences

- Adding a problem means choosing its stages; every type name follows from
  them. Adding a config field means choosing its stage and reusing the
  shared structs, not inventing a spelling.
- `crates/vision-calibration/tests/facade_compile_surface.rs` locks the
  shape: each problem's associated types, the `run_calibration` signature,
  and every step's options, seed and result types.
- The wire shape of every config is visible in its emitted schema
  (ADR 0014), which is the reference for field-level detail.
- Each rename breaks downstream code once.
