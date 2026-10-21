# ADR 0024: One API Vocabulary

## Context

Eight problem types expose the same concepts: step options, init
iterations, parameter fixing, the pose gauge, the solver budget, robust
losses, robot-pose priors. Spelled differently per problem, every consumer
(JSON, the schema-driven forms, Python, the docs) needs per-problem special
cases, and readers cannot scan the API.

## Decision

Each concept has one name and one shape across the workspace.

### Names

- Step options are `<Stage><Action>Options` with the stage mandatory and
  the action a full word: `IntrinsicsInitOptions`, `RigOptimizeOptions` —
  never a bare `InitOptions`, never `Optim`. They match the module's step
  family (`step_intrinsics_*` takes `Intrinsics*Options`).
- Step functions are `step_<action>`; problem types `<Name>Problem` with
  `<Name>Config` and `<Name>Export`; poses follow ADR 0009.

### Config shape

- Configs are grouped by stage (`init`, `intrinsics`, `rig`,
  `handeye_init`, `solver`, `optimize`, …), never flat boolean bags.
- Shared stages are defined once in
  `vision_calibration_pipeline::common::config` and reused:
  `IntrinsicsInitConfig`, `SolverConfig` (`max_iters`, `verbosity`,
  `robust_loss`, `backend`), `RobotPoseConfig`, `HandeyeInitConfig`,
  `RigConfig`.
- Parameter fixing uses one mask per parameter group: `CameraFixMask
  { intrinsics, distortion }` for cameras (never plain booleans),
  `ScheimpflugFixMask` as `fix_scheimpflug`, and `fix_poses: Vec<usize>` for
  per-view poses.
- A rig's gauge is removed by fixing the reference camera
  (`reference_camera_idx`) only; fixing a view pose as well is redundant.
- Iteration counts and verbosity are plain values with concrete
  per-problem defaults, documented on the field; never `Option`.
- Non-laser problems have one `robust_loss`. Laser stages keep separate
  target and laser losses and weights, because corners and laser stripes
  are different residual families; a loss scale is in its residual's unit.
- `distortion_model` is a live choice only in the single-camera intrinsics
  configs; rigs fix Brown–Conrady and reject other values at validation.

### Renames are hard

Pre-1.0, a vocabulary fix is a hard rename with no compatibility aliases,
recorded in the CHANGELOG with migration notes.

## Consequences

- Adding a config field means choosing its stage and reusing the shared
  structs, not inventing a spelling.
- The wire shape of every config is visible in its emitted schema
  (ADR 0014), which is the reference for field-level detail.
- Each rename breaks downstream code once.
