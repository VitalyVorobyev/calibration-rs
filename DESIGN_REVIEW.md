# Calibration Library Design Review

This document captures current issues observed in the pipeline/session design and recommended fixes.

## Findings
- **Planar intrinsics init/refine collapse** (`crates/calib-pipeline/src/session/problem_types.rs:57-101`, `crates/calib-pipeline/src/helpers.rs:201-276`): `initialize` and `optimize` both invoke `run_planar_intrinsics`, ignoring the initial-values stage and repeating a full solve. The advertised “inspect init, then refine” flow is misleading because `_init` is unused.
- **Hardcoded seeds in `run_planar_intrinsics`** (`crates/calib-pipeline/src/lib.rs:162-240`): The pipeline seeds optimization with fixed intrinsics/distortion/poses instead of using Zhang/iterative linear init and homography-derived pose seeds. Convergence and realism suffer, especially when focal length/scale differ from the baked guess.
- **Rig extrinsics init duplicates full solves** (`crates/calib-pipeline/src/session/problem_types.rs:234-420`): The init phase runs full per-camera intrinsics optimization, then optimize runs a joint BA, so the stage boundary is meaningless and compute is duplicated. Needs a true linear init (per-camera Zhang + poses) feeding the joint solve.
- **Session abstraction is too rigid** (`crates/calib-pipeline/src/session/mod.rs`): The state machine (set_observations → initialize → optimize → export) lacks hooks for real pipelines (feature detection, validation, branching). Because `ProblemType` impls already blur stages, sessions act as a thin wrapper over monolithic functions rather than a composable orchestrator.
- **Config naming vs model data** (`crates/calib-core/src/models/config.rs`): `CameraConfig`, `ProjectionConfig`, `DistortionConfig`, `SensorConfig`, `IntrinsicsConfig` are serialized parameter sets (calibrated models), not algorithm configs. Names duplicate math structs (`FxFyCxCySkew`, `BrownConrady5`), creating parallel representations and potential drift.
- **Hand-eye workflow gap**: No end-to-end example showing intrinsics + hand-eye using the public API. Hand-eye optimizer is not re-exported at `calib-pipeline`/`calib`, forcing users into subcrates and discouraging the recommended flow.

## Recommendations
- Split planar intrinsics into: (1) Zhang/iterative linear intrinsics + homography pose seeds, (2) nonlinear refinement via `PlanarIntrinsicsInit` → optimizer. Ensure helpers and `PlanarIntrinsicsProblem` consume the init rather than recomputing it.
- Rework session `ProblemType` semantics so `initialize` returns lightweight linear seeds and `optimize` consumes them. Treat the session as data + checkpoints; use helper pipeline functions for orchestration.
- Rebuild rig extrinsics init around per-camera linear intrinsics + pose estimation; run a single joint optimization afterward. Add tests proving init ≠ optimize and that joint BA improves cost.
- Rename config structs to indicate calibrated parameters (e.g., `CameraParams`) and wrap core math types instead of duplicating fields; preserve serde/backward compatibility.
- Re-export hand-eye optimizer through `calib-pipeline`/`calib` and add a `handeyesingle.rs` example: planar intrinsics → refine → per-view poses → hand-eye DLT seed → hand-eye nonlinear refine. Include a synthetic regression test and docs snippet.
- Update README/book/rustdoc with the new init flow, session guidance, and migration notes for any renamed types or behavior changes.
