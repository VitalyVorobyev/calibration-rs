# Calibration-RS Improvement Plan

This plan is tracked as we implement the fixes from the design review. Update statuses as work progresses.

Status legend: [TODO] not started, [IN PROGRESS], [DONE]

## 1) Planar intrinsics init/refine split
- [DONE] Replace hardcoded seeds in `crates/calib-pipeline/src/lib.rs::run_planar_intrinsics` with Zhang-based intrinsics + homography-derived pose seeds (fallback to iterative intrinsics when distortion is significant).
- [DONE] Adjust `PlanarIntrinsicsProblem` and `helpers::optimize_planar_intrinsics_from_init` to consume the computed `PlanarIntrinsicsInit` instead of recomputing full solves; ensure `_init` is used.
- [IN PROGRESS] Add robustness tests (noisy focal, skewed scales, few views) and regression coverage for the new init path.

## 2) Session semantics and problem boundaries
- [TODO] Clarify `ProblemType` contract: `initialize` returns linear seeds; `optimize` consumes them and observations. Update rustdoc and examples accordingly.
- [TODO] Document sessions as data + checkpoint containers; recommend orchestration helpers for real pipelines.

## 3) Rig extrinsics pipeline correctness
- [TODO] Implement true linear init: per-camera Zhang intrinsics + planar poses; remove full nonlinear per-camera solves from init.
- [TODO] Ensure optimize stage performs a single joint BA using init seeds. Add regression tests showing init ≠ optimize and cost improvement.

## 4) Config/model naming cleanup
- [TODO] Rename serialized camera structs to indicate calibrated parameters (e.g., `CameraParams`, `IntrinsicsParams`, etc.) while preserving serde/backward compatibility and `calib` public API stability.
- [TODO] Wrap core math structs (`FxFyCxCySkew`, `BrownConrady5`, etc.) instead of duplicating fields; add roundtrip/build tests.

## 5) Hand-eye pipeline and example
- [TODO] Re-export hand-eye optimizer (and needed types) through `calib-pipeline`/`calib`.
- [TODO] Implement `handeyesingle.rs` example: planar intrinsics init → refine → per-view poses → hand-eye DLT seed → hand-eye nonlinear refinement. Use deterministic seeds and real-data-friendly IO.
- [TODO] Add synthetic regression test covering the end-to-end hand-eye pipeline and minimal doc snippet.

## 6) Documentation and migration
- [TODO] Update README/book/rustdoc to reflect new init flow, session guidance, and any renamed param types.
- [TODO] Provide migration notes for users of old `run_planar_intrinsics` behavior and config type names.
