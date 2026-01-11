# Calibration-RS Improvement Plan

This plan is tracked as we implement the fixes from the design review. Update statuses as work progresses.

Status legend: [TODO] not started, [IN PROGRESS], [DONE]

## 1) Planar intrinsics init/refine split
- [DONE] Replace hardcoded seeds in `crates/calib-pipeline/src/lib.rs::run_planar_intrinsics` with Zhang-based intrinsics + homography-derived pose seeds (fallback to iterative intrinsics when distortion is significant).
- [DONE] Adjust `PlanarIntrinsicsProblem` and `helpers::optimize_planar_intrinsics_from_init` to consume the computed `PlanarIntrinsicsInit` instead of recomputing full solves; ensure `_init` is used.
- [IN PROGRESS] Add robustness tests (noisy focal, skewed scales, few views) and regression coverage for the new init path.

## 2) Session semantics and problem boundaries
- [DONE] Clarify `ProblemType` contract: `initialize` returns linear seeds; `optimize` consumes them and observations. Update rustdoc and examples accordingly.
- [DONE] Document sessions as data + checkpoint containers; recommend orchestration helpers for real pipelines.

## 3) Rig extrinsics pipeline correctness
- [DONE] Implement true linear init: per-camera Zhang intrinsics + planar poses; remove full nonlinear per-camera solves from init.
- [DONE] Ensure optimize stage performs a single joint BA using init seeds.
- [DONE] Add regression tests showing init ≠ optimize and cost improvement.

## 4) Config/model naming cleanup
- [DONE] Rename model config structs to params (`config.rs` → `params.rs`) and update all imports/call sites to the new names.
- [DONE] Use core math structs (`FxFyCxCySkew`, `BrownConrady5`) directly inside params enums and remove the optimizer-specific aliases/params types.

## 5) Hand-eye pipeline and example
- [DONE] Re-export hand-eye optimizer (and needed types) through `calib-pipeline`/`calib`.
- [DONE] Implement `handeyesingle.rs` example: planar intrinsics init → refine → per-view poses → hand-eye DLT seed → hand-eye nonlinear refinement. Use deterministic seeds and real-data-friendly IO.
- [DONE] Add synthetic regression test covering the end-to-end hand-eye pipeline and minimal doc snippet.

## 6) Documentation and migration
- [TODO] Update README/book/rustdoc to reflect new init flow, session guidance, and any renamed param types.
- [TODO] Provide migration notes for users of old `run_planar_intrinsics` behavior and config type names.
