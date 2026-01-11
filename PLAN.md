# Hand-Eye Example Revamp Plan

Status legend: [TODO] not started, [IN PROGRESS], [DONE]

## 1) Example data I/O module
- [DONE] Create `crates/calib/examples/handeye_io.rs` with KUKA dataset loaders:
  - load square size (meters), robot poses, and chessboard detections
  - return `Vec<ViewSample>` (2D/3D points + robot pose) plus summary stats
- [DONE] Configure chessboard detection with explicit expected rows/cols.

## 2) Stepwise hand-eye pipeline API
- [DONE] Add `crates/calib-pipeline/src/handeye_single.rs` with explicit step functions:
  - `init_intrinsics(...) -> IntrinsicsStage` (intrinsics + distortion + poses + mean reproj error)
  - `optimize_intrinsics(...) -> IntrinsicsStage`
  - `ransac_planar_poses(...) -> PoseRansacStage` (inlier filtering, poses, mean reproj error)
  - `init_handeye(...) -> HandEyeStage` (DLT, target poses, mean reproj error)
  - `optimize_handeye(...) -> HandEyeStage`
  - `run_handeye_single(...) -> HandEyeSingleReport` (high-level facade)
- [DONE] Add shared reprojection error helpers (mean pixel error) for planar and hand-eye chains.
- [DONE] Define RANSAC defaults explicitly (threshold 1 px, min_inliers 8, deterministic seed).

## 3) Manual step-by-step example
- [DONE] Rewrite `crates/calib/examples/handeyesingle.rs` to:
  - use `handeye_io` and step functions only
  - print mean reprojection error after each stage
  - show current best model at each stage (intrinsics, distortion, hand-eye)

## 4) Concise session + pipeline example
- [DONE] Add a second example (e.g., `handeye_session.rs`) that:
  - uses `CalibrationSession<PlanarIntrinsicsProblem>` for intrinsics
  - uses the pipeline facade for hand-eye
  - prints the same mean reprojection error stages

## 5) API wiring
- [DONE] Re-export `handeye_single` pipeline module via `calib-pipeline` and `calib`.
- [DONE] Update example docs/comments if needed (no migration notes).
