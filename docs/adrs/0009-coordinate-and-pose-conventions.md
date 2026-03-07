# ADR 0009: Coordinate and Pose Conventions

- Status: Accepted
- Date: 2026-03-07 (retroactive)

## Context

Camera calibration involves many coordinate frames (world, camera, gripper, base, rig). Inconsistent naming of transforms is a common source of bugs.

## Decision

Adopt the `frame_se3_frame` naming convention for all rigid transforms:

- `T_C_W` or `cam_se3_world`: transform **from** world **to** camera frame.
- `base_se3_gripper`: transform **from** gripper **to** base frame.
- `gripper_se3_camera`: transform **from** camera **to** gripper frame (hand-eye result).
- `cam_se3_rig`: transform **from** rig **to** camera frame.

SE3 storage order: `[qx, qy, qz, qw, tx, ty, tz]` (Hamilton quaternion, scalar-last, then translation).

Coordinate expectations:
- **Pixel coordinates**: used directly by homography, fundamental matrix, PnP solvers.
- **Normalized coordinates**: `K^-1 * pixel`; used by essential matrix.
- Solvers that need normalized coordinates perform the conversion internally.

## Consequences

- Variable names in code encode the transform direction unambiguously.
- SE3 storage matches tiny-solver's Lie group conventions.
- New problem types must follow these naming conventions in their API types.
