# ADR 0009: Coordinate and Pose Conventions

## Context

Calibration involves many frames (world, camera, gripper, base, rig,
target). Ambiguous transform naming is a common source of silent bugs.

## Decision

Rigid transforms are named `<to>_se3_<from>`: `a_se3_b` maps coordinates
from frame `b` to frame `a`. `T_C_W` and `cam_se3_world` are the same
world-to-camera transform. Common names: `base_se3_gripper`,
`gripper_se3_camera` (the eye-in-hand result), `cam_se3_rig`.

- **Storage:** SE(3) is stored as `[qx, qy, qz, qw, tx, ty, tz]` (Hamilton
  quaternion, scalar last, then translation), which is also the convention
  of both solver backends.
- **Units:** metres and radians inside the library. Wire formats written by
  hand may use other units only when the field name says so (ADR 0011).
- **Perturbation:** an SE(3) pose is perturbed on the right,
  `T · Exp([δθ, δt])`, rotation first, both in frame `b`. Pose covariances
  are 6×6 in this order (ADR 0026).
- **Planes:** `n·p + d = 0`; a plane's covariance is over `[nx, ny, nz, d]`.
- **Pixels vs normalized coordinates:** homography, fundamental-matrix and
  PnP solvers take pixels; essential-matrix solvers take normalized
  coordinates (`K⁻¹·pixel`). A solver that needs normalized coordinates
  converts internally.

## Consequences

- A variable's name states its direction; public API types follow it.
- Covariances are reported in the same tangent convention the solvers use.
