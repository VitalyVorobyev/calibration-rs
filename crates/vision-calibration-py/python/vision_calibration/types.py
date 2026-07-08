"""Low-level serde payload contracts for :mod:`vision_calibration`.

These aliases/TypedDicts model the raw Rust JSON payload schema. Most users
should use dataclasses from :mod:`vision_calibration.models` instead.

Warning
-------
This module is compatibility-oriented low-level surface. It is not the
recommended high-level API for new code.
"""

from __future__ import annotations

from typing import Any, Literal, TypeAlias, TypedDict

JsonObject: TypeAlias = dict[str, Any]
JsonArray: TypeAlias = list[Any]

# Nested geometry/model payloads are serde dictionaries coming from Rust.
Transform: TypeAlias = JsonObject
CameraModel: TypeAlias = JsonObject
IntrinsicsPayload: TypeAlias = JsonObject
DistortionPayload: TypeAlias = JsonObject
ScheimpflugPayload: TypeAlias = JsonObject
LaserPlanePayload: TypeAlias = JsonObject

Se3Delta: TypeAlias = tuple[float, float, float, float, float, float] | list[float]

HandEyeMode: TypeAlias = Literal["EyeInHand", "EyeToHand"]
LaserlineResidualType: TypeAlias = Literal["PointToPlane", "LineDistNormalized"]
ScheimpflugFixMask: TypeAlias = JsonObject


class _RobustLossScale(TypedDict):
    scale: float


class RobustLossHuber(TypedDict):
    Huber: _RobustLossScale


class RobustLossCauchy(TypedDict):
    Cauchy: _RobustLossScale


class RobustLossArctan(TypedDict):
    Arctan: _RobustLossScale


# Serde forms accepted by Rust:
# - "None"
# - {"Huber": {"scale": float}}
# - {"Cauchy": {"scale": float}}
# - {"Arctan": {"scale": float}}
RobustLoss: TypeAlias = (
    Literal["None"]
    | RobustLossHuber
    | RobustLossCauchy
    | RobustLossArctan
)

# Input payload aliases (workflow datasets).
PlanarInput: TypeAlias = JsonObject
SingleCamHandeyeInput: TypeAlias = JsonObject
RigExtrinsicsInput: TypeAlias = JsonObject
RigHandeyeInput: TypeAlias = JsonObject
LaserlineDeviceInput: TypeAlias = JsonObject
ScheimpflugIntrinsicsInput: TypeAlias = JsonObject


class SolveReport(TypedDict):
    final_cost: float


class PlanarIntrinsicsParams(TypedDict):
    camera: CameraModel
    camera_se3_target: list[Transform]


class PlanarIntrinsicsExport(TypedDict):
    params: PlanarIntrinsicsParams
    report: SolveReport
    mean_reproj_error: float
    per_cam_reproj_errors: list[float]


class SingleCamHandeyeExport(TypedDict):
    camera: CameraModel
    handeye_mode: HandEyeMode
    gripper_se3_camera: Transform | None
    camera_se3_base: Transform | None
    base_se3_target: Transform | None
    gripper_se3_target: Transform | None
    robot_deltas: list[Se3Delta] | None
    mean_reproj_error: float
    per_cam_reproj_errors: list[float]


class RigExtrinsicsExport(TypedDict):
    cameras: list[CameraModel]
    cam_se3_rig: list[Transform]
    mean_reproj_error: float
    per_cam_reproj_errors: list[float]


class RigHandeyeExport(TypedDict):
    cameras: list[CameraModel]
    cam_se3_rig: list[Transform]
    handeye_mode: HandEyeMode
    gripper_se3_rig: Transform | None
    rig_se3_base: Transform | None
    base_se3_target: Transform | None
    gripper_se3_target: Transform | None
    robot_deltas: list[Se3Delta] | None
    mean_reproj_error: float
    per_cam_reproj_errors: list[float]


class LaserlineParams(TypedDict):
    intrinsics: IntrinsicsPayload
    distortion: DistortionPayload
    sensor: ScheimpflugPayload
    poses: list[Transform]
    plane: LaserPlanePayload


class LaserlineEstimate(TypedDict):
    params: LaserlineParams
    report: SolveReport


class LaserlineStats(TypedDict):
    mean_reproj_error: float
    mean_laser_error: float
    per_view_reproj_errors: list[float]
    per_view_laser_errors: list[float]


class LaserlineDeviceExport(TypedDict):
    estimate: LaserlineEstimate
    stats: LaserlineStats
    mean_reproj_error: float
    per_cam_reproj_errors: list[float]


class ScheimpflugIntrinsicsParams(TypedDict):
    camera: CameraModel
    camera_se3_target: list[Transform]


class ScheimpflugIntrinsicsExport(TypedDict):
    params: ScheimpflugIntrinsicsParams
    report: SolveReport
    mean_reproj_error: float
    per_cam_reproj_errors: list[float]


class IntrinsicsInitConfig(TypedDict, total=False):
    """Shared per-camera linear-initialization stage (ADR 0024)."""

    init_iterations: int
    fix_k3: bool
    fix_tangential: bool
    zero_skew: bool


class SolverConfig(TypedDict, total=False):
    """Shared non-linear solve stage settings (ADR 0024)."""

    max_iters: int
    verbosity: int
    robust_loss: RobustLoss


class RobotPoseConfig(TypedDict, total=False):
    """Shared robot-pose refinement settings (ADR 0024)."""

    refine: bool
    rot_sigma: float
    trans_sigma: float


class HandeyeInitConfig(TypedDict, total=False):
    """Shared hand-eye linear-initialization settings (ADR 0024)."""

    handeye_mode: HandEyeMode
    min_motion_angle_deg: float


class CameraFixMask(TypedDict, total=False):
    """Combined per-camera intrinsics + distortion fix mask (ADR 0024)."""

    intrinsics: JsonObject
    distortion: JsonObject


class PlanarIntrinsicsConfig(TypedDict, total=False):
    init: IntrinsicsInitConfig
    solver: SolverConfig
    fix_camera: CameraFixMask
    fix_poses: list[int]


class SingleCamHandeyeConfig(TypedDict, total=False):
    intrinsics: IntrinsicsInitConfig
    handeye_init: HandeyeInitConfig
    solver: SolverConfig
    robot_poses: RobotPoseConfig


class RigConfig(TypedDict, total=False):
    """Shared multi-camera rig frame options (ADR 0024)."""

    reference_camera_idx: int
    refine_intrinsics_in_rig_ba: bool


class RigExtrinsicsConfig(TypedDict, total=False):
    intrinsics: IntrinsicsInitConfig
    rig: RigConfig
    solver: SolverConfig


class HandeyeBaConfig(TypedDict, total=False):
    """Final hand-eye bundle-adjustment options (ADR 0024)."""

    robot_poses: RobotPoseConfig
    refine_cam_se3_rig: bool
    refine_scheimpflug: bool


class RigHandeyeConfig(TypedDict, total=False):
    intrinsics: IntrinsicsInitConfig
    rig: RigConfig
    handeye_init: HandeyeInitConfig
    solver: SolverConfig
    handeye_ba: HandeyeBaConfig


class LaserlineDeviceOptimizeConfig(TypedDict, total=False):
    calib_loss: RobustLoss
    laser_loss: RobustLoss
    calib_weight: float
    laser_weight: float
    fix_camera: CameraFixMask
    fix_sensor: bool
    fix_poses: list[int]
    fix_plane: bool
    laser_residual_type: LaserlineResidualType


class LaserlineDeviceConfig(TypedDict, total=False):
    init: IntrinsicsInitConfig
    sensor_init: JsonObject
    solver: SolverConfig
    optimize: LaserlineDeviceOptimizeConfig


class RigLaserlineDeviceConfig(TypedDict, total=False):
    solver: SolverConfig
    laser_residual_type: LaserlineResidualType


class ScheimpflugIntrinsicsConfig(TypedDict, total=False):
    init: IntrinsicsInitConfig
    solver: SolverConfig
    fix_camera: CameraFixMask
    fix_scheimpflug: ScheimpflugFixMask
    fix_poses: list[int]
