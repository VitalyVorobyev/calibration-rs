"""Typed payload contracts for :mod:`vision_calibration`.

The Rust extension consumes and produces serde-compatible dictionaries.
These types model public config and export payloads so IDEs and type checkers
can reason about the API without requiring Rust internals.
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


class SolveReport(TypedDict):
    final_cost: float


class PlanarIntrinsicsParams(TypedDict):
    camera: CameraModel
    camera_se3_target: list[Transform]


class PlanarExport(TypedDict):
    params: PlanarIntrinsicsParams
    report: SolveReport
    mean_reproj_error: float


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


class PlanarConfig(TypedDict, total=False):
    init_iterations: int
    fix_k3_in_init: bool
    fix_tangential_in_init: bool
    zero_skew: bool
    max_iters: int
    verbosity: int
    robust_loss: RobustLoss
    fix_intrinsics: JsonObject
    fix_distortion: JsonObject
    fix_poses: list[int]


class SingleCamHandeyeConfig(TypedDict, total=False):
    intrinsics_init_iterations: int
    fix_k3: bool
    fix_tangential: bool
    zero_skew: bool
    handeye_mode: HandEyeMode
    min_motion_angle_deg: float
    max_iters: int
    verbosity: int
    robust_loss: RobustLoss
    refine_robot_poses: bool
    robot_rot_sigma: float
    robot_trans_sigma: float


class RigExtrinsicsConfig(TypedDict, total=False):
    intrinsics_init_iterations: int
    fix_k3: bool
    fix_tangential: bool
    zero_skew: bool
    reference_camera_idx: int
    max_iters: int
    verbosity: int
    robust_loss: RobustLoss
    refine_intrinsics_in_rig_ba: bool
    fix_first_rig_pose: bool


class RigHandeyeIntrinsicsConfig(TypedDict, total=False):
    init_iterations: int
    fix_k3: bool
    fix_tangential: bool
    zero_skew: bool


class RigHandeyeRigConfig(TypedDict, total=False):
    reference_camera_idx: int
    refine_intrinsics_in_rig_ba: bool
    fix_first_rig_pose: bool


class RigHandeyeInitConfig(TypedDict, total=False):
    handeye_mode: HandEyeMode
    min_motion_angle_deg: float


class RigHandeyeSolverConfig(TypedDict, total=False):
    max_iters: int
    verbosity: int
    robust_loss: RobustLoss


class RigHandeyeBaConfig(TypedDict, total=False):
    refine_robot_poses: bool
    robot_rot_sigma: float
    robot_trans_sigma: float
    refine_cam_se3_rig_in_handeye_ba: bool


class RigHandeyeConfig(TypedDict, total=False):
    intrinsics: RigHandeyeIntrinsicsConfig
    rig: RigHandeyeRigConfig
    handeye_init: RigHandeyeInitConfig
    solver: RigHandeyeSolverConfig
    handeye_ba: RigHandeyeBaConfig


class LaserlineDeviceInitConfig(TypedDict, total=False):
    iterations: int
    fix_k3: bool
    fix_tangential: bool
    zero_skew: bool
    sensor_init: JsonObject


class LaserlineDeviceSolverConfig(TypedDict, total=False):
    max_iters: int
    verbosity: int


class LaserlineDeviceOptimizeConfig(TypedDict, total=False):
    calib_loss: RobustLoss
    laser_loss: RobustLoss
    calib_weight: float
    laser_weight: float
    fix_intrinsics: bool
    fix_distortion: bool
    fix_k3: bool
    fix_sensor: bool
    fix_poses: list[int]
    fix_plane: bool
    laser_residual_type: LaserlineResidualType


class LaserlineDeviceConfig(TypedDict, total=False):
    init: LaserlineDeviceInitConfig
    solver: LaserlineDeviceSolverConfig
    optimize: LaserlineDeviceOptimizeConfig
