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
# Engine that linearizes the problem; both reach the same minimizer.
SolverBackend: TypeAlias = Literal["tiny_solver", "factrs"]
LaserlineResidualType: TypeAlias = Literal["PointToPlane", "LineDistNormalized"]

# Serde `snake_case` form of `DistortionKind`. `brown_conrady5` is the default;
# every workflow accepts every model.
DistortionModel: TypeAlias = Literal[
    "none",
    "brown_conrady5",
    "rational8",
    "thin_prism9",
    "division1",
]


class DistortionBrownConrady5(TypedDict):
    """Exported ``DistortionParams::BrownConrady5`` payload."""

    type: Literal["brown_conrady5"]
    k1: float
    k2: float
    k3: float
    p1: float
    p2: float
    iters: int


# `lambda` is a Python keyword, so the single-parameter division variant needs
# the functional TypedDict form.
DistortionDivision = TypedDict(
    "DistortionDivision",
    {"type": Literal["division"], "lambda": float},
)


class DistortionRational(TypedDict):
    """Exported ``DistortionParams::Rational`` payload."""

    type: Literal["rational"]
    k1: float
    k2: float
    k3: float
    k4: float
    k5: float
    k6: float
    p1: float
    p2: float
    iters: int


class DistortionThinPrism(TypedDict):
    """Exported ``DistortionParams::ThinPrism`` payload."""

    type: Literal["thin_prism"]
    k1: float
    k2: float
    k3: float
    p1: float
    p2: float
    s1: float
    s2: float
    s3: float
    s4: float
    iters: int


# Tagged (``type``) distortion payload emitted on the export side. Note the tags
# (``division``/``rational``/``thin_prism``) are the ``DistortionParams`` names,
# distinct from the ``DistortionModel`` config spellings above.
DistortionParamsPayload: TypeAlias = (
    DistortionBrownConrady5 | DistortionDivision | DistortionRational | DistortionThinPrism
)


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
    num_iters: int


ResidualKind: TypeAlias = Literal["reprojection", "laser_plane", "laser_line"]


class _ResidualNoiseRequired(TypedDict):
    kind: ResidualKind
    sigma: float
    count: int


class ResidualNoise(_ResidualNoiseRequired, total=False):
    camera: int


class ViewCovariance(TypedDict):
    view: int
    labels: list[str]
    covariance: list[list[float]]


class UncertaintyUnavailableReason(TypedDict, total=False):
    kind: Literal["not_converged", "rank_deficient", "at_bound", "no_redundancy", "not_computed"]
    unobservable: list[str]
    labels: list[str]


class Uncertainty(TypedDict, total=False):
    """`{"status": "available", labels, covariance, …}` or
    `{"status": "unavailable", "reason": {...}}`."""

    status: Literal["available", "unavailable"]
    labels: list[str]
    covariance: list[list[float]]
    views: list[ViewCovariance]
    fixed: list[str]
    noise: list[ResidualNoise]
    num_residuals: int
    dof: int
    reason: UncertaintyUnavailableReason


class ProjectionUncertainty(TypedDict):
    image_size: list[int]
    columns: int
    rows: int
    sigma: list[float | None]
    center: float
    corner: float | None
    max: float


class PlanarIntrinsicsParams(TypedDict):
    camera: CameraModel
    camera_se3_target: list[Transform]


class PlanarIntrinsicsExport(TypedDict):
    params: PlanarIntrinsicsParams
    report: SolveReport
    mean_reproj_error: float
    per_cam_reproj_errors: list[float]
    uncertainty: Uncertainty


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
    uncertainty: Uncertainty


class RigExtrinsicsExport(TypedDict):
    cameras: list[CameraModel]
    cam_se3_rig: list[Transform]
    mean_reproj_error: float
    per_cam_reproj_errors: list[float]
    uncertainty: Uncertainty


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
    uncertainty: Uncertainty


class LaserlineParams(TypedDict):
    intrinsics: IntrinsicsPayload
    distortion: DistortionPayload
    sensor: ScheimpflugPayload
    poses: list[Transform]
    plane: LaserPlanePayload


class LaserlineEstimate(TypedDict):
    params: LaserlineParams
    report: SolveReport
    uncertainty: Uncertainty


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


class IntrinsicsInitConfig(TypedDict, total=False):
    """Shared per-camera linear-initialization stage."""

    init_iterations: int
    fix_k3: bool
    fix_tangential: bool  # optional; absent = the camera's default


class SolverConfig(TypedDict, total=False):
    """Shared non-linear solve stage settings."""

    max_iters: int
    verbosity: int
    robust_loss: RobustLoss
    backend: SolverBackend


class RobotPoseConfig(TypedDict, total=False):
    """Shared robot-pose refinement settings."""

    refine: bool
    rot_sigma: float
    trans_sigma: float


class HandeyeInitConfig(TypedDict, total=False):
    """Shared hand-eye linear-initialization settings."""

    handeye_mode: HandEyeMode
    min_motion_angle_deg: float


ProjectionModel: TypeAlias = Literal["pinhole"]
SensorModel: TypeAlias = Literal["identity", "scheimpflug"]


class CameraModelSpec(TypedDict, total=False):
    """Camera model selector — serde ``CameraModelSpec``."""

    projection: ProjectionModel
    distortion: DistortionModel
    sensor: SensorModel


class CameraFix(TypedDict, total=False):
    """Parameter names fixed per slot; an absent slot keeps the model default."""

    intrinsics: list[str]
    projection: list[str]
    distortion: list[str]
    sensor: list[str]


class CameraConfig(TypedDict, total=False):
    model: CameraModelSpec
    fix: CameraFix


class _CameraOverrideRequired(TypedDict):
    camera: int


class CameraOverride(_CameraOverrideRequired, total=False):
    """Replaces the rig-wide camera config for one rig camera."""

    model: CameraModelSpec
    fix: CameraFix


class PlanarIntrinsicsConfig(TypedDict, total=False):
    init: IntrinsicsInitConfig
    solver: SolverConfig
    camera: CameraConfig
    fix_poses: list[int]


class SingleCamHandeyeConfig(TypedDict, total=False):
    intrinsics: IntrinsicsInitConfig
    camera: CameraConfig
    handeye_init: HandeyeInitConfig
    solver: SolverConfig
    robot_poses: RobotPoseConfig


class RigConfig(TypedDict, total=False):
    """Shared multi-camera rig frame options."""

    reference_camera_idx: int
    refine_intrinsics_in_rig_ba: bool
    refine_sensor: bool


class RigExtrinsicsConfig(TypedDict, total=False):
    intrinsics: IntrinsicsInitConfig
    camera: CameraConfig
    camera_overrides: list[CameraOverride]
    rig: RigConfig
    solver: SolverConfig


class HandeyeBaConfig(TypedDict, total=False):
    """Final hand-eye bundle-adjustment options."""

    robot_poses: RobotPoseConfig
    refine_cam_se3_rig: bool
    refine_sensor: bool


class RigHandeyeConfig(TypedDict, total=False):
    intrinsics: IntrinsicsInitConfig
    camera: CameraConfig
    camera_overrides: list[CameraOverride]
    rig: RigConfig
    handeye_init: HandeyeInitConfig
    solver: SolverConfig
    handeye_ba: HandeyeBaConfig


class LaserlineDeviceOptimizeConfig(TypedDict, total=False):
    calib_loss: RobustLoss
    laser_loss: RobustLoss
    calib_weight: float
    laser_weight: float
    fix_poses: list[int]
    fix_plane: bool
    laser_residual_type: LaserlineResidualType


class LaserlineDeviceConfig(TypedDict, total=False):
    init: IntrinsicsInitConfig
    camera: CameraConfig
    solver: SolverConfig
    optimize: LaserlineDeviceOptimizeConfig


class RigLaserlineDeviceConfig(TypedDict, total=False):
    solver: SolverConfig
    laser_residual_type: LaserlineResidualType


class RigHandeyeLaserlineBaConfig(TypedDict, total=False):
    """Final joint bundle-adjustment stage settings."""

    solver: SolverConfig
    laser_residual_type: LaserlineResidualType
    calib_loss: RobustLoss
    laser_loss: RobustLoss
    calib_weight: float
    laser_weight: float
    fix: list[CameraFix]
    fix_handeye: bool
    fix_target_ref: bool
    robot_poses: RobotPoseConfig


class RigHandeyeLaserlineConfig(TypedDict, total=False):
    """Joint rig hand-eye + laserline config: three warm-started stages."""

    handeye: RigHandeyeConfig
    laserline_init: RigLaserlineDeviceConfig
    joint_ba: RigHandeyeLaserlineBaConfig


# Input payload alias for the joint rig hand-eye laserline workflow.
RigHandeyeLaserlineInput: TypeAlias = JsonObject
