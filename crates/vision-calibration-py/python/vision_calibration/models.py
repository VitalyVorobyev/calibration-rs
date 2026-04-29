"""Public high-level models for :mod:`vision_calibration`.

These dataclasses provide a Python-native API surface with explicit fields,
defaults, and docstrings. Wrapper functions in :mod:`vision_calibration`
accept these models and convert them to the serde payloads expected by Rust.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, cast

from .types import HandEyeMode, LaserlineResidualType, RobustLoss

Vec2 = tuple[float, float]
Vec3 = tuple[float, float, float]
QuatXyzw = tuple[float, float, float, float]

_DEFAULT_INTRINSICS_FIX_MASK: dict[str, bool] = {
    "fx": False,
    "fy": False,
    "cx": False,
    "cy": False,
}
_DEFAULT_DISTORTION_FIX_MASK: dict[str, bool] = {
    "k1": False,
    "k2": False,
    "k3": True,
    "p1": False,
    "p2": False,
}
_DEFAULT_SENSOR_INIT: dict[str, float] = {
    "tilt_x": 0.0,
    "tilt_y": 0.0,
}
_DEFAULT_SCHEIMPFLUG_FIX_MASK: dict[str, bool] = {
    "tilt_x": False,
    "tilt_y": False,
}


def _as_floats(values: tuple[Any, ...] | list[Any], expected: int, name: str) -> tuple[float, ...]:
    if len(values) != expected:
        raise ValueError(f"{name} must have length {expected}, got {len(values)}")
    return tuple(float(v) for v in values)


def _as_vec2(values: tuple[Any, Any] | list[Any]) -> Vec2:
    x, y = _as_floats(values, 2, "Vec2")
    return (x, y)


def _as_vec3(values: tuple[Any, Any, Any] | list[Any]) -> Vec3:
    x, y, z = _as_floats(values, 3, "Vec3")
    return (x, y, z)


def _as_quat(values: tuple[Any, Any, Any, Any] | list[Any]) -> QuatXyzw:
    x, y, z, w = _as_floats(values, 4, "QuatXyzw")
    return (x, y, z, w)


def _payload_from_maybe_model(value: Any) -> Any:
    if hasattr(value, "to_payload"):
        return value.to_payload()
    return value


@dataclass(slots=True)
class Pose:
    """Rigid pose with quaternion rotation and translation.

    Parameters
    ----------
    rotation_xyzw:
        Quaternion as `(x, y, z, w)`.
    translation_xyz:
        Translation as `(x, y, z)` in meters.
    """

    rotation_xyzw: QuatXyzw = (0.0, 0.0, 0.0, 1.0)
    translation_xyz: Vec3 = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        self.rotation_xyzw = _as_quat(list(self.rotation_xyzw))
        self.translation_xyz = _as_vec3(list(self.translation_xyz))

    def to_payload(self) -> dict[str, list[float]]:
        """Convert to Rust/serde shape."""
        return {
            "rotation": [*self.rotation_xyzw],
            "translation": [*self.translation_xyz],
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "Pose":
        """Build pose from Rust/serde shape."""
        return cls(
            rotation_xyzw=cast(tuple[float, float, float, float], tuple(payload["rotation"])),
            translation_xyz=cast(tuple[float, float, float], tuple(payload["translation"])),
        )


@dataclass(slots=True)
class Observation:
    """2D-3D correspondences for one camera view.

    Parameters
    ----------
    points_3d:
        Target points as `(x, y, z)` tuples.
    points_2d:
        Pixel points as `(u, v)` tuples.
    weights:
        Optional per-point non-negative weights.
    """

    points_3d: list[Vec3]
    points_2d: list[Vec2]
    weights: list[float] | None = None

    def __post_init__(self) -> None:
        self.points_3d = [_as_vec3(list(p)) for p in self.points_3d]
        self.points_2d = [_as_vec2(list(p)) for p in self.points_2d]
        if len(self.points_3d) != len(self.points_2d):
            raise ValueError(
                "points_3d and points_2d must have identical length "
                f"(got {len(self.points_3d)} vs {len(self.points_2d)})"
            )
        if self.weights is not None:
            if len(self.weights) != len(self.points_3d):
                raise ValueError(
                    "weights must match point count "
                    f"(got {len(self.weights)} vs {len(self.points_3d)})"
                )
            self.weights = [float(w) for w in self.weights]

    def to_payload(self) -> dict[str, Any]:
        """Convert to Rust/serde shape."""
        payload: dict[str, Any] = {
            "points_3d": [[x, y, z] for x, y, z in self.points_3d],
            "points_2d": [[u, v] for u, v in self.points_2d],
        }
        if self.weights is not None:
            payload["weights"] = list(self.weights)
        return payload

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "Observation":
        """Build observation from Rust/serde shape."""
        return cls(
            points_3d=[cast(Vec3, tuple(p)) for p in payload["points_3d"]],
            points_2d=[cast(Vec2, tuple(p)) for p in payload["points_2d"]],
            weights=cast(list[float] | None, payload.get("weights")),
        )


@dataclass(slots=True)
class PlanarView:
    """Planar calibration view."""

    observation: Observation

    def to_payload(self) -> dict[str, Any]:
        return {"obs": self.observation.to_payload(), "meta": None}


@dataclass(slots=True)
class PlanarDataset:
    """Planar calibration dataset."""

    views: list[PlanarView]

    def to_payload(self) -> dict[str, Any]:
        return {"views": [view.to_payload() for view in self.views]}


@dataclass(slots=True)
class SingleCamHandeyeView:
    """Single-camera hand-eye view."""

    observation: Observation
    base_se3_gripper: Pose

    def to_payload(self) -> dict[str, Any]:
        return {
            "obs": self.observation.to_payload(),
            "meta": {"base_se3_gripper": self.base_se3_gripper.to_payload()},
        }


@dataclass(slots=True)
class SingleCamHandeyeDataset:
    """Single-camera hand-eye dataset."""

    views: list[SingleCamHandeyeView]

    def to_payload(self) -> dict[str, Any]:
        return {"views": [view.to_payload() for view in self.views]}


@dataclass(slots=True)
class RigExtrinsicsView:
    """One frame in a multi-camera rig dataset.

    Parameters
    ----------
    cameras:
        Per-camera observation for this frame. Use `None` for a missing camera.
    """

    cameras: list[Observation | None]

    def to_payload(self) -> dict[str, Any]:
        return {
            "obs": {
                "cameras": [
                    None if obs is None else obs.to_payload()
                    for obs in self.cameras
                ]
            },
            "meta": None,
        }


@dataclass(slots=True)
class RigExtrinsicsDataset:
    """Multi-camera rig dataset for extrinsics calibration."""

    num_cameras: int
    views: list[RigExtrinsicsView]

    def to_payload(self) -> dict[str, Any]:
        return {
            "num_cameras": int(self.num_cameras),
            "views": [view.to_payload() for view in self.views],
        }


@dataclass(slots=True)
class RigHandeyeView:
    """One frame in rig hand-eye calibration."""

    cameras: list[Observation | None]
    base_se3_gripper: Pose

    def to_payload(self) -> dict[str, Any]:
        return {
            "obs": {
                "cameras": [
                    None if obs is None else obs.to_payload()
                    for obs in self.cameras
                ]
            },
            "meta": {"base_se3_gripper": self.base_se3_gripper.to_payload()},
        }


@dataclass(slots=True)
class RigHandeyeDataset:
    """Multi-camera rig hand-eye dataset."""

    num_cameras: int
    views: list[RigHandeyeView]

    def to_payload(self) -> dict[str, Any]:
        return {
            "num_cameras": int(self.num_cameras),
            "views": [view.to_payload() for view in self.views],
        }


@dataclass(slots=True)
class LaserlineView:
    """Single view for laserline-device calibration."""

    observation: Observation
    laser_pixels: list[Vec2]
    laser_weights: list[float] | None = None

    def __post_init__(self) -> None:
        self.laser_pixels = [_as_vec2(list(p)) for p in self.laser_pixels]
        if self.laser_weights is not None:
            if len(self.laser_weights) != len(self.laser_pixels):
                raise ValueError(
                    "laser_weights must match laser_pixels length "
                    f"(got {len(self.laser_weights)} vs {len(self.laser_pixels)})"
                )
            self.laser_weights = [float(w) for w in self.laser_weights]

    def to_payload(self) -> dict[str, Any]:
        return {
            "obs": self.observation.to_payload(),
            "meta": {
                "laser_pixels": [[u, v] for u, v in self.laser_pixels],
                "laser_weights": self.laser_weights,
            },
        }


@dataclass(slots=True)
class LaserlineDataset:
    """Dataset for laserline-device calibration."""

    views: list[LaserlineView]

    def to_payload(self) -> list[dict[str, Any]]:
        return [view.to_payload() for view in self.views]


@dataclass(slots=True)
class PlanarCalibrationConfig:
    """Configuration for planar intrinsics calibration."""

    init_iterations: int = 2
    fix_k3_in_init: bool = True
    fix_tangential_in_init: bool = False
    zero_skew: bool = True
    max_iters: int = 50
    verbosity: int = 0
    robust_loss: RobustLoss = "None"
    fix_intrinsics: dict[str, bool] = field(default_factory=lambda: dict(_DEFAULT_INTRINSICS_FIX_MASK))
    fix_distortion: dict[str, bool] = field(default_factory=lambda: dict(_DEFAULT_DISTORTION_FIX_MASK))
    fix_poses: list[int] = field(default_factory=list)

    def to_payload(self) -> dict[str, Any]:
        fix_intrinsics = dict(_DEFAULT_INTRINSICS_FIX_MASK)
        fix_intrinsics.update(self.fix_intrinsics)
        fix_distortion = dict(_DEFAULT_DISTORTION_FIX_MASK)
        fix_distortion.update(self.fix_distortion)
        return {
            "init_iterations": int(self.init_iterations),
            "fix_k3_in_init": bool(self.fix_k3_in_init),
            "fix_tangential_in_init": bool(self.fix_tangential_in_init),
            "zero_skew": bool(self.zero_skew),
            "max_iters": int(self.max_iters),
            "verbosity": int(self.verbosity),
            "robust_loss": cast(Any, self.robust_loss),
            "fix_intrinsics": fix_intrinsics,
            "fix_distortion": fix_distortion,
            "fix_poses": [int(i) for i in self.fix_poses],
        }

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "PlanarCalibrationConfig":
        cfg = cls()
        for key, value in mapping.items():
            if not hasattr(cfg, key):
                raise ValueError(f"unknown PlanarCalibrationConfig field: {key}")
            setattr(cfg, key, value)
        return cfg


@dataclass(slots=True)
class SingleCamHandeyeCalibrationConfig:
    """Configuration for single-camera hand-eye calibration."""

    intrinsics_init_iterations: int = 2
    fix_k3: bool = True
    fix_tangential: bool = False
    zero_skew: bool = True
    handeye_mode: HandEyeMode = "EyeInHand"
    min_motion_angle_deg: float = 5.0
    max_iters: int = 50
    verbosity: int = 0
    robust_loss: RobustLoss = "None"
    refine_robot_poses: bool = True
    robot_rot_sigma: float = 0.5 * 3.141592653589793 / 180.0
    robot_trans_sigma: float = 0.001

    def to_payload(self) -> dict[str, Any]:
        return {
            "intrinsics_init_iterations": int(self.intrinsics_init_iterations),
            "fix_k3": bool(self.fix_k3),
            "fix_tangential": bool(self.fix_tangential),
            "zero_skew": bool(self.zero_skew),
            "handeye_mode": self.handeye_mode,
            "min_motion_angle_deg": float(self.min_motion_angle_deg),
            "max_iters": int(self.max_iters),
            "verbosity": int(self.verbosity),
            "robust_loss": cast(Any, self.robust_loss),
            "refine_robot_poses": bool(self.refine_robot_poses),
            "robot_rot_sigma": float(self.robot_rot_sigma),
            "robot_trans_sigma": float(self.robot_trans_sigma),
        }

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "SingleCamHandeyeCalibrationConfig":
        cfg = cls()
        for key, value in mapping.items():
            if not hasattr(cfg, key):
                raise ValueError(f"unknown SingleCamHandeyeCalibrationConfig field: {key}")
            setattr(cfg, key, value)
        return cfg


@dataclass(slots=True)
class RigExtrinsicsCalibrationConfig:
    """Configuration for rig extrinsics calibration."""

    intrinsics_init_iterations: int = 2
    fix_k3: bool = True
    fix_tangential: bool = False
    zero_skew: bool = True
    reference_camera_idx: int = 0
    max_iters: int = 50
    verbosity: int = 0
    robust_loss: RobustLoss = "None"
    refine_intrinsics_in_rig_ba: bool = False
    fix_first_rig_pose: bool = True

    def to_payload(self) -> dict[str, Any]:
        return {
            "intrinsics_init_iterations": int(self.intrinsics_init_iterations),
            "fix_k3": bool(self.fix_k3),
            "fix_tangential": bool(self.fix_tangential),
            "zero_skew": bool(self.zero_skew),
            "reference_camera_idx": int(self.reference_camera_idx),
            "max_iters": int(self.max_iters),
            "verbosity": int(self.verbosity),
            "robust_loss": cast(Any, self.robust_loss),
            "refine_intrinsics_in_rig_ba": bool(self.refine_intrinsics_in_rig_ba),
            "fix_first_rig_pose": bool(self.fix_first_rig_pose),
        }

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "RigExtrinsicsCalibrationConfig":
        cfg = cls()
        for key, value in mapping.items():
            if not hasattr(cfg, key):
                raise ValueError(f"unknown RigExtrinsicsCalibrationConfig field: {key}")
            setattr(cfg, key, value)
        return cfg


@dataclass(slots=True)
class RigHandeyeIntrinsicsConfig:
    """Per-camera intrinsics initialization options."""

    init_iterations: int = 2
    fix_k3: bool = True
    fix_tangential: bool = False
    zero_skew: bool = True

    def to_payload(self) -> dict[str, Any]:
        return {
            "init_iterations": int(self.init_iterations),
            "fix_k3": bool(self.fix_k3),
            "fix_tangential": bool(self.fix_tangential),
            "zero_skew": bool(self.zero_skew),
        }

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "RigHandeyeIntrinsicsConfig":
        cfg = cls()
        for key, value in mapping.items():
            if not hasattr(cfg, key):
                raise ValueError(f"unknown RigHandeyeIntrinsicsConfig field: {key}")
            setattr(cfg, key, value)
        return cfg


@dataclass(slots=True)
class RigHandeyeRigConfig:
    """Rig frame / gauge options."""

    reference_camera_idx: int = 0
    refine_intrinsics_in_rig_ba: bool = False
    fix_first_rig_pose: bool = True

    def to_payload(self) -> dict[str, Any]:
        return {
            "reference_camera_idx": int(self.reference_camera_idx),
            "refine_intrinsics_in_rig_ba": bool(self.refine_intrinsics_in_rig_ba),
            "fix_first_rig_pose": bool(self.fix_first_rig_pose),
        }

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "RigHandeyeRigConfig":
        cfg = cls()
        for key, value in mapping.items():
            if not hasattr(cfg, key):
                raise ValueError(f"unknown RigHandeyeRigConfig field: {key}")
            setattr(cfg, key, value)
        return cfg


@dataclass(slots=True)
class RigHandeyeInitConfig:
    """Linear hand-eye initialization settings."""

    handeye_mode: HandEyeMode = "EyeInHand"
    min_motion_angle_deg: float = 5.0

    def to_payload(self) -> dict[str, Any]:
        return {
            "handeye_mode": self.handeye_mode,
            "min_motion_angle_deg": float(self.min_motion_angle_deg),
        }

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "RigHandeyeInitConfig":
        cfg = cls()
        for key, value in mapping.items():
            if not hasattr(cfg, key):
                raise ValueError(f"unknown RigHandeyeInitConfig field: {key}")
            setattr(cfg, key, value)
        return cfg


@dataclass(slots=True)
class RigHandeyeSolverConfig:
    """Shared nonlinear solver settings."""

    max_iters: int = 50
    verbosity: int = 0
    robust_loss: RobustLoss = "None"

    def to_payload(self) -> dict[str, Any]:
        return {
            "max_iters": int(self.max_iters),
            "verbosity": int(self.verbosity),
            "robust_loss": cast(Any, self.robust_loss),
        }

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "RigHandeyeSolverConfig":
        cfg = cls()
        for key, value in mapping.items():
            if not hasattr(cfg, key):
                raise ValueError(f"unknown RigHandeyeSolverConfig field: {key}")
            setattr(cfg, key, value)
        return cfg


@dataclass(slots=True)
class RigHandeyeBaConfig:
    """Final hand-eye bundle-adjustment options."""

    refine_robot_poses: bool = True
    robot_rot_sigma: float = 0.5 * 3.141592653589793 / 180.0
    robot_trans_sigma: float = 0.001
    refine_cam_se3_rig_in_handeye_ba: bool = False

    def to_payload(self) -> dict[str, Any]:
        return {
            "refine_robot_poses": bool(self.refine_robot_poses),
            "robot_rot_sigma": float(self.robot_rot_sigma),
            "robot_trans_sigma": float(self.robot_trans_sigma),
            "refine_cam_se3_rig_in_handeye_ba": bool(self.refine_cam_se3_rig_in_handeye_ba),
        }

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "RigHandeyeBaConfig":
        cfg = cls()
        for key, value in mapping.items():
            if not hasattr(cfg, key):
                raise ValueError(f"unknown RigHandeyeBaConfig field: {key}")
            setattr(cfg, key, value)
        return cfg


@dataclass(slots=True)
class RigHandeyeCalibrationConfig:
    """Configuration for rig hand-eye calibration."""

    intrinsics: RigHandeyeIntrinsicsConfig = field(default_factory=RigHandeyeIntrinsicsConfig)
    rig: RigHandeyeRigConfig = field(default_factory=RigHandeyeRigConfig)
    handeye_init: RigHandeyeInitConfig = field(default_factory=RigHandeyeInitConfig)
    solver: RigHandeyeSolverConfig = field(default_factory=RigHandeyeSolverConfig)
    handeye_ba: RigHandeyeBaConfig = field(default_factory=RigHandeyeBaConfig)

    def to_payload(self) -> dict[str, Any]:
        return {
            "intrinsics": self.intrinsics.to_payload(),
            "rig": self.rig.to_payload(),
            "handeye_init": self.handeye_init.to_payload(),
            "solver": self.solver.to_payload(),
            "handeye_ba": self.handeye_ba.to_payload(),
        }

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "RigHandeyeCalibrationConfig":
        cfg = cls()
        for key, value in mapping.items():
            if key == "intrinsics":
                cfg.intrinsics = (
                    value
                    if isinstance(value, RigHandeyeIntrinsicsConfig)
                    else RigHandeyeIntrinsicsConfig.from_mapping(cast(Mapping[str, Any], value))
                )
            elif key == "rig":
                cfg.rig = (
                    value
                    if isinstance(value, RigHandeyeRigConfig)
                    else RigHandeyeRigConfig.from_mapping(cast(Mapping[str, Any], value))
                )
            elif key == "handeye_init":
                cfg.handeye_init = (
                    value
                    if isinstance(value, RigHandeyeInitConfig)
                    else RigHandeyeInitConfig.from_mapping(cast(Mapping[str, Any], value))
                )
            elif key == "solver":
                cfg.solver = (
                    value
                    if isinstance(value, RigHandeyeSolverConfig)
                    else RigHandeyeSolverConfig.from_mapping(cast(Mapping[str, Any], value))
                )
            elif key == "handeye_ba":
                cfg.handeye_ba = (
                    value
                    if isinstance(value, RigHandeyeBaConfig)
                    else RigHandeyeBaConfig.from_mapping(cast(Mapping[str, Any], value))
                )
            else:
                raise ValueError(f"unknown RigHandeyeCalibrationConfig field: {key}")
        return cfg


@dataclass(slots=True)
class LaserlineDeviceInitConfig:
    """Initialization options for laserline-device calibration."""

    iterations: int = 2
    fix_k3: bool = True
    fix_tangential: bool = False
    zero_skew: bool = True
    sensor_init: dict[str, float] = field(default_factory=lambda: dict(_DEFAULT_SENSOR_INIT))

    def to_payload(self) -> dict[str, Any]:
        sensor_init = dict(_DEFAULT_SENSOR_INIT)
        sensor_init.update(self.sensor_init)
        return {
            "iterations": int(self.iterations),
            "fix_k3": bool(self.fix_k3),
            "fix_tangential": bool(self.fix_tangential),
            "zero_skew": bool(self.zero_skew),
            "sensor_init": sensor_init,
        }

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "LaserlineDeviceInitConfig":
        cfg = cls()
        for key, value in mapping.items():
            if not hasattr(cfg, key):
                raise ValueError(f"unknown LaserlineDeviceInitConfig field: {key}")
            setattr(cfg, key, value)
        return cfg


@dataclass(slots=True)
class LaserlineDeviceSolverConfig:
    """Nonlinear solver options for laserline-device calibration."""

    max_iters: int = 50
    verbosity: int = 0

    def to_payload(self) -> dict[str, Any]:
        return {
            "max_iters": int(self.max_iters),
            "verbosity": int(self.verbosity),
        }

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "LaserlineDeviceSolverConfig":
        cfg = cls()
        for key, value in mapping.items():
            if not hasattr(cfg, key):
                raise ValueError(f"unknown LaserlineDeviceSolverConfig field: {key}")
            setattr(cfg, key, value)
        return cfg


@dataclass(slots=True)
class LaserlineDeviceOptimizeConfig:
    """Bundle-adjustment options for laserline-device calibration."""

    calib_loss: RobustLoss = field(default_factory=lambda: {"Huber": {"scale": 1.0}})
    laser_loss: RobustLoss = field(default_factory=lambda: {"Huber": {"scale": 0.01}})
    calib_weight: float = 1.0
    laser_weight: float = 1.0
    fix_intrinsics: bool = False
    fix_distortion: bool = False
    fix_k3: bool = True
    fix_sensor: bool = True
    fix_poses: list[int] = field(default_factory=lambda: [0])
    fix_plane: bool = False
    laser_residual_type: LaserlineResidualType = "LineDistNormalized"

    def to_payload(self) -> dict[str, Any]:
        return {
            "calib_loss": cast(Any, self.calib_loss),
            "laser_loss": cast(Any, self.laser_loss),
            "calib_weight": float(self.calib_weight),
            "laser_weight": float(self.laser_weight),
            "fix_intrinsics": bool(self.fix_intrinsics),
            "fix_distortion": bool(self.fix_distortion),
            "fix_k3": bool(self.fix_k3),
            "fix_sensor": bool(self.fix_sensor),
            "fix_poses": [int(i) for i in self.fix_poses],
            "fix_plane": bool(self.fix_plane),
            "laser_residual_type": self.laser_residual_type,
        }

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "LaserlineDeviceOptimizeConfig":
        cfg = cls()
        for key, value in mapping.items():
            if not hasattr(cfg, key):
                raise ValueError(f"unknown LaserlineDeviceOptimizeConfig field: {key}")
            setattr(cfg, key, value)
        return cfg


@dataclass(slots=True)
class LaserlineDeviceCalibrationConfig:
    """Configuration for laserline-device calibration."""

    init: LaserlineDeviceInitConfig = field(default_factory=LaserlineDeviceInitConfig)
    solver: LaserlineDeviceSolverConfig = field(default_factory=LaserlineDeviceSolverConfig)
    optimize: LaserlineDeviceOptimizeConfig = field(default_factory=LaserlineDeviceOptimizeConfig)

    def to_payload(self) -> dict[str, Any]:
        return {
            "init": self.init.to_payload(),
            "solver": self.solver.to_payload(),
            "optimize": self.optimize.to_payload(),
        }

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "LaserlineDeviceCalibrationConfig":
        cfg = cls()
        for key, value in mapping.items():
            if key == "init":
                cfg.init = (
                    value
                    if isinstance(value, LaserlineDeviceInitConfig)
                    else LaserlineDeviceInitConfig.from_mapping(cast(Mapping[str, Any], value))
                )
            elif key == "solver":
                cfg.solver = (
                    value
                    if isinstance(value, LaserlineDeviceSolverConfig)
                    else LaserlineDeviceSolverConfig.from_mapping(cast(Mapping[str, Any], value))
                )
            elif key == "optimize":
                cfg.optimize = (
                    value
                    if isinstance(value, LaserlineDeviceOptimizeConfig)
                    else LaserlineDeviceOptimizeConfig.from_mapping(cast(Mapping[str, Any], value))
                )
            else:
                raise ValueError(f"unknown LaserlineDeviceCalibrationConfig field: {key}")
        return cfg


@dataclass(slots=True)
class ScheimpflugIntrinsicsCalibrationConfig:
    """Configuration for planar Scheimpflug intrinsics calibration."""

    init_iterations: int = 2
    fix_k3_in_init: bool = True
    zero_skew: bool = True
    max_iters: int = 120
    verbosity: int = 0
    robust_loss: RobustLoss = field(default_factory=lambda: "None")
    fix_intrinsics: dict[str, bool] = field(default_factory=lambda: dict(_DEFAULT_INTRINSICS_FIX_MASK))
    fix_distortion: dict[str, bool] = field(
        default_factory=lambda: {
            "k1": False,
            "k2": False,
            "k3": True,
            "p1": True,
            "p2": True,
        }
    )
    fix_scheimpflug: dict[str, bool] = field(default_factory=lambda: dict(_DEFAULT_SCHEIMPFLUG_FIX_MASK))
    fix_first_pose: bool = True

    def to_payload(self) -> dict[str, Any]:
        fix_intrinsics = dict(_DEFAULT_INTRINSICS_FIX_MASK)
        fix_intrinsics.update(self.fix_intrinsics)
        fix_distortion = {
            "k1": False,
            "k2": False,
            "k3": True,
            "p1": True,
            "p2": True,
        }
        fix_distortion.update(self.fix_distortion)
        fix_scheimpflug = dict(_DEFAULT_SCHEIMPFLUG_FIX_MASK)
        fix_scheimpflug.update(self.fix_scheimpflug)
        return {
            "init_iterations": int(self.init_iterations),
            "fix_k3_in_init": bool(self.fix_k3_in_init),
            "zero_skew": bool(self.zero_skew),
            "max_iters": int(self.max_iters),
            "verbosity": int(self.verbosity),
            "robust_loss": cast(Any, self.robust_loss),
            "fix_intrinsics": fix_intrinsics,
            "fix_distortion": fix_distortion,
            "fix_scheimpflug": fix_scheimpflug,
            "fix_first_pose": bool(self.fix_first_pose),
        }

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "ScheimpflugIntrinsicsCalibrationConfig":
        cfg = cls()
        for key, value in mapping.items():
            if not hasattr(cfg, key):
                raise ValueError(f"unknown ScheimpflugIntrinsicsCalibrationConfig field: {key}")
            setattr(cfg, key, value)
        return cfg


@dataclass(slots=True)
class PinholeIntrinsics:
    """Typed pinhole intrinsics model."""

    fx: float
    fy: float
    cx: float
    cy: float
    skew: float

    def to_payload(self) -> dict[str, float]:
        """Convert to serde payload shape."""
        return {
            "fx": float(self.fx),
            "fy": float(self.fy),
            "cx": float(self.cx),
            "cy": float(self.cy),
            "skew": float(self.skew),
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "PinholeIntrinsics":
        """Parse intrinsics from serde payload shape."""
        return cls(
            fx=float(payload["fx"]),
            fy=float(payload["fy"]),
            cx=float(payload["cx"]),
            cy=float(payload["cy"]),
            skew=float(payload["skew"]),
        )


@dataclass(slots=True)
class BrownConradyDistortion:
    """Typed Brown-Conrady distortion model."""

    k1: float
    k2: float
    k3: float
    p1: float
    p2: float
    iters: int

    def to_payload(self) -> dict[str, float | int]:
        """Convert to serde payload shape."""
        return {
            "k1": float(self.k1),
            "k2": float(self.k2),
            "k3": float(self.k3),
            "p1": float(self.p1),
            "p2": float(self.p2),
            "iters": int(self.iters),
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "BrownConradyDistortion":
        """Parse distortion from serde payload shape."""
        return cls(
            k1=float(payload["k1"]),
            k2=float(payload["k2"]),
            k3=float(payload["k3"]),
            p1=float(payload["p1"]),
            p2=float(payload["p2"]),
            iters=int(payload["iters"]),
        )


@dataclass(slots=True)
class ScheimpflugSensor:
    """Typed Scheimpflug sensor tilt model."""

    tilt_x: float
    tilt_y: float

    def to_payload(self) -> dict[str, float]:
        """Convert to serde payload shape."""
        return {
            "tilt_x": float(self.tilt_x),
            "tilt_y": float(self.tilt_y),
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "ScheimpflugSensor":
        """Parse Scheimpflug tilt, accepting legacy `tau_x`/`tau_y` aliases."""
        tilt_x = payload.get("tilt_x", payload.get("tau_x"))
        tilt_y = payload.get("tilt_y", payload.get("tau_y"))
        if tilt_x is None:
            raise ValueError("Scheimpflug sensor payload missing tilt_x/tau_x")
        if tilt_y is None:
            raise ValueError("Scheimpflug sensor payload missing tilt_y/tau_y")
        return cls(tilt_x=float(tilt_x), tilt_y=float(tilt_y))


@dataclass(slots=True)
class PinholeBrownConradyCamera:
    """Typed pinhole camera model with Brown-Conrady distortion."""

    intrinsics: PinholeIntrinsics
    distortion: BrownConradyDistortion

    def to_payload(self) -> dict[str, Any]:
        """Convert to the `PinholeCamera` serde payload shape."""
        return {
            "k": self.intrinsics.to_payload(),
            "dist": self.distortion.to_payload(),
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "PinholeBrownConradyCamera":
        """Parse from either `PinholeCamera` or `CameraParams`-style payload."""
        if "k" in payload and "dist" in payload:
            return cls(
                intrinsics=PinholeIntrinsics.from_payload(cast(Mapping[str, Any], payload["k"])),
                distortion=BrownConradyDistortion.from_payload(
                    cast(Mapping[str, Any], payload["dist"])
                ),
            )

        if "intrinsics" in payload and "distortion" in payload:
            intrinsics_payload = cast(Mapping[str, Any], payload["intrinsics"])
            distortion_payload = cast(Mapping[str, Any], payload["distortion"])

            intrinsics_type = intrinsics_payload.get("type")
            if intrinsics_type not in (None, "fx_fy_cx_cy_skew"):
                raise ValueError(
                    f"unsupported intrinsics payload type for pinhole camera: {intrinsics_type!r}"
                )

            distortion_type = distortion_payload.get("type")
            if distortion_type not in (None, "brown_conrady5"):
                raise ValueError(
                    f"unsupported distortion payload type for pinhole camera: {distortion_type!r}"
                )

            return cls(
                intrinsics=PinholeIntrinsics.from_payload(intrinsics_payload),
                distortion=BrownConradyDistortion.from_payload(distortion_payload),
            )

        raise ValueError("camera payload missing expected intrinsics/distortion fields")


@dataclass(slots=True)
class PinholeBrownConradyScheimpflugCamera:
    """Typed pinhole + Brown-Conrady + Scheimpflug camera model."""

    intrinsics: PinholeIntrinsics
    distortion: BrownConradyDistortion
    sensor: ScheimpflugSensor

    def to_payload(self) -> dict[str, Any]:
        """Convert to `CameraParams` serde payload shape."""
        return {
            "projection": {"type": "pinhole"},
            "distortion": {"type": "brown_conrady5", **self.distortion.to_payload()},
            "sensor": {"type": "scheimpflug", **self.sensor.to_payload()},
            "intrinsics": {"type": "fx_fy_cx_cy_skew", **self.intrinsics.to_payload()},
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "PinholeBrownConradyScheimpflugCamera":
        """Parse from `CameraParams` payload with Scheimpflug sensor."""
        intrinsics_payload = cast(Mapping[str, Any], payload["intrinsics"])
        distortion_payload = cast(Mapping[str, Any], payload["distortion"])
        sensor_payload = cast(Mapping[str, Any], payload["sensor"])

        intrinsics_type = intrinsics_payload.get("type")
        if intrinsics_type not in (None, "fx_fy_cx_cy_skew"):
            raise ValueError(
                f"unsupported intrinsics payload type for Scheimpflug camera: {intrinsics_type!r}"
            )

        distortion_type = distortion_payload.get("type")
        if distortion_type not in (None, "brown_conrady5"):
            raise ValueError(
                f"unsupported distortion payload type for Scheimpflug camera: {distortion_type!r}"
            )

        sensor_type = sensor_payload.get("type")
        if sensor_type not in (None, "scheimpflug"):
            raise ValueError(
                f"unsupported sensor payload type for Scheimpflug camera: {sensor_type!r}"
            )

        return cls(
            intrinsics=PinholeIntrinsics.from_payload(intrinsics_payload),
            distortion=BrownConradyDistortion.from_payload(distortion_payload),
            sensor=ScheimpflugSensor.from_payload(sensor_payload),
        )


@dataclass(slots=True)
class LaserlinePlane:
    """Typed laser plane representation."""

    normal_xyz: Vec3
    distance: float

    def __post_init__(self) -> None:
        self.normal_xyz = _as_vec3(list(self.normal_xyz))
        self.distance = float(self.distance)

    def to_payload(self) -> dict[str, Any]:
        """Convert to Rust/serde shape.

        Rust's ``LaserPlane`` (``nalgebra::Unit<Vector3<f64>>``) deserializes
        ``normal`` as a plain array ``[x, y, z]``.
        """
        return {
            "normal": list(self.normal_xyz),
            "distance": float(self.distance),
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "LaserlinePlane":
        """Parse laser plane from serde payload."""
        normal_value = payload["normal"]
        if isinstance(normal_value, Mapping):
            if "coords" in normal_value:
                normal_value = normal_value["coords"]
            elif "data" in normal_value:
                normal_value = normal_value["data"]
            else:
                raise ValueError("unsupported laser plane normal payload shape")
        return cls(
            normal_xyz=cast(Vec3, tuple(normal_value)),
            distance=float(payload["distance"]),
        )


@dataclass(slots=True)
class LaserlineEstimateParams:
    """Typed parameter payload for laserline optimization results."""

    intrinsics: PinholeIntrinsics
    distortion: BrownConradyDistortion
    sensor: ScheimpflugSensor
    poses: list[Pose]
    plane: LaserlinePlane

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "LaserlineEstimateParams":
        return cls(
            intrinsics=PinholeIntrinsics.from_payload(
                cast(Mapping[str, Any], payload["intrinsics"])
            ),
            distortion=BrownConradyDistortion.from_payload(
                cast(Mapping[str, Any], payload["distortion"])
            ),
            sensor=ScheimpflugSensor.from_payload(cast(Mapping[str, Any], payload["sensor"])),
            poses=[
                Pose.from_payload(cast(Mapping[str, Any], pose))
                for pose in cast(list[Any], payload["poses"])
            ],
            plane=LaserlinePlane.from_payload(cast(Mapping[str, Any], payload["plane"])),
        )


@dataclass(slots=True)
class LaserlineEstimate:
    """Typed laserline optimizer estimate payload."""

    params: LaserlineEstimateParams
    final_cost: float

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "LaserlineEstimate":
        report = cast(Mapping[str, Any], payload["report"])
        return cls(
            params=LaserlineEstimateParams.from_payload(
                cast(Mapping[str, Any], payload["params"])
            ),
            final_cost=float(report["final_cost"]),
        )


@dataclass(slots=True)
class LaserlineStats:
    """Typed summary statistics for laserline results."""

    mean_reproj_error: float
    mean_laser_error: float
    per_view_reproj_errors: list[float]
    per_view_laser_errors: list[float]

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "LaserlineStats":
        return cls(
            mean_reproj_error=float(payload["mean_reproj_error"]),
            mean_laser_error=float(payload["mean_laser_error"]),
            per_view_reproj_errors=[
                float(v) for v in cast(list[Any], payload["per_view_reproj_errors"])
            ],
            per_view_laser_errors=[
                float(v) for v in cast(list[Any], payload["per_view_laser_errors"])
            ],
        )


@dataclass(slots=True)
class PlanarCalibrationResult:
    """Result from :func:`vision_calibration.run_planar_intrinsics`."""

    camera: PinholeBrownConradyCamera
    camera_se3_target: list[Pose]
    final_cost: float
    mean_reproj_error: float
    per_cam_reproj_errors: list[float]

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "PlanarCalibrationResult":
        params = cast(Mapping[str, Any], payload["params"])
        report = cast(Mapping[str, Any], payload["report"])
        poses = [
            Pose.from_payload(cast(Mapping[str, Any], p))
            for p in cast(list[Any], params["camera_se3_target"])
        ]
        return cls(
            camera=PinholeBrownConradyCamera.from_payload(
                cast(Mapping[str, Any], params["camera"])
            ),
            camera_se3_target=poses,
            final_cost=float(report["final_cost"]),
            mean_reproj_error=float(payload["mean_reproj_error"]),
            per_cam_reproj_errors=[
                float(v) for v in cast(list[Any], payload["per_cam_reproj_errors"])
            ],
        )


@dataclass(slots=True)
class SingleCamHandeyeResult:
    """Result from :func:`vision_calibration.run_single_cam_handeye`."""

    camera: PinholeBrownConradyCamera
    handeye_mode: HandEyeMode
    gripper_se3_camera: Pose | None
    camera_se3_base: Pose | None
    base_se3_target: Pose | None
    gripper_se3_target: Pose | None
    robot_deltas: list[list[float]] | None
    mean_reproj_error: float
    per_cam_reproj_errors: list[float]

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "SingleCamHandeyeResult":
        def _pose(name: str) -> Pose | None:
            value = payload.get(name)
            if value is None:
                return None
            return Pose.from_payload(cast(Mapping[str, Any], value))

        return cls(
            camera=PinholeBrownConradyCamera.from_payload(
                cast(Mapping[str, Any], payload["camera"])
            ),
            handeye_mode=cast(HandEyeMode, payload["handeye_mode"]),
            gripper_se3_camera=_pose("gripper_se3_camera"),
            camera_se3_base=_pose("camera_se3_base"),
            base_se3_target=_pose("base_se3_target"),
            gripper_se3_target=_pose("gripper_se3_target"),
            robot_deltas=cast(list[list[float]] | None, payload.get("robot_deltas")),
            mean_reproj_error=float(payload["mean_reproj_error"]),
            per_cam_reproj_errors=[float(v) for v in cast(list[Any], payload["per_cam_reproj_errors"])],
        )


@dataclass(slots=True)
class RigExtrinsicsResult:
    """Result from :func:`vision_calibration.run_rig_extrinsics`."""

    cameras: list[PinholeBrownConradyCamera]
    cam_se3_rig: list[Pose]
    mean_reproj_error: float
    per_cam_reproj_errors: list[float]

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "RigExtrinsicsResult":
        return cls(
            cameras=[
                PinholeBrownConradyCamera.from_payload(cast(Mapping[str, Any], c))
                for c in cast(list[Any], payload["cameras"])
            ],
            cam_se3_rig=[Pose.from_payload(cast(Mapping[str, Any], p)) for p in cast(list[Any], payload["cam_se3_rig"])],
            mean_reproj_error=float(payload["mean_reproj_error"]),
            per_cam_reproj_errors=[float(v) for v in cast(list[Any], payload["per_cam_reproj_errors"])],
        )


@dataclass(slots=True)
class RigHandeyeResult:
    """Result from :func:`vision_calibration.run_rig_handeye`."""

    cameras: list[PinholeBrownConradyCamera]
    cam_se3_rig: list[Pose]
    handeye_mode: HandEyeMode
    gripper_se3_rig: Pose | None
    rig_se3_base: Pose | None
    base_se3_target: Pose | None
    gripper_se3_target: Pose | None
    robot_deltas: list[list[float]] | None
    mean_reproj_error: float
    per_cam_reproj_errors: list[float]

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "RigHandeyeResult":
        def _pose(name: str) -> Pose | None:
            value = payload.get(name)
            if value is None:
                return None
            return Pose.from_payload(cast(Mapping[str, Any], value))

        return cls(
            cameras=[
                PinholeBrownConradyCamera.from_payload(cast(Mapping[str, Any], c))
                for c in cast(list[Any], payload["cameras"])
            ],
            cam_se3_rig=[Pose.from_payload(cast(Mapping[str, Any], p)) for p in cast(list[Any], payload["cam_se3_rig"])],
            handeye_mode=cast(HandEyeMode, payload["handeye_mode"]),
            gripper_se3_rig=_pose("gripper_se3_rig"),
            rig_se3_base=_pose("rig_se3_base"),
            base_se3_target=_pose("base_se3_target"),
            gripper_se3_target=_pose("gripper_se3_target"),
            robot_deltas=cast(list[list[float]] | None, payload.get("robot_deltas")),
            mean_reproj_error=float(payload["mean_reproj_error"]),
            per_cam_reproj_errors=[float(v) for v in cast(list[Any], payload["per_cam_reproj_errors"])],
        )


@dataclass(slots=True)
class LaserlineDeviceResult:
    """Result from :func:`vision_calibration.run_laserline_device`."""

    estimate: LaserlineEstimate
    stats: LaserlineStats
    mean_reproj_error: float
    per_cam_reproj_errors: list[float]

    @property
    def mean_laser_error(self) -> float:
        """Mean laser residual (units depend on residual type)."""
        return self.stats.mean_laser_error

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "LaserlineDeviceResult":
        return cls(
            estimate=LaserlineEstimate.from_payload(
                cast(Mapping[str, Any], payload["estimate"])
            ),
            stats=LaserlineStats.from_payload(cast(Mapping[str, Any], payload["stats"])),
            mean_reproj_error=float(payload["mean_reproj_error"]),
            per_cam_reproj_errors=[float(v) for v in cast(list[Any], payload["per_cam_reproj_errors"])],
        )


@dataclass(slots=True)
class ScheimpflugIntrinsicsResult:
    """Result from :func:`vision_calibration.run_scheimpflug_intrinsics`."""

    camera: PinholeBrownConradyScheimpflugCamera
    camera_se3_target: list[Pose]
    final_cost: float
    mean_reproj_error: float
    per_cam_reproj_errors: list[float]

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "ScheimpflugIntrinsicsResult":
        params = cast(Mapping[str, Any], payload["params"])
        report = cast(Mapping[str, Any], payload["report"])
        return cls(
            camera=PinholeBrownConradyScheimpflugCamera.from_payload(
                cast(Mapping[str, Any], params["camera"])
            ),
            camera_se3_target=[
                Pose.from_payload(cast(Mapping[str, Any], pose))
                for pose in cast(list[Any], params["camera_se3_target"])
            ],
            final_cost=float(report["final_cost"]),
            mean_reproj_error=float(payload["mean_reproj_error"]),
            per_cam_reproj_errors=[float(v) for v in cast(list[Any], payload["per_cam_reproj_errors"])],
        )


# ─── Scheimpflug rig extrinsics ─────────────────────────────────────────────


@dataclass(slots=True)
class RigScheimpflugExtrinsicsDataset:
    """Multi-camera Scheimpflug rig dataset for extrinsics calibration.

    Mirrors :class:`RigHandeyeDataset` but without per-view robot pose metadata.
    """

    num_cameras: int
    views: list[RigExtrinsicsView]

    def to_payload(self) -> dict[str, Any]:
        return {
            "num_cameras": int(self.num_cameras),
            "views": [view.to_payload() for view in self.views],
        }


@dataclass(slots=True)
class RigScheimpflugExtrinsicsCalibrationConfig:
    """Configuration for Scheimpflug rig extrinsics calibration."""

    intrinsics_init_iterations: int = 2
    fix_k3: bool = True
    fix_tangential: bool = False
    zero_skew: bool = True
    init_tilt_x: float = 0.0
    init_tilt_y: float = 0.0
    fix_scheimpflug_in_intrinsics: dict[str, bool] = field(
        default_factory=lambda: dict(_DEFAULT_SCHEIMPFLUG_FIX_MASK)
    )
    reference_camera_idx: int = 0
    max_iters: int = 80
    verbosity: int = 0
    robust_loss: RobustLoss = "None"
    refine_intrinsics_in_rig_ba: bool = False
    refine_scheimpflug_in_rig_ba: bool = False
    fix_first_rig_pose: bool = True

    def to_payload(self) -> dict[str, Any]:
        fix_s = dict(_DEFAULT_SCHEIMPFLUG_FIX_MASK)
        fix_s.update(self.fix_scheimpflug_in_intrinsics)
        return {
            "intrinsics_init_iterations": int(self.intrinsics_init_iterations),
            "fix_k3": bool(self.fix_k3),
            "fix_tangential": bool(self.fix_tangential),
            "zero_skew": bool(self.zero_skew),
            "init_tilt_x": float(self.init_tilt_x),
            "init_tilt_y": float(self.init_tilt_y),
            "fix_scheimpflug_in_intrinsics": fix_s,
            "reference_camera_idx": int(self.reference_camera_idx),
            "max_iters": int(self.max_iters),
            "verbosity": int(self.verbosity),
            "robust_loss": cast(Any, self.robust_loss),
            "refine_intrinsics_in_rig_ba": bool(self.refine_intrinsics_in_rig_ba),
            "refine_scheimpflug_in_rig_ba": bool(self.refine_scheimpflug_in_rig_ba),
            "fix_first_rig_pose": bool(self.fix_first_rig_pose),
        }


@dataclass(slots=True)
class RigScheimpflugExtrinsicsResult:
    """Result from :func:`vision_calibration.run_rig_scheimpflug_extrinsics`."""

    cameras: list[PinholeBrownConradyCamera]
    sensors: list[ScheimpflugSensor]
    cam_se3_rig: list[Pose]
    rig_se3_target: list[Pose]
    mean_reproj_error: float
    per_cam_reproj_errors: list[float]

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "RigScheimpflugExtrinsicsResult":
        return cls(
            cameras=[
                PinholeBrownConradyCamera.from_payload(cast(Mapping[str, Any], c))
                for c in cast(list[Any], payload["cameras"])
            ],
            sensors=[
                ScheimpflugSensor.from_payload(cast(Mapping[str, Any], s))
                for s in cast(list[Any], payload["sensors"])
            ],
            cam_se3_rig=[
                Pose.from_payload(cast(Mapping[str, Any], p))
                for p in cast(list[Any], payload["cam_se3_rig"])
            ],
            rig_se3_target=[
                Pose.from_payload(cast(Mapping[str, Any], p))
                for p in cast(list[Any], payload["rig_se3_target"])
            ],
            mean_reproj_error=float(payload["mean_reproj_error"]),
            per_cam_reproj_errors=[
                float(v) for v in cast(list[Any], payload["per_cam_reproj_errors"])
            ],
        )


# ─── Scheimpflug rig hand-eye ────────────────────────────────────────────────


@dataclass(slots=True)
class RigScheimpflugHandeyeIntrinsicsConfig:
    """Per-camera intrinsics initialization options for Scheimpflug hand-eye."""

    init_iterations: int = 2
    fix_k3: bool = True
    fix_tangential: bool = False
    zero_skew: bool = True
    init_tilt_x: float = 0.0
    init_tilt_y: float = 0.0
    fix_scheimpflug: dict[str, bool] = field(
        default_factory=lambda: dict(_DEFAULT_SCHEIMPFLUG_FIX_MASK)
    )
    fallback_to_shared_init: bool = True
    fix_intrinsics_when_overridden: bool = True
    fix_intrinsics_in_percam_ba: dict[str, bool] = field(
        default_factory=lambda: dict(_DEFAULT_INTRINSICS_FIX_MASK)
    )
    fix_distortion_in_percam_ba: dict[str, bool] = field(
        default_factory=lambda: dict(_DEFAULT_DISTORTION_FIX_MASK)
    )

    def to_payload(self) -> dict[str, Any]:
        fix_s = dict(_DEFAULT_SCHEIMPFLUG_FIX_MASK)
        fix_s.update(self.fix_scheimpflug)
        fix_i = dict(_DEFAULT_INTRINSICS_FIX_MASK)
        fix_i.update(self.fix_intrinsics_in_percam_ba)
        fix_d = dict(_DEFAULT_DISTORTION_FIX_MASK)
        fix_d.update(self.fix_distortion_in_percam_ba)
        return {
            "init_iterations": int(self.init_iterations),
            "fix_k3": bool(self.fix_k3),
            "fix_tangential": bool(self.fix_tangential),
            "zero_skew": bool(self.zero_skew),
            "init_tilt_x": float(self.init_tilt_x),
            "init_tilt_y": float(self.init_tilt_y),
            "fix_scheimpflug": fix_s,
            "initial_cameras": None,
            "initial_sensors": None,
            "fallback_to_shared_init": bool(self.fallback_to_shared_init),
            "fix_intrinsics_when_overridden": bool(self.fix_intrinsics_when_overridden),
            "fix_intrinsics_in_percam_ba": fix_i,
            "fix_distortion_in_percam_ba": fix_d,
        }


@dataclass(slots=True)
class RigScheimpflugHandeyeRigConfig:
    """Rig frame / gauge options for Scheimpflug hand-eye."""

    reference_camera_idx: int = 0
    refine_intrinsics_in_rig_ba: bool = False
    refine_scheimpflug_in_rig_ba: bool = False
    fix_first_rig_pose: bool = True

    def to_payload(self) -> dict[str, Any]:
        return {
            "reference_camera_idx": int(self.reference_camera_idx),
            "refine_intrinsics_in_rig_ba": bool(self.refine_intrinsics_in_rig_ba),
            "refine_scheimpflug_in_rig_ba": bool(self.refine_scheimpflug_in_rig_ba),
            "fix_first_rig_pose": bool(self.fix_first_rig_pose),
        }


@dataclass(slots=True)
class RigScheimpflugHandeyeInitConfig:
    """Linear hand-eye initialization settings for Scheimpflug."""

    handeye_mode: HandEyeMode = "EyeInHand"
    min_motion_angle_deg: float = 5.0

    def to_payload(self) -> dict[str, Any]:
        return {
            "handeye_mode": self.handeye_mode,
            "min_motion_angle_deg": float(self.min_motion_angle_deg),
        }


@dataclass(slots=True)
class RigScheimpflugHandeyeSolverConfig:
    """Shared nonlinear solver settings for Scheimpflug hand-eye."""

    max_iters: int = 80
    verbosity: int = 0
    robust_loss: RobustLoss = "None"

    def to_payload(self) -> dict[str, Any]:
        return {
            "max_iters": int(self.max_iters),
            "verbosity": int(self.verbosity),
            "robust_loss": cast(Any, self.robust_loss),
        }


@dataclass(slots=True)
class RigScheimpflugHandeyeBaConfig:
    """Final hand-eye bundle-adjustment options for Scheimpflug."""

    refine_robot_poses: bool = True
    robot_rot_sigma: float = 0.5 * 3.141592653589793 / 180.0
    robot_trans_sigma: float = 0.001
    refine_cam_se3_rig_in_handeye_ba: bool = False
    refine_scheimpflug_in_handeye_ba: bool = False

    def to_payload(self) -> dict[str, Any]:
        return {
            "refine_robot_poses": bool(self.refine_robot_poses),
            "robot_rot_sigma": float(self.robot_rot_sigma),
            "robot_trans_sigma": float(self.robot_trans_sigma),
            "refine_cam_se3_rig_in_handeye_ba": bool(self.refine_cam_se3_rig_in_handeye_ba),
            "refine_scheimpflug_in_handeye_ba": bool(self.refine_scheimpflug_in_handeye_ba),
        }


@dataclass(slots=True)
class RigScheimpflugHandeyeCalibrationConfig:
    """Configuration for Scheimpflug rig hand-eye calibration."""

    intrinsics: RigScheimpflugHandeyeIntrinsicsConfig = field(
        default_factory=RigScheimpflugHandeyeIntrinsicsConfig
    )
    rig: RigScheimpflugHandeyeRigConfig = field(
        default_factory=RigScheimpflugHandeyeRigConfig
    )
    handeye_init: RigScheimpflugHandeyeInitConfig = field(
        default_factory=RigScheimpflugHandeyeInitConfig
    )
    solver: RigScheimpflugHandeyeSolverConfig = field(
        default_factory=RigScheimpflugHandeyeSolverConfig
    )
    handeye_ba: RigScheimpflugHandeyeBaConfig = field(
        default_factory=RigScheimpflugHandeyeBaConfig
    )

    def to_payload(self) -> dict[str, Any]:
        return {
            "intrinsics": self.intrinsics.to_payload(),
            "rig": self.rig.to_payload(),
            "handeye_init": self.handeye_init.to_payload(),
            "solver": self.solver.to_payload(),
            "handeye_ba": self.handeye_ba.to_payload(),
        }


@dataclass(slots=True)
class RigScheimpflugHandeyeDataset:
    """Multi-camera Scheimpflug rig hand-eye dataset."""

    num_cameras: int
    views: list[RigHandeyeView]

    def to_payload(self) -> dict[str, Any]:
        return {
            "num_cameras": int(self.num_cameras),
            "views": [view.to_payload() for view in self.views],
        }


@dataclass(slots=True)
class RigScheimpflugHandeyeResult:
    """Result from :func:`vision_calibration.run_rig_scheimpflug_handeye`."""

    cameras: list[PinholeBrownConradyCamera]
    sensors: list[ScheimpflugSensor]
    cam_se3_rig: list[Pose]
    handeye_mode: HandEyeMode
    gripper_se3_rig: Pose | None
    rig_se3_base: Pose | None
    base_se3_target: Pose | None
    gripper_se3_target: Pose | None
    robot_deltas: list[list[float]] | None
    mean_reproj_error: float
    per_cam_reproj_errors: list[float]

    def to_payload(self) -> dict[str, Any]:
        """Convert to Rust/serde shape (``RigScheimpflugHandeyeExport``)."""

        def _pose(p: Pose | None) -> Any:
            return None if p is None else p.to_payload()

        return {
            "cameras": [c.to_payload() for c in self.cameras],
            "sensors": [s.to_payload() for s in self.sensors],
            "cam_se3_rig": [p.to_payload() for p in self.cam_se3_rig],
            "handeye_mode": self.handeye_mode,
            "gripper_se3_rig": _pose(self.gripper_se3_rig),
            "rig_se3_base": _pose(self.rig_se3_base),
            "base_se3_target": _pose(self.base_se3_target),
            "gripper_se3_target": _pose(self.gripper_se3_target),
            "robot_deltas": self.robot_deltas,
            "mean_reproj_error": float(self.mean_reproj_error),
            "per_cam_reproj_errors": [float(v) for v in self.per_cam_reproj_errors],
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "RigScheimpflugHandeyeResult":
        def _pose(name: str) -> Pose | None:
            value = payload.get(name)
            if value is None:
                return None
            return Pose.from_payload(cast(Mapping[str, Any], value))

        return cls(
            cameras=[
                PinholeBrownConradyCamera.from_payload(cast(Mapping[str, Any], c))
                for c in cast(list[Any], payload["cameras"])
            ],
            sensors=[
                ScheimpflugSensor.from_payload(cast(Mapping[str, Any], s))
                for s in cast(list[Any], payload["sensors"])
            ],
            cam_se3_rig=[
                Pose.from_payload(cast(Mapping[str, Any], p))
                for p in cast(list[Any], payload["cam_se3_rig"])
            ],
            handeye_mode=cast(HandEyeMode, payload["handeye_mode"]),
            gripper_se3_rig=_pose("gripper_se3_rig"),
            rig_se3_base=_pose("rig_se3_base"),
            base_se3_target=_pose("base_se3_target"),
            gripper_se3_target=_pose("gripper_se3_target"),
            robot_deltas=cast(list[list[float]] | None, payload.get("robot_deltas")),
            mean_reproj_error=float(payload["mean_reproj_error"]),
            per_cam_reproj_errors=[
                float(v) for v in cast(list[Any], payload["per_cam_reproj_errors"])
            ],
        )


# ─── Rig laserline device ─────────────────────────────────────────────────────


@dataclass(slots=True)
class RigLaserlineUpstreamCalibration:
    """Frozen upstream calibration for rig laserline device.

    Typically derived from a :class:`RigScheimpflugHandeyeResult` export.
    """

    intrinsics: list[dict[str, float]]
    distortion: list[dict[str, Any]]
    sensors: list[dict[str, float]]
    cam_se3_rig: list[Pose]
    rig_se3_target: list[Pose]

    def to_payload(self) -> dict[str, Any]:
        return {
            "intrinsics": self.intrinsics,
            "distortion": self.distortion,
            "sensors": self.sensors,
            "cam_se3_rig": [p.to_payload() for p in self.cam_se3_rig],
            "rig_se3_target": [p.to_payload() for p in self.rig_se3_target],
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "RigLaserlineUpstreamCalibration":
        return cls(
            intrinsics=cast(list[dict[str, float]], payload["intrinsics"]),
            distortion=cast(list[dict[str, Any]], payload["distortion"]),
            sensors=cast(list[dict[str, float]], payload["sensors"]),
            cam_se3_rig=[
                Pose.from_payload(cast(Mapping[str, Any], p))
                for p in cast(list[Any], payload["cam_se3_rig"])
            ],
            rig_se3_target=[
                Pose.from_payload(cast(Mapping[str, Any], p))
                for p in cast(list[Any], payload["rig_se3_target"])
            ],
        )


@dataclass(slots=True)
class RigLaserlineView:
    """One frame in a rig laserline dataset.

    Parameters
    ----------
    cameras:
        Per-camera target correspondences. Use ``None`` for a missing camera.
    laser_pixels:
        Per-camera laser pixel observations. Use ``None`` for a missing camera.
    """

    cameras: list[Observation | None]
    laser_pixels: list[list[Vec2] | None]

    def to_payload(self) -> dict[str, Any]:
        return {
            "cameras": [None if obs is None else obs.to_payload() for obs in self.cameras],
            "laser_pixels": [
                None if px is None else [[u, v] for u, v in px]
                for px in self.laser_pixels
            ],
        }


@dataclass(slots=True)
class RigLaserlineDataset:
    """Multi-camera rig laserline dataset."""

    num_cameras: int
    views: list[RigLaserlineView]

    def to_payload(self) -> dict[str, Any]:
        return {
            "num_cameras": int(self.num_cameras),
            "views": [view.to_payload() for view in self.views],
        }


@dataclass(slots=True)
class RigLaserlineDeviceInput:
    """Input for rig laserline device calibration."""

    dataset: RigLaserlineDataset
    upstream: RigLaserlineUpstreamCalibration
    initial_planes_cam: list[dict[str, Any]] | None = None

    def to_payload(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset.to_payload(),
            "upstream": self.upstream.to_payload(),
            "initial_planes_cam": self.initial_planes_cam,
        }


@dataclass(slots=True)
class RigLaserlineDeviceCalibrationConfig:
    """Configuration for rig laserline device calibration."""

    max_iters: int | None = None
    verbosity: int | None = None
    laser_residual_type: LaserlineResidualType = "LineDistNormalized"

    def to_payload(self) -> dict[str, Any]:
        return {
            "max_iters": self.max_iters,
            "verbosity": self.verbosity,
            "laser_residual_type": self.laser_residual_type,
        }


@dataclass(slots=True)
class RigLaserlineDeviceResult:
    """Result from :func:`vision_calibration.run_rig_laserline_device`."""

    laser_planes_rig: list[LaserlinePlane]
    laser_planes_cam: list[LaserlinePlane]
    per_camera_stats: list[LaserlineStats]

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "RigLaserlineDeviceResult":
        return cls(
            laser_planes_rig=[
                LaserlinePlane.from_payload(cast(Mapping[str, Any], p))
                for p in cast(list[Any], payload["laser_planes_rig"])
            ],
            laser_planes_cam=[
                LaserlinePlane.from_payload(cast(Mapping[str, Any], p))
                for p in cast(list[Any], payload["laser_planes_cam"])
            ],
            per_camera_stats=[
                LaserlineStats.from_payload(cast(Mapping[str, Any], s))
                for s in cast(list[Any], payload["per_camera_stats"])
            ],
        )


def normalize_input_payload(input_value: Any) -> Any:
    """Convert high-level model inputs to serde payloads."""
    return _payload_from_maybe_model(input_value)
